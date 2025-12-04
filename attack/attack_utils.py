import torch
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM, PegasusTokenizer, PegasusForConditionalGeneration
from evaluation.tools.text_editor import WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution, GPTParaphraser, DipperParaphraser, TextEditor, BackTranslationTextEditor
import gc
from nltk.tokenize import sent_tokenize
from vllm import LLM, SamplingParams

from collections import Counter
import openai
import backoff
import torch
import re
from tqdm import trange
from bert_score import BERTScorer
from parrot import Parrot
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import StoppingCriteria
from nltk.tokenize import sent_tokenize
from string import punctuation
from itertools import groupby


scorer = BERTScorer(model_type = "microsoft/deberta-xlarge-mnli", rescale_with_baseline=True, device="cuda", lang = "en")
def run_bert_score(gen_sents, para_sents):
    P, R, F1 = scorer.score(gen_sents, para_sents)
    return torch.mean(F1).item()

MAX_TRIALS = 100
if torch.cuda.is_available():
    rng = torch.Generator("cuda")
else: 
    rng = torch.Generator("cpu")
hash_key = 15485863
PUNCTS = '!.?'
device = "cuda" if torch.cuda.is_available() else "cpu"


class SentenceEndCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.current_num_sentences = 0

    def update(self, current_text):
        self.current_num_sentences = len(sent_tokenize(current_text))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.size(0) == 1
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return len(sent_tokenize(text)) > self.current_num_sentences + 1


def discard_final_token_in_outputs(outputs):
    # (bz, seqlen)
    return outputs[:, :-1]


def extract_prompt_from_text(text, len_prompt):
    tokens = text.split(' ')
    tokens = tokens[:len_prompt]
    new_text = ' '.join(tokens)
    prompts = []
    for p in PUNCTS:
        idx = new_text.find(p)
        if idx != -1:
            tokens = new_text[:idx + 1].split(" ")
            # has to be greater than a minimum prompt
            if len(tokens) > 3:
                prompts.append(new_text[:idx + 1])
    if len(prompts) == 0:
        prompts.append(new_text + ".")
    # select first (sub)sentence, deliminated by any of PUNCTS
    prompt = list(sorted(prompts, key=lambda x: len(x)))[0]
    return prompt

def gen_sent(model, tokenizer, text_ids, gen_config, stopping_criteria):
    outputs = model.generate(
            # input_ids,
            text_ids,
            gen_config,
            stopping_criteria=stopping_criteria,
        )
    outputs = discard_final_token_in_outputs(outputs)
    new_text_ids = outputs
    new_text = tokenizer.decode(
        new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    # print(new_text, new_text_ids)
    return new_text, new_text_ids

def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return clean_text(sent)

def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)

def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]

stops = []
class SParrot(Parrot):
    def __init__(self, model_tag="../models/parrot_paraphraser_on_T5", use_gpu=True):
        super().__init__(model_tag, use_gpu)
    
    def augment(self, input_phrase, use_gpu=False, diversity_ranker="levenshtein", do_diverse=False, max_return_phrases = 10, max_length=32, adequacy_threshold = 0.90, fluency_threshold = 0.90):
      if use_gpu:
        device= "cuda"
      else:
        device = "cpu"

      self.model     = self.model.to(device)

      import re

      save_phrase = input_phrase
      if len(input_phrase) >= max_length:
         max_length += 32	
			
      input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
      input_phrase = "paraphrase: " + input_phrase
      input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
      input_ids = input_ids.to(device)

      if do_diverse:
        for n in range(2, 9):
          if max_return_phrases % n == 0:
            break          
        preds = self.model.generate(
              input_ids,
              do_sample=False, 
              max_length=max_length, 
              num_beams = max_return_phrases,
              num_beam_groups = n,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=max_return_phrases)
      else: 
        preds = self.model.generate(
                input_ids,
                do_sample=True, 
                max_length=max_length, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=max_return_phrases) 
        

      paraphrases= set()

      for pred in preds:
        gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
        gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
        paraphrases.add(gen_pp)


      adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold, device )
      if len(adequacy_filtered_phrases) == 0 :
        adequacy_filtered_phrases = paraphrases
      fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold, device )
      if len(fluency_filtered_phrases) == 0 :
          fluency_filtered_phrases = adequacy_filtered_phrases
      diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
      para_phrases = []
      for para_phrase, diversity_score in diversity_scored_phrases.items():
          para_phrases.append((para_phrase, diversity_score))
      para_phrases.sort(key=lambda x:x[1], reverse=True)
      para_phrases = [x[0] for x in para_phrases]
      return para_phrases


def tokenize(tokenizer, text):
    return tokenizer(text, return_tensors='pt').input_ids[0].to(device)

def build_bigrams(input_ids):
    bigrams = []
    for i in range(len(input_ids) - 1):
        bigram = tuple(input_ids[i:i+2].tolist())
        bigrams.append(bigram)
    return bigrams

def compare_ngram_overlap(input_ngram, para_ngram):
    input_c = Counter(input_ngram)
    para_c = Counter(para_ngram)
    intersection = list(input_c.keys() & para_c.keys())
    overlap = 0
    for i in intersection:
        overlap += para_c[i]
    return overlap

def accept_by_bigram_overlap(sent, para_sents, tokenizer, bert_threshold = 0.03):
    input_ids = tokenize(tokenizer, sent)
    input_bigram = build_bigrams(input_ids)
    para_ids = [tokenize(tokenizer, para) for para in para_sents]
    para_bigrams = [build_bigrams(para_id) for para_id in para_ids]
    min_overlap = len(input_ids)

    bert_scores = [run_bert_score([sent], [para_sent]) for para_sent in para_sents]
    max_score = bert_scores[0]
    best_paraphrased = para_sents[0]
    score_threshold = bert_threshold * max_score
    for i in range(len(para_bigrams)):
        para_bigram = para_bigrams[i]
        overlap = compare_ngram_overlap(input_bigram, para_bigram)
        bert_score = bert_scores[i]
        diff = max_score - bert_score
        if overlap < min_overlap and len(para_ids[i]) <= 1.5 * len(input_ids) and (diff <= score_threshold):
            min_overlap = overlap
            best_paraphrased = para_sents[i]
    return best_paraphrased

def gen_prompt(sent, context):
  prompt = f'''Previous context: {context} \n Current sentence to paraphrase: {sent}'''
  return prompt

def gen_bigram_prompt(sent, context, num_beams):
  prompt = f'''Previous context: {context} \n Paraphrase in {num_beams} different ways and return a numbered list : {sent}'''
  return prompt
  
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai(client, prompt):
  while True:
    try:
      response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except openai.APIError:
      continue
    break
  return response.choices[0].message.content

# use long context
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai_bigram(client, prompt):
  while True:
    try:
      response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except openai.APIError:
      continue
    break
  return response.choices[0].message.content

# pick the paraphrases by openai
def pick_para(sent_list, tokenizer, all_paras, thres):
    # all_paras is a list shape: num of texts X num of paraphrases X number of beams
    data_len, para_texts, para_texts_bigram = [], [], [], []
    data_len = [len(t) for t in sent_list]

    for i in trange(len(sent_list), desc = "Picking paraphrases"):
        sents = sent_list[i]
        for j in range(len(sents)):
            sent = sents[j]
            # each sent has num_beams paraphrases
            paraphrases = all_paras[i][j] # all beams
            para = accept_by_bigram_overlap(sent, paraphrases, tokenizer, bert_threshold=thres)
            para_texts_bigram.append(para)
            para_texts.append(paraphrases[0])
    output_no_bigram = []
    output_bigram = []
    # new_texts = []
    start_pos = 0
    for l in data_len:
        output_no_bigram.append(para_texts[start_pos: start_pos+l])
        output_bigram.append(para_texts_bigram[start_pos: start_pos+l])
        start_pos+=l
    return output_no_bigram, output_bigram

class PegasusParaphraseConfig:
    model_path: str = "../models/pegasus_paraphrase"
    device: str = "auto" 
    num_beams: int = 10
    temperature: float = 2.0
    max_length: int = 60
    batch_size: int = 0 
    use_bigram_filter: bool = True
    bert_threshold: float = 0.03
    bigram_tokenizer_name: str="../models/opt-1.3b"

class PegasusParaphrase(TextEditor):
    def __init__(self, cfg: PegasusParaphraseConfig):
        self.cfg = cfg
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = PegasusForConditionalGeneration.from_pretrained(cfg.model_path).to(self.device)
        self.tokenizer = PegasusTokenizer.from_pretrained(cfg.model_path)
        self.cfg.bigram_tokenizer=AutoTokenizer.from_pretrained(cfg.bigram_tokenizer_name)

    def _generate_beams(self, sents):
        batch = self.tokenizer(
            sents, truncation=True, padding="longest", return_tensors="pt", max_length=self.cfg.max_length
        ).to(self.device)

        ids = self.model.generate(
            **batch,
            max_length=self.cfg.max_length,
            num_beams=self.cfg.num_beams,
            num_return_sequences=self.cfg.num_beams,
            repetition_penalty=1.03,
        )

        beams_per_sent = []
        for i in range(len(sents)):
            cand_ids = ids[i * self.cfg.num_beams : (i + 1) * self.cfg.num_beams]
            decoded = [self.tokenizer.decode(x, skip_special_tokens=True) for x in cand_ids]
            decoded = [well_formed_sentence(d) for d in decoded]
            beams_per_sent.append(decoded)
        return beams_per_sent

    def edit(self, text: str, reference=None) -> str:
        text = " ".join(text.split())
        sents = sent_tokenize(text)
        if not sents:
            return text

        outputs = []

        if self.cfg.batch_size and self.cfg.batch_size > 0:
            for i in range(0, len(sents), self.cfg.batch_size):
                chunk = sents[i : i + self.cfg.batch_size]
                beams_list = self._generate_beams(chunk)
                for src, beams in zip(chunk, beams_list):
                    if self.cfg.use_bigram_filter:
                        best = accept_by_bigram_overlap(
                            src, beams, self.cfg.bigram_tokenizer, self.cfg.bert_threshold
                        )
                        outputs.append(best)
                    else:
                        outputs.append(beams[0])
        else:
            for src in sents:
                beams = self._generate_beams([src])[0]
                if self.cfg.use_bigram_filter:
                    best = accept_by_bigram_overlap(
                        src, beams, self.cfg.bigram_tokenizer, self.cfg.bert_threshold
                    )
                    outputs.append(best)
                else:
                    outputs.append(beams[0])

        return " ".join(outputs)


class ParrotParaphraseConfig:
    use_gpu: bool = True
    num_beams: int = 10
    max_length: int = 128
    adequacy_threshold: float = 0.8
    fluency_threshold: float = 0.8
    use_bigram_filter: bool = True
    bert_threshold: float = 0.03
    bigram_tokenizer_name: str="../models/opt-1.3b"

class ParrotParaphrase(TextEditor):
    def __init__(self, cfg: ParrotParaphraseConfig, parrot):
        self.cfg = cfg
        self.parrot = parrot if parrot is not None else SParrot()
        self.cfg.bigram_tokenizer=AutoTokenizer.from_pretrained(self.cfg.bigram_tokenizer_name)

    def _normalize_candidates(self, cands):
        out = []
        for c in cands:
            if isinstance(c, tuple) and len(c) > 0:
                out.append(str(c[0]))
            else:
                out.append(str(c))
        return [well_formed_sentence(p, end_sent=True) for p in out]

    def edit(self, text: str, reference=None) -> str:
        text = " ".join(text.split())
        sents = sent_tokenize(text)
        if not sents:
            return text

        outputs = []

        for src in sents:
            cands = self.parrot.augment(
                input_phrase=src,
                use_gpu=self.cfg.use_gpu,
                diversity_ranker="levenshtein",
                do_diverse=True,
                max_return_phrases=self.cfg.num_beams,
                max_length=self.cfg.max_length,
                adequacy_threshold=self.cfg.adequacy_threshold,
                fluency_threshold=self.cfg.fluency_threshold,
            )
            beams = self._normalize_candidates(cands)
            if not beams:
                outputs.append(src)
                continue

            if self.cfg.use_bigram_filter:
                best = accept_by_bigram_overlap(
                    src, beams, self.cfg.bigram_tokenizer, bert_threshold=self.cfg.bert_threshold
                )
                outputs.append(best)
            else:
                outputs.append(beams[0])

        return " ".join(outputs)
    
class HFTranslator:
    def __init__(self, source_lang, target_lang, llm=None):
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, text):
        input_tokens = self.tokenizer(text, return_tensors="pt").input_ids
        input_length = input_tokens.shape[1]
        max_new_tokens = 256
        _min_new_tokens = int(input_length * 1.8)

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.2,
            top_p=1.0,
            stop=["\nUSER:", "USER:"]
        )

        prompt = f"USER: Translate the following text from {self.source_lang} to {self.target_lang}: {text}\nASSISTANT: "
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        translated_text = generated_text.split("\n")[0].replace("1. ","")

        print("Generated text:", generated_text)
        print("-------------")
        print("Translated text:", translated_text)
        print("*************")

        return translated_text



def get_attacker(attack_name,ratio,device="cuda"):
    if attack_name == 'Word-D':
      attack = WordDeletion(ratio=ratio)
    elif attack_name == 'Word-S':
      attack = SynonymSubstitution(ratio=ratio)
    elif attack_name == 'Word-S(Context)':
      attack = ContextAwareSynonymSubstitution(ratio=ratio,
                                               tokenizer=BertTokenizer.from_pretrained('../models/bert-large-uncased'),
                                               model=BertForMaskedLM.from_pretrained('../models/bert-large-uncased').to(device))
    elif attack_name=="Doc-T":
        shared_llm = LLM(model="../models/Meta-Llama-3.1-8B",
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="float16")

        en2es = HFTranslator("English", "Spanish", llm=shared_llm)
        es2en = HFTranslator("Spanish", "English", llm=shared_llm)
        attack = BackTranslationTextEditor(en2es.translate, es2en.translate)
    elif attack_name == 'Doc-P(GPT)':
        attack = GPTParaphraser(openai_model='gpt-3.5-turbo',
                                prompt='Please rewrite the following text: ')
    elif attack_name == 'Doc-P(Dipper)':
        attack = DipperParaphraser(tokenizer=T5Tokenizer.from_pretrained('../models/t5-v1_1-xxl/'),
                                   model=T5ForConditionalGeneration.from_pretrained('../models/dipper-paraphraser-xxl').to(device),
                                   lex_diversity=60, order_diversity=0, sent_interval=1,
                                   max_new_tokens=100, do_sample=True, top_p=0.75, top_k=None)
    elif attack_name == 'Doc-P(Pegasus)':
        attack = PegasusParaphrase(cfg=PegasusParaphraseConfig())
    elif attack_name == 'Doc-P(Parrot)':
        attack = ParrotParaphrase(cfg=ParrotParaphraseConfig(),parrot=SParrot())
    return attack


if __name__ == '__main__':
    attack=get_attacker('Doc-P(Parrot)',0.3)
    res=attack.edit("Hello world!")
    print(res)