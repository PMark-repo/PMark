from typing import List
import time
import random
import os

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.tokenization_utils import PreTrainedTokenizer
from vllm import SamplingParams
import torch.nn.functional as F

from utils.rand import get_random_vector, get_random_vectors, get_median, secret_sbit, secret_mbit

#########################
# Sentence utils
#########################
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

def get_text_embeddings(texts: List[str], embedder):
    embeddings = embedder.encode(texts, convert_to_tensor=True).cpu()
    return embeddings

def get_cosine_similarities(A, B):
    return F.cosine_similarity(A[:, None, :], B[None, :, :], dim=-1)


def generate_next_sentences_api(
    prompt: str,
    model_name: str = "Qwen2.5-3B",
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_samples: int = 1,
    openai_api_key: str = "EMPTY",
    openai_api_base: str = "http://localhost:8000/v1",
    dedup=True
) -> List[str]:
    """
    Generate a single sentence using vllm api.
    """
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    for i in range(5):
        try:
            completion = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_samples,
                stream=False,
                timeout=60
            )
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    
    generated_sentences = []
    
    # process outputs
    for choice in completion.choices:
        full_text = prompt + choice.text
        sentences = sent_tokenize(full_text)
        prompt_sentences = sent_tokenize(prompt)
        # cleaning
        if len(sentences) > len(prompt_sentences):
            first_new_sentence = sentences[len(prompt_sentences)]
            sentence=first_new_sentence.replace("\n","").replace("“","\"").replace("”","\"").rstrip()
            ending_punctuations = {'.', '?', '!', '\"'}
            if not sentence or sentence[-1] not in ending_punctuations:
                sentence += '.' 
            generated_sentences.append(sentence)
    if dedup:
        generated_sentences = list(set(generated_sentences))
    return generated_sentences

def generate_next_sentences_vllm(
    prompt: str,
    llm,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_samples: int = 1,
    dedup=True
) -> List[str]:
    """
    Generate next sentences using the vLLM's llm.generate method.
    """
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

    generated_sentences = []
    
    for output in outputs:
        for generated_text in output.outputs:
            # cleaning
            full_text = prompt + generated_text.text
            sentences = sent_tokenize(full_text)
            prompt_sentences = sent_tokenize(prompt)
            if len(sentences) > len(prompt_sentences):
                first_new_sentence = sentences[len(prompt_sentences)]
                sentence=first_new_sentence.replace("\n","").replace("“","\"").replace("”","\"").rstrip()
                ending_punctuations = {'.', '?', '!', '\"'}
                if not sentence or sentence[-1] not in ending_punctuations:
                    sentence += '.'
                generated_sentences.append(sentence)
    if dedup:
        generated_sentences = list(set(generated_sentences))
    return generated_sentences


def generate_next_sentences_hf(model, tokenizer, prompt, num_sentences, debug=False, dedup=True):
    text_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    stop_criteria = SentenceEndCriteria(tokenizer)
    stopping_criteria = StoppingCriteriaList([stop_criteria])

    generated_texts=[]
    for i in range(num_sentences):
        stop_criteria.update(prompt)
        outputs = model.generate(
                    text_ids,
                    stopping_criteria=stopping_criteria,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id
                )
        # cleaning
        new_text_ids = outputs
        new_text = tokenizer.decode(new_text_ids[0, text_ids.size(1):-1], skip_special_tokens=True)
        full_text = tokenizer.decode(new_text_ids[0], skip_special_tokens=True)
        sentence=new_text.replace("\n","").replace("“","\"").replace("”","\"").rstrip()
        ending_punctuations = {'.', '?', '!', '\"'}
        if not sentence or sentence[-1] not in ending_punctuations:
            sentence += '.'
        generated_texts.append(new_text)

        if debug:
            print(f"Generated text: {new_text}\nFull text: {full_text}")
            print("-------------")
    if dedup:
        generated_sentences = list(set(generated_sentences))
    return generated_texts


def speculative_sample_next_sentence(embedder, prompt, gen_cossim=None,num_samples=100, debug=False, savefig=None,  pivot="rand", min_seqs=10, sen_id=None, median_method="torch", llm=None, fast_llm=None):
    # generate texts
    start=time.time()
    for i in range(5):
        fast_texts = generate_next_sentences_vllm(prompt=prompt, llm=fast_llm, num_samples=num_samples)
        if len(fast_texts)>min_seqs:# if not enough samples, return None
            break
    end=time.time()
    if len(fast_texts)<min_seqs: # too few samples, end the loop
        return None
    fast_embeddings = get_text_embeddings(fast_texts, embedder)

    if pivot=="rand":# random pivot
        pivot_vec = get_random_vector(fast_embeddings.shape[1], seed=42)
    elif pivot=="mean":# mean pivot
        pivot_vec = torch.nn.functional.normalize(fast_embeddings, dim=0).mean(dim=0).cpu()
    else:
        pivot_vec = get_random_vector(fast_embeddings.shape[1], seed=42)
    fast_cossim = get_cosine_similarities(fast_embeddings, pivot_vec)

    # split into two parts
    fast_median = get_median(fast_cossim, method=median_method)
    rand_flag=secret_sbit[sen_id]

    for i in range(5):
        gen_texts = generate_next_sentences_vllm(prompt=prompt, llm=fast_llm, num_samples=num_samples)
        if len(gen_texts):# if not enough samples, return None
            break
    random.Random(42).shuffle(gen_texts)
    print(f"Generated texts: {len(gen_texts)}")
    gen_embeddings = get_text_embeddings(gen_texts, embedder)
    gen_cossim = get_cosine_similarities(gen_embeddings, pivot_vec)
    gen_samples=[[gen_texts[i], gen_embeddings[i], gen_cossim[i]] for i in range(len(gen_texts))]
    for idx,sample in enumerate(gen_samples):
        if rand_flag and sample[2]>fast_median:
            next_sample=sample
            break
        elif not rand_flag and sample[2]<fast_median:
            next_sample=sample
            break
        else:
            next_sample=None
    if next_sample is None:
        return None
    next_sample={"text":next_sample[0], "cossim":next_sample[2].item(), "rand_flag":rand_flag, "sampled_median":fast_median.item(), "sen_id":sen_id, "gen_cossim":fast_cossim.tolist(),"reject_num":idx }

    if debug:#print debug infomation
        print(f"Generation time: {end-start} s")
        print(f"Generated texts: {gen_texts[:3]}")
        print(f"Pivot vector: {pivot_vec}")
        print(f"Generated cossim: {gen_cossim}")
        print(f"Next sample:\n Text: {next_sample['text']}\n Cossim: {next_sample['cossim']}\n Rand_flag: {next_sample['rand_flag']}\n Median_cossim: {next_sample['sampled_median']}\n Reject_num: {next_sample['reject_num']}")
        print("-------------")

    if savefig:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.kdeplot(fast_cossim, fill = True, label="Fast")
        sns.kdeplot(gen_cossim, fill = True, label="Slow")
        plt.axvline(fast_median, color='r', linestyle='dashed', linewidth=1)
        plt.title("Proxy Value Distribution")
        plt.xlabel("Proxy Value")
        plt.ylabel("Density")
        plt.savefig(savefig)
        plt.close()

    return next_sample



def sample_next_sentence_msignal(
    model, tokenizer, embedder, prompt, num_samples=100, debug=False,
    model_name=None, openai_api_base=None, min_seqs=10, sen_id=None,
    median_method="torch", llm=None, dedup=True, parallel=False
):
    # candidate sampling
    start = time.time()
    for _ in range(5):
        if llm:
            gen_texts = generate_next_sentences_vllm(prompt=prompt, llm=llm, num_samples=num_samples, dedup=dedup)
        elif openai_api_base:
            gen_texts = generate_next_sentences_api(prompt=prompt, num_samples=num_samples, model_name=model_name, openai_api_base=openai_api_base, dedup=dedup)
        else:
            gen_texts = generate_next_sentences_hf(model, tokenizer, prompt, num_samples, dedup=dedup)
        end = time.time()
        if len(gen_texts) > min_seqs:
            break
    print("Generated texts num: ", len(gen_texts))
    if len(gen_texts) == 0:
        return None

    # embedding
    gen_embeddings = get_text_embeddings(gen_texts, embedder)                       # [N, d]
    pivot_vec = get_random_vectors(gen_embeddings.shape[1], len(secret_mbit[0]), seed=42)  # [b, d]
    gen_cossim = get_cosine_similarities(gen_embeddings, pivot_vec)                 # [N, b]
    rand_flags = torch.tensor(secret_mbit[sen_id], dtype=torch.bool)                # [b]

    if not parallel:
        # -------- multi-channel sequential sampling (online) --------
        N, b = gen_cossim.shape
        alive_idx = list(range(N))
        step_medians = []

        for j in range(b):
            if len(alive_idx) == 0:
                break
            vals = gen_cossim[alive_idx, j]
            m_j = get_median(vals, method=median_method)
            step_medians.append(float(m_j))

            if rand_flags[j]:
                keep_mask = (vals >= m_j)
            else:
                keep_mask = (vals < m_j)

            if keep_mask.sum().item() == 0:
                if rand_flags[j]:
                    keep_mask = (vals > m_j)
                else:
                    keep_mask = (vals <= m_j)

            if keep_mask.sum().item() == 0:
                eq_mask = (vals == m_j)
                eq_idx = [alive_idx[k] for k, is_eq in enumerate(eq_mask.tolist()) if is_eq]
                if len(eq_idx) > 0:
                    random.shuffle(eq_idx)
                    half = max(1, len(eq_idx) // 2)
                    chosen = set(eq_idx[:half])
                    keep_mask = torch.tensor([alive_idx[k] in chosen for k in range(len(alive_idx))], dtype=torch.bool)
                else:
                    keep_mask = torch.zeros(len(alive_idx), dtype=torch.bool)
                    keep_mask[random.randrange(len(alive_idx))] = True

            alive_idx = [alive_idx[k] for k, keep in enumerate(keep_mask.tolist()) if keep]

        if len(alive_idx) == 0:
            chosen_i = random.randrange(len(gen_texts))
            select_num = len(gen_texts)
        else:
            chosen_i = random.choice(alive_idx)
            select_num = len(alive_idx)

        b = gen_cossim.shape[1]
        global_medians = torch.tensor([
            get_median(gen_cossim[:, j], method=median_method) for j in range(b)
        ])

        next_sample = {
            "text": gen_texts[chosen_i],
            "cossim": gen_cossim[chosen_i].tolist(),
            "signal": [True]*len(rand_flags),
            "total_num": len(gen_texts),
            "select_num": select_num,
            "rand_flag": rand_flags.tolist(),
            "sampled_median": global_medians.tolist(),   
            "step_medians": step_medians,               
            "sen_id": sen_id,
            "gen_cossim": gen_cossim.tolist()
        }

    else:
        # -------- multi-channel parallel sampling (offline) --------
        median_cossims = torch.tensor([
            get_median(gen_cossim[:, i], method=median_method) for i in range(gen_cossim.shape[1])
        ])
        sample_flags = [sim >= median_cossims for sim in gen_cossim]  # [N] of [b]
        gen_samples = [[gen_texts[i], gen_embeddings[i], gen_cossim[i], torch.eq(sample_flags[i], rand_flags)]
                       for i in range(len(gen_texts))]

        select_num = 0
        next_sample_row = None
        for signal_num in range(len(rand_flags), -1, -1):
            selected = [s for s in gen_samples if abs(torch.sum(s[3]) - signal_num) < 1e-3]
            if len(selected):
                next_sample_row = random.choice(selected)
                select_num = len(selected)
                break

        if next_sample_row is None:
            ridx = random.randrange(len(gen_texts))
            next_sample_row = [gen_texts[ridx], gen_embeddings[ridx], gen_cossim[ridx], torch.zeros_like(rand_flags)]

        next_sample = {
            "text": next_sample_row[0],
            "cossim": next_sample_row[2].tolist(),
            "signal": next_sample_row[3].tolist(),
            "total_num": len(gen_texts),
            "select_num": select_num,
            "rand_flag": rand_flags.tolist(),
            "sampled_median": median_cossims.tolist(),
            "sen_id": sen_id,
            "gen_cossim": gen_cossim.tolist()
        }

    # debug info
    if debug:
        print(f"Generation time: {end - start} s")
        print(f"Generated texts (head): {gen_texts[:3]}")
        print(f"Next sample:\n Text: {next_sample['text']}\n Cossim: {next_sample['cossim']}\n Rand_flag: {next_sample['rand_flag']}")
        if not parallel:
            print(f"Step-wise medians: {next_sample.get('step_medians')}")
        else:
            print(f"Median_cossim: {next_sample['sampled_median']}")
        print("-------------")

    return next_sample
