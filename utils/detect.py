import torch
import numpy as np
from scipy import stats
import math

from utils.sample import get_cosine_similarities, get_text_embeddings,  generate_next_sentences_vllm, generate_next_sentences_hf
from utils.rand import secret_sbit, get_median, get_random_vector, get_random_vectors, secret_mbit

def within_green(curr_sen, prompt, rand_flag, llm, embedder, num_samples, api_base=None, model_name=None, pivot="rand", median_method="torch", debug=False, dedup=False):
    for _ in range(3):
        sampled_texts = generate_next_sentences_vllm(prompt=prompt, llm=llm, num_samples=num_samples, dedup=dedup)
        if len(sampled_texts)>0:
            break
    if len(sampled_texts)==0:
        return {"within_green":True,"rand_flag":rand_flag,"sampled_median":None,"curr_cossim":None, "detect_cossim":None}
    sampled_embeddings = get_text_embeddings(sampled_texts, embedder)
    if pivot=="rand":# random pivot
        pivot_vec = get_random_vector(sampled_embeddings.shape[1], seed=42)
    elif pivot=="mean":
        pivot_vec = torch.nn.functional.normalize(sampled_embeddings, dim=0).mean(dim=0)
    sampled_cossim = get_cosine_similarities(sampled_embeddings, pivot_vec)
    try:
        sampled_median=get_median(sampled_cossim, method=median_method)
    except:
        sampled_median=get_median(sampled_cossim, method="torch")

    curr_embedding=get_text_embeddings([curr_sen], embedder)
    curr_cossim = get_cosine_similarities(curr_embedding, pivot_vec)[0]
    if rand_flag and curr_cossim>=sampled_median-1e-4:#numerical stability
        within_green=True
    elif not rand_flag and curr_cossim<=sampled_median+1e-4:
        within_green=True
    else:
        within_green=False
    if debug:
        print(f"Prompt: {prompt}")
        print(f"rand_flag: {rand_flag}")
        print(f"curr_cossim: {curr_cossim}")
        print(f"sampled_median: {sampled_median}")
        print(f"within_green: {within_green}")
    
    res={"within_green":[within_green],"rand_flag":rand_flag,"sampled_median":sampled_median.item(),"curr_cossim":curr_cossim.item(), "detect_cossim":sampled_cossim.tolist()}
    return res

def within_green_msig(curr_sen, prompt, rand_flags, llm, embedder, num_samples, median_method="torch",K=250,threshold=0.001, debug=False, dedup=False, min_seqs=10,model=None, tokenizer=None):
    if median_method!="prior":
        for _ in range(3):
            if model:
                sampled_texts = generate_next_sentences_hf(prompt=prompt, model=model,tokenizer=tokenizer, num_sentences=num_samples, dedup=dedup)
            else:
                sampled_texts = generate_next_sentences_vllm(prompt=prompt, llm=llm, num_samples=num_samples, dedup=dedup)
            if len(sampled_texts)>min_seqs:
                break
        if len(sampled_texts)==0:
            return {"within_green":[True]*len(rand_flags),"rand_flag":rand_flags,"sampled_median":None,"curr_cossim":None, "detect_cossim":None}
        sampled_embeddings = get_text_embeddings(sampled_texts, embedder)
        pivot_vec = get_random_vectors(sampled_embeddings.shape[1],len(rand_flags))
        sampled_cossim = get_cosine_similarities(sampled_embeddings, pivot_vec)
        sampled_median=[]
        for i in range(len(rand_flags)):
            try:
                sampled_median.append(get_median(sampled_cossim.T[i], method=median_method))
            except:
                sampled_median.append(get_median(sampled_cossim.T[i], method="torch"))
        sampled_median = torch.tensor(sampled_median)
    else:
        sampled_median=torch.tensor([0]*len(rand_flags))

    curr_embedding=get_text_embeddings([curr_sen], embedder)
    pivot_vec = get_random_vectors(curr_embedding.shape[-1],len(rand_flags))
    curr_cossim = get_cosine_similarities(curr_embedding, pivot_vec)[0]#[1,4]
    sampled_up=(curr_cossim>sampled_median)
    
    # soft counting
    within_green=[]
    for sm, cs, rf in zip(sampled_median, curr_cossim, rand_flags):
        if cs is None:
            ing = 1.0
        elif rf and cs >= sm - threshold:
            ing = 1.0
        elif (not rf) and cs <= sm + threshold:
            ing = 1.0
        else:
            ing = math.exp(-K * abs(cs - sm)) if K is not None else 0.0
        within_green.append(float(ing))

    if debug:
        print(f"Prompt: {prompt}")
        print(f"rand_flag: {rand_flags}")
        print(f"curr_cossim: {curr_cossim}")
        print(f"sampled_median: {sampled_median}")
        print(f"within_green: {within_green}")
    
    res={"within_green":within_green,"rand_flags":rand_flags,"sampled_median":sampled_median.tolist(),"curr_cossim":curr_cossim.tolist(), "sampled_up":sampled_up.tolist()}
    return res

def watermark_z_test(within_greens, alpha=0.05, debug=False):
    """
    z-test for watermark detection
    """
    n = len(within_greens)
    x = sum(within_greens)
    if debug:
        print(f"Total Sentences: {n}")
        print(f"Sentences within green: {x}")
    p0 = 0.5
    p_hat = x / n
    standard_error = np.sqrt(p0 * (1 - p0) / n)
    z_score = (p_hat - p0) / standard_error
    p_value = 1 - stats.norm.cdf(z_score)
    is_watermarked = p_value < alpha
    return is_watermarked.item(), z_score.item(), p_value.item()


def detect_paragraph(sentences, num_samples=None, api_base=None, model_name=None, llm=None, embedder=None, pivot=None, median_method=None, debug=False, dedup=False, msig=False, model=None, tokenizer=None):
    """
    Detect watermark in a paragraph
    """
    within_greens = []
    detect_logs=[]
    for i in range(1, len(sentences)):
        curr_sen = sentences[i]
        prompt = " ".join(sentences[:i])
        if msig<=1:
            rand_flag=secret_sbit[i]
            res = within_green(curr_sen=sentences[i], prompt=prompt, rand_flag=rand_flag, llm=llm, embedder=embedder, num_samples=num_samples, api_base=api_base, model_name=model_name, pivot=pivot, median_method=median_method, debug= False, dedup=dedup)
        else:
            rand_flags=secret_mbit[i]
            res = within_green_msig(curr_sen=sentences[i], prompt=prompt, rand_flags=rand_flags, llm=llm, embedder=embedder, num_samples=num_samples, median_method=median_method, debug= False, dedup=dedup, model=model, tokenizer=tokenizer)
        det_log={"curr_sen":curr_sen,"prompt":prompt,**res}
        within_greens.extend(res['within_green'])
        detect_logs.append(det_log)
    try:
        is_watermarked, z_score, p_value = watermark_z_test(within_greens, debug=debug)
    except:
        is_watermarked, z_score, p_value = False, 0.0, 1.0
    res={"within_greens":within_greens,"is_watermarked":is_watermarked,"z_score":z_score,"p_value":p_value, "detect_logs":detect_logs}
    return res