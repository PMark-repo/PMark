import json 
import os
import argparse
from tqdm import tqdm
from collections import Counter
from itertools import groupby
import re
from attack_utils import get_attacker


if __name__ == "__main__":      
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../PMark/c4_logs/opt-1.3b_log")
    parser.add_argument("--sub_dir", type=str, default="all-mpnet-base-v2_hd5")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--attack_name", type=str, default="Doc-P(Pegasus)")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--prompt_attack", type=bool, default=False)
    args = parser.parse_args()

    log_dir=args.log_dir
    sub_dir=args.sub_dir
    attack_name=args.attack_name
    attack=get_attacker(attack_name, args.ratio, "cuda")
    if args.prompt_attack:
        attack_name=f"{attack_name}{args.ratio}-{int(args.prompt_attack)}"
    else:
        attack_name=f"{attack_name}{args.ratio}"
    print(args.start,args.end)
    for i in tqdm(range(args.start,args.end)):
        json_path=f"{log_dir}/{sub_dir}/{i}.json"
        if not os.path.exists(json_path):
            print(f"Sample {i} not found!")
            continue
        if os.path.exists(f"{log_dir}/{sub_dir}/attack/{attack_name}/{i}.json"):
            print(f"Sample {i} already attacked!")
            continue
        with open(json_path,"r",encoding="utf-8") as f:
            data=json.load(f)
        log=data["log"]
        text=data['generated_text']
        texts=[item['text'] for item in log]
        attacked_texts=[attack.edit(texts[j]) for j in range(len(texts))]
        if args.prompt_attack:# prompt attacked
            prompt_attacked=attack.edit(data['prompt'])
        new_log=[]
        for j,item in enumerate(log):
            item['gen_text']=item['text']
            item["gen_cossim"]=item['cossim']
            item['gen_median']=item['sampled_median']
            item.pop("text")
            item.pop("cossim")
            item.pop("sampled_median")
            item['text']=attacked_texts[j]
            new_log.append(item)
        paraphrased_text=" "+" ".join(attacked_texts)
        data["log"]=new_log
        data["generated_text"]=paraphrased_text
        print(data["generated_text"])
        para_path=f"{log_dir}/{sub_dir}/attack/{attack_name}/{i}.json"
        os.makedirs(os.path.dirname(para_path), exist_ok=True)
        with open(para_path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=4)