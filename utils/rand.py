import os
import time
import random
import hashlib
import json
import torch
import numpy as np 
from typing import List
from scipy.stats import beta, gaussian_kde, mstats
import copy

#########################
# Random utils
#########################
with open("secret_mbit.json",'r') as f:
    secret_sbit = json.load(f)

class SecretBits:
    def __init__(self):
        with open("secret_mbit.json",'r') as f:
            self.data = json.load(f)
            self.signum=1
            self.secret_bits=copy.deepcopy(self.data)
            
    def set_signum(self, signum):
        self.signum=signum
        self.secret_bits=[]

        bits=self.data[:(len(self.data)//self.signum)*self.signum]
        for i in range(0,len(bits),self.signum):
            self.secret_bits.append(bits[i:i+self.signum])
        return self.secret_bits
    
    def __getitem__(self, indices):
        return self.secret_bits[indices]

    def __len__(self):
        return len(self.secret_bits)
    
    def __iter__(self):
        return iter(self.secret_bits)

secret_mbit=SecretBits()

def set_random_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_random_vector(dim, seed=42):
    """
    Returns a random vector of the given dimension.
    """
    set_random_seed(seed)
    return torch.rand(dim)

def get_random_vectors(dim, num, seed=42):
    """
    Returns a random vector of the given dimension.
    """
    rng = np.random.default_rng(seed) 
    A = rng.standard_normal((dim, num))
    Q, R = np.linalg.qr(A)  

    # G = Q.T @ Q 
    # print(G)
    return torch.tensor(Q.T)

def get_random_flag(seed=42):
    """
    Returns a random bool var.
    """
    set_random_seed(seed)
    return random.choice([True, False])

def extract_seed_from_str(s:str, mod:int = 1000000000):
    """
    Extract the seed from the given str through hashing.
    """
    s=s.replace(" ","")
    s_encoded = s.encode('utf-8')
    hash_object = hashlib.sha256(s_encoded)
    hash_digest = hash_object.hexdigest()
    seed_value = int(hash_digest, 16)
    if mod:
        seed_value = seed_value % mod
    return seed_value

def kde_median(x, bandwidth=None):
    x = np.asarray(x)
    kde = gaussian_kde(x, bw_method=bandwidth)
    
    grid = np.linspace(np.min(x), np.max(x), 10000)
    cdf_vals = np.cumsum(kde(grid))
    cdf_vals /= cdf_vals[-1]
    
    median_estimate = grid[np.argmin(np.abs(cdf_vals - 0.5))]
    return median_estimate

# https://aakinshin.net/posts/sthdme/
def harrell_davis_median(x, q=0.5): 
    x = np.sort(np.asarray(x))
    n = len(x)
    i = np.arange(1, n+1)
    a = (n + 1) * q
    b = (n + 1) * (1 - q)

    w = beta.cdf(i / n, a, b) - beta.cdf((i - 1) / n, a, b)
    return np.sum(w * x)

def get_median(dist, method="torch"):
    if len(dist)==1:
        return dist[0]
    if method=="torch":
        res=torch.median(dist)
    elif method=="kde":
        res=kde_median(dist)
    elif method=="hd":
        res=harrell_davis_median(dist)
    elif method=="prior":
        res=torch.tensor(0)
    else:
        raise ValueError(f"Unknown median estimation method: {method}")
    return res

