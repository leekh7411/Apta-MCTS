import io
import os
import json
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict
import RNA
import pandas as pd

def shuffle2(a,b):
    s = np.arange(len(a))
    np.random.shuffle(s)
    return a[s], b[s]

def shuffle3(a,b,c):
    s = np.arange(len(a))
    np.random.shuffle(s)
    return a[s], b[s], c[s]

def shuffle4(a,b,c,d):
    s = np.arange(len(a))
    np.random.shuffle(s)
    return a[s], b[s], c[s], d[s]

# Improved CTF 
class improvedCTF:
    def __init__(self, letters, length):
        self.letters = letters
        self.length = length
        self.dict = {}
        self.generate_feature_dict()
        
    def generate_feature_dict(self):
        def generate(cur_key, depth):
            if depth == self.length:
                return
            for k in self.letters:
                next_key = cur_key + k
                self.dict[next_key] = 0
                generate(next_key, depth+1)
                
        generate(cur_key="",depth=0)
        
        #print("iterate letters : {}".format(self.letters))
        #print("number of keys  : {}".format(len(self.dict.keys())))
        
    
    def get_feature_dict(self):
        for k in self.dict.keys():
            self.dict[k] = 0
        
        self.dict = OrderedDict(sorted(self.dict.items()))
        return deepcopy(self.dict)
    
    def get_feature_dict_keys(self):
        for k in self.dict.keys():
            self.dict[k] = 0
        
        self.dict = OrderedDict(sorted(self.dict.items()))
        keys = list(self.dict.keys())
        return deepcopy(keys)
    
def get_reduced_protein_letter_dict():
    rpdict = {}
    """
    ["[R]","[H]","[K]","[D]","[E]","[S]","[T]",
            "[N]","[Q]","[C]","U","[G]","[P]","[A]",
            "[V]","[I]","[L]","[M]","[F]","[Y]","[W]"]
            
    """
    reduced_letters = [["A","G","V"],
                       ["I","L","F","P"],
                       ["Y","M","T","S"],
                       ["H","N","Q","W"],
                       ["R","K"],
                       ["D","E"],
                       ["C"]]
    changed_letter = ["A","B","C","D","E","F","G"]
    for class_idx, class_letters in enumerate(reduced_letters):
        for letter in class_letters:
            rpdict[letter] = changed_letter[class_idx]
    
    return rpdict

def get_reduced_protein_letter_defaultdict():
    rpdict = {}
    """
    ["[R]","[H]","[K]","[D]","[E]","[S]","[T]",
            "[N]","[Q]","[C]","U","[G]","[P]","[A]",
            "[V]","[I]","[L]","[M]","[F]","[Y]","[W]"]
            
    """
    reduced_letters = [["A","G","V"],
                       ["I","L","F","P"],
                       ["Y","M","T","S"],
                       ["H","N","Q","W"],
                       ["R","K"],
                       ["D","E"],
                       ["C"]]
    changed_letter = ["A","B","C","D","E","F","G"]
    for class_idx, class_letters in enumerate(reduced_letters):
        for letter in class_letters:
            rpdict[letter] = changed_letter[class_idx]
    
    return defaultdict(lambda: "X", rpdict)


def get_reduced_protein_seqs(p_seqs):
    rpdict = get_reduced_protein_letter_dict()
    reduced_pseqs = []
    for seq in p_seqs:
        r_seq = ""
        for s in seq:
            try: rs = rpdict[s]
            except: rs = s
            r_seq += rs
            
        reduced_pseqs.append(r_seq)
    return reduced_pseqs

def rna2feature_iCTF(seq_list, k=4):
    r_mer = k
    r_CTF = improvedCTF(letters=["A","C","G","U"], length= r_mer)
    r_features = []
    for seq in seq_list:
        seq = seq.upper()
        seq = seq.replace("T","U")
        r_feature_dict = r_CTF.get_feature_dict()
        seq_len = len(seq)
        for mer in range(1,r_mer+1):
            for i in range(0,len(seq)-mer):
                pattern = seq[i:i+mer]
                try:
                    r_feature_dict[pattern] += 1
                except:
                    continue
        
        r_feature = np.array(list(r_feature_dict.values()))
        r_feature = r_feature / seq_len
        r_features.append(r_feature)
        
    r_features = np.array(r_features)
        
    return r_features

def rss2feature_iCTF(seq_list, k=4):
    r_mer = k
    r_CTF = improvedCTF(letters=["(",".",")"], length= r_mer)
    r_features = []
    for seq in seq_list:
        r_feature_dict = r_CTF.get_feature_dict()
        ss, mfe = RNA.fold(seq)
        seq_len = len(ss)
        for mer in range(1,r_mer+1):
            for i in range(0,len(ss)-mer):
                pattern = ss[i:i+mer]
                try:
                    r_feature_dict[pattern] += 1
                except:
                    continue
        
        r_feature = np.array(list(r_feature_dict.values()))
        r_feature = r_feature / seq_len
        r_feature = np.array(list(r_feature) + [mfe])
        r_features.append(r_feature)
        
    r_features = np.array(r_features)
        
    return r_features


def pro2feature_iCTF(seq_list, k=3):
    rpdict = get_reduced_protein_letter_dict()
    p_mer = k
    
    p_CTF = improvedCTF(letters=["A","B","C","D","E","F","G"],length=p_mer)
    
    p_features = []
    for seq in seq_list:
        seq_len = len(seq)
        p_feature_dict = p_CTF.get_feature_dict()
        rpseq = []
        for p in seq:
            if p=="X": rpseq.append(p)
            else:rpseq.append(rpdict[p])
        
        pseq = rpseq
        temp_pseq = ""
        for p in pseq:
            temp_pseq += p
        pseq = temp_pseq
        
        for mer in range(1,p_mer+1):
            for i in range(0,len(pseq)-mer):
                pattern = pseq[i:i+mer]
                try:
                    p_feature_dict[pattern] += 1
                except:
                    continue
        
        p_feature = np.array(list(p_feature_dict.values()))
        p_feature = p_feature / seq_len
        p_features.append(p_feature)
       
    p_features = np.array(p_features)
    
    return p_features

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    with io.open(fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj


def preprocess_class(labels):
    _labels = []
    for label in labels:
        if label == "positive":
            _labels.append(1)
        else:
            _labels.append(0)
    return np.array(_labels)

def load_benchmark_dataset(path):
    #df = pd.read_csv(path)
    #pseqs  = list(df["protein"])
    #rseqs  = list(df["rna"])
    #labels = list(df["class"])
    
    d = load_json(path)
    pseqs  = d["protein-seq"]
    rseqs  = d["rna-aptamer-seq"]
    labels = d["label"]
    
    px = pro2feature_iCTF(pseqs, k=3)
    rx = rna2feature_iCTF(rseqs, k=4)
    labels = preprocess_class(labels)
    print("> Benchmark        : {}".format(path.split('/')[-1]))
    print("- protein features : {}".format(px.shape))
    print("- rna features     : {}".format(rx.shape))
    print("- labels           : {}".format(labels.shape))
    px, rx, labels = shuffle3(px,rx,labels)
    return px, rx, labels

def load_docking_benchmark_dataset(path):
    #df = pd.read_csv(path)
    #pseqs  = list(df["protein"])
    #rseqs  = list(df["rna"])
    
    d = load_json(path)
    pseqs = d["protein-seq"]
    rseqs = d["rna-aptamer-seq"]
    px = pro2feature_iCTF(pseqs, k=3)
    rx = rna2feature_iCTF(rseqs, k=4)
    print("> Benchmark        : {}".format(path.split('/')[-1]))
    print("- protein features : {}".format(px.shape))
    print("- rna features     : {}".format(rx.shape))
    return pseqs, rseqs, px, rx

