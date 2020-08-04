import os
import fire
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Manager, Queue
   

def aptamer_candidate_parser(path):
    f = open(path, "r")
    lines = f.readlines()
    cur_vals = []
    for line in lines:
        line = line.replace("\n","")
        if line[0] == ">":
            cur_key = line[2:]
        else:
            cur_vals.append((cur_key, line))
    _dict = defaultdict(lambda: [])
    
    for k,v in cur_vals:
        _dict[k].append(v)
    
    return _dict


def load_aptamers_of_testset(model_tag, num_pairs):
    base_dir = "aptamers/{}/output-{:02d}.txt"
    sequence_tag = "APTAMER-{}"
    candidates = []
    for i in range(num_pairs):
        cand_file_path = "aptamers/{}/output-{:02d}.txt".format(model_tag, i)
        cand_file_dict = aptamer_candidate_parser(cand_file_path)
        seq, ss, mfe = cand_file_dict["RNA-Aptamer"]
        candidates.append((sequence_tag.format(i), 0, seq, ss, i, 0))
    return candidates


def load_aptamers(model_tag, num_pairs, top_k):
    base_dir = "aptamers/{}/output-{:02d}.txt"
    sequence_tag = "{}-{}-{}"
    candidates = []
    for i in range(num_pairs):
        cand_file_path = "aptamers/{}/output-{:02d}.txt".format(model_tag, i)
        cand_file_dict = aptamer_candidate_parser(cand_file_path)
        for j, cand_vals in enumerate(cand_file_dict["Predicted RNA-Aptamer Candidates"]):
            if j == top_k: break
            seq, ss, mfe, score = cand_vals.split("\t")
            seq, ss, mfe, score = seq, ss, float(mfe), float(score)
            seq_tag = sequence_tag.format(model_tag, i, j)
            candidates.append((seq_tag, score, seq, ss, i, j))
    return candidates


def modeling_RNA_structure_using_SimRNA(seq, ss, tag):
    out_dirs = "/tmp/SIMRNA/outputs/{}".format(tag)
    if not os.path.exists(out_dirs):
        os.makedirs(out_dirs)
        
    SimRNA_command = "./SimRNA.sh {} '{}' '{}'".format(tag, seq, ss)
    print(SimRNA_command)
    os.system(SimRNA_command)


def main(model_tag, num_pairs, top_k):
    if model_tag == "APTAMER":
        candidates = load_aptamers_of_testset("RAND_A", num_pairs) # RAND-A or anything else..
    else:
        candidates = load_aptamers(model_tag, num_pairs, top_k)
        
    for c in candidates:
        cand_tag, score, seq, ss, pid, rank = c
        print("> simrna-modeling / tag: {} / score: {} / pid: {} / rank: {}".format(cand_tag, score, pid, rank))
        modeling_RNA_structure_using_SimRNA(seq, ss, cand_tag)
        
        
def __experiments_modeling_all_structures():
    candidates = []
    candidates += load_aptamers_of_testset("RAND_A", 56) # RAND-A or anything else..
    candidates += load_aptamers("RAND_A", 56, 10)
    candidates += load_aptamers("RAND_B", 56, 10)
    candidates += load_aptamers("MCTS-A-0-100", 56, 10)
#     candidates += load_aptamers("MCTS-A-10-100", 56, 10)
    candidates += load_aptamers("MCTS-A-20-100", 56, 10)
    candidates += load_aptamers("MCTS-A-30-100", 56, 10)
#     candidates += load_aptamers("MCTS-A-40-100", 56, 10)
    candidates += load_aptamers("MCTS-A-50-100", 56, 10)
#     candidates += load_aptamers("MCTS-A-60-100", 56, 10)
    candidates += load_aptamers("MCTS-A-70-100", 56, 10)
#     candidates += load_aptamers("MCTS-A-80-100", 56, 10)
#     candidates += load_aptamers("MCTS-A-90-100", 56, 10)
#     candidates += load_aptamers("MCTS-A-100-100", 56, 10)
    candidates += load_aptamers("MCTS-B-0-100", 56, 10)
#     candidates += load_aptamers("MCTS-B-10-100", 56, 10)
    candidates += load_aptamers("MCTS-B-20-100", 56, 10)
    candidates += load_aptamers("MCTS-B-30-100", 56, 10)
#     candidates += load_aptamers("MCTS-B-40-100", 56, 10)
    candidates += load_aptamers("MCTS-B-50-100", 56, 10)
#     candidates += load_aptamers("MCTS-B-60-100", 56, 10)
    candidates += load_aptamers("MCTS-B-70-100", 56, 10)
#     candidates += load_aptamers("MCTS-B-80-100", 56, 10)
#     candidates += load_aptamers("MCTS-B-90-100", 56, 10)
#     candidates += load_aptamers("MCTS-B-100-100", 56, 10)
    random.shuffle(candidates)
    
    n_process = 30
    
    with Manager() as manager:
        
        print("- Predict RNA 3D structure using SimRNA with batch-multi-processing")
        print("- number of processes : {}".format(n_process))
    
        for i in range(0, len(candidates), n_process):
            if i+n_process < len(candidates):
                print("> current job {} to {}".format(i, i+n_process))
                cur_jobs = candidates[i:i+n_process]
            else:
                print("> current job {} ~ ".format(i))
                cur_jobs = candidates[i:]
        
            processes = []
            for pi, (cand_tag, score, seq, ss, pid, rank) in enumerate(cur_jobs):
                p = Process(target=modeling_RNA_structure_using_SimRNA, args=(seq, ss, cand_tag))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                
            print("- batch {} finished".format(i))
            print("")

        
if __name__ == "__main__":
    # usage - python3 gen_aptamer_structure.py [model_tag] [num_pairs] [top_k] 
    # i.g. python3 gen_cand_str.py MCTS-A-0-100 56 10
    # >> candidates = load_aptamers("MCTS-A-0-100", num_pairs=56, top_k=10)
    # fire.Fire(main)
    __experiments_modeling_all_structures()