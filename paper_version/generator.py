import os
import json
import fire
import requests
import itertools
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Manager
from query import QueryManager
from apta_mcts import Apta_MCTS
from rand_hue import RandomHeuristicSampling


def print_string_multilines(x, s):
    chunks, chunk_size = len(x), s
    for i in range(0, chunks, chunk_size):
        print(x[i:i+chunk_size])


def get_clf_path():
    return ""


class Generator():
    def __init__(self, q):
        self.qm           = QueryManager(q)
        self.num_targets  = self.qm.get_num_targets()
        self.target_names = self.qm.get_target_names()
        self.model_infos  = [self.qm.get_model_info(t) for t in self.target_names] # m_method, m_type, top_k, bp, n_iter
        self.p_infos      = [self.qm.get_protein_info(t) for t in self.target_names]
        self.a_infos      = [self.qm.get_aptamer_info(t) for t in self.target_names]
        self.ps_infos     = [self.qm.get_protein_specificity_info(t) for t in self.target_names]
        self.n_jobs       = self.qm.get_num_jobs()
        
    def generate(self):
        with Manager() as manager:
            for i in range(0, self.num_targets, self.n_jobs):
                if i+self.n_jobs >= self.num_targets:
                    model_infos = self.model_infos[i:]
                    p_infos     = self.p_infos[i:]
                    t_names     = self.target_names[i:]
                    ps_infos    = self.ps_infos[i:]
                else:
                    model_infos = self.model_infos[i:i+self.n_jobs]
                    p_infos     = self.p_infos[i:i+self.n_jobs]
                    t_names     = self.target_names[i:i+self.n_jobs]
                    ps_infos    = self.ps_infos[i:i+self.n_jobs]
                    
                L = manager.list()
                processes = []    
                for t_name, model_info, p_info, ps_info      in zip(t_names, model_infos, p_infos, ps_infos):
                    method, score_function, top_k, bp, n_iter = model_info
                    p_name, p_seq = p_info
                    ps_names, ps_seqs = ps_info
                    
                    print("> Target task name is {}".format(t_name))
                    print("- taret protein is {}".format(p_name))
                    print("- generative model {} with length {} bp".format(method, bp))
                    print("- score function : {}".format(score_function))
                    print("- generative model will save top {} candidates".format(top_k))
                    print("- (when model is Apta-MCTS, number of iteration is {})".format(n_iter))
                    print("- proteins for checking binding specificity (#proteins: {})".format(len(ps_names)))
                    print("- target protein sequence")
                    print_string_multilines(p_seq, 70)
                    print("")
                    
                    if method == "Apta-MCTS":
                        p = Process(target=self.apta_mcts, 
                                    args=(L, t_name, p_seq, score_function, bp, top_k, n_iter, ps_names, ps_seqs))
                    elif method == "Lee_and_Han_2019":
                        p = Process(target=self.leeandhan2019, 
                                    args=(L, t_name, p_seq, score_function, top_k))
                    else:
                        raise ValueError("unreachable error")
                        
                    p.start()
                    processes.append(p)
                    
                for p in processes:
                    p.join()
                                
                for t_name, candidates in L:
                    self.qm.set_candidate_info(t_name, candidates)
                self.qm.update_and_reload()
                
    def apta_mcts(self, L, t_name, p_seq, score_function, bp, k, n_iter, ps_names, ps_seqs):
        G = Apta_MCTS(score_function)
        #candidate_aptamers = G.sampling(p_seq, bp, k, n_iter) # debugging
        # updated - considering binding specificity
        p_spes = (ps_names, ps_seqs)
        candidate_aptamers = G.sampling(p_seq, bp, k, n_iter, p_spes) # debugging
        
        # self.qm.set_candidate_info(t_name, candidate_aptamers)
        L.append((t_name, candidate_aptamers))
        
    def leeandhan2019(self, L, t_name, p_seq, score_function, k):
        # fixed parameters
        n_samples, bp = 6000000, 27
        G = RandomHeuristicSampling(score_function)
        G.pre_sampling(n_samples, self.n_jobs, bp)
        candidate_aptamers = G.post_sampling(p_seq, k)
        # self.qm.set_candidate_info(t_name, candidate_aptamers)
        L.append((t_name, candidate_aptamers))


if __name__ == "__main__":
    fire.Fire(Generator)
