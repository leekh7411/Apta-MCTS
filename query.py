import os
import io
import json
import fire
import requests
import itertools
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Manager

class QueryManager():
    def __init__(self, query_file):
        self.query_file = query_file
        self.d = self.load_json(query_file)
        self.t = self.d["targets"]
        self.n_jobs = self.d["n_jobs"]
        
        
    def write_json(self, data, fname):
        def _conv(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            raise TypeError

        with io.open(fname, "w", encoding="utf-8") as f:
            json_str = json.dumps(data, ensure_ascii=False, default=_conv, indent=4)
            f.write(json_str)
        
    def update_and_reload(self):
        self.write_json(self.d, self.query_file)
        self.d = self.load_json(self.query_file)
        self.t = self.d["targets"]
        self.n_jobs = self.d["n_jobs"]
        
    def load_json(self, fname):
        with open(fname, encoding="utf-8") as f:
            json_obj = json.load(f)
        return json_obj
        
        
    def get_num_targets(self):
        target_names = list(self.t.keys())
        return len(target_names)
        
        
    def get_target_names(self):
        target_names = list(self.t.keys())
        return target_names
        
        
    def get_num_jobs(self):
        return self.n_jobs
        
        
    def get_model_info(self, target_name):
        m_info   = self.d["targets"][target_name]["model"]
        m_method = m_info["method"]
        if m_method in ("Lee_and_Han_2019","Apta-MCTS") :
            score_function = m_info["score_function"]
            top_k  = m_info["k"]
            bp     = m_info["bp"]
            n_iter = m_info["n_iter"]
        else:
            raise ValueError("QueryError - please select model-method one Lee_and_Han_2019|Apta-MCTS, not {}".format(m_method))
        
        return m_method, score_function, top_k, bp, n_iter
    
    
    def get_unique_tag(self, target_name):
        m_method, m_type, k, bp, n_iter = self.get_model_info(target_name)
        tag = str(target_name)+str(m_method)+str(m_type)+str(k)+str(bp)+str(n_iter)
        return tag
    
    
    def get_structure(self, structure):
        modeling = structure["modeling"]
        filename = structure["file_name"]
        filename_ms = structure["marksur"]
        return modeling, filename, filename_ms
    
    
    def get_protein_info(self, target_name):
        info = self.d["targets"][target_name]["protein"]
        name = info["name"]
        seq  = info["seq"]
        return name, seq
        
    
    def get_aptamer_info(self, target_name):
        info = self.d["targets"][target_name]["aptamer"]
        names = info["name"]
        seqs  = info["seq"]
        return names, seqs
    
    
    def set_candidate_info(self, target_name, candidates):
        self.d["targets"][target_name]["candidate-aptamer"]["score"] = []
        self.d["targets"][target_name]["candidate-aptamer"]["seq"]   = []
        self.d["targets"][target_name]["candidate-aptamer"]["ss"]    = []
        self.d["targets"][target_name]["candidate-aptamer"]["mfe"]   = []
        for score, seq, ss, mfe in candidates:
            self.d["targets"][target_name]["candidate-aptamer"]["score"].append(score)
            self.d["targets"][target_name]["candidate-aptamer"]["seq"].append(seq)
            self.d["targets"][target_name]["candidate-aptamer"]["ss"].append(ss)
            self.d["targets"][target_name]["candidate-aptamer"]["mfe"].append(mfe)
        print("++ The candidate aptamers of target {} are updated".format(target_name))
        
    
    def get_candidate_info(self, target_name):
        scores = self.d["targets"][target_name]["candidate-aptamer"]["score"]
        seqs   = self.d["targets"][target_name]["candidate-aptamer"]["seq"]
        sss    = self.d["targets"][target_name]["candidate-aptamer"]["ss"]
        mfes   = self.d["targets"][target_name]["candidate-aptamer"]["mfe"]
        return scores, seqs, sss, mfes
    
    
    def get_protein_specificity_info(self, target_name):
        protein_specificity = self.d["targets"][target_name]["protein-specificity"]
        ps_names = protein_specificity["name"]
        ps_seqs  = protein_specificity["seq"]
        return ps_names, ps_seqs
        