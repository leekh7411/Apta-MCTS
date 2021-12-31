import os
import io
import json
import fire
import requests
import itertools
import pandas as pd
from collections import defaultdict
from multiprocessing import Process, Manager

class QueryMaker():
    def __init__(self):
        print("-- init query maker")
        
    def write_json(self, data, fname):
        def _conv(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            raise TypeError

        with io.open(fname, "w", encoding="utf-8") as f:
            json_str = json.dumps(data, ensure_ascii=False, default=_conv, indent=4)
            f.write(json_str)
        
    def load_json(self, fname):
        with open(fname, encoding="utf-8") as f:
            json_obj = json.load(f)
        return json_obj
        
    def generate(self, q, t, method, bp, score_function, k, n_iter, n_jobs, bp_same_with_aptamer=False):
        # This method updated for calculating binding-affinity and specificity with other proteins
        # the comparative proteins that used for binding specificity are not chosen proteins all
        inp_target_file = t
        inp_target_json = self.load_json(inp_target_file)
        out_query_file  = q
        
        # pythonic template
        out_query = {"targets": defaultdict(lambda: {
            "model": {
                "method"         : None,
                "score_function" : None,
                "k"              : None,
                "bp"             : None,
                "n_iter"         : None
            },
            "protein": {
                "name" : None,
                "seq"  : None,
            },
            "aptamer": {
                "name" : [],
                "seq"  : []
            },
            "candidate-aptamer": {
                "score" : [],
                "seq"   : [],
                "ss"    : [],
                "mfe"   : []
            },
            "protein-specificity": {
                "name"  : [],
                "seq"   : []
            }
        }), "n_jobs" : n_jobs}
        
        sf_dict = {
            "li2014"  : "classifiers/rf-ictf-li2014/mcc0.484-ppv1.000-acc0.822-sn0.290-sp1.000-npv0.809-yd0.290-77trees",
            "lee2019" : "classifiers/rf-ictf-leeandhan2019/mcc0.607-ppv0.696-acc0.777-sn0.982-sp0.571-npv0.970-yd0.554-49trees"
        }
        
        p_dict = {pname: vals["sequence"] for pname, vals in inp_target_json.items()}
        for m in method:
            
            for target_score_function in score_function:
                for pname, vals in inp_target_json.items():
                    
                    aptamers = vals["aptamer"]
                    pseq = vals["sequence"]
                    aptamer_names = aptamers["name"]
                    aptamer_seqs = aptamers["sequence"]
                    aptamer_lens = [len(s) for s in aptamer_seqs]
                    aptamer_lens = list(set(aptamer_lens))

                    if bp_same_with_aptamer:
                        for aptamer_bp in aptamer_lens:
                            q_name = "{}+{}+{}+{}bp".format(pname, m, target_score_function, aptamer_bp)
                            out_query["targets"][q_name]["model"]["method"] = m
                            out_query["targets"][q_name]["model"]["score_function"] = sf_dict[target_score_function]
                            out_query["targets"][q_name]["model"]["k"] = k
                            out_query["targets"][q_name]["model"]["bp"] = aptamer_bp
                            out_query["targets"][q_name]["model"]["n_iter"] = n_iter
                            out_query["targets"][q_name]["protein"]["name"] = pname
                            out_query["targets"][q_name]["protein"]["seq"] = pseq
                            out_query["targets"][q_name]["aptamer"]["name"] = aptamer_names
                            out_query["targets"][q_name]["aptamer"]["seq"] = aptamer_seqs

                            for spname, spseq in p_dict.items():
                                if spname == pname: continue
                                out_query["targets"][q_name]["protein-specificity"]["name"].append(spname)
                                out_query["targets"][q_name]["protein-specificity"]["seq"].append(spseq)

                            print("--- query {} saved".format(q_name))

                    else:
                        for target_bp in bp:
                            q_name = "{}+{}+{}+{}bp".format(pname, m, target_score_function, target_bp)
                            out_query["targets"][q_name]["model"]["method"] = m
                            out_query["targets"][q_name]["model"]["score_function"] = sf_dict[target_score_function]
                            out_query["targets"][q_name]["model"]["k"] = k
                            out_query["targets"][q_name]["model"]["bp"] = target_bp
                            out_query["targets"][q_name]["model"]["n_iter"] = n_iter
                            out_query["targets"][q_name]["protein"]["name"] = pname
                            out_query["targets"][q_name]["protein"]["seq"] = pseq
                            out_query["targets"][q_name]["aptamer"]["name"] = aptamer_names
                            out_query["targets"][q_name]["aptamer"]["seq"] = aptamer_seqs

                            for spname, spseq in p_dict.items():
                                if spname == pname: continue
                                out_query["targets"][q_name]["protein-specificity"]["name"].append(spname)
                                out_query["targets"][q_name]["protein-specificity"]["seq"].append(spseq)

                            print("--- query {} saved".format(q_name))
                        
        # save query
        self.write_json(out_query, out_query_file)
        print("-- out query saved in {}".format(out_query_file))
        
            
if __name__ == "__main__":
    os.system("clear")
    qm = QueryMaker()
    qm.generate(q                    = "queries/v2-test-our-collections.json", 
                t                    = "targets/target_proteins_collected_ours.json", 
                method               = ["Apta-MCTS"],
                bp                   = [30, 50, 70, 90], 
                score_function       = ["li2014","lee2019"], 
                k                    = 10, 
                n_iter               = 2000, 
                n_jobs               = 30, 
                bp_same_with_aptamer = True)
    