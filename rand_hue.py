import os
import RNA
import numpy as np
from sklearn.externals import joblib
from collections import defaultdict
from classifier import RandomForestModel
from preprocess import load_benchmark_dataset, rna2feature_iCTF, pro2feature_iCTF
from multiprocessing import Process, Manager

class RandomHeuristicSampling():
    def __init__(self, score_function_path):
        self.sf_path = score_function_path # pre-trained rf-model path
        self.__load_score_function()
        
    def __load_score_function(self):
        self.score_function = joblib.load(self.sf_path)
        print("")
        print("> Load RF model with {} trees".format(len(self.score_function.estimators_)))
        print("- from : {}".format(self.sf_path))
        print("")
        
    def __generate_sequences(self, n_samples, bp, letters):
        r_seqs = []
        r_ss  = []
        r_mfe = []
        
        for i in range(n_samples):

            rseq = np.random.rand(bp, len(letters))
            seq = ""
            for r in rseq:
                seq += letters[np.argmax(r)]
            r_seqs.append(seq)
            ss, mfe = RNA.fold(seq)
            r_ss.append(ss)
            r_mfe.append(mfe)

        return r_seqs, r_ss, r_mfe
    
    def __heuristic_filtering(self, rand_seqs, r_ss, r_mfe):
        rand_seqs = np.array(rand_seqs)
        r_ss = np.array(r_ss)
        r_mfe = np.array(r_mfe)
        ret_ss_dict = defaultdict(lambda: [])

        for i, (seq, ss, mfe) in enumerate(zip(rand_seqs, r_ss, r_mfe)):
            pre_ss1 = ss[:3]
            pre_ss2 = ss[:4]
            post_ss1 = ss[-3:]
            post_ss2 = ss[-4:]
            rule_01 = False
            rule_02 = False
            rule_03 = False

            if pre_ss1 == "((("  and post_ss1 == ")))"  : rule_01 = True
            if pre_ss2 == ".(((" and post_ss1 == ")))"  : rule_01 = True
            if pre_ss1 == "((("  and post_ss2 == ")))." : rule_01 = True
            if pre_ss2 == ".(((" and post_ss2 == ")))." : rule_01 = True        

            if mfe <= -5.7 : rule_02 = True

            unpaired_bases = [1 if s == "." else 0 for s in ss]
            if sum(unpaired_bases) >= 11 : rule_03 = True 

            if rule_01 and rule_02 and rule_03:
                ret_ss_dict[ss].append(i)

        # Last rule filtering
        # - the number of same secondary structure in the pool should not exceed 150
        final_indices = []
        for ss_key, ss_list in ret_ss_dict.items():
            final_indices += ss_list
        
        final_indices = np.array(final_indices)
        
        print("> Finally selected {} items from {} items".format(len(final_indices), len(rand_seqs)))
        
        return rand_seqs[final_indices], r_ss[final_indices], r_mfe[final_indices]
    
    def __batch_sampling(self, L, rseed):
        np.random.seed(rseed)
        r_seqs, r_ss, r_mfe = self.__generate_sequences(self.n_batch_samples, bp=self.bp, letters=["A","C","G","U"])
        f_seqs, f_ss, f_mfe = self.__heuristic_filtering(r_seqs, r_ss, r_mfe)
        L.append((f_seqs, f_ss, f_mfe))
    
    def pre_sampling(self, n_samples, n_jobs, bp):
        pre_sample_path = "aptamers/pre-samples-n{}-bp{}.txt".format(n_samples, bp)
        self.pre_sample_path = pre_sample_path
        self.n_samples       = n_samples # number of samples
        self.n_jobs          = n_jobs # multiprocessing cores
        self.n_batch_samples = n_samples // n_jobs
        self.bp = bp
                
        if os.path.exists(pre_sample_path):
            print("- pre-sampling results already exists : {}".format(pre_sample_path))
            return
        
        f = open(pre_sample_path, "w")
        
        with Manager() as manager:
            L = manager.list()
            processes = []
            for i in range(self.n_jobs):
                p = Process(target=self.__batch_sampling, args=(L, i))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                
            all_seqs, all_sss, all_mfes = [],[],[]
            for ret in L:
                seqs, sss, mfes = ret[0], ret[1], ret[2]
                all_seqs += list(seqs)
                all_sss  += list(sss)
                all_mfes += list(mfes)

            all_seqs = np.array(all_seqs)
            all_sss = np.array(all_sss)
            all_mfes = np.array(all_mfes)
            ss_dict = defaultdict(lambda: [])
            for i, ss in enumerate(all_sss):
                ss_dict[ss].append(i)

            filtered_indices = []
            for ss_key, ss_list in ss_dict.items():
                if len(ss_list) < 150:
                    filtered_indices += list(ss_list)

            #filtered_indices = np.array(filtered_indices)
            all_seqs = all_seqs[filtered_indices]
            all_sss  = all_sss[filtered_indices]
            all_mfes = all_mfes[filtered_indices]

            for seq, ss, mfe in zip(all_seqs, all_sss, all_mfes):
                f.write("{}\t{}\t{}\n".format(seq, ss, mfe))
            
            print("- Number of pre-sampled sequences : {}".format(len(all_seqs)))
        f.close()
        
    def __load_candidates(self):
        # have to be excuted after call the method 'candidate_sampling'
        f = open(self.pre_sample_path, "r")
        flines = f.readlines()
        seqs, sss, mfes = [],[],[]
        for line in flines:
            line = line.replace("\n","")
            seq, ss, mfe = line.split("\t")
            mfe = float(mfe)
            seqs.append(seq)
            sss.append(ss)
            mfes.append(mfe)
        print("- Load pre-sampled sequences : {}".format(self.pre_sample_path))
        return np.array(seqs), np.array(sss), np.array(mfes)
    
    def post_sampling(self, p_seq, top_k):
        cand_rseqs, cand_sss, cand_mfes = self.__load_candidates()
        
        px = pro2feature_iCTF([p_seq])[0]
        cand_rx_list = rna2feature_iCTF(cand_rseqs)
        
        num_cands = len(cand_rx_list)
        cand_px = np.array([px] * num_cands)
        cand_rx = np.array(cand_rx_list)
        cand_x  = np.concatenate([cand_px, cand_rx], axis=-1)
        
        cand_preds = self.score_function.predict_proba(cand_x)[:,1]
        cand_list  = []
        
        for reward, seq, ss, mfe in zip(cand_preds, cand_rseqs, cand_sss, cand_mfes):
            cand_list.append((reward, seq, ss, mfe))
        cand_list = list(sorted(cand_list, key=lambda x:-x[0]))
        cand_list = cand_list[:top_k]
        
        return cand_list
