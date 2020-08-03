import os
import time
import math
import random
import operator
import numpy as np
from functools import reduce
from copy import deepcopy
from sklearn.externals import joblib
from collections import defaultdict
import RNA
from preprocess import *
ENV = None
class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0
        self.children = {}
        
def random_policy(state):
    while not state.is_terminal():
        try:
            action = random.choice(state.get_possible_actions())
        except IndexError:
            raise Exception("No possible action in this state {}".format(str(state)))
        state = state.take_action(action)
    return state.get_reward()


class MCTS():
    def __init__(self, time_limit=None, iteration_limits=None,
                 exploration_constant=1/math.sqrt(2), rollout_policy=random_policy):
        
        if time_limit != None:
            if iteration_limits != None:
                raise ValueError("Cannot have both a time limit and an iteration limit!")
            
            # time taken for each MCTS search in milliseconds
            self.time_limit = time_limit
            self.limit_type = "time"
        else:
            if iteration_limits == None:
                raise ValueError("Must have either a time limit or an iteraction limit!")
            
            # number of iteractions of the search
            if iteration_limits < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.search_limit = iteration_limits
            self.limit_type   = 'iteration'
        
        self.exploration_constant = exploration_constant
        self.rollout = rollout_policy
        self.candidates = []
    
    def get_candidates(self):
        return self.candidates
        
    def get_root_childs(self):
        return self.root.get_children()
    
    def search(self, initial_state):
        self.root = treeNode(initial_state, None) # parent -> none
        
        if self.limit_type == "time":
            time_limit = time.time() + self.time_limit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()
                
        best_child = self.get_best_child(self.root, 0)
        return self.get_action(self.root, best_child)
    
    
    def execute_round(self):
        
        # selection
        node = self.select_node(self.root) 
        
        # expansion & simulation
        reward, candidate_data = self.rollout(node.state)
        self.candidates.append(candidate_data)
        
        # backpropagation
        self.backpropagation(node, reward)
        
    
    def select_node(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                return self.expand(node)
        return node
    
    
    def expand(self, node):
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                next_state = node.state.take_action(action)
                parent_node = node
                new_node = treeNode(next_state, parent_node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node
        raise Exception("Non reachable error")
    
    
    def backpropagation(self, node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent
    
    
    def get_best_child(self, node, exploration_value):
        best_value = float("-inf")
        best_nodes = []
        for child in node.children.values():
            
            node_value  = child.total_reward / child.num_visits
            node_value += exploration_value * math.sqrt(2*math.log(node.num_visits)/child.num_visits)
            
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        
        return random.choice(best_nodes)
    
    
    def get_action(self, root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action
            
class Action():
    def __init__(self, cur_bp, next_letter):
        self.cur_bp = cur_bp
        self.next_letter = next_letter
    
    def get_next_letter(self):
        return self.next_letter
    
    def __str__(self):
        return str((self.cur_bp, self.next_letter))
        
    def __repr__(self):
        return str(self)
        
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.cur_bp == other.cur_bp and self.next_letter == other.next_letter

    def __hash__(self):
        return hash((self.cur_bp, self.next_letter))

    
class Environment():
    def __init__(self, model):
        self.model = model # scikit-learn random forest classifier
        self.p_mer = 3
        self.r_mer = 4
        self.p_iCTF = improvedCTF(letters=["A","B","C","D","E","F","G"],length=self.p_mer)
        self.r_iCTF = improvedCTF(letters=["A","C","G","U"],length=self.r_mer)
        self.reduced_p_dict = get_reduced_protein_letter_defaultdict()
            
    def init_target_protein(self, seq):
        self.target_p_seq = seq
        self.encoder_p(seq)
        
    def encoder_p(self, seq):
        seq_len = len(seq)
        pf_dict = self.p_iCTF.get_feature_dict()
        rpseq = []
        for p in seq:
            rpseq.append(self.reduced_p_dict[p])
        
        temp_pseq = ""
        for p in rpseq:
            temp_pseq += p
        seq = temp_pseq
        
        for mer in range(1,self.p_mer+1):
            for i in range(0,len(seq)-mer):
                pattern = seq[i:i+mer]
                try:
                    pf_dict[pattern] += 1
                except:
                    continue
        pf = np.array(list(pf_dict.values()))
        pf = pf / seq_len
        self.px = pf
        
    
    def encoder_r(self, seq):
        seq_len = len(seq)
        rf_dict = self.r_iCTF.get_feature_dict()
        for mer in range(1,self.r_mer+1):
            for i in range(0,len(seq)-mer):
                pattern = seq[i:i+mer]
                try:
                    rf_dict[pattern] += 1
                except:
                    continue
                    
        rf = np.array(list(rf_dict.values()))
        rf = rf / seq_len
        return rf
    
    def get_reward(self, aptamer_sequence):
        rx = self.encoder_r(aptamer_sequence)
        x  = np.array([list(self.px) + list(rx)])
        y_pred_prob = self.model.predict_proba(x)[0][1]
        return y_pred_prob
    
   
    
def act8_aptamer_to_string(best_aptamer):
    reordered_aptamer = ""
    for apt in best_aptamer:
        if apt in "acgu":
            reordered_aptamer = reordered_aptamer + apt
        elif apt in "ACGU":
            reordered_aptamer = apt + reordered_aptamer
        else:
            continue
    return reordered_aptamer.upper()

class AptamerStates():
    def __init__(self, bp=27, letters=["A","C","G","U","a","c","g","u"], current_aptamer=""):
        self.n_letters = len(letters)
        self.bp = bp
        self.aptamer = current_aptamer
        self.actions = letters
        
        
    def get_possible_actions(self):
        """ 
        State is sequene of the Aptamer
        
        - aptamer has 4 letters (DNA case ACGT, RNA case ACGU)
        - that means possible actions are only 4 actions
        - But! in this version, we choose 8 actions which is multiplied left or right directions
        
        """
        
        possible_actions = [Action(len(self.aptamer), nl) for nl in self.actions]
        return possible_actions
    
    def take_action(self, action):
        """
        Select possible action and update states
        
        - In this case, aptamer is sequence of 4 letters x 2 direction(left, right) of concatenation 
        - so the next state(aptamer string) just add action(next letter) to the current state
        
        """
        new_state = deepcopy(self)
        new_state.aptamer += action.get_next_letter()
        return new_state
    
    def is_terminal(self):
        """ Check the end of state
        - we already specified target length of aptamer
        - if length of aptamer(state) is same with the length then terminal
        """
        if len(self.aptamer) == self.bp:
            return True
        else:
            return False
    
    def get_reward(self):
        
        reordered_aptamer = act8_aptamer_to_string(self.aptamer)                
        #self.aptamer = reordered_aptamer.upper()
        reward = ENV.get_reward(reordered_aptamer)
        ss, mfe = RNA.fold(reordered_aptamer)
        candidate_data = (reward, reordered_aptamer, ss, mfe)
        
        return reward, candidate_data




class MCTSeq():
    def __init__(self, score_function_path, tag):
        self.tag = tag
        self.base_dir = "aptamers/{}".format(tag)
        self.sf_path = score_function_path # pre-trained rf-model path
        self.__load_score_function()
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
    def __load_score_function(self):
        self.score_function = joblib.load(self.sf_path)
        self.score_function.set_params(n_jobs=1)
        print("")
        print("> Load RF model with {} trees".format(len(self.score_function.estimators_)))
        print("- from : {}".format(self.sf_path))
        print("")
    
    def __redundancy_removal(self, candidates):
        ss_dict = defaultdict(lambda: {"score": -1, "seq": None, "mfe": None})
        for score, seq, ss, mfe in candidates:
            if ss_dict[ss]["score"] < score:
                ss_dict[ss]["score"] = score
                ss_dict[ss]["seq"]   = seq
                ss_dict[ss]["mfe"]   = mfe
        
        _candidates = []
        for ss, vals in ss_dict.items():
            score = vals["score"]
            seq   = vals["seq"]
            mfe   = vals["mfe"]
            _candidates.append((score, seq, ss, mfe))
        
        return _candidates
    
    def __save_candidates_with_truth(self, path, candidates, target_data, top_k):
        pseq, rseq, px, rx = target_data
        f = open(path, "w")
        f.write("> Target Protein\n")
        f.write("{}\n".format(pseq))
        f.write("> RNA-Aptamer\n")
        f.write("{}\n".format(rseq))
        rss, rmfe = RNA.fold(rseq)
        f.write("{}\n".format(rss))
        f.write("{}\n".format(rmfe))
        x = np.array([np.concatenate([px,rx],axis=-1)])
        pred = self.score_function.predict_proba(x)[0]
        f.write("> True Predict Proba : {}\n".format(pred[1]))
        f.write("> Predicted RNA-Aptamer Candidates\n")
        samples = list(sorted(candidates, key=lambda x: x[0]))
        max_reward = samples[-1][0]
        true_reward = pred[1]
        for i, (reward, aptamer, ss, mfe) in enumerate(reversed(samples)):
            if top_k > 0 and i == top_k: break
            f.write("{}\t{}\t{}\t{}\n".format(aptamer, ss, mfe, reward))
        f.close()
        print("- process complete : {} / reward : {} (original {})".format(path, max_reward, true_reward))
        
    def __save_candidates(self, path, candidates, target_data, top_k):
        pseq, px = target_data
        f = open(path, "w")
        f.write("> Target Protein\n")
        f.write("{}\n".format(pseq))
        f.write("> RNA-Aptamer\n")
        f.write("None\n")
        f.write("None\n")
        f.write("None\n")
        f.write("> True Predict Proba : None\n")
        f.write("> Predicted RNA-Aptamer Candidates\n")
        samples = list(sorted(candidates, key=lambda x: x[0]))
        max_reward = samples[-1][0]
        for i, (reward, aptamer, ss, mfe) in enumerate(reversed(samples)):
            if i == top_k: break
            f.write("{}\t{}\t{}\t{}\n".format(aptamer, ss, mfe, reward))
        f.close()
        print("- process complete : {}".format(path))
        print("- process complete : {} / reward : {}".format(path, max_reward))
    
    def __save_candidates_sequential(self, path, orig_cands):
        f = open(path, "w")
        for i, cands in enumerate(orig_cands):
            N = i+1
            f.write(">N-{}\n".format(N))
            for reward, aptamer, ss, mfe in cands:
                f.write("{}\t{}\t{}\t{}\n".format(aptamer, ss, mfe, reward))        
        f.close()
        print("- process complete : {}".format(path))
                
    def sampling_with_truth(self, target_pseqs, target_rseqs, top_k, n_iter, bp=0):
        global ENV
        # This method used for sampling test
        # We need true aptamer sequence pairs about target proteins
        # So, in this method you dont need to specify aptamer sequence length
        # aptamer's length will be matched about true aptamer sequence for testing
        
        #aptamer_letters = ["A","C","G","U"] # original ribonucleotide sequence bases
        aptamer_letters = ["A","C","G","U", "a","c","g","u"] # directional ribonucleotide bases
        
        rx_list      = rna2feature_iCTF(target_rseqs)
        px_list      = pro2feature_iCTF(target_pseqs)
        
        for i, (px, rx, pseq, rseq) in enumerate(zip(px_list, rx_list, target_pseqs, target_rseqs)):
            print("> MCTS-seq sampling start processing {} / {}".format(i+1, len(px_list)))
            print("- True aptamer length : {}".format(len(rseq)))
            ENV = Environment(self.score_function) # global parameter (as a score function) initialization
            ENV.init_target_protein(pseq)
            TRUE_RSEQ    = rseq
            TRUE_RSS, _  = RNA.fold(rseq)
            best_aptamer = ""
            if bp > 0:
                target_bp = bp
            else:
                target_bp = len(rseq)
            cand_path    = "aptamers/{}/output-{:02d}.txt".format(self.tag, i)
            orig_cand_path = "aptamers/{}/output-{:02d}-sequential.txt".format(self.tag, i)
            candidates   = []
            orig_candidates = []
            while len(best_aptamer) < target_bp:
                CUR_RSEQ      = act8_aptamer_to_string(best_aptamer)
                initial_state = AptamerStates(bp              = target_bp,
                                              current_aptamer = best_aptamer,
                                              letters         = aptamer_letters)
                
                mcts          = MCTS(time_limit=None, iteration_limits=n_iter)
                action        = mcts.search(initial_state=initial_state)
                next_letter   = action.get_next_letter()
                best_aptamer += next_letter
                mcts_cands    = mcts.get_candidates()
                orig_candidates.append(mcts_cands)
                candidates   += mcts_cands
                print(next_letter, end="")
            print("")
            _candidates = self.__redundancy_removal(candidates)
            target_data = (pseq, rseq, px, rx)
            print("- Total number of candidates : {} (original {})".format(len(_candidates), len(candidates)))
            self.__save_candidates_with_truth(cand_path, _candidates, target_data, top_k)
            self.__save_candidates_sequential(orig_cand_path, orig_candidates)
            
    def sampling(self, target_pseqs, target_bp, top_k, n_iter):
        global ENV
        px_list = pro2feature_iCTF(target_pseqs)
        
        for i, (px, pseq) in enumerate(zip(px_list, target_pseqs)):
            print("> MCTS-seq sampling start processing {} / {}".format(i+1, len(px_list)))
            print("- Target aptamer length : {}".format(target_bp))
            ENV = Environment(self.score_function) # global parameter (as a score function) initialization
            ENV.init_target_protein(pseq)
            best_aptamer = ""
            cand_path    = "aptamers/{}/output-{:02d}.txt".format(self.tag, i)
            candidates   = []
            
            while len(best_aptamer) < target_bp:
                CUR_RSEQ      = act8_aptamer_to_string(best_aptamer)
                initial_state = AptamerStates(bp=target_bp, current_aptamer=best_aptamer)
                mcts          = MCTS(time_limit=None, iteration_limits=n_iter)
                action        = mcts.search(initial_state=initial_state)
                next_letter   = action.get_next_letter()
                best_aptamer += next_letter
                candidates   += mcts.get_candidates()
                print(next_letter, end="")
            print("")
            
            target_data = (pseq, px)
            _candidates = self.__redundancy_removal(candidates)
            print("- Total number of candidates : {} (original {})".format(len(_candidates), len(candidates)))
            self.__save_candidates(cand_path, _candidates, target_data, top_k)