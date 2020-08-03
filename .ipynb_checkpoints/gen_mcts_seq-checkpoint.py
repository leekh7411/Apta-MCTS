import fire
from preprocess import load_docking_benchmark_dataset
from mcts_seq import MCTSeq

def __load_docking_benchmark():
    docking_benchmark_dataset_path = "__benchmark_dataset/benchmark_docking.csv"
    pseqs, rseqs, px, rx, df       = load_docking_benchmark_dataset(docking_benchmark_dataset_path)
    return pseqs, rseqs, px, rx, df

def __load_score_function_A():
    score_function_path = "classifiers/TESTING-A/mcc0.496-ppv1.000-acc0.826-sn0.303-sp1.000-npv0.812-yd0.303-61trees"
    return score_function_path

def __load_score_function_B():
    score_function_path = "classifiers/TESTING-B/mcc0.593-ppv0.688-acc0.768-sn0.982-sp0.554-npv0.969-yd0.536-79trees"
    return score_function_path

def main(model_type, top_k, n_iter, bp):
    pseqs, rseqs, px, rx, df = __load_docking_benchmark()
    if model_type=="A":
        score_function_path = __load_score_function_A()
    elif model_type=="B":
        score_function_path = __load_score_function_B()
    else:
        raise ValueError("Unknown model type {} (please select the model type A or B)".format(model_type))
        
    
    tag     = "MCTS-{}-{}-{}".format(model_type, bp, top_k)
    sampler = MCTSeq(score_function_path=score_function_path, tag=tag)
    
    sampler.sampling_with_truth(target_pseqs = pseqs,  # target protein sequence (for evaluation)
                                target_rseqs = rseqs,  # target rna-aptamer sequences (for evaluation)
                                top_k        = top_k,  # when k=0 then save all candidates
                                n_iter       = n_iter, # default iteration is 1000
                                bp           = bp)     # when bp=0 the length of samples is same with target rna-aptamer sequence
    
    
if __name__ == "__main__":
    # usg. python3 gen_mcts_seq [model_type] [top_k] [n_iter] [bp]
    # i.g.1. python3 gen_mcts_seq A 100 1000 10
    # i.g.2. python3 gen_mcts_seq B 100 1000 100
    fire.Fire(main)