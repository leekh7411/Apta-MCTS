import fire
from preprocess import load_docking_benchmark_dataset
from mcts_seq import MCTSeq
from multiprocessing import Process, Manager

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
    

def __experiments_56proteins_same_length_aptamers_AB():
    n_process = 2
    # usg. python3 gen_mcts_seq [model_type] [top_k] [n_iter] [bp]
    experiments =[
        ("A", 100, 1000, 0),
        ("B", 100, 1000, 0)
    ]
    return experiments, n_process


def __experiments_56proteins_fixed_length_10_to_100_aptamers_AB():
    n_process = 20
    # usg. python3 gen_mcts_seq [model_type] [top_k] [n_iter] [bp]
    experiments =[
        ("A", 100, 1000, 10),
        ("A", 100, 1000, 20),
        ("A", 100, 1000, 30),
        ("A", 100, 1000, 40),
        ("A", 100, 1000, 50),
        ("A", 100, 1000, 60),
        ("A", 100, 1000, 70),
        ("A", 100, 1000, 80),
        ("A", 100, 1000, 90),
        ("A", 100, 1000, 100),
        ("B", 100, 1000, 10),
        ("B", 100, 1000, 20),
        ("B", 100, 1000, 30),
        ("B", 100, 1000, 40),
        ("B", 100, 1000, 50),
        ("B", 100, 1000, 60),
        ("B", 100, 1000, 70),
        ("B", 100, 1000, 80),
        ("B", 100, 1000, 90),
        ("B", 100, 1000, 100)
    ]
    return experiments, n_process

def __example_through_mutiprocessing(args):
    experiments, n_process = args[0], args[1]
    with Manager() as manager:
        L = manager.list()
        processes = []
        while True:
            if len(processes) == n_process or len(experiments) == 0:
                for p in processes:
                    p.join()
                
                if len(experiments) == 0:
                    break
                else:
                    continue
            else:
                arg_exp = experiments[0]
                print(arg_exp)
                p       = Process(target=main, args=arg_exp)
                p.start()
                processes.append(p)
                if len(experiments) == 1: 
                    experiments = [] # no more jobs here
                else:
                    experiments = experiments[1:]
                                
if __name__ == "__main__":
    # usg. python3 gen_mcts_seq [model_type] [top_k] [n_iter] [bp]
    # i.g.1. python3 gen_mcts_seq A 100 1000 10
    # i.g.2. python3 gen_mcts_seq B 100 1000 100
    # fire.Fire(main)
    
    #__example_through_mutiprocessing(__experiments_56proteins_same_length_aptamers_AB())
    __example_through_mutiprocessing(__experiments_56proteins_fixed_length_10_to_100_aptamers_AB())
    