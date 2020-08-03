import fire
from preprocess import load_docking_benchmark_dataset
from rand_hue import RandomHeuristicSampling

def __load_RAND_A():
    score_function_path = "classifiers/TESTING-A/mcc0.496-ppv1.000-acc0.826-sn0.303-sp1.000-npv0.812-yd0.303-61trees"
    tag                 = "RAND_A"
    sampler             = RandomHeuristicSampling(score_function_path=score_function_path, tag=tag)
    return sampler

def __load_RAND_B():
    score_function_path = "classifiers/TESTING-B/mcc0.593-ppv0.688-acc0.768-sn0.982-sp0.554-npv0.969-yd0.536-79trees"
    tag                 = "RAND_B"
    sampler             = RandomHeuristicSampling(score_function_path=score_function_path, tag=tag)
    return sampler

def __load_docking_benchmark():
    docking_benchmark_dataset_path = "__benchmark_dataset/benchmark_docking.csv"
    pseqs, rseqs, px, rx, df       = load_docking_benchmark_dataset(docking_benchmark_dataset_path)
    return pseqs, rseqs, px, rx, df

def main(model_type, n_samples, top_k, n_jobs):
    pseqs, rseqs, px, rx, df = __load_docking_benchmark()
    if model_type=="A":
        sampler = __load_RAND_A()
    elif model_type=="B":
        sampler = __load_RAND_B()
    else:
        raise ValueError("Unknown model type {} (please select the model type A or B)".format(model_type))
        
    # In our paper, n_samples 6000000, n_jobs 30
    sampler.pre_sampling(n_samples=n_samples, n_jobs=n_jobs, bp=27)
    sampler.post_sampling(target_pseqs=pseqs, target_rseqs=rseqs, top_k=top_k)

if __name__ == "__main__":
    # ex. python3 gen_rand_hue A 100
    fire.Fire(main)