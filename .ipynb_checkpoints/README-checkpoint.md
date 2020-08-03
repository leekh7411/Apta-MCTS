# **MCTS-seq**
Supporting code for the paper **"Predicting aptamer sequences that interact with target proteins using an Aptamer-Protein Interaction classifier and a Monte-Carlo tree search approach"**(currently being prepared).

## Scripts
### Train API classifier
Implementations of how to train the Aptamer-Protein Interaction classifier using Random Forest are described in a `examples/classifier.ipynb`. The trained models are separated as A and B according to the dataset that trained. The benchmark datasets are available in the `__benchmark_dataset/` folder
- API classifier benchmark dataset A from [Li et al, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086729)
- API classifier benchmark dataset B from [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705)

### Prediction using **MCTS-A** and **-B**
We designed the candidate aptamer sequence sampling algorithm that narrow down its search space with Monte-Carlo tree search and the API classifiers A and B. We separated the version of models A and B according to the applied API classifier and we called **MCTS-A** and **MCTS-B**

- The method is implemented in the `mcts_seq.py` 
- The sampling job is available using the `gen_mcts_seq.py`
    - Usage : `python3 gen_mcts_seq.py [model_type] [top_k] [n_iter] [bp]`
        - model_type : set `A` or `B`  
        - top_k : number of output sequences (recommends set `10` or `100`)
        - n_iter : number of iterations in the MCTS based sampling process (default `1000`)
        - bp : the length of candidate sequences (and the model repeats 'bp' times)
        - i.g. `python3 gen_mcts_seq A 100 1000 10` or `python3 gen_mcts_seq B 100 1000 10`


### Prediction using **RAND-A** and **-B**
The candidate RNA aptamer sampling algorithm of [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) that generate candidate sequences randomly with constraints. The method used the API classifier and we separated the version of models A and B according to the applied API classifier. We defined the models of [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) as **RAND-A** and **RAND-B**. We compared the performance of generated sequences of our method with the **RAND-A** and **-B**.

- The method is implemented in the `rand_hue.py` 
- The sampling job is available using the `gen_rand_hue.py`
    - Usage : `python3 gen_rand_hue.py [model_type] [num_sequences] [top_k] [n_jobs]`
        - model_type : set `A` or `B` 
        - num_sequences : number of sequences that pre-sampled (set `6,000,000`) 
        - top_k : number of output sequences
        - n_jobs : number of process for multiprocessing (set `4 ~ 30`)
        - i.g. `python3 gen_rand_hue.py A 6000000 100 30` or `python3 gen_rand_hue.py B 6000000 10 30`