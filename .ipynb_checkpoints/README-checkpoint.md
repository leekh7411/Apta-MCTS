# **MCTS-seq**
Supporting code for the paper **"Predicting aptamer sequences that interact with target proteins using an Aptamer-Protein Interaction classifier and a Monte-Carlo tree search approach"**(currently being prepared).

## **Generate Aptamers** 
### Train API classifier
Implementations of how to train the Aptamer-Protein Interaction classifier using Random Forest are described in a `examples/classifier.ipynb`. The trained models are separated as A and B according to the dataset that trained. The benchmark datasets are available in the `__benchmark_dataset/` folder
- API classifier benchmark dataset A from [Li et al, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086729)
- API classifier benchmark dataset B from [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705)

### Prediction using **MCTS-A** and **-B**
We designed the candidate aptamer sequence sampling algorithm that narrow down its search space with Monte-Carlo tree search and the API classifiers A and B. We separated the version of models A and B according to the applied API classifier and we called **MCTS-A** and **MCTS-B**

- The method is implemented in the `mcts_seq.py` 
- The sampling job is available using the `gen_mcts_seq.py`
    - Usage : `python3 gen_mcts_seq.py [model_type] [top_k] [n_iter] [bp]`
        - ***model_type*** : set `A` or `B`  
        - ***top_k*** : number of output sequences (recommends set `10` or `100`)
        - ***n_iter*** : number of iterations in the MCTS based sampling process (default `1000`)
        - ***bp*** : the length of candidate sequences (and the model repeats 'bp' times)
        - i.g. `python3 gen_mcts_seq A 100 1000 10` or `python3 gen_mcts_seq B 100 1000 10`
    - Re-implementation of the experiments in our paper also available in `gen_mcts_seq.py`

### Prediction using **RAND-A** and **-B**
The candidate RNA aptamer sampling algorithm of [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) that generate candidate sequences randomly with constraints. The method used the API classifier and we separated the version of models A and B according to the applied API classifier. We defined the models of [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) as **RAND-A** and **RAND-B**. We compared the performance of generated sequences of our method with the **RAND-A** and **-B**.

- The method is implemented in the `rand_hue.py` 
- The sampling job is available using the `gen_rand_hue.py`
    - Usage : `python3 gen_rand_hue.py [model_type] [num_sequences] [top_k] [n_jobs]`
        - ***model_type*** : set `A` or `B` 
        - ***num_sequences*** : number of sequences that pre-sampled (set `6,000,000`) 
        - ***top_k*** : number of output sequences
        - ***n_jobs*** : number of process for multiprocessing (set `4 ~ 30`)
        - i.g. `python3 gen_rand_hue.py A 6000000 100 30` or `python3 gen_rand_hue.py B 6000000 10 30`
        
  
## **Evaluate Aptamers**  
### 3D structure (.pdb format) prediction of RNA aptamers (optional)
If you want to calculate docking score through ZDOCK or else, the candidate sequences that predicted need to convert to the pdb format structure file. We provide a simple script that predict 3D structure of RNA using **[SimRNA](https://genesilico.pl/SimRNAweb)** program (please install a standalone version). The script `gen_aptamer_structure.py` takes as **the tag of model** (check the tag here, `aptamer/[model_type]/output...`), **number of pairs** (number of .txt files in `aptamer/[model_type]/`) and **top-k** (pick the top k sequences in `aptamer/[model_type]/??.txt` file) value as inputs. 

- The modeling job is available using the `gen_aptamer_structure.py`
    - Usage : `python3 gen_aptamer_structure.py [model_tag] [num_pairs] [top_k] `
        - ***model_type*** : check the tag in here `aptamer/[model_type]/output...`
        - ***num_pairs*** : number of .txt files in `aptamer/[model_type]/`
        - ***top_k*** : pick the top k sequences in `aptamer/[model_type]/??.txt` file
        - i.g. `i.g. python3 gen_cand_str.py MCTS-A-0-100 56 10`
        
        
        