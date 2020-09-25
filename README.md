# Apta-MCTS 
![](https://github.com/leekh7411/Apta-MCTS/blob/master/figs/overview.png?raw=true)
Supporting code for the paper **"Predicting aptamer sequences that interact with target proteins using Aptamer-Protein Interaction classifiers and a Monte-Carlo tree search approach"**(currently being prepared).

## Requirements
- Test environment : Ubuntu 16.04 server
- Recent versions of numpy, scipy, sklearn, and pandas are required. 
- Additionally, you need to install the [ViennaRNA packages](https://github.com/ViennaRNA/ViennaRNA) for predicting RNA secondary structures ***(Please check if you can `import RNA` in your python script)***.

## **(A)** Aptamer-Protein-Interaction (API) classifiers
Implementations of how to train the Aptamer-Protein Interaction classifier using Random Forest are described in a `classifier.py`. We train two versions of API classifiers according to the datasets. The benchmark datasets are available in the `datasets/` folder with json format files.
#### *List of benchmark dataset*
| Source | Positives | Negatives | Description |
|--|--|--|--|
| [Li et al, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086729) | 580 | 1740 | Training data for API classifiers |
| [Li et al, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086729) | 145 | 435 | Validation data for API classifiers |
| [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) | 157 | 493 | Training data for API classifiers |
| [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) | 56 | 56 | Validation data for API classifiers and Apta-MCTS |

#### *How to train API classifiers*
```sh
python3 classifier.py \
-dataset_dir=datasets/li2014 \
-tag=rf-iCTF-li2014 \
-min_trees=35 \
-max_trees=200 \
-n_jobs=20 \
-num_models=1000
```
The `classifier.py` takes inputs as a sequence list written of json format (see the .json files in `datasets/<source>/train.json or test.json`) through the parameter `dataset_dir`. Second parameter `tag` is an identifier for your API classifiers.  The range for number of trees in Random Forest algorithm are feed through the `min_trees` and `max_trees` (integer values). Due to this classifiers built on the scikit-learn packages, `n_jobs` has same meaning with the scikit-learn models. We select best scored API classifier for the score function of Apta-MCTS, so there are multiple models (number of models = `num_models`) gerneated for selection.  Below table is binary classifier evaluation scores with respect to the our appoaches (RF with iCTF) and others for two benchmark datasets 

#### *Performace references*
| API classifier    | Training set     | Sensitivity | Specificity | Accuracy | Yuden's Index | MCC   |
|-------------------|------------------|-------------|-------------|----------|---------------|-------|
| Li et al. 2014    | Li et al. 2014   | 0.483       | 0.871       | 0.774    | 0.354         | 0.372 |
| Zhang et al. 2016 | Li et al. 2014   | **0.738**       | 0.713       | 0.719    | **0.451**         | 0.398 |
| **RF with iCTF**      | Li et al. 2014   | 0.303       | **0.999**       | **0.826**    | 0.303         | **0.496** |
| Lee and Han 2019  | Lee and Han 2019 | 0.768       | **0.661**       | 0.714    | 0.429         | 0.431 |
| **RF with iCTF**      | Lee and Han 2019 | **0.982**       | 0.554       | **0.768**    | **0.536**         | **0.593** |

With the perspective of machine learning models, the result of **RF with iCTF** is not a good performaces, but we just designed the classifiers only takes the patterns of sequences as inputs (looking forward to the day when the data volume [#aptamer-protein complexs] grows). 

## **(B)** Generative model for candidate aptamers
The candidate aptamer sequence generation model Apta-MCTS takes the **target protein sequence** and **length of candidate aptamer** as inputs. With the 8-node tree structure for RNA sequences what we desigend, Monte-Carlo tree search (MCTS) algorithm finds next base for each step. Whole processes of sequence sampling are based on an iterative forward algorithm. So, what is the outputs? All candidates in each MCTS processes are collected and sorted based on the score (note that the scores are from the score function that is the best performance API classifier what we trained before). After some post-processing (*i.g.* remove redundant secondary structure sequences), Apta-MCTS returns only top-k candidates. 

#### *How to generate candidate aptamers for a target protein*
Apta-MCTS simply takes the configuration json file as an input for generation.
```sh
python3 generator.py -q=queries/sample.json
```
How to write the configuration file? First, you need a target protein name and sequence. Specify the target protein information `name` and `seq` as below,
```json
"protein": {
	"name": "<target protein name>",
	"seq" : "<target protein sequence>"
}
```
and set the parameters of generative models as below,
```json
"model": {
	"method"         : "Lee_and_Han_2019|Apta-MCTS",
	"score_function" : "<path of the weights of the pre-trained API classifer>",
	"k"              : "<number of top scored candidates>",
	"bp"             : "<length of candidate RNA-aptamer sequences>",
	"n_iter"         : "<number of iterations for each base when method is Apta-MCTS>"
}
```
Our model name is the Apta-MCTS (and Lee and Han 2019 method also available, note that it is for validation of our paper experiments), set the `method` as Apta-MCTS. The `score_function` is the path of pre-trained API classifier's weight (available in `classifiers/<model_name>/<weight_file>`). The parameter `k` is the number of final candidate output sequences. The parameter `bp` is a length of candidate aptamer sequence, and remember that our iterative sampling algorithm repeats `bp` times. The number of repeatition of MCTS in each iteration is a `n_iter` parameter.    

Almost finished, next job is just construct other empty template as below,
```json
{
    "targets": {
        "6GOF-Apta-MCTS": {
            "model": {
                "method"        : "Apta-MCTS",
                "score_function": "classifiers/rf-ictf-li2014/mcc0.484-ppv1.000-acc0.822-sn0.290-sp1.000-npv0.809-yd0.290-77trees",
                "k"             : 5,
                "bp"            : 30,
                "n_iter"        : 100
            },
            "protein": {
                "name": "6GOF",
                "seq": "STEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSK"
            },
            "aptamer": {
                "name": [],
                "seq" : []
            },
            "candidate-aptamer": {
                "score": [],
                "seq"  : [],
                "ss"   : [],
                "mfe"  : []
            }
        }
    },
    "n_jobs" : 10
}
```
The parameter `experiment_name`(in here, `6GOF-Apta-MCTS`) is an identifier of the task. You don't need to fill the information of `aptamer`(not used) and `candidate-aptamer`(outputs) fields. If you want multiple tasks then initialize mutiple experiment templates as a single file and write the available number of processes in `n_jobs`, this script support the multiprocessing of tasks through default python multiprocessing library. 

After finished the tasks, the candidates of each task are saved in the `candidate-aptamer` field like this,
```json
"candidate-aptamer": {
    "score": [
        0.4675324675324675,
        0.45454545454545453,
        0.45454545454545453,
        0.45454545454545453,
        0.45454545454545453
    ],
    "seq": [
        "GUUAGACGUGGACGAACCUAGGGUGUAGAA",
        "AAGACGUGGACGAACCUAGGGUGUAGCAAC",
        "AAAGACGUGGACGAACCUAGGGUGUAGCAG",
        "GCAAGACGUGGACGAACCUAGGGUGUAGAG",
        "GAAGACGUGGACGAACCUAGGGUGUAGCAC"
    ],
    "ss": [
        ".(((.((.(((......)))..)).)))..",
        "....(.(((......))).)..........",
        ".....(.(((......))).).........",
        "(((...(.(((......))).).)))....",
        "......(((.....(((...)))....)))"
    ],
    "mfe": [
        -2.4000000953674316,
        -2.200000047683716,
        -2.200000047683716,
        -2.4000000953674316,
        -3.0
    ]
}
```
Please check the details in the sample configuration file in `quries/sample.json`.
