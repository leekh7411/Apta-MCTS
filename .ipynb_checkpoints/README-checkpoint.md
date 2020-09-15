# Apta-MCTS 
Supporting code for the paper **"Predicting aptamer sequences that interact with target proteins using Aptamer-Protein Interaction classifiers and a Monte-Carlo tree search approach"**(currently being prepared).

## Aptamer-Protein-Interaction (API) classifiers
Implementations of how to train the Aptamer-Protein Interaction classifier using Random Forest are described in a `classifier.py`. We train two versions of API classifiers according to the datasets. The benchmark datasets are available in the `datasets/` folder with json format files.
#### *List of benchmark dataset*
| Source | Positives | Negatives | Description |
|--|--|--|--|
| [Li et al, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086729) | 580 | 1740 | Training data for API classifiers |
| [Li et al, 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0086729) | 145 | 435 | Validation data for API classifiers |
| [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) | 157 | 493 | Training data for API classifiers |
| [Lee and Han, 2019](https://ieeexplore.ieee.org/document/8890705) | 56 | 56 | Validation data for API classifiers and Apta-MCTS |

#### *How to train API classifiers*
```Bash
python3 classifier.py \
-dataset_dir=datasets/li2014 \
-tag=rf-iCTF-li2014 \
-min_trees=35 \
-max_trees=200 \
-n_jobs=20 \
-num_models=1000
```
The `classifier.py` takes inputs as a sequence list written of json format (see the .json files in `datasets/<source>/train.json or test.json`) through the parameter `dataset_dir`. Second parameter `tag` is an identifier for your API classifiers.  The range for number of trees in Random Forest algorithm are feed through the `min_trees` and `max_trees` (integer values). Due to this classifiers built on the scikit-learn packages, `n_jobs` has same meaning with the scikit-learn models. We select best scored API classifier for the score function of Apta-MCTS, so there are multiple models (number of models = `num_models`) gerneated for selection.  Below table is binary classifier evaluation scores with respect to the our appoaches (RF with iCTF) and others for two benchmark datasets 

| API classifier    | Training set     | Sensitivity | Specificity | Accuracy | Yuden's Index | MCC   |
|-------------------|------------------|-------------|-------------|----------|---------------|-------|
| Li et al. 2014    | Li et al. 2014   | 0.483       | 0.871       | 0.774    | 0.354         | 0.372 |
| Zhang et al. 2016 | Li et al. 2014   | **0.738**       | 0.713       | 0.719    | **0.451**         | 0.398 |
| **RF with iCTF**      | Li et al. 2014   | 0.303       | **0.999**       | **0.826**    | 0.303         | **0.496** |
| Lee and Han 2019  | Lee and Han 2019 | 0.768       | **0.661**       | 0.714    | 0.429         | 0.431 |
| **RF with iCTF**      | Lee and Han 2019 | **0.982**       | 0.554       | **0.768**    | **0.536**         | **0.593** |

With the perspective of machine learning models, the result of **RF with iCTF** is not a good performaces, but we just designed the classifiers only takes the patterns of sequences as inputs (looking forward to the day when the data volume [aptamer-protein complexes] grows). 