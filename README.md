# Aptamer Monte-Carlo Tree Search (Apta-MCTS)
- Search candidate aptamer sequences that expected to have high interaction with a target protein sequence
  - **Score function** 
    - Random Forest based Aptamer-Protein Interaction Classifier
  - **Search method** 
    - Monte-Carlo Tree Search for Nucleotide Sequence
  - *Note that our method is an activation maximization framework that composed score function and search method, so these components can be replaced any modules*

## How to use?
```text
usage: apta_mcts.py [-h] -i INPUT_PROTEIN [-k TOP_K] [-bp BP_SIZE] [-n NUM_ITERATIONS] -s SCORE_FUNCTION [-e EX_PROTEIN] [-o OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  -i INPUT_PROTEIN, --input-protein INPUT_PROTEIN
                        Target protein sequences (.fa or .fasta) file path
  -k TOP_K, --top-k TOP_K
                        Number of candidate aptamer sequences to be selected for each iteration
  -bp BP_SIZE, --bp-size BP_SIZE
                        Length of candidate aptamer sequences
  -n NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        Number of iterations for each MCTS process
  -s SCORE_FUNCTION, --score-function SCORE_FUNCTION
                        Score function (pre-trained Aptamer-Protein Interaction Classifier) for candidate aptamer samples
  -e EX_PROTEIN, --ex-protein EX_PROTEIN
                        The protein sequences (.fa or .fasta) file path do not want to interact with candidate aptamers in sampling process
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save result .csv file
```
## Example
```bash
cd src
python apta_mcts.py \
-i 'examples/rcsb_pdb_6GOF.fasta' \
-k 100 \
-bp 10 \
-n 1000 \
-s 'score_functions/rf-ictf-li2014/mcc0.484-ppv1.000-acc0.822-sn0.290-sp1.000-npv0.809-yd0.290-35trees' \
-e 'examples/rcsb_pdb_7JTL.fasta' \
-o './examples'
```

## Output (.csv)
- Candidate results will be saved for each protein sequence in the input `.fa` file
```text
aptamer_protein_interaction_score,primary_sequence,secondary_structure,minimum_free_energy
0.5428571428571428,ACUGCUACCAGUACGACAACGCCAGGUUUG,((((....))))..................,-1.7000000476837158
0.5428571428571428,GGAUGCUACCAGUACGACAACGCCAGGUUG,((......)).......((((.....)))),-3.0999999046325684
0.5142857142857142,GACGCUACCAGUACGACAACGCCAGGUUGU,...............(((((.....))))),-4.0
0.5142857142857142,CACUGCUACCAGUACGACAACGCCAGGUUC,.((((....)))).................,-1.7999999523162842
0.5142857142857142,UGCUACCAGUACGACAACGCCAGGUUUGGA,.....((((.((...........)))))).,-2.5999999046325684
0.5142857142857142,UGCUACCAGUACGACAACGCCAGGUUUGGC,..................((((....)))),-4.0
0.5142857142857142,UGAGGCUACCAGUACGACAACGCCAGGUUG,...(((...............)))......,-2.700000047683716
0.5142857142857142,CGAACCAGUACGACAACGCCAGUUUGCGCU,(((((..((........))..)))))....,-2.4000000953674316
0.5142857142857142,CACUUUGACCAGUACGACAACGCCAGCGUA,............((((.(.......))))),-0.699999988079071
0.5142857142857142,UGCUACCAGUACGACAACGCCAGGUUUGAC,....(((....((....))...))).....,-0.699999988079071
```

## Install ViennaRNA package
```bash
# https://github.com/ViennaRNA/ViennaRNA
tar -zxvf ViennaRNA-2.5.0.tar.gz
cd ViennaRNA-2.5.0
./configure --prefix=/<any>/<path>/ --without-perl
make install

# If you have import error when `import RNA`
# copy the installed python site-packages to your python site-packages dir
cp -r /<any>/<path>/lib/<your-python-ver>/site-packages/RNA /<your>/<python-site-packages>/
```

## Citation
If you apply this library or model to any project and research, please cite our article:
```text
@article{lee2021predicting,
  title={Predicting aptamer sequences that interact with target proteins using an aptamer-protein interaction classifier and a Monte Carlo tree search approach},
  author={Lee, Gwangho and Jang, Gun Hyuk and Kang, Ho Young and Song, Giltae},
  journal={PloS one},
  volume={16},
  number={6},
  pages={e0253760},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA}
}
```