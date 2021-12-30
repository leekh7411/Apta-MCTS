# Aptamer Monte-Carlo Tree Search (Apta-MCTS)
- Search candidate aptamer sequences that expected to have high interaction with a target protein sequence
  - Score function - Random Forest based Aptamer-Protein Interaction Classifier
  - Search method - Monte-Carlo Tree Search for Nucleotide Sequence
    - Note that our method is an activation maximization framework that composed with a score function and search method, so the components can be replaced others

## How to use?
```bash
python apta_mcts.py \
-i examples/rcsb_pdb_6GOF.fasta \
-k 100 \
-bp 10 \
-n 1000 \
-s 'score_functions/rf-ictf-li2014/mcc0.484-ppv1.000-acc0.822-sn0.290-sp1.000-npv0.809-yd0.290-35trees' \
-e examples/rcsb_pdb_7JTL.fasta
```

### ViennaRNA package
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