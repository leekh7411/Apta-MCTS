#!/bin/bash
cd /packages/simrna/intel64bit
mkdir /tmp/SIMRNA/outputs/$1
echo "$2" > /tmp/SIMRNA/outputs/$1/seq.txt
echo "$3" > /tmp/SIMRNA/outputs/$1/ss.txt
./SimRNA \
-s /tmp/SIMRNA/outputs/$1/seq.txt \
-S /tmp/SIMRNA/outputs/$1/ss.txt \
-c configSA.dat \
-o $1 > $1.log \
./trafl_extract_lowestE_frame.py $1.trafl
./SimRNA_trafl2pdbs $1*.pdb $1_minE.trafl 1 AA
mv $1* /tmp/SIMRNA/outputs/$1/