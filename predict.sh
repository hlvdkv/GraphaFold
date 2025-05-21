#!/bin/bash

# D="/home/mjustyna/data/graphafold_data/casp"
# O="casp"
D="/home/mjustyna/graphafold-analysis/large"
O="large"

for f in $D/*.idx; do
  base=$(basename $f .idx)
  # echo $f >>times.txt
  echo $base.fasta >>times.txt
  mkdir -p $O/$base/cmt
  mkdir -p $O/$base/idx
  cp $D/$base.cmt $O/$base/cmt
  cp $D/$base.idx $O/$base/idx
  (time python src/graphafold/script.py $O/$base) 2>>times.txt
  # (time python ufold_predict.py --nc True) 2>>times.txt
done
