#!/bin/bash
mkdir -p exports

i=0
while [ -f exports/$i/G_matrix_S273.csv ]; do
    ((i++))
done

if [ $i -eq 0 ]; then
    mkdir -p exports/$i
    python ./train.py train --model exports/$i/model.pth --eval-freq 1000 --train-episodes 5000
    python ./train.py export --model exports/$i/model.pth --export-dir exports/$i
    ((i++))
fi

while [ 1 ]; do
    echo "################ Running sim number $i ################"
    mkdir -p exports/$i
    cp exports/$((i-1))/model.pth exports/$((i-1))/Simulated_G_best_S273.csv exports/$i
    python ./train.py retrain --model exports/$i/model.pth --eval-freq 1000 --train-episodes 5000
    python ./train.py export --model exports/$i/model.pth --export-dir exports/$i
    ((i++))
done