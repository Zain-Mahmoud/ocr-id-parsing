#!/bin/bash
if test $# -lt 2
then
    echo "Usage: ./generate_synthetic.sh [train | val] [num_samples] [num_augments_per_sample]" 2>&1
    exit 1
fi

rm -rf ./data/synthetic-ids/${1}/images/*
mkdir ./data/synthetic-ids/${1}/images/augment
rm -rf ./data/synthetic-ids/${1}/IDLabels.csv

cd ./src/data_generation

uv run ./EGID.py $1 $2 no 
uv run ./EGID.py $1 $2 yes $3

cd ../../

cat ./data/synthetic-ids/${1}/augment_IDLabels.csv >> ./data/synthetic-ids/${1}/IDLabels.csv
rm -rf ./data/synthetic-ids/${1}/augment_IDLabels.csv
rm -rf ./data/synthetic-ids/${1}/images/augment
