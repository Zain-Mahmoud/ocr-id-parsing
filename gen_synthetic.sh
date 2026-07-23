#!/bin/sh
rm -rf ./data/synthetic-ids/images/*
mkdir ./data/synthetic-ids/images/augment
rm -rf ./data/synthetic-ids/IDLabels.csv

cd datagen

uv run ./EGID.py $1 no 
uv run ./EGID.py $1 yes $2

cd ..

cat ./data/synthetic-ids/augment_IDLabels.csv >> ./data/synthetic-ids/IDLabels.csv
rm -rf ./data/synthetic-ids/augment_IDLabels.csv
rm -rf ./data/synthetic-ids/images/augment
