#!/usr/bin/env bash

if [ "$#" -lt 1 ]
then
    echo "usage: $0 [train | val | all]"
    exit 1
fi

while [ $# -gt 0 ]
do 
    if [ $1 == "all" ]
    then
        rm -rf ./data/synthetic-ids/train/images/*
        rm -rf ./data/synthetic-ids/train/line/*
        rm -rf ./data/synthetic-ids/train/*.csv

        rm -rf ./data/synthetic-ids/val/images/*
        rm -rf ./data/synthetic-ids/val/line/*
        rm -rf ./data/synthetic-ids/val/*.csv
        exit 0
    fi
    rm -rf ./data/synthetic-ids/${1}/images/*
    rm -rf ./data/synthetic-ids/${1}/line/*
    rm -rf ./data/synthetic-ids/${1}/*.csv
    shift 1
done

