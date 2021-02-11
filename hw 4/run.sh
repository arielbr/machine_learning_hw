#!/bin/bash -e

DATADIR="datasets/"
TOPICS="earn acq"
KERNEL="ngram"

echo "Fitting..."
python3 classify.py \
    --mode train \
    --datadir $DATADIR \
    --topics $TOPICS \
    --kernel $KERNEL \
    --model-file "$KERNEL.${TOPICS%% *}.model"

echo "Predicting..."
python3 classify.py \
    --mode test \
    --datadir $DATADIR \
    --topics $TOPICS \
    --model-file "$KERNEL.${TOPICS%% *}.model"  \
    --kernel $KERNEL \
    --predictions-file "$KERNEL.${TOPICS%% *}.predictions"

echo "Computing Accuracy..."
python3 compute_accuracy.py \
    --datadir $DATADIR \
    --topics $TOPICS \
    --predictions-file "$KERNEL.${TOPICS%% *}.predictions" 