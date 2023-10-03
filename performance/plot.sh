#!/bin/bash

for framework in "heterofl" "fjord" "feddrop" "pdhfl"; do
    python main.py --rounds 500 --steps-per-epoch 1 --dataset mnist --framework $framework --allocation sim --seed 1 --gen-plot-data
done

python plot.py
