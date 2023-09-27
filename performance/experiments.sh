#!/bin/bash

for framework in "pdhfl" "heterofl" "fjord" "feddrop" "local" "fedavg"; do
    for dataset in "mnist" "har" "nbaiot_10" "cifar10" "cifar100"; do
        if [[ $framework == "fedavg" ]] || [[ $framework == "local" ]]; then
            allocations=("full")
        else
            allocations=("cyclic" "sim")
        fi

        for allocation in ${allocations[@]}; do
            for seed in {1..5}; do
                python main.py --rounds 300 --steps-per-epoch 1 --dataset $dataset --framework $framework --seed $seed --allocation $allocation
            done
        done
    done
done