#!/bin/bash


frameworks=("pdhfl" "heterofl" "fjord" "feddrop" "local" "fedavg")
datasets=("mnist" "har" "nbaiot" "cifar10" "cifar100")


for framework in ${frameworks[@]}; do
    for dataset in ${datasets[@]}; do
        if [[ $framework == "fedavg" ]] || [[ $framework == "local" ]]; then
            allocations=("full")
        else
            allocations=("cyclic" "sim")
        fi

        for allocation in ${allocations[@]}; do
            for seed in {1..5}; do
                python main.py --rounds 50 --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size 128

                if [[ $dataset == "nbaiot" ]]; then
                    python main.py --rounds 50 --clients 90 --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size 128
                    python main.py --rounds 50 --clients 90 --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size 128 --proportion-clients 0.1
                elif [[ $dataset == "cifar100" ]]; then
                    python main.py --rounds 50 --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size 128 --proportion-clients 0.1
                fi
            done
        done
    done
done
