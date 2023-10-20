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

        if [[ $dataset == "nbaiot" ]]; then
            rounds=10
            batch_size=128
        elif [[ $dataset == "cifar100" ]]; then
            rounds=100
            batch_size=32
        else
            rounds=50
            batch_size=128
        fi

        for allocation in ${allocations[@]}; do
            for seed in {1..5}; do
                python main.py --rounds $rounds --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size $batch_size

                if [[ $dataset == "nbaiot" ]]; then
                    python main.py --rounds $rounds --clients 90 --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size $batch_size
                    python main.py --rounds $rounds --clients 90 --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size $batch_size --proportion-clients 0.1
                elif [[ $dataset == "cifar100" ]]; then
                    python main.py --rounds $rounds --dataset $dataset --framework $framework --seed $seed --allocation $allocation --batch-size $batch_size --proportion-clients 0.1
                fi
            done
        done
    done
done
