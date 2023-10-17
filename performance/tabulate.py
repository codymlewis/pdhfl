import argparse
import os
import re
import json
import jax
import numpy as np
import pandas as pd


def formatter(cell_value):
    cell_value = cell_value.replace('%', '\\%')
    # Format algorithms
    cell_value = cell_value.replace('fedavg', 'FedAVG')
    cell_value = cell_value.replace('feddrop', 'FedDrop')
    cell_value = cell_value.replace('fjord', 'FjORD')
    cell_value = cell_value.replace('heterofl', 'HeteroFL')
    cell_value = cell_value.replace('local', 'Local')
    cell_value = cell_value.replace('pdhfl', 'PDHFL')
    # Format datasets
    cell_value = cell_value.replace('cifar', 'CIFAR-')
    cell_value = cell_value.replace('mnist', 'MNIST')
    cell_value = cell_value.replace('har', 'HAR')
    cell_value = cell_value.replace('nbaiot', 'N-BAIoT')
    cell_value = cell_value.replace('tinyimagenet', 'Tiny-ImageNet')
    return cell_value


def format_final_table(styler):
    styler = styler.hide()
    styler = styler.format(formatter=formatter)
    return styler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LaTeX table from the experiments results.")
    parser.add_argument("-a", "--allocation", type=str, default="cyclic", help="The allocation type to look at the results from.")
    args = parser.parse_args()

    result_data_fns = [fn for fn in os.listdir("results/") if "rounds=50" in fn]
    datasets = set(fn[re.search('dataset=', fn).end():re.search('dataset=[a-z0-9]+_', fn).end() - 1] for fn in result_data_fns)
    frameworks = set(fn[re.search('framework=', fn).end():re.search('framework=[a-z0-9]+_', fn).end() - 1] for fn in result_data_fns)
    allocations = set(fn[re.search('allocation=', fn).end():re.search('allocation=[a-z0-9]+_', fn).end() - 1] for fn in result_data_fns)

    all_data = {d: {f: [] for f in frameworks} for d in datasets}
    for dataset in datasets:
        for framework in frameworks:
            for result_data_fn in filter(lambda fn: dataset in fn and framework in fn and args.allocation in fn, result_data_fns):
                with open(f"results/{result_data_fn}", 'r') as f:
                    all_data[dataset][framework].append(json.load(f))
            if len(all_data[dataset][framework]):
                all_data[dataset][framework] = jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *all_data[dataset][framework])

    full_results = {"Dataset": [], "Algorithm": [], "Client Mean (STD)": [], "Global": []}
    for dataset in datasets:
        for framework in frameworks:
            if all_data[dataset][framework]:
                full_results['Dataset'].append(dataset)
                full_results['Algorithm'].append(framework)
                full_results['Client Mean (STD)'].append(
                    f"{all_data[dataset][framework]['analytics'][-1]['mean']:.3%} ({all_data[dataset][framework]['analytics'][-1]['std']:.3%})"
                )
                full_results['Global'].append(f"{all_data[dataset][framework]['evaluation'][-1]:.3%}")
    full_results = pd.DataFrame(full_results)
    full_results = full_results.sort_values(by=["Dataset", "Algorithm"])
    print(full_results.style.pipe(format_final_table).to_latex(position_float='centering'))