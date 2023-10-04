import os
import re
import json
import numpy as np
import pandas as pd


if __name__ == "__main__":
    allocation = "sim"
    result_data_fns = [fn for fn in os.listdir("results/") if "plot" not in fn]
    datasets = set(fn[re.search('dataset=', fn).end():re.search('dataset=[a-z0-9]+_', fn).end() - 1] for fn in result_data_fns)
    frameworks = set(fn[re.search('framework=', fn).end():re.search('framework=[a-z0-9]+_', fn).end() - 1] for fn in result_data_fns)
    allocations = set(fn[re.search('allocation=', fn).end():re.search('allocation=[a-z0-9]+_', fn).end() - 1] for fn in result_data_fns)

    analytics_data = {d: {f: None for f in frameworks} for d in datasets}
    for dataset in datasets:
        for framework in frameworks:
            for result_data_fn in filter(lambda fn: dataset in fn and framework in fn and allocation in fn, result_data_fns):
                with open(f"results/{result_data_fn}", 'r') as f:
                    result_data = json.load(f)
                if analytics_data[dataset][framework] is None:
                    analytics_data[dataset][framework] = {k: [v] for k, v in result_data['analytics'].items()}
                else:
                    for k, v in result_data['analytics'].items():
                        analytics_data[dataset][framework][k].append(v)

    full_results = {"dataset": [], "framework": [], "analytics mean": [], "analytics std": []}
    for dataset in datasets:
        for framework in frameworks:
            full_results['dataset'].append(dataset)
            full_results['framework'].append(framework)
            full_results['analytics mean'].append(np.mean(analytics_data[dataset][framework]['mean']))
            full_results['analytics std'].append(np.mean(analytics_data[dataset][framework]['std']))
    full_results = pd.DataFrame(full_results)
    print(full_results)