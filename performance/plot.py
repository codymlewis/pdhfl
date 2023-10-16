import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np


def create_plot(plot_data, save_filename):
    analytics = {k: np.array([pd[k] for pd in plot_data['analytics']]) for k in plot_data['analytics'][0].keys()}
    evaluations = np.array(plot_data['evaluation'])
    rounds = np.arange(len(evaluations)) + 1
    plt.plot(rounds, analytics['mean'], label="Local", marker='s', markevery=5)
    plt.fill_between(rounds, analytics['min'], analytics['max'], alpha=0.2)
    if "pdhfl" not in save_filename:
        plt.plot(rounds, evaluations, label="Global", marker='^', markevery=5)
        plt.legend(title="Model", loc='lower right')
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Top-1 Accuracy")
    plt.xlabel("Round")
    plt.tight_layout()
    plt.savefig(save_filename, dpi=320)
    print(f"Saved to {save_filename}")
    plt.clf()


if __name__ == "__main__":
    # TODO: Generate the data by the average of 5 experiments
    plot_data_fns = [fn for fn in os.listdir("results") if re.search("dataset=mnist.*seed=\d.*allocation=sim.*framework=(pdhfl|feddrop|heterofl|fjord)", fn)]
    print(plot_data_fns)
    for plot_data_fn in plot_data_fns:
        with open(f"results/{plot_data_fn}", "r") as f:
            create_plot(json.load(f), plot_data_fn[re.search("framework=", plot_data_fn).end():re.search("framework=[a-z]+_", plot_data_fn).end() - 1] + ".png")