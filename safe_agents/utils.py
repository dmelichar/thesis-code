import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def reject_outliers(data, m=2):
    mean = np.mean(data)
    std = np.std(data)
    distance = abs(data - mean)
    not_outlier = distance < m * std
    return data[not_outlier]


def plot_visuals(agent, scores, safety, loc="./results/"):
    name = agent.__class__.__name__ + "#" + str(len(scores))
    Path(loc).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="darkgrid")

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
    scores = reject_outliers(np.array(scores))
    scoredf = pd.DataFrame(
        [(i, v) for i, v in enumerate(scores)], columns=["episode", "score"]
    )
    sns.regplot(
        x="episode",
        y="score",
        data=scoredf,
        robust=True,
        ci=None,
        scatter_kws={"s": 10},
        ax=axs[0],
    )
    sns.boxplot(scores, showfliers=False, ax=axs[1])
    fig.savefig(loc + name + "-Scores.png", dpi=400)

    fig, axs = plt.subplots(ncols=1)
    safe_bin = np.array(safety)
    plt.step(np.arange(0, len(safe_bin)), safe_bin)
    plt.savefig(loc + name + "-Safety.png", dpi=400)
