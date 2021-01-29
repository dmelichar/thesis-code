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

def plot_visuals(agent, scores):
    name = agent.__class__.__name__ + '#' + str(len(scores))
    loc = './results/'
    Path(loc).mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
    scores = reject_outliers(np.array(scores))
    df = pd.DataFrame([(i, v) for i, v in enumerate(scores)], columns=["episode", "score"])
    sns.regplot(x="episode", y="score", data=df, robust=True, ci=None, scatter_kws={"s": 10}, ax=axs[0])
    sns.boxplot(scores, showfliers=False, ax=axs[1])
    fig.savefig(loc + name + '-Scores.png', dpi=400)

    bounds = bounds = agent.get_safe_bounds()
    df = pd.DataFrame([(abs(v[0]-v[1])) for i, v in enumerate(bounds)], columns=["b"])
    sns.displot(df, x="b", kind="kde")
    plt.savefig(loc + name + '-Safety.png', dpi=400)