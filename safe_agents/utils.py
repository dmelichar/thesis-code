import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
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

    fig, ax = plt.subplots(ncols=1)
    risk_rates = []
    for j in safety:
        safe = j.count(1)
        unsafe = j.count(0)
        r = 0 if unsafe == 0 else (unsafe / (safe + unsafe))
        risk_rates.append(r)
    ax.plot(risk_rates)
    plt.savefig(loc + name + "-Safety.png", dpi=400)


def plot_comparisson(data, episodes, loc="./results/"):
    df = pd.concat([pd.DataFrame(d) for d in data])
    df["episode"] = [i for _ in range(len(data)) for i in range(episodes)]
    r = []
    for i, row in df.iterrows():
        u = row["unsafe"]
        s = row["safe"]
        r.append(0 if u<1 else u/(u+s))
    df["risk_rate"] = r

    #f, axs = plt.subplots(ncols=2, figsize=(15,5))
    sns.lmplot(
       x="episode", y="scores", data=df, hue="agent", scatter_kws={"s": 10}, height=7
    )
    plt.savefig(loc + "".join([i["agent"] for i in data]) + "-Scores.png", dpi=400)
    sns.lmplot(
       x="episode", y="risk_rate", data=df, hue="agent", scatter_kws={"s": 10}, height=7
    )
    plt.savefig(loc + "".join([i["agent"] for i in data]) + "-Safety.png", dpi=400)
