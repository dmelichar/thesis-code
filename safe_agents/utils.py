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

    fig, axs = plt.subplots(ncols=1)
    labels = 'unsafe', 'safe' # 0=unsafe, 1=safe
    sizes2 = [safety.count(0), safety.count(1)]
    plt.pie(x=sizes2, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.savefig(loc + name + "-Safety.png", dpi=400)

def plot_multi_scores(agents: list, scores: list, loc="./results/"):
    list_len = len(scores)
    bins = int(list_len/len(agents))
    l = []
    for i in range(len(agents)):
        for j in range(bins):
            l.append(agents[i])
    m = []
    for i in range(len(agents)):
        for j in range(bins):
            m.append(j)
    data = {"episodes": m,
        "agent": l,
        "scores": scores}
    df = pd.DataFrame.from_dict(data)
    scores = reject_outliers(np.array(scores))
    sns.lmplot(
        x="episodes",
        y="scores",
        data=df,
        hue="agent",
        scatter_kws={"s": 10},
        size=7
    )
    plt.savefig(loc + 'multi' + str(list_len) + '-Scores.png', dpi=400)


def plot_multi_safety(agents: list, loc="./results/"):
    widths = [2, 3]
    heights = [2 for _ in range(len(agents))]
    fig = plt.figure(constrained_layout=True, figsize=(15,8))
    spec = fig.add_gridspec(ncols=2, nrows=len(agents), width_ratios=widths, height_ratios=heights)
    agent_v = list(agents.values())

    for row in range(len(agents)):
        agent_str = str(list(agents.keys())[row])
        for col in range(2):
            ax = fig.add_subplot(spec[row, col])
            if col == 0:
                labels = 'unsafe', 'safe'
                merged = list(itertools.chain.from_iterable(agent_v[row]))
                sizes = [merged.count(0), merged.count(1)]
                ax.pie(x=sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
                ax.set_title(agent_str + ' safety piechart')
            if col == 1:
                s = [(i.count(0), i.count(1)) for i in agent_v[row]]
                unsafe = [i[0] for i in s]
                safe = [i[1] for i in s]
                labels = [str(i) for i in range(len(agent_v[row]))]
                x = np.arange(len(labels))  # the label locations
                width = 0.35  # the width of the bars
                p1 = ax.bar(x - width/2, unsafe, width, label='unsafe')
                p2 = ax.bar(x - width/2, safe, width, label='safe')
                ax.legend()
                ax.set_title(agent_str + ' time in safe bounds per episode')

    
    
