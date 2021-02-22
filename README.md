# Learning from heuristics to improve safety of Deep Reinforcement Agents

This repository is the official implementation of my [bachelor thesis](https://).


## Requirements

For development I used Python 3.6 and Tensorflow 1.4.

I recommend to use the provided Conda environment config to install all requirements and the package itself.

```setup
conda env create -f env.yml -n safe_agents
conda run -n safe_agents pip install -e .
```


## Training

To train the model(s) in the paper, run this command:

```train
(safe_agents) python scripts/runner.py train ENV_NAME AGENT EPISODES
```


## Evaluation

To evaluate the model(s), run this command:

```eval
(safe_agents) python scripts/runner.py evaluate ENV_NAME AGENT EPISODES
```


## Pre-trained Models

The pretrained models are stored in models. You can use them during evaluation.


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.


## Contributing

Please open a pull request for any issues. You can also contactm me directly, see my [website](https://melichar.xyz).

You can use, modify, or distribute the code however you see fit.

---

This README is using the [Papers With Code](https://paperswithcode.com/) template from [GitHub](https://github.com/paperswithcode/releasing-research-code).
