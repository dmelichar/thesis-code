# Learning from heuristics to improve safety of Deep Reinforcement Agents

This repository is the official implementation of my [bachelor thesis](https://gitlab.com/danielmelichar/thesis-doc).


## Requirements

For development I used Python 3.6 and Tensorflow 1.4 on Conda 4.9.2.

I recommend to use the provided Conda environment config to install all requirements and the package itself.

```setup
./make_env safe_agents
```


## Training


```train
(safe_agents) python scripts/runner.py train --help
Usage: runner.py train [OPTIONS]

Options:
  --agent TEXT              Any of ['A2CAgent', 'A2CControlAgent',
                            'A2CSafeAgent', 'BaselineAgent', 'DQNAgent',
                            'DQNControlAgent', 'DQNSafeAgent']  [required]

  --episodes INTEGER        Number of training episodes  [required]
  --env-name TEXT           Environment to run.  [default: LunarSafe-v0]
  --save-loc TEXT           Path to save to after training  [default:
                            ./models/]

  --plot-loc TEXT           Path to save plots to  [default: ./results/]
  --compare / --no-compare  Generate comparission plots  [default: False]
  --help                    Show this message and exit.
```

To train the models in the paper, run these commands:

```train
(safe_agents) python scripts/runner.py train --agent DQN --episodes 300
```




## Evaluation

To evaluate a trained model visually, run this command:

```eval
(safe_agents) python scripts/runner.py evaluate --help
Usage: runner.py evaluate [OPTIONS]

Options:
  --agent TEXT              Any of ['A2CAgent', 'A2CControlAgent',
                            'A2CSafeAgent', 'BaselineAgent', 'DQNAgent',
                            'DQNControlAgent', 'DQNSafeAgent']  [required]

  --episodes INTEGER        Number of training episodes  [required]
  --env-name TEXT           Environment to run.  [default: LunarSafe-v0]
  --save-loc TEXT           Path to save to after training  [default:
                            ./models/]

  --compare / --no-compare  [default: False]
  --help                    Show this message and exit.
```

For example:

```train
(safe_agents) python scripts/runner.py evaluate --agent DQN --episodes 10
```


## Pre-trained Models

The pretrained models are stored in models. You can use them during evaluation.


## Results

The trained models achieved the following performance:


| Model name  | Top Score | Minimal Risk |
| ----------- | --------- | ------------ |
| DQN         |           |              |
| DQN Control |           |              |
| DQN Safe    |           |              |
| A2C         |           |              |
| A2C Control |           |              |
| A2C Safe    |           |              |


## Contributing

Please open a pull request for any issues. You can also contactm me directly, see my [website](https://melichar.xyz).

You can use, modify, or distribute the code however you see fit.

---

This README is using the [Papers With Code](https://paperswithcode.com/) template from [GitHub](https://github.com/paperswithcode/releasing-research-code).
