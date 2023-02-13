
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)

# Zeroth-Order Based Deep Reinforcement Learning Library 

DRL is gaining popularity in data science in combination with the advances in deep learning. It starts gaining prominence as viable and practical tools for both research and industry applications. There are several issues faced by researcher and practitioners:
- The volume and complexity of RL algorithms have risen substantially over time. This has made it more difficult for researchers/practitioners to quickly prototype concepts and has resulted in major concerns with the issues in reproductivity and productivity. 
- There are no first zeroth-order based DRL framework/platform that possess robustness-seeking property and naturally supports multi-agent environment.
- Existing RL algorithms were evaluated in a limited set of simulated environments (e.g., Atari games). How these algorithms perform in a new environment/a simulator of a different problem domain remained unknown. 
- There exist various simulators in various domains, integrating them with RL algorithms is not straightforward. 


Zeroth-Order Based Deep Reinforcement Learning Library (ZOO-RL) is a reliable RL library for developing, training, 
evaluating, and benchmarking zeroth-order based deep reinforcement learning. To the best of our knowledge, this is 
**the First zeroth-order based DRL Research Library** in the world with the following features:

- Fast prototyping of zeroth-order based DRL algorithms via simple YAML configuration files.
- Parallel training/testing DRL at various scales of execution.
- Supporting third-party simulation environments and NIWA's existing and future simulators (e.g., the dynamic 
coastal environment and the energy-aware data centre simulation).
- Provide user cases, such as fishery management, which integrates existing fishery simulation models and DRL techniques 
to achieve a sustainable fishery 

# Different from First Ordered Optimization
- ZOO is not restricted to differentiable policies; 
- ZOO perturbs the policy in parameter space rather than in action space, which leads to state-dependent and temporally-extended exploration; 
- Zeroth-order population-based optimisation possesses robustness-seeking property and diverse policy behaviours.


# YAML your DRL algorithm
A DRL algorithms is created by configuring three ingredients, i.e., environments, optimization 
methods, and policies in a YAML file. An example of a DRL algorithms based on Open AI ES optimization method for a use 
case on cloud-based workflow resource is provided below: 

```python
env:                            #enviromental simulator 
  name: WorkflowScheduling-v0       #simulator name
  seed: 0                           #random seed
  traffic_pattern: CONSTANT         #job arrived at data center at a constant rate
  gamma: 5                          #SLA (service level agreement) penalization factor
  wf_size: S                        #Workflow size, varied from small(S), medium(M), large(L), and extra large(XL 
  wf_num: 30                        #Total number of workflow sent to data center within an episode
policy:                         #policy model
  agent_0                        #agent id
    name: model_rnn              #policy name
    add_gru: True                    #the use of RNN(GRU)
  shared_params: None                #the settings for shared-parameter multiple agents
optim:                          #optimization method
  name: es_openai                   #optimization name
  population_size: 40               #population size
  maximization: True                #It is maximization problem?
  generation_num: 3000              #total number of generations
  input_running_mean_std: True      #Normalize runtime mean and standard deviation for observations 
  reward_shaping: True              #shape reward values?
  reward_norm: False                #normalize reward values?
  sigma_init: 0.01                  #noise standard deviation in open AI ES
  sigma_decay: 0.999                #decay of the above noise
  learning_rate: 0.001              #step size of optimizer
  learning_rate_decay: 0.9999       #decay of the above step size
```
## The first YAML ingredient: optimization method 
We have included and tested each optimization algorithms.


| **Optimization Methods** | **Progress**        | Discrete Action | Continous Action | Recurrent Model | Parallel Training |
|--------------------------|---------------------|---------------------|---------------------|---------------------|-------------------|
| Open AI ES               | ✓                   |✓                   |✓                   |✓                   | ✓                 |
| Uber AI Deep GA          | ✓                   |✓                   |✓                   |✓                   | ✓                 |
| Deep Mind NES            | ✓                   |✓                   |✓                   |✓                   | ✓                 |

## The second YAML ingredient: environmental simulator

Environments that support the OpenAI Gym's interface (`__init__`, `reset`, `step`, and `close` methods) can be used.

| **Evironmental Simulators** | **Progress** | Discrete Action | Continous Action | Static Action Space | POMDP              |
|-----------------------------|--------------|-----------------|------------------|------------------|--------------------|
| Cartpole                    | ✓            | ✓               | x                |✓                | ✓                |
| Lunarlander                 | ✓            | x               | ✓                |✓                | ✓                |
| Workflow scheduling         | ✓            | ✓               | x                |x               | x               |
| Fish management             | ✓            | x               | ✓                |✓                | x               |

## The third YAML ingredient: policy


| **Policy Models** | **Progress**        |
|-------------------|---------------------|
| Simple ANN        | ✓                   |
| Simple GRU        | ✓                   |

## Class Diagrams in all ingredients
❗This won't be frequently updated.
![](test/diagram.png)

## Performance report on Benchmark problems

| Discrete Action                                           | Continous Action                                                            | 
|-----------------------------------------------------------|-----------------------------------------------------------------------------|
| <br /><img src="test/training_cartpole.png" width="480"/> | <br /><img src="test/training_lunarlander.png" alt="cartpole" width="480"/> |

## Getting started

### Prerequisites
* This repository is tested on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.6.1+
    ```
    $ conda create -n zoo-rl python=3.7.9
    $ conda activate zoo-rl
    ```
### Installation
First, clone the repository.
```
git clone git@git.niwa.co.nz:rl-group/zoo-rl.git
cd zoo-rl
conda install --file requirements.txt
```


### Arguments for training DRL algorithms

In addition, there are various argument settings for running algorithms. If you check the options to run file you should command 
```
python run_rl
```
- `--config`
    - Set YAML path.
- `--processor-num <int>`
    - Specify processor number for multiprocessing.
- `--log`
    - Turn on logging.
- `--eval-ep-num <int>`
    - Set the number of episodes for training.
- `--save-period <int>`
    - Set saving period of model.
- `--seed <int>`
    - Overwrite seed number in YAML

### Arguments for testing DRL algorithms
```
python eval_rl
```
just change which_log variable in eval_rl.py to switch testing for different use cases.
```
    which_log = 'logs/WorkflowScheduling-v0'
```
## Cite this platform
```
@misc{wang2022niwazoo,
  title={NIWA ZOO-RL Platform},
  author={Wang, Chen and Huang, Victoria and Gavin, Tasker},
  year={2022}
}
```
## References