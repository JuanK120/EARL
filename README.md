# EARL: Explanations using Alternative Realities for Reinforcement Learning 

Code for the paper Explaining Reinforcement Learning Decisions in Self-adaptive Systems.

This repository provides code for the Python package EARL which implements methods for generating counterfactual explanations for RL tasks.

### Counterfactual Methods

EARL implements four counterfactual methods:
* *GANterfactual-RL*: implemented and modififed from the original version (https://github.com/hcmlab/GANterfactual-RL) to adapt it to more general use-cases.
* *RACCER*: implemented and modified to multi-discrete action spaces from the original version
* *RACCER-Advance and RACCER-Rewind*: adaptations of the RACCER algorithm that uses generative algorithm instead of heuristic tree search for faster counterfactual search.

### Recreating Experiments on CitiBikes task

To recreate the experiments from the original paper, please follow the steps below.

#### Requirements

```bash
python >= 3.8

```
#### Installation

```bash

git clone git@github.com:anonymous902109/earl.git
conda create -n earl python=3.8
conda activate earl
pip install -r requirements.txt  

```

#### Running Experiments

```python

python citibikes/run_citibikes.py

```
