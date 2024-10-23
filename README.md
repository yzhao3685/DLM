## Installation and run
```
conda create -n py311 python=3.11                                                             
conda activate py311
pip install -e .
mv /home/ezipe/.miniconda3/compiler_compat/ld /home/ezipe/.miniconda3/compiler_compat/ld-backup
sudo apt install libopenmpi-dev gcc

pip install mpi4py

./make_dirs.sh
python run/gemini.py   
```

## Train base rewards
```python run/gemini.py --config '{"reward_from": "base"}' --wandb | tee base-01-28.log```


**A Decision-Language Model (DLM) for Dynamic Restless Multi-Armed Bandit Tasks in Public Health**
==================================

Restless multi-armed bandits (RMAB) have demonstrated success in optimizing resource allocation for large beneficiary populations in public health settings. Unfortunately, RMAB models lack flexibility to adapt to evolving public health policy priorities. Concurrently, Large Language Models (LLMs) have emerged as adept automated planners across domains of robotic control and navigation. In this paper, we propose a Decision Language Model (DLM) for RMABs, enabling dynamic fine-tuning of RMAB policies in public health settings using human-language commands. We propose using LLMs as automated planners to (1) interpret human policy preference prompts, (2) propose reward functions as code for a multi-agent RMAB environment, and (3) iterate on the generated reward functions using feedback from grounded RMAB simulations. We illustrate the application of DLM in collaboration with ARMMAN, an India-based non-profit promoting preventative care for pregnant mothers, that currently relies on RMAB policies to optimally allocate health worker calls to low-resource populations. We conduct a technology demonstration in simulation using the Gemini Pro model, showing DLM can dynamically shape policy outcomes using only human prompts as input.


## Setup

Main file for PreFeRMAB, the main algorithm is `agent_oracle.py`

- Clone the repo:
- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`

To run Synthetic dataset from the paper, run 
`bash run/job.run_rmabppo_counterexample.sh`

Code adapted from https://github.com/killian-34/RobustRMAB, the github repo accompanying the paper "Restless and Uncertain: Robust Policies for Restless Bandits via Deep Multi-Agent Reinforcement Learning" in UAI 2023. 
