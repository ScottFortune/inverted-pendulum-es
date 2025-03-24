# Reinforcement Learning with Evolutionary Strategies for Inverted Pendulum Domain

<p align="center">
  <img src="figs/trained_pendulum.gif" alt="Trained Pendulum GIF" width="300"/>
</p>

This project showcases a custom implementation of the Evolution Strategies (ES) algorithm applied to a continuous control problem inspired by the classic **Inverted Pendulum Swing-Up** task.

---

## Overview

- Custom JIT-compiled pendulum environment
- Lightweight neural network policy
- Evolution strategy for black-box optimization
- Notebook-driven experiments with plots, rollouts, and animations
- Grid search over hyperparameters to optimize training performance

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| **01_environment_walkthrough.ipynb** | Builds and explores the custom pendulum environment. Visualizes behavior under random and scripted controllers. |
| **02_es_walkthrough.ipynb** | Step-by-step breakdown of how evolution strategy works using small population and manual perturbations. |
| **02_es_training_experiments.ipynb** | Runs a grid search across σ, α, population size, and network size. Produces performance plots and logs. |
| **03_final_evaluation.ipynb** | Loads the best config, retrains, and animates the resulting pendulum behavior with a saved `.gif`. |


## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/inverted-pendulum-es.git
cd inverted-pendulum-es
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install as a package

```bash
pip install -e .
```

### 4. Run a notebook or training script

Explore the notebooks in [notebooks/](notebooks/)

Or run main.py

```bash
python main.py
```

--- 

## Evolution Strategy Overview

This project implements a simple but effective **black-box optimization** algorithm known as **Evolution Strategy (ES)** to train the neural network policy.

### Core idea:

1. **Perturb** the policy's parameters with random noise
2. **Evaluate** how each perturbation performs in the environment
3. **Estimate a gradient** by comparing good vs bad perturbations
4. **Update** the policy parameters using a weighted average of the noise

This approach requires no gradients from the environment or model — it's simple, parallelizable, and works surprisingly well on continuous control tasks.

The neural network observes a 3D state vector:

`[ sin(θ), cos(θ), angular velocity ]`

and outputs a **torque value** to balance the pendulum.

## Reward Function Explained

The reward function used in this environment is designed to encourage the pendulum to stay upright, minimize wobbling, and avoid excessive force:

```python
reward = - (omega_normalized**2 + 0.1 * dot_omega**2 + 0.001 * action**2)
```
Here's what each part means:
| Component                      | Description                                             |
|-------------------------------|---------------------------------------------------------|
| `omega_normalized**2`         | Penalizes the angle error from upright (θ = 0)          |
| `0.1 * dot_omega**2`          | Penalizes large angular velocity (reduces wobbling)     |
| `0.001 * action**2`           | Penalizes strong torque (encourages smooth control)     |



The maximum reward is 0 when the pendulum is perfectly balanced upright, not moving, and no torque is applied. The more it deviates from this, the more negative the reward becomes.