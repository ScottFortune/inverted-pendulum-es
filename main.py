from rl_env.pendulum_env import CustomInvertedPendulum
from agent.utils import SimplePolicyNet
from agent.evolution_strategy import EvolutionStrategy
import numpy as np
import matplotlib.pyplot as plt

def train_and_plot():
    env = CustomInvertedPendulum()
    policy = SimplePolicyNet(input_dim=3, hidden_dim=16, output_dim=1)

    es = EvolutionStrategy(
        policy=policy,
        env=env,
        n_perturb=100,
        sigma=0.2,
        alpha=0.02,
        episodes=5
    )

    rewards = es.optimize(300)

    plt.plot(rewards)
    plt.title("Training Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png")
    print("Final reward:", rewards[-1])

if __name__ == "__main__":
    train_and_plot()
