import numpy as np
from numba import float64, int64
from numba.experimental import jitclass
from rl_env.pendulum_env import CustomInvertedPendulum
from agent.utils import SimplePolicyNet

spec_es = [
    ('policy', SimplePolicyNet.class_type.instance_type),
    ('env', CustomInvertedPendulum.class_type.instance_type),
    ('n_perturb', int64),
    ('sigma', float64),
    ('alpha', float64),
    ('episodes', int64),
    ('damping', float64),
    ('noise', float64),
    ('torque_penalty', float64)
    
]

@jitclass(spec_es)
class EvolutionStrategy:
    def __init__(self, policy, env, n_perturb=100, sigma=0.1, alpha=0.02, episodes=10):
        self.policy = policy
        self.env = env
        self.n_perturb = n_perturb
        self.sigma = sigma
        self.alpha = alpha
        self.episodes = episodes

    def evaluate(self):
        total = 0.0
        for _ in range(self.episodes):
            state = self.env.reset()
            episode_return = 0.0
            for _ in range(200):
                action = self.policy.get_action(state)
                state, reward, done = self.env.step(action)
                episode_return += reward
                if done:
                    break
            total += episode_return
        return total / self.episodes

    def optimize(self, iterations):
        dim = self.policy.param_dim()
        performance_log = []

        for t in range(iterations):
            base_params = self.policy.get_flat_params()
            rewards = np.zeros(self.n_perturb)
            noises = np.random.randn(self.n_perturb, dim)
            base_score = self.evaluate()

            for i in range(self.n_perturb):
                new_params = base_params + self.sigma * noises[i]
                self.policy.set_flat_params(new_params)
                rewards[i] = self.evaluate()

            gradient = np.dot((rewards - base_score), noises) / (self.n_perturb * self.sigma)
            updated_params = base_params + self.alpha * gradient
            self.policy.set_flat_params(updated_params)

            performance_log.append(base_score)

        return performance_log