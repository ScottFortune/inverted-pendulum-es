import numpy as np
from numba import jit, float64
from numba.experimental import jitclass

spec = [
    ('G', float64),
    ('L', float64),
    ('M', float64),
    ('MAX_SPEED', float64),
    ('MAX_TORQUE', float64),
    ('dt', float64),
    ('omega', float64),
    ('dot_omega', float64),
    ('damping', float64),
    ('noise', float64)
    
]

@jit(nopython=True)
def clip(x, min_val, max_val):
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x

@jitclass(spec)
class CustomInvertedPendulum:
    def __init__(self):
        self.G = 9.81
        self.L = 1.0
        self.M = 1.0
        self.MAX_SPEED = 8.0
        self.MAX_TORQUE = 2.0
        self.dt = 0.05
        self.damping = 0.99

        self.reset()

    def reset(self):
        self.omega = np.random.uniform(-5 * np.pi / 6, 5 * np.pi / 6)
        self.dot_omega = np.random.uniform(-1, 1)
        return np.array([self.omega, self.dot_omega], dtype=np.float64)

    def step(self, action):
        action = clip(action, -self.MAX_TORQUE, self.MAX_TORQUE)

        ddot_omega = (3 * self.G / (2 * self.L)) * np.sin(self.omega) + (3 * action / (self.M * self.L ** 2))
        self.dot_omega += ddot_omega * self.dt
        self.dot_omega = clip(self.dot_omega, -self.MAX_SPEED, self.MAX_SPEED)
        self.dot_omega *= self.damping
        noise = np.random.normal(0, 0.01)
        self.dot_omega += noise
        self.omega += self.dot_omega * self.dt

        omega_normalized = (self.omega + np.pi) % (2 * np.pi) - np.pi
        
        reward = - (omega_normalized ** 2 + 0.1 * self.dot_omega ** 2 + 0.001 * action ** 2)

        done = False
        return np.array([self.omega, self.dot_omega], dtype=np.float64), reward, done
