import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

spec_nn = [
    ('input_dim', int64),
    ('hidden_dim', int64),
    ('output_dim', int64),
    ('weights1', float64[:, :]),
    ('bias1', float64[:]),
    ('weights2', float64[:, :]),
    ('bias2', float64[:])
]

@jitclass(spec_nn)
class SimplePolicyNet:
    def __init__(self, input_dim=3, hidden_dim=6, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.weights1 = np.ascontiguousarray(np.random.randn(input_dim, hidden_dim))
        self.bias1 = np.ascontiguousarray(np.random.randn(hidden_dim) * 0.01)
        self.weights2 = np.ascontiguousarray(np.random.randn(hidden_dim, output_dim))
        self.bias2 = np.ascontiguousarray(np.random.randn(output_dim) * 0.01)


    def tanh(self, x):
        return np.tanh(x)

    def forward(self, state):
        features = np.ascontiguousarray(np.array([
            np.sin(state[0]),
            np.cos(state[0]),
            state[1]
        ], dtype=np.float64))

        hidden = self.tanh(np.dot(features, self.weights1) + self.bias1)
        output = np.dot(hidden, self.weights2) + self.bias2
        return output[0]

    def get_action(self, state):
        return self.tanh(self.forward(state)) * 2.0

    def get_flat_params(self):
        return np.concatenate((
            self.weights1.flatten(),
            self.bias1,
            self.weights2.flatten(),
            self.bias2
        ))

    def set_flat_params(self, params):
        w1_size = self.input_dim * self.hidden_dim
        b1_size = self.hidden_dim
        w2_size = self.hidden_dim * self.output_dim
        b2_size = self.output_dim

        self.weights1 = np.ascontiguousarray(params[:w1_size].reshape((self.input_dim, self.hidden_dim)))
        self.bias1 = np.ascontiguousarray(params[w1_size:w1_size + b1_size])
        self.weights2 = np.ascontiguousarray(params[w1_size + b1_size:w1_size + b1_size + w2_size].reshape((self.hidden_dim, self.output_dim)))
        self.bias2 = np.ascontiguousarray(params[-b2_size:])


    def param_dim(self):
        return (
            self.input_dim * self.hidden_dim +
            self.hidden_dim +
            self.hidden_dim * self.output_dim +
            self.output_dim
        )
