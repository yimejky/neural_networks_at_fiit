import numpy as np


class MultilayerPerceptron:

    def __init__(self, dim_input, dim_hidden, dim_output):
        self.w_1 = self.xavier(dim_hidden, dim_input)
        self.b_1 = np.zeros(dim_hidden)
        self.w_2 = self.xavier(dim_output, dim_hidden)
        self.b_2 = np.zeros(dim_output)

    def xavier(self, a, b):
        return np.math.sqrt(6/(a + b)) * (np.random.rand(a, b) * 2 - 1)

    def loss(self, xs, ys):
        return np.mean([
            np.sum([
                (y - self.predict(x)[0])**2
            ])
            for x, y
            in zip(xs, ys)
        ])

    def stochastic_gradient_descent(self, xs, ys, num_epochs=200, learning_rate=0.003):
        for _ in range(num_epochs):
            self.epoch(xs, ys, learning_rate)

    def epoch(self, xs, ys, learning_rate):
        order = np.random.permutation(len(ys))
        xs, ys = xs[order], ys[order]
        for x, y in zip(xs, ys):
            self.step(x, y, learning_rate)
        print(self.loss(xs, ys))

    def step(self, x, y, learning_rate):
        params = [self.w_1, self.b_1, self.w_2, self.b_2]
        dparams = self.gradient(x, y)
        for param, dparam in zip(params, dparams):
            param -= learning_rate * dparam

    def sigma(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigma(self, x):
        sigma = self.sigma(x)
        return sigma * (1 - sigma)

    def predict(self, x):
        z_1 = ...  # FIXME: 3.4.1
        h = ...
        z_2 = ...
        y_hat = ...
        return y_hat, z_2, h, z_1

    def gradient(self, x, y):
        y_hat, z_2, h, z_1 = self.predict(x)
        dz_2 = ...  # FIXME: 3.4.2
        db_2 = ...
        dw_2 = ...
        ...  # Continue with first layer parameters
        return dw_1, db_1, dw_2, db_2
