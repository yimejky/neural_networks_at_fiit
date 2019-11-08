import numpy as np


def sin_dataset(num_samples, len_sample, skip):

    def sample(len_sample, skip):
        lin = np.arange(len_sample + skip) / 20.0
        a = np.random.rand() * 1.5 + 0.5
        b = np.random.randint(0, 100)
        c = (np.random.rand() - 0.5) * 0.1
        data = np.sin(lin * a + b) + c
        return np.expand_dims(data[:len_sample], axis=-1), data[-1]

    dataset = [sample(len_sample, skip) for _ in range(num_samples)]
    return np.array([data for data, _ in dataset]), np.array([target for _, target in dataset])
