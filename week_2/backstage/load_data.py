import types

import numpy as np


def load_data(filename, normalize=False, *args):
    data = np.genfromtxt(f'data/{filename}', *args)
    if normalize:
        means, stds = np.mean(data, axis=0), np.std(data, axis=0)
        data = (data - means) / stds
    output = types.SimpleNamespace()
    output.x = data[:, :-1]
    output.y = np.squeeze(data[:, -1])
    return output
