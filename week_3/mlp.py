import numpy as np

from week_3.backstage.load_data import load_data





a = MultilayerPerceptron(4, 10, 3)
data = load_data('iris.csv', 3)
a.stochastic_gradient_descent(data.x, data.y)
