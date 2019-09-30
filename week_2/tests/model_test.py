import unittest

import numpy as np

from week_2.model import LinearRegressionModel


class TestLinearRegressionModel(unittest.TestCase):

    def setUp(self):
        self.model_1d = LinearRegressionModel(
            input_dim=1,
            w=np.array([0.5]),
            b=0.2)
        self.model_3d = LinearRegressionModel(
            input_dim=3,
            learning_rate=0.1,
            w=np.array([0.5, 0.4, 0.3]),
            b=0.2)
        self.data_1d = (
            np.array([
                [1.2],
                [1.5],
                [1.3],
                [1.7]
            ]),
            np.array([0.2, 0.4, 0.5, 3.8])
        )
        self.data_3d = (
            np.array([
                [1.2, 1.5, 1.3],
                [2.2, 2.5, 2.3],
                [0.2, 0.5, 0.3],
                [3.2, 0.5, 0.3],
            ]),
            np.array([0.2, 0.4, 0.5, 0.8])
        )

    def test_predict(self):
        for predicted, expected in zip(
            self.model_1d.predict(self.data_1d[0]),
            [0.8, 0.95, 0.85, 1.05]
        ):
            self.assertAlmostEqual(predicted, expected)

        for predicted, expected in zip(
            self.model_3d.predict(self.data_3d[0]),
            [1.79, 2.99, 0.59, 2.09]
        ):
            self.assertAlmostEqual(predicted, expected)

    def test_gradient(self):
        dw, db = self.model_1d.gradient(*self.data_1d)
        self.assertEqual(len(dw), 1)
        self.assertAlmostEqual(dw[0], -1.3375)
        self.assertAlmostEqual(db, -0.625)

        dw, db = self.model_3d.gradient(*self.data_3d)
        self.assertEqual(len(dw), 3)
        for desired, computed in zip([5.876, 4.775, 4.219], dw):
            self.assertAlmostEqual(desired, computed)
        self.assertAlmostEqual(db, 2.78)

    def test_step(self):
        self.model_1d.step(*self.data_1d)
        w, b = self.model_1d.w, self.model_1d.b
        self.assertEqual(len(w), 1)
        self.assertAlmostEqual(w[0], 0.540125)
        self.assertAlmostEqual(b, 0.21875)

        self.model_3d.step(*self.data_3d)
        w, b = self.model_3d.w, self.model_3d.b
        self.assertEqual(len(w), 3)
        for desired, computed in zip([-0.0876, -0.0775, -0.1219], w):
            self.assertAlmostEqual(desired, computed)
        self.assertAlmostEqual(b, -0.078)

    def test_loss(self):
        self.assertAlmostEqual(
            self.model_1d.loss(*self.data_1d),
            2.086875)
        self.assertAlmostEqual(
            self.model_3d.loss(*self.data_3d),
            2.7271)
