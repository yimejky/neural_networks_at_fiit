import os

from week_4.backstage.utils import timestamp

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp


class MultilayerPerceptron(keras.Model):

    def __init__(self, dim_output, dim_hidden):
        super(MultilayerPerceptron, self).__init__(name='multilayer_perceptron')
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.layer_1 = keras.layers.Dense(
            units=dim_hidden)
        self.layer_2 = keras.layers.Dense(
            units=dim_output)

    def call(self, x):
        h = self.layer_1(x)
        y = self.layer_2(h)
        return y

# g_hparams = {
#     'dim_hidden': hp.HParam('dim_hidden', hp.IntInterval(1, 1000)),
#     'loss_function': hp.HParam('loss_function', hp.Discrete(['mse', 'categorial_crossentropy'])),
#     'learning_rate': hp.HParam('learning_rate', hp.RealInterval(1e-10, 1.))
# }

# with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
#   hp.hparams_config(
#     hparams=g_hparams.values(),
#     metrics=metrics,
#   )


def train(x, y, **hparams):

    model = MultilayerPerceptron(
        dim_output=hparams['dim_output'],
        dim_hidden=hparams['dim_hidden'])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=hparams['learning_rate']),
        loss=hparams['loss_function'],
        metrics=['accuracy'])

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", timestamp()),
        histogram_freq=1)
    hparams_callback = hp.KerasCallback(
        os.path.join("logs", timestamp()),
        hparams),

    model.fit(
        x=x,
        y=y,
        batch_size=hparams['batch_size'],
        epochs=hparams['epoch'],
        validation_split=0.2,
        callbacks=[tensorboard_callback, hparams_callback],
        verbose=0)
