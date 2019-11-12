import tensorflow.keras as keras
from tensorflow.keras.layers import concatenate, Conv2D, Dense, Flatten, Layer, MaxPooling2D

from week_7.backstage.utils import *


class Inception(Layer):

    """
    This is an example of how can you define your own layer from existing layers. It is very similar to how we define
    models.

    Change the __init__ and call methods so that the Inception layer looks like the layer depicted in the figure
    in the notebook. The Inception layer consists of four branches, that are concatenated at the end.

    You should be able to build this layer from the layers you have already seen - Conv2D, MaxPooling2D and concatenate.

    Several notes:
        1) The 3x3 max-pooling layer in the 4th branch has 1x1 stride
        2) You should use concatenate with lowercase `c` at the start. Concatenate with uppercase `C` is slightly
        different and does not work properly in Layer definitions.
    """

    def __init__(self, filters, activation):
        """
        :param filters: How many filters are in the convolutional layers within this layer.
        :param activation: What is the activation function used
        """
        super(Inception, self).__init__()

        # first
        self.conv1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)

        # second
        self.conv2 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.conv3 = Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation=activation)
        
        # third
        self.conv4 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.conv5 = Conv2D(
            filters=filters,
            kernel_size=5,
            padding='same',
            activation=activation)
        
        # fourth
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv6 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)

    def call(self, x):
        c1 = self.conv1(x)
       
        c2 = self.conv2(x)
        c2 = self.conv3(c2)
        
        c3 = self.conv4(x)
        c3 = self.conv5(c3)
        
        c4 = self.pool1(x)
        c4 = self.conv6(c4)
        
        x = concatenate([c1, c2, c3, c4])

        return x


class InceptionNet(keras.Model):
    """
    Inception version of the simple CNN we used previously. You do not need to change anything here.
    """

    def __init__(self, filters, dim_output):
        super(InceptionNet, self).__init__()
        self.model_layers = [
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(
                units=512,
                activation='relu'),
            Dense(
                units=dim_output,
                activation='softmax')
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
