import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


class Prototypical_network(Model):
    """
    Implemenation of Prototypical Network.
    """

    def __init__(self, h, w, c):
        """
        Args:
            h (int): image height.
            w (int): image width .
            c (int): number of channels.
        """
        super(Prototypical_network, self).__init__()
        self.w, self.h, self.c = w, h, c

        # Encoder as ResNet like CNN with 4 blocks
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2)), Flatten()]
        )

    def call(self, input):
        z = self.encoder(input)
        return z

    def save(self, model_path):
        """
        Save encoder to the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder.save(model_path)

    def load(self, model_path):
        """
        Load encoder from the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))
        self.encoder.load_weights(model_path)
