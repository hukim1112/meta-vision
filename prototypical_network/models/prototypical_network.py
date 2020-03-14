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

    def call(self, support, query):
        n_class = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]

        # merge support and query to forward through encoder
        cat = tf.concat([
            tf.reshape(support, [n_class * n_support,
                                 self.w, self.h, self.c]),
            tf.reshape(query, [n_class * n_query,
                               self.w, self.h, self.c])], axis=0)
        z = self.encoder(cat)

        # Divide embedding into support and query
        z_prototypes = tf.reshape(z[:n_class * n_support],
                                  [n_class, n_support, z.shape[-1]])
        # Prototypes are means of n_support examples
        z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1)
        z_query = z[n_class * n_support:]
        return z_prototypes, z_query

    @staticmethod
    def calc_euclidian_dists(x, y):
        """
        Calculate euclidian distance between two 3D tensors.

        Args:
            x (tf.Tensor):
            y (tf.Tensor):

        Returns (tf.Tensor): 2-dim tensor with distances.

        """
        n = x.shape[0]
        m = y.shape[0]
        x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
        y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
        return tf.reduce_mean(tf.math.pow(x - y, 2), 2)

    @staticmethod
    def calc_probability_with_dists(dists, n_class, n_query):
        log_p_y = tf.nn.log_softmax(-dists, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])
        return log_p_y

    @staticmethod
    def loss_func(log_p_y, n_class, n_query):
        y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)
        loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(
                               tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        pred = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
        return loss, pred

    @staticmethod
    def cal_metric(log_p_y, n_class, n_query):
        y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
        eq = tf.cast(tf.equal(
            tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
            tf.cast(y, tf.int32)), tf.int32)
        acc = tf.reduce_mean(eq)
        return eq, acc

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
