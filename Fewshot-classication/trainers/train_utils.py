import numpy as np
import tensorflow as tf

def calc_prototypes(embeddings, n_class, n_data):
    # [n_class*n_data, n_dim]
    z_prototypes = tf.reshape(embeddings,
                              [n_class, n_data, -1])
    z_prototypes = tf.math.reduce_mean(z_prototypes, axis=1) # [n_class, n_dim]
    return z_prototypes


def calc_euclidian_dists(x, y):
    # x : [n_class_x, n_dim], y : [n_class_y, n_dim]

    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2) # [n_class_x, n_class_y]

def calc_probability_with_dists(dists, n_class, n_query):
    log_p_y = tf.nn.log_softmax(-dists, axis=-1)
    log_p_y = tf.reshape(log_p_y, [n_class, n_query, -1])
    return log_p_y

def loss_func(log_p_y, n_class, n_query):
    y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
    y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)
    loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(
                               tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
    pred = tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32)
    return loss, pred

def cal_metric(log_p_y, n_class, n_query):
    y = np.tile(np.arange(n_class)[:, np.newaxis], (1, n_query))
    eq = tf.cast(tf.equal(
            tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32),
            tf.cast(y, tf.int32)), tf.int32)

    acc = tf.reduce_mean(tf.cast(eq, tf.float32))
    return eq, acc
