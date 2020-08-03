import tensorflow as tf
import numpy as np


def ThinPlateSpline(coord, vector, out_size):
    """Thin Plate Spline Spatial Transformer Layer
    TPS control points are arranged in arbitrary positions given by `coord`.
    coord : float Tensor [num_batch, num_point, 2]
      Relative coordinate of the control points.
    vector : float Tensor [num_batch, num_point, 2]
      The vector on the control points.
    out_size: tuple of two integers [height, width]
      The size of the output of the network (height, width)
    ----------
    Reference :
      1. Spatial Transformer Network implemented by TensorFlow
        https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
      2. Thin Plate Spline Spatial Transformer Network with regular grids.
        https://github.com/iwyoo/TPS_STN-tensorflow
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _meshgrid(height, width, coord):
        x_t = tf.tile(
            tf.reshape(tf.linspace(-1.0, 1.0, width), [1, width]), [height, 1])
        y_t = tf.tile(
            tf.reshape(tf.linspace(-1.0, 1.0, height), [height, 1]), [1, width])

        x_t_flat = tf.reshape(x_t, (1, 1, -1))
        y_t_flat = tf.reshape(y_t, (1, 1, -1))

        num_batch = tf.shape(coord)[0]
        px = tf.expand_dims(coord[:, :, 0], 2)  # [bn, pn, 1]
        py = tf.expand_dims(coord[:, :, 1], 2)  # [bn, pn, 1]
        d2 = tf.square(x_t_flat - px) + tf.square(y_t_flat - py)
        r = d2 * tf.math.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = tf.tile(x_t_flat, tf.stack(
            [num_batch, 1, 1]))  # [bn, 1, h*w]
        y_t_flat_g = tf.tile(y_t_flat, tf.stack(
            [num_batch, 1, 1]))  # [bn, 1, h*w]
        ones = tf.ones_like(x_t_flat_g)  # [bn, 1, h*w]

        grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r],
                         1)  # [bn, 3+pn, h*w]
        return grid

    def _transform(T, coord, out_size):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = out_size[0]
        out_width = out_size[1]
        out_height_f = tf.cast(out_height, 'float32')
        out_width_f = tf.cast(out_width, 'float32')

        grid = _meshgrid(out_height, out_width, coord)  # [2, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = tf.matmul(T, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_meshgrid = tf.reshape(x_s, [-1, out_height_f, out_width_f])
        y_s_meshgrid = tf.reshape(y_s, [-1, out_height_f, out_width_f])
        x_s_meshgrid = (x_s_meshgrid + 1) * out_width_f / 2
        y_s_meshgrid = (y_s_meshgrid + 1) * out_height_f / 2

        return x_s_meshgrid, y_s_meshgrid

    def _solve_system(coord, vector):
        num_batch = tf.shape(coord)[0]
        num_point = tf.shape(coord)[1]

        ones = tf.ones([num_batch, num_point, 1], dtype="float32")
        p = tf.concat([ones, coord], 2)  # [bn, pn, 3]

        p_1 = tf.reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = tf.reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = tf.reduce_sum(tf.square(p_1 - p_2), 3)  # [bn, pn, pn]
        r = d2 * tf.math.log(d2 + 1e-6)  # [bn, pn, pn]

        zeros = tf.zeros([num_batch, 3, 3], dtype="float32")
        W_0 = tf.concat([p, r], 2)  # [bn, pn, 3+pn]
        W_1 = tf.concat([zeros, tf.transpose(p, [0, 2, 1])],
                        2)  # [bn, 3, pn+3]
        W = tf.concat([W_0, W_1], 1)  # [bn, pn+3, pn+3]
        W_inv = tf.linalg.inv(W)

        tp = tf.pad(coord + vector,
                    [[0, 0], [0, 3], [0, 0]], "CONSTANT")  # [bn, pn+3, 2]
        T = tf.matmul(W_inv, tp)  # [bn, pn+3, 2]
        T = tf.transpose(T, [0, 2, 1])  # [bn, 2, pn+3]

        return T

    T = _solve_system(coord, vector)
    output = _transform(T, coord, out_size)
    return output
