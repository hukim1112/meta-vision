from geo_transform.tf_tps import ThinPlateSpline as tps
import numpy as np
import cv2
import tensorflow as tf

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
    
    #W_inv = tf.linalg.inv(W)

    #tp = tf.pad(coord + vector,
    #                [[0, 0], [0, 3], [0, 0]], "CONSTANT")  # [bn, pn+3, 2]
    #T = tf.matmul(W_inv, tp)  # [bn, pn+3, 2]
    #T = tf.transpose(T, [0, 2, 1])  # [bn, 2, pn+3]

    #return T




img_name = "sample_dataset/060_0000.png"
img = cv2.imread(img_name)[:,:,::-1]
img = cv2.resize(img, (200, 200) ,interpolation=cv2.INTER_AREA)

p = np.array([
  [-0.5, -0.5],
  [0.5, -0.5],
  [-0.0, 0.0],
  [-0.5, 0.5],
  [0.5, 0.5]])

v = np.array([
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.0],
  [0.2, 0.2]])

p = tf.constant(p.reshape([1, 5, 2]), dtype=tf.float32)
v = tf.constant(v.reshape([1, 5, 2]), dtype=tf.float32)
#t_img = tf.constant(img.reshape(shape), dtype=tf.float32)
img = np.reshape(img, (1, 200, 200, 3))

out_size=(20, 20)

_solve_system(p, v)

tf.linalg.inv(
    tf.random.uniform([10, 10]), adjoint=False, name=None
)
#x_s, y_s = tps(p, -v, out_size)


