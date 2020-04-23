from models.cnn_geo import CNN_geo
import tensorflow as tf
import numpy as np
import math
import geo_transform as tps


class CNN_align(tf.keras.Model):
    def __init__(self, backbone_name, thresh=1.0):
        super(CNN_align, self).__init__()
        self.model_name = 'CNNalign'
        self.cnn_geo = CNN_geo(backbone_name)
        self.thresh = thresh

    def call(self, image_A, image_B):
        geo_parameters, corr_scores = self.cnn_geo(image_A, image_B)
        map_size = corr_scores.shape[-2:]
        inlier_masks = tf.map_fn(lambda x: tf.py_function(generate_inlier_mask, [
                                 x, map_size, self.thresh], tf.float32), geo_parameters)
        inlier_matching = corr_scores * inlier_masks  # B, H, W, H, W
        return inlier_matching, tf.reduce_sum(inlier_matching, axis=(1, 2, 3, 4))

    def save(self, ckpt_path):
        self.cnn_geo.save(ckpt_path)

    def load(self, ckpt_path):
        self.cnn_geo.load(ckpt_path)


def generate_inlier_mask(moving_vectors, map_size, thresh):
    moving_vectors = moving_vectors.numpy()
    height, width = map_size
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])
    dst_points = src_points + moving_vectors
    theta = tps.tps_theta_from_points(src_points, dst_points, reduced=True)
    grid = tps.tps_grid(theta, dst_points, map_size)
    grid = grid * (height - 1, width - 1)
    # coordinates of a_x, a_y, b_x, b_y from feature map A (a_x,a_y) and feature map B (b_x,b_y)
    mask = np.zeros([height, width, height, width])
    for b_y in range(height):
        for b_x in range(width):
            # (a_x, a_y) of feature map A matched (b_x, b_y), coord of feature map B
            a_x, a_y = grid[b_y, b_x]
            range_a_x = range(math.ceil(a_x - thresh),
                              math.floor(a_x + thresh) + 1)
            range_a_x = list(
                filter(lambda x: x >= 0 and x <= width - 1, range_a_x))
            range_a_y = range(math.ceil(a_y - thresh),
                              math.floor(a_y + thresh) + 1)
            range_a_y = list(
                filter(lambda x: x >= 0 and x <= height - 1, range_a_y))
            for a_x in range_a_x:
                for a_y in range_a_y:
                    mask[a_y, a_x, b_y, b_x] = 1

    return mask
