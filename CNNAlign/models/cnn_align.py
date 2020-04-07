from .modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
import tensorflow as tf
import geo_transform as tps


class CNN_geo(tf.keras.Model):
    def __init__(self, backbone_name):
        super(CNN_geo, self).__init__()
        self.feature_extractor = Feature_Extractor(backbone_name)
        self.correlation_network = Correlation_network()
        self.geo_parameter_regressor = Spatial_transformer_regressor(18)

    def call(self, image_A, image_B):
        feature_A = self.feature_extractor(image_A)
        feature_B = self.feature_extractor(image_B)
        feature_A = self.feature_extractor.channelwise_l2_normalize(feature_A)
        feature_B = self.feature_extractor.channelwise_l2_normalize(feature_B)
        corr_scores = self.correlation_network(feature_A, feature_B)
        geo_parameters = self.geo_parameter_regressor(corr_scores)
        return geo_parameters, corr_scores


class CNN_align(tf.keras.Model):
    def __init__(self, backbone_name):
        super(CNN_align, self).__init__()
        self.CNN_geo = CNN_geo(backbone_name)

    def call(self, image_A, image_B):
        geo_parameters, corr_scores = self.CNN_geo(image_A, image_B)
        map_size = corr_scores.shape[-2:]
        inlier_masks = tf.map_fn(lambda x: tf.numpy_function(generate_inlier_mask,
                                                             [x, map_size], tf.float32), geo_parameters)
        inlier_matching = corr_scores * inlier_masks  # B, H, W, H, W
        return tf.reduce_sum(inlier_matching, axis=(1, 2, 3, 4))


def generate_inlier_mask(moving_vectors, map_size=(16, 16)):
    moving_vectors = np.reshape(moving_vectors, (9, 2))
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])
    dst_points = src_points + moving_vectors
    theta = tps.tps_theta_from_points(src_points, dst_points, reduced=True)
    grid = tps.tps_grid(theta, dst_points, map_size)
    grid = grid * (map_size[0] - 1, map_size[1] - 1)
    height, width = map_size
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
