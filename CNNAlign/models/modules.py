import tensorflow as tf
from .backbones import resnet, prototypical_network


class Feature_Extractor(tf.keras.layers.Layer):
    def __init__(self, backbone_name):
        super(Feature_Extractor, self).__init__()
        if backbone_name == 'resnet':
            self.backbone = resnet()
        elif backbone_name == 'prototypical_network':
            self.backbone = prototypical_network()
        else:
            raise ValueError("Wrong backbone name : {}".format(backbone_name))

    def call(self, input):
        return self.backbone(input)

    @staticmethod
    def channelwise_l2_normalize(features):
        l2_norm = tf.math.sqrt(tf.reduce_sum(
            tf.pow(features, 2), axis=-1))  # (N, H, W)
        l2_norm = tf.expand_dims(l2_norm, axis=-1)  # (N, H, W, 1)
        # (N, H, W, D) tensor divided by (N, H, W, 1) tensor
        return features / l2_norm

    @staticmethod
    def channelwise_l2_normalize_debug(normalized_features):
        # all element of returned tensor must be 1.0
        return tf.math.sqrt(tf.reduce_sum(tf.pow(normalized_features, 2), axis=-1))


class Correlation_network(tf.keras.layers.Layer):
    def __init__(self):
        super(Correlation_network, self).__init__()

    def call(self, feature_A, feature_B):
        # featureA : feature information from source image
        # featureB : feature information from target image
        #assert feature_A.shape == feature_B.shape
        # new feature A and feature B have new shape of tensors.
        # featureA has tensor shape as [batch, HA, WA, 1, 1, depth]
        # featureB has tensor shape as [batch, 1, 1, HB, WB, depth]
        feature_A = feature_A[:, :, :, tf.newaxis, tf.newaxis, :]
        feature_B = feature_B[:, tf.newaxis, tf.newaxis, :, :, :]
        # correlation score has tensor shape as [batch, HA, WA, HB, WB]
        corr_score = tf.reduce_mean(tf.multiply(feature_A, feature_B), axis=-1)
        ambiguous_match_penalty = tf.math.sqrt(
            tf.reduce_sum(tf.pow(corr_score, 2), axis=[1, 2], keepdims=True))
        return tf.math.divide(corr_score, ambiguous_match_penalty)
        # see eq (2) in "End-to-end weakly-supervised semantic alignment"


'''
class Spatial_transformer_regressor(tf.keras.layers.Layer):
    def __init__(self, num_param):
        super(Spatial_transformer_regressor, self).__init__()
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Conv3D(64, (5, 5, 5), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_param),
        ])

    def call(self, correlation):
        return self.regressor(correlation)

'''


class Spatial_transformer_regressor(tf.keras.layers.Layer):
    def __init__(self, num_param):
        super(Spatial_transformer_regressor, self).__init__()
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Conv3D(128, (7, 7, 7), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(64, (5, 5, 5), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_param),
        ])

    def call(self, correlation):
        return self.regressor(correlation)
