from .modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
import tensorflow as tf


class CNN_geo(tf.keras.Model):
    def __init__(self, backbone_name, tentative_penalty=True):
        super(CNN_geo, self).__init__()
        self.model_name = 'CNNgeo'
        self.feature_extractor = Feature_Extractor(backbone_name)
        self.correlation_network = Correlation_network()
        self.geo_parameter_regressor = Spatial_transformer_regressor(18)
        self.tentative_penalty = tentative_penalty
    def call(self, image_A, image_B):
        feature_A = self.feature_extractor(image_A)
        feature_B = self.feature_extractor(image_B)
        feature_A = self.feature_extractor.channelwise_l2_normalize(feature_A)
        feature_B = self.feature_extractor.channelwise_l2_normalize(feature_B)
        corr_scores = self.correlation_network(feature_A, feature_B)
        if self.tentative_penalty:
            corr_scores = self.correlation_network.tentative_penalty(corr_scores)
        geo_parameters = self.geo_parameter_regressor(corr_scores)
        geo_parameters = tf.reshape(geo_parameters, [-1, 9, 2])
        return geo_parameters, corr_scores
    def save(self, ckpt_path):
        self.save_weights(ckpt_path)
    def load(self, ckpt_path):
        self.call(tf.ones([1,64,64,3]), tf.ones([1,64,64,3]))
        self.load_weights(ckpt_path)
