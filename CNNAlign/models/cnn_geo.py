from .modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
import tensorflow as tf

class CNN_geo(tf.keras.Model):
    def __init__(self):
        super(CNN_geo, self).__init__()
        self.feature_extractor = Feature_Extractor()
        self.correlation_network = Correlation_network()
        self.geo_parameter_regressor = Spatial_transformer_regressor(18)
    def call(self, _input):
        num_image_pair = _input.shape[0]/2
        features = self.feature_extractor(_input)
        features = self.feature_extractor.channelwise_l2_normalize(features)
        featureA, featureB = features[:num_image_pair], features[num_image_pair:]
        corr_scores = Correlation_network(featureA, featureB)
        geo_parameters = geo_parameter_regressor(corr_scores)
        return geo_parameters

