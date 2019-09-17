import tensorflow as tf
from tensorflow import keras

layer = keras.layers


config = {'image_size' : [240, 240, 3], 'transfer_learning' : True}


class Feature_Extractor(tf.keras.Model):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        self.inputs = tf.keras.Input(shape=config['image_size'])

        if config['transfer_learning'] == True:
            resnet101 = tf.keras.applications.ResNet101(include_top=False,
            weights='imagenet',
            input_tensor=self.inputs)
        else:
            resnet101 = tf.keras.applications.ResNet101(include_top=False,
            weights=None,
            input_tensor=self.inputs)

        extractor_input = resnet101.get_layer('input_1').input
        extractor_output = model_output = resnet101.get_layer('conv4_block23_out').output
        self.feature_extractor = tf.keras.Model(extractor_input, extractor_output)
        del resnet101
    def call(self, input):
        x = self.feature_extractor(input)
        return x

class FeatureL2Norm(keras.layers.Layer):
    #Channel-wise L2 normalization
    def __init__(self):
        super(FeatureL2Norm, self).__init__()
    def call(self, feature):
        epsilon = 1e-6
        norm = tf.expand_dims(tf.pow(tf.reduce_sum(tf.pow(feature, 2), axis=-1)+epsilon, 0.5), axis=-1)
        return tf.math.divide(feature, norm)
