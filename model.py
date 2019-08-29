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

class FeatureL2Norm(tf.keras.layers.Layer):
