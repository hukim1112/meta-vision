import tensorflow as tf

def vgg16(input_size):
    vgg16 = tf.keras.applications.VGG16(weights='imagenet', input_shape=(input_size[0], input_size[1], 3), include_top=False)
    output_layer = vgg16.get_layer("block4_conv3")
    output_layer.activation = None
    splited_model = tf.keras.Model(inputs=vgg16.input, outputs=output_layer.output)

    input_layer = tf.keras.Input(shape=(input_size[0], input_size[1],3))
    normalized_input = tf.keras.applications.vgg16.preprocess_input(input_layer)
    output = splited_model(normalized_input)
    feature_extractor = tf.keras.Model(inputs=normalized_input, outputs=output)
    return feature_extractor
