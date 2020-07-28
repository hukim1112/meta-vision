import tensorflow as tf


def resnet():
    resnet101 = tf.keras.applications.ResNet101(
        include_top=False, weights='imagenet')
    return tf.keras.Model(resnet101.input, resnet101.layers[312].output)


def vgg16():
    vgg16 = tf.keras.applications.VGG16(
        weights='imagenet', input_shape=(200, 200, 3), include_top=False)
    return tf.keras.Model(inputs=vgg16.layers[0].input, outputs=vgg16.layers[9].output)


def prototypical_network():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2, 2)),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                               activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D((2, 2))])
