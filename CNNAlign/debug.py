import tensorflow as tf
import numpy as np
import cv2
def data_process(image):
    return tf.py_function(image_process, [image], [tf.float32])
def image_process(image):
    image = image.numpy()
    return cv2.resize(image, (56, 56))

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

ds = tf.data.Dataset.from_tensor_slices(train_images)
ds = ds.map(data_process)

for i in ds.take(1):
    print(i)

