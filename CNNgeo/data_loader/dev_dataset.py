import os
import cv2
import numpy as np
import tensorflow as tf
from functools import partial
from geo_transform.tf_tps import ThinPlateSpline as tps


def pad_image(image, pad_ratio):
    '''
        input : original image, padding_ratio( ragne 0~1 )
        output : padded_image(reflected boundary)
    '''
    original_size = (image.shape[0], image.shape[1])
    top = bottom = int(original_size[0] * pad_ratio)
    left = right = int(original_size[1] * pad_ratio)
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_REFLECT)
    return padded_image


def py_image_process(image, motion_vectors, tps_random_rate, output_size):
    image = tf.keras.applications.vgg16.preprocess_input(image)
    image = image.numpy()
    tps_random_rate = tps_random_rate.numpy()
    #image = image / 255.
    padded_image = pad_image(image, tps_random_rate)

    ratio = 1 / (1 + tps_random_rate)
    control_points = tf.constant([[-ratio, -ratio], [0.0, -ratio], [ratio, -ratio],
                                  [-ratio, 0.0], [0.0, 0.0], [ratio, 0.0],
                                  [-ratio, ratio], [0.0, ratio], [ratio, ratio]], dtype=tf.float32)
    x_s, y_s = tps(control_points[tf.newaxis, ::], -
                   motion_vectors[tf.newaxis, ::], padded_image.shape[0:2])
    synth_image = cv2.remap(
        padded_image, x_s[0].numpy(), y_s[0].numpy(), cv2.INTER_CUBIC)
    dH = (synth_image.shape[0] - image.shape[0]) / 2
    dW = (synth_image.shape[1] - image.shape[1]) / 2
    dH, dW = int(dH), int(dW)
    synth_image = synth_image[dH:-dH, dW:-dW]
    return image, synth_image, motion_vectors


def tf_image_process(image, tps_random_rate, output_size):
    motion_vectors = (tf.random.uniform([9, 2]) - 0.5) * 2 * tps_random_rate
    return tf.py_function(py_image_process, [image, motion_vectors, tps_random_rate, output_size], [tf.float32, tf.float32, tf.float32])
