import os
import cv2
import numpy as np
import tensorflow as tf
from functools import partial
from geo_transform.tf_tps import ThinPlateSpline as tps

def tf_image_process(image, tps_random_rate, output_size):
    motion_vectors = (tf.random.uniform([9, 2]) - 0.5) * 2 * tps_random_rate
    return tf.py_function(py_image_process, [image, motion_vectors, output_size], [tf.float32, tf.float32, tf.float32])

def py_image_process(image, motion_vectors, output_size):
    image = image.numpy()
    #image = preprocess_input(image)
    control_points = tf.constant([[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
                               [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                               [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
    x_s, y_s = tps(control_points[tf.newaxis,::], -motion_vectors[tf.newaxis,::], output_size)
    synth_image = cv2.remap(image, x_s[0].numpy(), y_s[0].numpy(), cv2.INTER_CUBIC)
    return image, synth_image, motion_vectors

def load_dev_dataset():
    _datapath = "sample_dataset"
    filelist = os.listdir(_datapath)
    input_size = (200, 200)

    images = []

    for f in filelist:
        _path = os.path.join(_datapath, f)
        img = cv2.imread(_path)[:,:,::-1]
        img = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
        images.append(img)

    images = np.array(images, dtype=np.float32)

    tps_random_rate = 0.4
    output_size = (200, 200)
    map_func = partial(tf_image_process, tps_random_rate=tps_random_rate,
                          output_size=output_size)

    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(16)
    return ds