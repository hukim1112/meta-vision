import tensorflow as tf
import numpy as np
from .mini_imagenet import load_mini_imagenet
from utils.image import make_synthesized_image_pair


def load_data(splits, config):
    if config['dataset_name'] == 'mini_imagenet':
        ds = load_mini_imagenet(splits, config)
    else:
        raise ValueError("Wrong dataset name : {}".format(
            config['dataset_name']))
    if config['processing'] == 'random_TPS':
        tf_func = data_process(use_py_function, config)
        for split in splits:
            ds[split] = ds[split].map(tf_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif config['processing'] == 'debug':
        for split in splits:
            ds[split] = ds[split]
    else:
        raise ValueError("Wrong data processing type : {}".format(
            config['processing']))
    return ds

def data_process(function, config):
    tps_random_rate = config['train']['tps_random_rate']
    pad_ratio = config['train']['pad_ratio']
    output_size = config['image_shape'][:2]
    def wrapper(image):
        return function(image, tps_random_rate, pad_ratio, output_size)
    return wrapper

def use_py_function(image, tps_random_rate, pad_ratio, output_size):
    moving_vectors = (tf.random.uniform([9, 2]) - 0.5) * 2 * tps_random_rate
    return tf.py_function(make_synthesized_image_pair, [image, moving_vectors, pad_ratio, output_size], [tf.float32, tf.float32, tf.float32])
