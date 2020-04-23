import os
import pickle
import numpy as np
import tensorflow as tf
from utils.image import make_synthesized_pair
from functools import partial
from. episode import Episode_generator


def load_mini_imagenet(splits, config):
    """
    Load miniImagenet dataset.

    Args:
        splits (list): list of strings 'train'|'val'|'test'
        config (dict): general dict with program settings.

    Returns (dict): dictionary with keys as splits and values as tf.Dataset
    """
    ds = {}
    for split in splits:
        # Load images as numpy
        ds_filename = os.path.join(config['data_dir'], 'fsl',
                                   f'mini-imagenet-cache-{split}.pkl')
        # load dict with 'class_dict' and 'image_data' keys
        with open(ds_filename, 'rb') as f:
            data_dict = pickle.load(f)

        # Convert original data to format [n_classes, n_img, w, h, c]
        first_key = list(data_dict['class_dict'])[0]
        data = np.zeros((len(data_dict['class_dict']), len(
            data_dict['class_dict'][first_key]), 84, 84, 3))
        for i, (k, v) in enumerate(data_dict['class_dict'].items()):
            data[i, :, :, :, :] = data_dict['image_data'][v, :]

        if config['data']['method'] == 'synthesized_pair':
            ds[split] = get_dataset_of_synthesized_pair(split, data, config)
        elif config['data']['method'] == 'episodic_learning':
            ds[split] = get_dataset_of_episodes(split, data, config)
        else:
            raise ValueError("Wrong data processing type : {}".format(
                config['processing']))
    return ds


def get_dataset_of_synthesized_pair(split, data, config):
    data = np.reshape(data, [-1, 84, 84, 3])
    #np.random.shuffle(data)
    data = data[:config[split]['n_examples']]
    n_examples = len(data)
    print("{} dataset amount : {}".format(split, n_examples))
    np.random.shuffle(data)
    data /= 255.
    data = tf.data.Dataset.from_tensor_slices(data)
    if split == 'train':
        if n_examples > 1000:
            shuffle_buffer = 1000
        else:
            shuffle_buffer = n_examples
        data = data.shuffle(shuffle_buffer)

    tps_random_rate = config['data']['tps_random_rate']
    pad_ratio = config['data']['pad_ratio']
    output_size = config['image_shape'][:2]
    py_func = partial(use_py_function, tps_random_rate=tps_random_rate,
                      pad_ratio=pad_ratio, output_size=output_size)
    data = data.map(py_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return data


def use_py_function(image, tps_random_rate, pad_ratio, output_size):
    moving_vectors = (tf.random.uniform([9, 2]) - 0.5) * 2 * tps_random_rate
    return tf.py_function(make_synthesized_pair, [image, moving_vectors, pad_ratio, output_size], [tf.float32, tf.float32, tf.float32])


def get_dataset_of_episodes(split, data, config):
    data /= 255.
    n_way, n_support, n_query = config[split]['n_way'], config[split]['n_support'], config[split]['n_query']
    h, w, c = data.shape[-3:]
    h_out, w_out = config['image_shape'][:2]
    gen = Episode_generator(data, n_way, n_support, n_query)

    def generate_episode():
        for epi in range(config[split]['episodes']):
            support, query = gen.get_next_episode()
            yield support, query

    data = tf.data.Dataset.from_generator(
        generate_episode, (tf.float32, tf.float32))

    def transform_data(support, query):
        support = tf.reshape(support, [n_way * n_support, w, h, c])
        support = tf.image.resize(support, (h_out, w_out))
        query = tf.reshape(query, [n_way * n_query, w, h, c])
        query = tf.image.resize(query, (h_out, w_out))
        return support, query
    data = data.map(transform_data)

    return data
