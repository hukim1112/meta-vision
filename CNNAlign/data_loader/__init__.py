import tensorflow as tf
from .mini_imagenet import load_mini_imagenet
from utils.image import make_synthesized_image_pair


def load_data(splits, config):
    if config['dataset_name'] == 'mini_imagenet':
        ds = load_mini_imagenet(splits, config)
    else:
        raise ValueError("Wrong dataset name : {}".format(
            config['dataset_name']))
    if config['processing'] == 'random_TPS':
        for split in splits:
            ds[split] = ds[split].map(make_pair_with_random_TPS)
    elif config['processing'] == 'debug':
        for split in splits:
            ds[split] = ds[split]
    else:
        raise ValueError("Wrong data processing type : {}".format(
            config['processing']))
    return ds


def make_pair_with_random_TPS(image):
    return tf.py_function(make_synthesized_image_pair, [image, (64, 64), 0.2], [tf.float32, tf.float32, tf.float32])
