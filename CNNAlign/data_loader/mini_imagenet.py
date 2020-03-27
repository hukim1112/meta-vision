import tensorflow as tf
import os
import pickle
import numpy as np
from utils.image import make_synthesized_image_pair
import cv2


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
        data = np.reshape(data, [-1, 84, 84, 3])
        np.random.shuffle(data)
        data = data[:config[split]['n_examples']]
        n_examples = len(data)
        data /= 255.
        data = tf.data.Dataset.from_tensor_slices(data)
        if split == 'train':
            if n_examples > 1000:
                shuffle_buffer = 1000
            else:
                shuffle_buffer = n_examples
            data = data.shuffle(shuffle_buffer)
        ds[split] = data
    return ds
