import tensorflow as tf
import os, pickle
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
        data = np.zeros((len(data_dict['class_dict']), len(data_dict['class_dict'][first_key]), 84, 84, 3))
        for i, (k, v) in enumerate(data_dict['class_dict'].items()):
            data[i, :, :, :, :] = data_dict['image_data'][v, :]
        data /= 255.
        data = np.reshape(data, [-1, 84, 84, 3])
        data = tf.data.Dataset.from_tensor_slices(data)
        if split == 'train':
            data = data.shuffle(4000).repeat()
        ds[split] = data.map(data_process)
    return ds

def data_process(image):
    return tf.py_function(make_synthesized_image_pair, [image, (64, 64), 0.2], [tf.float32, tf.float32, tf.float32])