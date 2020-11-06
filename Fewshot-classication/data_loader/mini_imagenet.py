import os
import numpy as np
import pickle
from PIL import Image
from functools import partial
import tensorflow as tf

class DataLoader(object):
    def __init__(self, data, n_way, n_support, n_query):
        self.data = data
        self.n_classes = data.shape[0]
        self.n_examples = data.shape[1]
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        support = np.zeros([self.n_way, self.n_support, 84, 84, 3], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 84, 84, 3], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(self.n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query

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

        if split in ['val', 'test']:
            n_way = config['test']['n_way']
            n_support = config['test']['n_support']
            n_query = config['test']['n_query']
        else:
            n_way = config['train']['n_way']
            n_support = config['train']['n_support']
            n_query = config['train']['n_query']

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
        data /= 255
        ds[split] = data
    return ds
