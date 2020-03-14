from data_loader import load_data
from matplotlib import pyplot as plt
import tensorflow as tf

def main():
    print(tf.executing_eagerly())
    config = {}
    config['data_dir'] = '/home/dan/prj/datasets/mini_imagenet'
    config['dataset_name'] = 'mini_imagenet'

    ds = load_data(['train', 'val'], config)
    train_ds = ds['train'].batch(64)

if __name__ == '__main__':
    main()
