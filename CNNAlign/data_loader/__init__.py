from .mini_imagenet import load_mini_imagenet

def load_data(splits, config):
    if config['dataset_name'] == 'mini_imagenet':
        ds = load_mini_imagenet(splits, config)
    else:
        raise ValueError("Wrong dataset name : {}".format(config['dataset_name']))

def data_process(image):
    return tf.py_function(make_synthesized_image_pair, [image, (64, 64), 0.2], [tf.float32, tf.float32, tf.float32])