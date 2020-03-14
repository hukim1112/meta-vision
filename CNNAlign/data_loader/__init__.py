from .mini_imagenet import load_mini_imagenet

def load_data(splits, config):
    if config['dataset_name'] == 'mini_imagenet':
        return load_mini_imagenet(splits, config)
    else:
        raise ValueError("Wrong dataset name : {}".format(config['dataset_name']))

