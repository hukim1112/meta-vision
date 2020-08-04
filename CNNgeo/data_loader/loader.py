from .mini_imagenet import load_mini_imagenet
from .dev import load_dev_dataset

def load(splits, config):
    if config['dataset_name'] == 'mini_imagenet':
        ds = load_mini_imagenet(splits, config)
    elif config['dataset_name'] == 'dev':
        ds = load_dev_dataset(config)
    else:
        raise ValueError("Wrong dataset name : {}".format(
            config['dataset_name']))
    return ds
