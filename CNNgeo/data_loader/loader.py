from .mini_imagenet import load_mini_imagenet


def load(splits, config):
    if config['dataset_name'] == 'mini_imagenet':
        ds = load_mini_imagenet(splits, config)
    else:
        raise ValueError("Wrong dataset name : {}".format(
            config['dataset_name']))
    return ds
