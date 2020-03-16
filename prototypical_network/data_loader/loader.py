from .mini_imagenet import load_mini_imagenet
from .omniglot import load_omniglot


def load_data(splits, config):
    """
    Load specific dataset.

    Args:
        splits (list): list of strings 'train'|'val'|'test'.
        config (dict): general dict with settings.
        
    Returns (dict): dictionary with keys 'train'|'val'|'test'| and values
    as tensorflow Dataset objects.

    """
    if config['dataset_name'] == "mini_imagenet":
        ds = load_mini_imagenet(splits, config)
    # if config['data.dataset'] == "omniglot":
    #     ds = load_omniglot(splits, config)
    else:
        raise ValueError(f"Unknow dataset: {config['data.dataset']}")
    return ds
