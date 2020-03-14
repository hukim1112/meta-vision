from .prototypical_network import Prototypical_network


def model_load(split, config):
    """
    Load specific dataset.
checkpoint_load(model, pretrained_type, checkpoint_name)
    Args:
        split (string): strings 'train'|'test'.
        config (dict): general dict with settings.

    Returns (dict): dictionary with keys 'train'|'val'|'test'| and values
    as tensorflow Dataset objects.

    """
    if config['model_name'] == "prototypical_networks":
        h, w, c = config['input_shape']
        model = Prototypical_network(h, w, c)
        if split not in ['train', 'test']:
            raise ValueError('Unknown split name {}'.format(split))
        model = checkpoint_load(model, split, config)
    else:
        raise ValueError(f"Unknow model: {config['model_name']}")
    return model


def checkpoint_load(model, split, config):
    checkpoint_dir = config['checkpoint_dir']
    pretrained_type = config[split]['pretrained_type']
    checkpoint_name = config[split]['checkpoint_name']
    if pretrained_type == 'base':
        pass
    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, name)
                     for name in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            model.load(latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
            net.init_vgg16()
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')
    elif pretrained_type == 'specified':
        checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.isfile(checkpoint):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint))
        try:
            model.load(checkpoint)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint, arch))
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    return model