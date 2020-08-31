from .datasets import PF_Pascal

def load_data(name, splits=None):
    if name == "PF_Pascal":
        data_dir = config['data_dir']
        dataset = PF_Pascal(data_dir)
    else:
        raise ValueError("Wrong dataset name : {}".format(
            config["dataset_name"]))
    return dataset
