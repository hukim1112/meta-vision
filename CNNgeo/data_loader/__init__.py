from .datasets import PF_Pascal

def load_data(name):
    if name == "PF_Pascal":
        return PF_Pascal
    else:
        raise ValueError("Wrong dataset name : {}".format(
            config["dataset_name"]))
    return dataset
