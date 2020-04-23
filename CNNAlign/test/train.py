import numpy as np
from data_loader import load_data
from models.cnn_geo import CNN_geo
from utils import image
from . import visualize
from CNNgeo import train

def overfit(config, splits):
    datasets = load_data(splits, config)
    train(config)
    model = CNN_geo("prototypical_network")
    ckpt_dir = os.path.join(
        'checkpoints', config['model_name'], config['exp_desc'])
    ckpt = os.path.join(ckpt_dir, "CNNgeo-100.h5")
    model.load(ckpt)
    print(model.get_weight())