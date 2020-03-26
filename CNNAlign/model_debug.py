import os
import json
from data_loader import load_data
from models.cnn_geo import CNN_geo
from models.modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
from geo_transform import tps
from utils import image

from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np

def main():
    with open("configs/labtop_cnngeo.json") as file:
        config = json.load(file)
    batch_size = 1
    splits = ['val']
    datasets = load_data(splits, config)
    val_ds = datasets['val'].batch(batch_size)

    model = Feature_Extractor(backbone_name='prototypical_network')

    for c, w, p in val_ds.take(1):
        print(c.shape, w.shape)
        fm_A = model(c)
        fm_B = model(w)

    # print(fm_A.shape)
    # print(fm_B.shape)
    # print(p)

if __name__ == '__main__':
    main()
