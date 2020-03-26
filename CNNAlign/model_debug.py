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
    batch_size = 10
    splits = ['val']
    datasets = load_data(splits, config)
    train_ds = datasets['val'].batch(batch_size)

    model = Feature_Extractor(backbone_name='prototypical_network')
    correlation_network = Correlation_network()
    for c, w, p in train_ds.take(1):
        print(c.shape, w.shape)
        fm_A = model(c)
        fm_B = model(w)
        fm_A_norm = model.channelwise_l2_normalize(fm_A)
        fm_B_norm = model.channelwise_l2_normalize(fm_B)
        score = correlation_network(feature_A, feature_B)
    print(model.channelwise_l2_normalize_debug(fm_A_norm))

if __name__ == '__main__':
    main()
