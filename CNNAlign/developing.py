import os
import json
import argparse
from data_loader import load_data
from models.cnn_geo import CNN_geo
from models.cnn_align import CNN_align
from models.modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
from geo_transform import tps

from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from utils import image, manage_checkpoint
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def test1():
    model = CNN_geo("prototypical_network")
    output = model(tf.zeros([1,64,64,3]), tf.zeros([1,64,64,3]))
    model.load_weights("model-24.h5")
    model.summary()

def test2():
    model = CNN_align("prototypical_network")
    model.cnn_geo.load("model-24.h5")


if __name__ == '__main__':
    test2()
