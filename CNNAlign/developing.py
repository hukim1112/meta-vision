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
    model.cnn_geo.load("trained_cnngeo.h5")

def loss_fn(pred):
    return tf.reduce_mean(-pred)

#@tf.function
def train_step(image_A, image_B, label, model, optimizer):
    inlier_matching, sum_of_inlier_matching = model(image_A, image_B)
    loss = loss_fn(sum_of_inlier_matching)
    print(loss)
    #gradients = tape.gradient(loss, model.trainable_variables)
    #optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    config = args.config
    with open(config) as file:
        config = json.load(file)

    batch_size = config['train']['batch_size']
    splits = ['train', 'val']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = datasets['val'].batch(batch_size)

    model = CNN_align("prototypical_network", config['thresh'])
    model.load(config['ckpt']['trained_cnngeo'])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['train']['learning_rate'])
    for epoch in range(config['train']['epochs']):
        print("start of epoch {}".format(epoch + 1))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            t_loss = train_step(
                image_a, image_b, label, model, optimizer)
if __name__ == '__main__':
    main()
