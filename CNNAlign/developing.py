import os
import json
from data_loader import load_data
from models.cnn_geo import CNN_geo
from models.modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
from geo_transform import tps
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np


def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)


def show_warped(img1, img2, name1, name2):
    cv2.imwrite(name1, img1)
    cv2.imwrite(name2, img2)


def loss_fn(pred, label):
    return tf.reduce_mean(tf.keras.losses.MSE(pred, label))


@tf.function
def train_step(image_A, image_B, label, model, optimizer):
    with tf.GradientTape() as tape:
        pred = model(image_A, image_B)
        loss = loss_fn(pred, tf.reshape(label, [-1, 18]))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred, loss


def main():
    with open("configs/cnngeo.json") as file:
        config = json.load(file)
    batch_size = 1
    splits = ['train', 'val', 'test']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(batch_size)
    model = CNN_geo("prototypical_network")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    epochs = 1

    for epoch in range(epochs):
        print("start of epoch {}".format(epoch))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            pred, loss = train_step(image_a, image_b, label, model, optimizer)
            print('Training loss (for one batch) at step {}: {}'.format(
                step, loss.numpy()))

    for a, b, l in train_ds.take(1):
        pred = model(a, b)
        label = tf.reshape(l, [-1, 18])
        print(pred.numpy())
        print(label.numpy())

    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])
    mv = l.numpy()[0]
    print(mv.shape)
    #mv = np.reshape(mv, [2, 9])
    # b = b[0].numpy()

    # print(a.shape, b.shape)
    # show_warped(a.copy(), b.copy())

    print((src_points + mv).shape)
    warped = warp_image_cv(b.numpy()[0], src_points,
                           src_points + mv)
    show_warped(a.numpy()[0] * 255, warped * 255, 'a.jpg', 'b.jpg')


if __name__ == '__main__':
    main()
