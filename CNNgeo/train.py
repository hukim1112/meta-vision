from geo_transform.tps import ThinPlateSpline as tps

import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from functools import partial
import os
import session_config
session_config.setup_gpus(True, 0.95)

import models


def py_image_process(image, motion_vectors, output_size):
    image = tf.keras.applications.vgg16.preprocess_input(image)
    image = image.numpy()
    #image = preprocess_input(image)
    control_points = tf.constant([[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
                                  [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                                  [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
    x_s, y_s = tps(control_points[tf.newaxis, ::], -
                   motion_vectors[tf.newaxis, ::], output_size)
    synth_image = cv2.remap(
        image, x_s[0].numpy(), y_s[0].numpy(), cv2.INTER_CUBIC)
    return image, synth_image, motion_vectors


def tf_image_process(image, tps_random_rate, output_size):
    motion_vectors = (tf.random.uniform([9, 2]) - 0.5) * 2 * tps_random_rate
    return tf.py_function(py_image_process, [image, motion_vectors, output_size], [tf.float32, tf.float32, tf.float32])


def loss_fn(preds, labels):
    control_points = tf.constant([[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
                                  [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                                  [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
    num_batch = preds.shape[0]
    pred_grid_x, pred_grid_y = tps(tf.tile(control_points[tf.newaxis, ::], [
                                   num_batch, 1, 1]), preds, (20, 20))
    gt_grid_x, gt_grid_y = tps(tf.tile(control_points[tf.newaxis, ::], [
                               num_batch, 1, 1]), labels, (20, 20))

    dist = tf.sqrt(tf.pow(pred_grid_x - gt_grid_x, 2) +
                   tf.pow(pred_grid_y - gt_grid_y, 2))
    print("distshape", dist.shape)
    loss_mean = tf.reduce_mean(dist)
    return loss_mean


@tf.function
def train_step(image_A, image_B, labels, model, optimizer):
    with tf.GradientTape() as tape:
        preds = model(image_A, image_B)
        loss = loss_fn(preds, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def main():
    _datapath = "sample_dataset"
    filelist = os.listdir(_datapath)
    input_size = (200, 200)

    images = []

    for f in filelist:
        _path = os.path.join(_datapath, f)
        img = cv2.imread(_path)[:, :, ::-1]
        img = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
        images.append(img)

    images = np.array(images)
    tps_random_rate = 0.2
    output_size = (200, 200)
    map_func = partial(tf_image_process, tps_random_rate=tps_random_rate,
                       output_size=output_size)

    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(16)

    vgg16 = tf.keras.applications.VGG16(weights='imagenet', input_shape=(input_size[0], input_size[1], 3),
                                        include_top=False)
    output_layer = vgg16.get_layer("block4_conv3")
    output_layer.activation = None
    feature_extractor = tf.keras.Model(
        inputs=vgg16.input, outputs=output_layer.output)
    cnngeo = models.CNN_geotransform(feature_extractor, 9 * 2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4)

    train_loss = tf.metrics.Mean(name='train_loss')
    x_axis = []
    y_loss = []
    for epoch in range(200):
        for step, (image_a, image_b, labels) in enumerate(ds):
            t_loss = train_step(image_a, image_b, labels, cnngeo, optimizer)
            train_loss(t_loss)
        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
        x_axis.append(epoch)
        y_loss.append(train_loss.result().numpy())
        train_loss.reset_states()
        cnngeo.save_weights(os.path.join(
            "checkpoints", "model-" + epoch + ".h5"))
