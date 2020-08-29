import os, argparse, json
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from functools import partial
from utils import tf_session
from geo_transform.tf_tps import ThinPlateSpline as tps
from data_loader.dev_dataset import tf_image_process

from models.cnngeo import CNN_geotransform

os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf_session.setup_gpus(True, 0.95)

def train_geo(config):
    def loss_fn(preds, labels):
        control_points = tf.constant([[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
                                   [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                                   [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
        num_batch = preds.shape[0]
        pred_grid_x, pred_grid_y = tps(tf.tile(control_points[tf.newaxis,::], [num_batch,1,1]), -preds, (20, 20))
        gt_grid_x, gt_grid_y = tps(tf.tile(control_points[tf.newaxis,::], [num_batch,1,1]), -labels, (20, 20))
        
        dist = tf.sqrt(tf.pow(pred_grid_x - gt_grid_x, 2) + tf.pow(pred_grid_y - gt_grid_y, 2))
        loss_mean = tf.reduce_mean(dist)
        return loss_mean

    @tf.function
    def train_step(image_A, image_B, labels, model, optimizer):
        with tf.GradientTape() as tape:
            preds, corr = model(image_A, image_B)
            loss = loss_fn(preds, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    _datapath = "/home/files/datasets/PF-dataset-PASCAL/JPEGImages"
    filelist = os.listdir(_datapath)
    random.shuffle(filelist)
    input_size = (200, 200)
    images = []
    for f in filelist:
        _path = os.path.join(_datapath, f)
        img = cv2.imread(_path)[:,:,::-1]
        img = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
        images.append(img)
    images = np.array(images, dtype=np.float32)
    tps_random_rate = 0.4
    output_size = (200, 200)
    map_func = partial(tf_image_process, tps_random_rate=tps_random_rate,
                          output_size=output_size)
    ds = tf.data.Dataset.from_tensor_slices(images).shuffle(2000)
    ds = ds.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(10)
    for A, B, p in ds.take(1):
        print(A.shape, B.shape)
        print(p.shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4)

    train_loss = tf.metrics.Mean(name='train_loss')
    x_axis = []
    y_loss = []
    for epoch in range(4000):
        for step, (image_a, image_b, labels) in enumerate(ds):
            t_loss = train_step(image_a, image_b, labels, cnngeo, optimizer)
            train_loss(t_loss)
        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
        x_axis.append(epoch)
        y_loss.append(train_loss.result().numpy())
        train_loss.reset_states()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='config_file',
                        help='path to config file.')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as file:
        config = json.load(file)

    if config['model'] == "cnngeo":
        train_cnngeo(config)
    elif config['model'] == "cnnalign":
        train_cnnalign(config)
    else:
        raise ValueError("Make sure the model name is correct in your config file.")


