import os
import json
import argparse
from data_loader import load_data
from models.cnn_geo import CNN_geo
from models.modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
from geo_transform import tps

from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from utils import image, manage_checkpoint
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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


@tf.function
def val_step(image_A, image_B, label, model):
    pred = model(image_A, image_B)
    loss = loss_fn(pred, tf.reshape(label, [-1, 18]))
    return pred, loss


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
    train_ds = datasets['train'].batch(batch_size)
    val_ds = datasets['val'].batch(batch_size)

    model = CNN_geo("prototypical_network")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['train']['learning_rate'])
    epochs = 100

    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')

    ckpt_dir = os.path.join('checkpoints', config['model_name'], config['exp_desc'])
    log_dir = os.path.join('logs', config['model_name'], config['exp_desc'])
    saver1 = manage_checkpoint.Saver(ckpt_dir+'/latest', save_type='latest', max_to_keep=10)
    saver2 = manage_checkpoint.Saver(ckpt_dir+'/best', save_type='best', max_to_keep=10)
    saver3 = manage_checkpoint.Saver(ckpt_dir+'/local_minimum', save_type='local_minimum', interval=10, max_to_keep=10)
    summary_writer = tf.summary.create_file_writer(log_dir)

    for epoch in range(epochs):
        print("start of epoch {}".format(epoch + 1))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            pred, t_loss = train_step(
                image_a, image_b, label, model, optimizer)
            train_loss(t_loss)
            if step % 20 == 0:
                print('Training loss (for one batch) at step {}: {}'.format(
                    step, t_loss.numpy()))
        for image_a, image_b, label in val_ds:
            pred, v_loss = val_step(image_a, image_b, label, model)
            val_loss(v_loss)
        template = 'Epoch {}, Loss: {}, ' \
                   'Val Loss: {}'
        print(template.format(epoch+1, train_loss.result(), val_loss.result()))
        print("end of epoch.")
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch+1)
            summary_writer.flush()
        # Save your model
        saver1.save_or_not(model, epoch+1, val_loss.result())
        saver2.save_or_not(model, epoch+1, val_loss.result())
        saver3.save_or_not(model, epoch+1, val_loss.result())
        train_loss.reset_states()
        val_loss.reset_states()

if __name__ == '__main__':
    main()
