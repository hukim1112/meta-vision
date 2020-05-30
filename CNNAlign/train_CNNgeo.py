import os
import sys
import json
import argparse
from data_loader import load_data
from models.cnn_geo import CNN_geo
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from utils import image, manage_checkpoint
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def loss_fn(pred, label):
    loss = tf.sqrt(tf.reduce_sum(tf.pow(pred - label, 2), axis=[-1, -2]))
    loss_mean = tf.reduce_mean(loss)
    return loss_mean


@tf.function
def train_step(image_A, image_B, label, model, optimizer):
    with tf.GradientTape() as tape:
        pred, score = model(image_A, image_B)
        loss = loss_fn(pred, label)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred, loss, score


@tf.function
def val_step(image_A, image_B, label, model):
    pred, _ = model(image_A, image_B)
    loss = loss_fn(pred, label)
    return pred, loss


def train(config):
    batch_size = config['train']['batch_size']
    splits = ['train', 'val']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = datasets['val'].batch(batch_size)

    model = CNN_geo(config['backbone'])
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['train']['learning_rate'])

    train_loss = tf.metrics.Mean(name='train_loss')
    train_score_std = tf.metrics.Mean(name='train_score_std')
    val_loss = tf.metrics.Mean(name='val_loss')

    ckpt_dir = os.path.join(
        'checkpoints', config['model_name'], config['exp_desc'])
    log_dir = os.path.join('logs', config['model_name'], config['exp_desc'])
    saver = manage_checkpoint.Saver(
        ckpt_dir, save_type=config['ckpt']['save_type'], max_to_keep=config['ckpt']['max_to_keep'])
    summary_writer = tf.summary.create_file_writer(log_dir)

    for epoch in range(config['train']['epochs']):
        print("start of epoch {}".format(epoch + 1))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            pred, t_loss, score = train_step(
                image_a, image_b, label, model, optimizer)
            score_std = tf.math.reduce_std(score)
            train_loss(t_loss)
            train_score_std(score_std)

            if step % config['train']['print_step'] == 0:
                print('Training loss (for one batch) at step {}: {}'.format(
                    step, t_loss.numpy()))
        for image_a, image_b, label in val_ds:
            pred, v_loss = val_step(image_a, image_b, label, model)
            val_loss(v_loss)
        template = 'Epoch {}, Loss: {}, Val Loss: {}'
        print(template.format(epoch + 1, train_loss.result(), val_loss.result()))
        print("end of epoch.")
        with summary_writer.as_default():
            tf.summary.scalar(
                'train_loss', train_loss.result(), step=epoch + 1)
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch + 1)
            tf.summary.scalar(
                'score std', train_score_std.result(), step=epoch + 1)
            summary_writer.flush()
        # Save your model
        saver.save_or_not(model, epoch + 1, val_loss.result())
        train_loss.reset_states()
        train_score_std.reset_states()
        val_loss.reset_states()
    print("Checkpoint directory : ", ckpt_dir)
    print("Tensorboard log directory : ", log_dir)
    return model


def train_debug(config, tentative_penalty):
    batch_size = config['train']['batch_size']
    splits = ['train']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    model = CNN_geo(config['backbone'], tentative_penalty)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['train']['learning_rate'])

    train_loss = tf.metrics.Mean(name='train_loss')
    train_score_std = tf.metrics.Mean(name='train_score_std')

    x = []
    y_loss = []
    y_score_std = []
    for epoch in range(config['train']['epochs']):
        for step, (image_a, image_b, label) in enumerate(train_ds):
            pred, t_loss, score = train_step(
                image_a, image_b, label, model, optimizer)
            score_std = tf.math.reduce_std(score)
            train_loss(t_loss)
            train_score_std(score_std)
        template = 'Epoch {}, Loss: {}, score std {}'
        print(template.format(epoch + 1, train_loss.result(), train_score_std.result()))
        x.append(epoch)
        y_loss.append(train_loss.result().numpy())
        y_score_std.append(train_score_std.result().numpy())

        train_loss.reset_states()
        train_score_std.reset_states()

    return model, x, y_loss, y_score_std


if __name__ == '__main__':
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
    train(config)
