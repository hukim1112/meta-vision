import os
import json
import argparse
from data_loader import load_data
from models.cnn_align import CNN_align

from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from utils import image, manage_checkpoint
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def loss_fn(pred):
    return -pred


@tf.function
def train_step(image_A, image_B, label, model, optimizer):
    with tf.GradientTape() as tape:
        inlier_matching, sum_of_inlier_matching = model(image_A, image_B)
        loss = loss_fn(sum_of_inlier_matching)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def val_step(image_A, image_B, label, model):
    inlier_matching, sum_of_inlier_matching = model(image_A, image_B)
    loss = loss_fn(sum_of_inlier_matching)
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

    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')

    ckpt_dir = os.path.join(
        'checkpoints', config['model_name'], config['exp_desc'])
    log_dir = os.path.join('logs', config['model_name'], config['exp_desc'])
    saver = manage_checkpoint.Saver(ckpt_dir, config['ckpt']['save_type'], config['ckpt']['max_to_keep'])
    summary_writer = tf.summary.create_file_writer(log_dir)

    for epoch in range(config['train']['epochs']):
        print("start of epoch {}".format(epoch + 1))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            t_loss = train_step(
                image_a, image_b, label, model, optimizer)
            train_loss(t_loss)
            if step % 20 == 0:
                print('Training loss (for one batch) at step {}: {}'.format(
                    step, t_loss.numpy()))
        for image_a, image_b, label in val_ds:
            v_loss = val_step(image_a, image_b, label, model)
            val_loss(v_loss)
        template = 'Epoch {}, Loss: {}, Val Loss: {}'
        print(template.format(epoch + 1, train_loss.result(), val_loss.result()))
        print("end of epoch.")
        with summary_writer.as_default():
            tf.summary.scalar(
                'train_loss', train_loss.result(), step=epoch + 1)
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch + 1)
            summary_writer.flush()
        # Save your model
        saver.save_or_not(model, epoch + 1, val_loss.result())
        train_loss.reset_states()
        val_loss.reset_states()
    print("Checkpoint directory : ", ckpt_dir)
    print("Tensorboard log directory : ", log_dir)

if __name__ == '__main__':
    main()
