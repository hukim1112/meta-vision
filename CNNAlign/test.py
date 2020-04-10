import argparse
import json
from data_loader import load_data
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from utils import image
from models.cnn_geo import CNN_geo


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

    ds = load_data(['train'], config)
    val_ds = ds['train']
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for image1, image2, moving_vector in val_ds.take(1):
        ax1.imshow(image1)
        ax2.imshow(image2)
        print(moving_vector)
    plt.show()


def loss_fn(pred, label):
    return tf.reduce_mean(tf.keras.losses.MSE(pred, label))


def test1():
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

    model = CNN_geo("prototypical_network")
    ckpt_path = "checkpoints/CNNgeo/200405_get_best_cnngeo/best/model-1.h5"
    model.load(ckpt_path)
    batch_size = config['train']['batch_size']
    ds = load_data(['train'], config)
    train_ds = ds['train'].batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    for image_A, image_B, label in train_ds.take(1):
        pred, _ = model(image_A, image_B)
        loss = loss_fn(pred, tf.reshape(label, [-1, 18]))
        print("pred and label", pred, label)
        print("LOSS", loss)


def loss_fn(pred, label):
    return tf.reduce_mean(tf.keras.losses.MSE(pred, label))


@tf.function
def train_step(image_A, image_B, label, model, optimizer):
    with tf.GradientTape() as tape:
        pred, _ = model(image_A, image_B)
        loss = loss_fn(pred, tf.reshape(label, [-1, 18]))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred, loss


def test2():
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
    splits = ['train']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    model = CNN_geo("prototypical_network")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['train']['learning_rate'])

    train_loss = tf.metrics.Mean(name='train_loss')

    for epoch in range(config['train']['epochs']):
        print("start of epoch {}".format(epoch + 1))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            pred, t_loss = train_step(
                image_a, image_b, label, model, optimizer)
            print('Training loss (for one batch) at step {}: {}'.format(
                step, t_loss.numpy()))
            print("label : {} , pred : {}".format(label.numpy(), pred.numpy()))


def test3():
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
    splits = ['train']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    for step, (image_a, image_b, label) in enumerate(train_ds):
        print(label)


def test4():
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
    splits = ['train']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for image_a, image_b, label in train_ds.take(1):
        image_a = image_a.numpy()[0]
        image_b = image_b.numpy()[0]
        label = label.numpy()[0]
        ax1.imshow(image_a)
        ax2.imshow(image_b)
    pad_ratio = 0.25
    warped_image, mv2 = image.synthesize_image(
        image_a.copy(), label.copy(), config['image_shape'][:2], bbox=None, pad_ratio=pad_ratio)
    ax3.imshow(warped_image)
    plt.show()


def test5():
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
    splits = ['train']
    datasets = load_data(splits, config)
    train_ds = datasets['train']

    for i, j in train_ds.take(5):
        print(i.shape, j.shape)
        image_a = i.numpy()
        image_b = j.numpy()

    print(image_a[0])
    fig = plt.figure()
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)

    ax1.imshow(image_a[0])
    ax2.imshow(image_b[0])
    ax3.imshow(image_a[10])
    ax4.imshow(image_b[10])
    ax5.imshow(image_a[20])
    ax6.imshow(image_b[20])
    ax7.imshow(image_a[30])
    ax8.imshow(image_b[30])

    plt.show()


if __name__ == '__main__':
    test5()
