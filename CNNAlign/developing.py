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
from utils import image
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def loss_fn(pred, label):
    return tf.reduce_sum(tf.keras.losses.MSE(pred, label))


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

        log_dir = 'logs/train'
    summary_writer = tf.summary.create_file_writer(log_dir)

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

    for epoch in range(epochs):
        print("start of epoch {}".format(epoch + 1))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            pred, t_loss = train_step(
                image_a, image_b, label, model, optimizer)
            train_loss(t_loss)
            if step % 100 == 0:
                print('Training loss (for one batch) at step {}: {}'.format(
                    step, loss.numpy()))
        for image_a, image_b, label in val_ds:
            pred, v_loss = val_step(image_a, image_b, label, model)
            val_loss(v_loss)
        template = 'Epoch {}, Loss: {}, ' \
                   'Val Loss: {}'
        print(template.format(epoch + 1, train_loss.result(), val_loss.result()))
        print("end of epoch.")
        with summary_writer.as_default():
            tf.summary.scalar(
                'train_loss', train_loss.result(), step=epoch + 1)
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch + 1)
            summary_writer.flush()
        train_loss.reset_states()
        val_loss.reset_states()

        # Save your model
        if epoch == 0 or (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(
                config['checkpoint_dir'], config['model_name'] + "_{}.h5".format(epoch + 1)))


'''
    for a, b, p in train_ds.take(1):
        pred = model(a, b)
        print(pred)
        print(p)
    image_a = a.numpy()[0]*255
    image_b = b.numpy()[0]*255

    np_pred = np.reshape(pred.numpy()[0], (9, 2))
    warp_image, _ = image.synthesize_image(image_a.copy(), np_pred.copy(), (64, 64), bbox=None, pad_ratio=None)
    print(warp_image.shape)
    cv2.imwrite('image_a.jpg', image_a)
    cv2.imwrite('image_b.jpg', image_b)
    cv2.imwrite('warped_image_a.jpg', warp_image)
'''

if __name__ == '__main__':
    main()
