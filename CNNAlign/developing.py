import os
import json
from data_loader import load_data
from models.cnn_geo import CNN_geo
from models.modules import Feature_Extractor, Correlation_network, Spatial_transformer_regressor
from matplotlib import pyplot as plt
import tensorflow as tf


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
    batch_size = 4
    datasets = load_data(['val', 'train'], config)
    train_ds = datasets['train'].batch(batch_size)
    model = CNN_geo("prototypical_network")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    epochs = 3

    for epoch in range(epochs):
        print("start of epoch {}".format(epoch))
        for step, (image_a, image_b, label) in enumerate(train_ds):
            pred, loss = train_step(image_a, image_b, label, model, optimizer)
            print('Training loss (for one batch) at step {}: {}'.format(
                step, loss.numpy()))


if __name__ == '__main__':
    main()
