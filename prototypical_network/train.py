import os
import json
from utils import parser, session_config
from data_loader import data_load
from models import model_load
from trainers.trainer import Protonet_trainer
from matplotlib import pyplot as plt
from pathlib import Path
import tensorflow as tf


def main():
    args = parser.get_args()
    config = args.config
    with open(config, "r") as file:
        config = json.load(file)
    train_config = config['train']

    os.environ['CUDA_VISIBLE_DEVICES'] = train_config['gpu_id']
    session_config.setup_gpus(True, 0.9)

    ds = data_load(['train', 'val'], config)
    train_ds, val_ds = ds['train'], ds['val']
    model = model_load('train', config)
    optimizer = tf.keras.optimizers.Adam(train_config['learning_rate'])
    trainer = Protonet_trainer(model, train_step, val_step, config, optimizer)

    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    # Training process
    print("Training started.")
    for epoch in range(train_config['n_epoch']):
        trainer.on_start_epoch()
        for epi in range(train_config['n_episode']):
            support, query = train_ds.get_next_episode()
            trainer.on_start_episode(support, query, epi)
            trainer.on_end_episode(val_ds)
        trainer.on_end_epoch(epoch)
    print("Training ended.")
    return


@tf.function
def train_step(support, query, model, optimizer):
    with tf.GradientTape() as tape:
        n_class, n_query = support.shape[0], query.shape[1]
        z_prototypes, z_query = model(support, query)
        dists = model.calc_euclidian_dists(z_query, z_prototypes)
        log_p_y = model.calc_probability_with_dists(dists, n_class, n_query)
        loss, pred = model.loss_func(log_p_y, n_class, n_query)
        eq, acc = model.cal_metric(log_p_y, n_class, n_query)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, pred, eq, acc


@tf.function
def val_step(support, query, model):
    n_class, n_query = support.shape[0], query.shape[1]
    z_prototypes, z_query = model(support, query)
    dists = model.calc_euclidian_dists(z_query, z_prototypes)
    log_p_y = model.calc_probability_with_dists(dists, n_class, n_query)
    loss, pred = model.loss_func(log_p_y, n_class, n_query)
    eq, acc = model.cal_metric(log_p_y, n_class, n_query)
    return loss, pred, eq, acc


if __name__ == "__main__":
    main()