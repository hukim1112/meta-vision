import os, argparse, json
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from utils import tf_session
from geo_transform.tf_tps import ThinPlateSpline as tps
#from models.cnngeo import CNN_geotransform
from models.cnngeo import CNN_geotransform
from data_loader import load_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf_session.setup_gpus(True, 0.95)

def train_cnngeo(config):
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

    #1. dataset pipeline
    PF_Pascal = load_data("PF_Pascal")
    dataset = PF_Pascal(config["data_dir"])
    input_shape = config["image_shape"][:2]
    n_examples = config["n_examples"]
    data_normalize = tf.keras.applications.vgg16.preprocess_input
    ds = dataset.load_pipeline("SynthesizedImagePair", input_shape, n_examples, data_normalize)
    ds = ds.shuffle(1000).batch(config["train"]["batch_size"])
    for A, B, p in ds.take(1):
        print(A.shape, B.shape)
        print(p.shape)

    # 2. load model
    if config["backbone"] == "vgg16":
        vgg16 = tf.keras.applications.VGG16(weights='imagenet', input_shape=(input_size[0], input_size[1], 3),
                                            include_top=False)
        output_layer = vgg16.get_layer("block4_conv3")
        output_layer.activation = None
        feature_extractor = tf.keras.Model(inputs=vgg16.input, outputs=output_layer.output)
    cnngeo = CNN_geotransform(feature_extractor, 18)

    # 3. Training
    model_name = config["model_name"]
    exp_desc = config["exp_desc"]

    log_dir = os.path.join('logs', model_name, exp_desc)
    summary_writer = tf.summary.create_file_writer(log_dir)
    if os.path.isdir(log_dir):
        raise ValueError("log directory exists. checkout your experiment name in configure file.")

    ckpt_dir = os.path.join('checkpoints', model_name, exp_desc)
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.isdir(ckpt_dir):
        raise ValueError("checkpoint directory exists. checkout your experiment name in configure file.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    train_loss = tf.metrics.Mean(name='train_loss')
    for epoch in range(config["train"]["epochs"]):
        for step, (image_a, image_b, labels) in enumerate(ds):
            t_loss = train_step(image_a, image_b, labels, cnngeo, optimizer)
            train_loss(t_loss)
        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)
        train_loss.reset_states()
        if (epoch+1)%20 == 0:
            model.save(os.path.join(ckpt_dir,
                                    "{}-{}.h5".format(model_name, epoch)))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='config_file',
                        help='path to config file.')
    args = parser.parse_args()
    config = args.config
    with open(config, 'r') as file:
        config = json.load(file)

    if config['model_name'] == "cnngeo":
        train_cnngeo(config)
    elif config['model_name'] == "cnnalign":
        train_cnnalign(config)
    else:
        raise ValueError("Make sure the model name is correct in your config file.")
