import numpy as np
import os
from data_loader import load_data
from models.cnn_geo import CNN_geo
from utils import image
from . import visualize
from CNNgeo import train
import tensorflow as tf
import numpy as np

def overfit(config, splits):
    train(config) # mode is overfitted.

def result_test(config, splits):
    model = CNN_geo("prototypical_network")
    ckpt_dir = os.path.join(
        'checkpoints', config['model_name'], config['exp_desc'])
    ckpt = os.path.join(ckpt_dir, "{}-{}.h5".format(config['model_name'], str(config['train']['epochs'])))
    model.load(ckpt)
    datasets = load_data(splits, config)
    ds = datasets['train'].batch(1)
    for image_A, image_B, parameters in ds.take(1):
        image_A = image_A.numpy()
        image_B = image_B.numpy()
        parameters = parameters.numpy()
    pred, _ = model(image_A, image_B)
    print("compare gt : {} and pred : {}".format(parameters, pred))
    loss = tf.reduce_sum(tf.keras.losses.MSE(pred, parameters), axis=1)
    print("loss : {}".format(loss))

    pred = pred.numpy()
    image_C = list(map(lambda x : image.synthesize_image(x[0], x[1], (64, 64), bbox=None, pad_ratio=None),
                   zip(image_A.copy(), pred.copy())))    
    image_C = np.array(image_C)
    visualize.show_TPS_image([image_A, image_B, image_C], [np.ones_like(parameters), parameters, pred])    

if __name__ == "__main__":
    pass
