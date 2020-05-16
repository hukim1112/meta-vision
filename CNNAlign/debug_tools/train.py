import numpy as np
import os
from data_loader import load_data
from models.cnn_geo import CNN_geo
from utils import image
from . import visualize
from CNNgeo_debug import train
import tensorflow as tf
import numpy as np

def overfit(config, splits):
    return train(config) # mode is overfitted.

def predict_test(config, splits):
    model = CNN_geo(config['backbone'])
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
    #image_B_hat = np.ones([1, 64, 64, 3])
    pred, score = model(image_A, image_B)
    #print(score.shape)
    print("compare gt : {} and pred : {}".format(parameters, pred))
    print("score : {}".format(score[0,0,0]))
    loss = tf.reduce_sum(tf.keras.losses.MSE(pred, parameters), axis=1)
    print("loss : {}".format(loss))

    pred = pred.numpy()
    image_C = list(map(lambda x : image.synthesize_image(x[0], x[1], (64, 64), bbox=None, pad_ratio=None),
                   zip(image_A.copy(), pred.copy())))    
    image_C = np.array(image_C)
    visualize.show_TPS_image([image_A, image_B, image_C], [np.ones_like(parameters), parameters, pred])    

def draw_grid(image, grid_coord, color = (0, 100, 0), grid_shape = (16, 16)):
    draw_image = image.copy()
    H, W, C = draw_image.shape
    grid_size = H/grid_shape[0], W/grid_shape[1]

    start_pix_h = int(grid_size[0]*grid_coord[0])
    end_pix_h = int(grid_size[0]*(1+grid_coord[0]))

    start_pix_w = int(grid_size[1]*grid_coord[1])
    end_pix_w = int(grid_size[1]*(1+grid_coord[1]))

    draw_image[start_pix_h:end_pix_h, start_pix_w:end_pix_w, 2] = 100
    return draw_image




if __name__ == "__main__":
    pass
