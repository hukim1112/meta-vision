import tensorflow as tf
import numpy as np
import cv2
import json
from data_loader import load_data
from models.cnn_geo import CNN_geo
from matplotlib import pyplot as plt
from utils import image


def visualize(image_A, image_B, image_C, batch_size):
    print(image_A.shape, image_B.shape, image_C.shape)
    images = np.concatenate([image_A, image_B, image_C], axis=0)
    print(images.shape)
    num_images = images.shape[0]
    fig = plt.figure()
    fig_axis = []
    col = batch_size
    row = num_images//col
    
    for i in range(num_images):
        fig_axis.append(fig.add_subplot(row, col,i+1))
        fig_axis[i].imshow(images[i])
    plt.show()

def data_test_module(image_A, image_B, parameters):
    batch_size = parameters.shape[0]
    _list = list(map(lambda x : image.synthesize_image(x[0], x[1], (64,64), bbox=None, pad_ratio=None)
    ,zip(image_A.copy(), parameters.copy())))
    image_C = np.array([img  for (img, p) in _list])
    visualize(image_A, image_B, image_C, batch_size)

def data_test():
    with open("configs/CNNgeo/overfit.json") as file:
        config = json.load(file)
    splits = ['train', 'val']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(4)
    for image_A, image_B, parameters in train_ds.take(1):
        image_A = image_A.numpy()
        image_B = image_B.numpy()
        parameters = parameters.numpy()
    data_test_module(image_A, image_B, parameters)


def model_test_module(image_A, image_B, parameters, model):
    batch_size = parameters.shape[0]
    preds = model(image_A, image_B).numpy()

    print("Comparison parameters and pred")
    for i, j in parameters, preds:
        print("parameters : {}, pred : {}".format(i, j))

    _list = list(map(lambda x : image.synthesize_image(x[0], x[1], (64,64), bbox=None, pad_ratio=None)
    ,zip(image_A.copy(), preds.copy())))
    image_C = np.array([img  for (img, p) in _list])
    visualize(image_A, image_B, image_C, batch_size)

def model_test()
    with open("configs/CNNgeo/overfit.json") as file:
        config = json.load(file)
    splits = ['train', 'val']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(4)

    model = CNN_geo("prototypical_network")
    model.load(config['ckpt']['trained_cnngeo'])

    for image_A, image_B, parameters in train_ds.take(1):
        image_A = image_A.numpy()
        image_B = image_B.numpy()
        parameters = parameters.numpy()
    data_test_module(image_A, image_B, parameters, model)
if __name__ == "__main__":
    data_pipeline_test()