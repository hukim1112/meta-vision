import tensorflow as tf
import numpy as np
import cv2
import json
from data_loader import load_data
from matplotlib import pyplot as plt
from utils import image


def show_warped(img1, img2, name1, name2):
    cv2.imwrite(name1, img1)
    cv2.imwrite(name2, img2)


def test1():
    img = cv2.imread('origin.jpg')[:,:,::-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    output_size = (42, 42)
    cropped_img, bbox = image.crop_image_randomly(img.copy(), (42, 42))

    moving_vectors = (np.random.rand(9, 2) - 0.5) * 2 * tps_random_rate

    pad_ratio = None
    warped_image1, mv2 = image.synthesize_image(
        cropped_img.copy(), moving_vectors.copy(), bbox=None, output_size, pad_ratio=pad_ratio)
    pad_ratio = 0.5
    warped_image2, mv1 = image.synthesize_image(
        img.copy(), moving_vectors.copy(), output_size, bbox, pad_ratio)

    ax1.imshow(cropped_img)
    ax2.imshow(warped_image1)
    ax3.imshow(warped_image2)
    plt.show()

def test2():
    with open("configs/labtop_cnngeo.json") as file:
        config = json.load(file)
    batch_size = 1
    splits = ['val']
    datasets = load_data(splits, config)
    val_ds = datasets['val'].batch(batch_size)

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for c, w, p in val_ds.take(1):
        cropped = c.numpy()[0]
        warped = w.numpy()[0]
        parameter = p.numpy()[0]
        ax1.imshow(cropped)
        ax2.imshow(warped)

    warped_image, mv2 = image.synthesize_image(
        cropped.copy(), parameter.copy(), output_size, bbox=None, pad_ratio=None)
    ax3.imshow(warped_image)
    plt.show()

if __name__ == "__main__":
    test1()
