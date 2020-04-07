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
    img = cv2.imread('origin.jpg')[:, :, ::-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    output_size = (42, 42)
    cropped_img, bbox = image.crop_image_randomly(img.copy(), (42, 42))
    tps_random_rate = 0.2
    moving_vectors = (np.random.rand(9, 2) - 0.5) * 2 * tps_random_rate

    pad_ratio = None
    warped_image1, mv2 = image.synthesize_image(
        cropped_img.copy(), moving_vectors.copy(), output_size, bbox=None, pad_ratio=pad_ratio)
    pad_ratio = 0.5
    warped_image2, mv1 = image.synthesize_image(
        img.copy(), moving_vectors.copy(), output_size, bbox, pad_ratio)

    ax1.imshow(cropped_img)
    ax2.imshow(warped_image1)
    ax3.imshow(warped_image2)
    plt.show()
    print(cropped_img.shape, warped_image1.shape, warped_image2.shape)


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
        cropped.copy(), parameter.copy(), config['image_shape'][:2], bbox=None, pad_ratio=None)
    ax3.imshow(warped_image)
    plt.show()


def test3():
    with open("configs/labtop_cnngeo.json") as file:
        config = json.load(file)
    batch_size = 10
    splits = ['train']
    datasets = load_data(splits, config)
    train_ds = datasets['train'].batch(batch_size)

    # model = Feature_Extractor(backbone_name='prototypical_network')

    for c, w, p in train_ds.take(10):
        print(c.shape, w.shape)


def show_warped(img1, img2, name1, name2):
    cv2.imwrite(name1, img1)
    cv2.imwrite(name2, img2)


def test4():
    with open("configs/cnngeo.json") as file:
        config = json.load(file)
    splits = ['val']
    datasets = load_data(splits, config)
    split = 'val'
    train_ds = datasets[split].batch(1)
    for i in train_ds.take(1):
        print(i.shape)
        # a = i.numpy() * 255
        # b = j.numpy() * 255
        # show_warped(a, b, '{}_1.jpg'.format(
        #     split), '{}_2.jpg'.format(split))

    img = i.numpy()[0]

    cropped_img, bbox = image.crop_image_randomly(img, (64, 64))

    tps_random_rate = 0.4
    moving_vectors = (np.random.rand(9, 2) - 0.5) * 2 * tps_random_rate

    pad_ratio = 0.25
    # Test 1.
    warped_image1, _ = image.synthesize_image(img, bbox=bbox,
                                              pad_ratio=pad_ratio, moving_vectors=moving_vectors, tps_random_rate=tps_random_rate)
    # Test 2.
    warped_image2, _ = image.synthesize_image(cropped_img, bbox=None,
                                              pad_ratio=pad_ratio, moving_vectors=moving_vectors, tps_random_rate=tps_random_rate)
    # Test 3.
    warped_image3, _ = image.synthesize_image(cropped_img, bbox=None,
                                              pad_ratio=None, moving_vectors=moving_vectors, tps_random_rate=tps_random_rate)

    cv2.imwrite('test1.jpg', warped_image1 * 255)
    cv2.imwrite('test2.jpg', warped_image2 * 255)
    cv2.imwrite('test3.jpg', warped_image3 * 255)


if __name__ == "__main__":
    test2()
