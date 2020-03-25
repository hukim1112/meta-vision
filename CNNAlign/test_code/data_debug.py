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


def main():
    '''
    with open("configs/labtop_cnngeo.json") as file:
        config = json.load(file)
    splits = ['val']
    datasets = load_data(splits, config)
    split = 'val'
    train_ds = datasets[split].batch(1)
    for i in train_ds.take(1):
        print(i.shape)

    img = i.numpy()[0]

    '''
    img = cv2.imread('origin.jpg')[:,:,::-1]
    pad_ratio = 0.5
    tps_random_rate = 0.2

    cropped_img, bbox = image.crop_image_randomly(img.copy(), (42, 42))

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # moving_vectors = np.zeros([9,2])
    moving_vectors = (np.random.rand(9, 2) - 0.5) * 2 * tps_random_rate
    warped_image1, mv1 = image.synthesize_image(
        img.copy(), bbox=bbox, pad_ratio=pad_ratio, moving_vectors=moving_vectors.copy(), tps_random_rate=0)
    pad_ratio = None
    warped_image2, mv2 = image.synthesize_image(
        cropped_img.copy(), bbox=None, pad_ratio=pad_ratio, moving_vectors=moving_vectors.copy(), tps_random_rate=0)
    #print("hello", mv1, mv2)
    ax1.imshow(cropped_img)
    ax2.imshow(warped_image2)
    ax3.imshow(warped_image1)
    plt.show()

    # test 1.
    '''
        image -> crop -> pad -> synthesize -> crop -> warped
    '''
    # print(bbox)

    # # Test 1.
    # warped_image1, _ = image.synthesize_image(img, bbox=bbox,
    #                                           pad_ratio=pad_ratio, moving_vectors=moving_vectors, tps_random_rate=tps_random_rate)
    # # Test 2.
    # warped_image2, _ = image.synthesize_image(cropped_img, bbox=None,
    #                                           pad_ratio=pad_ratio, moving_vectors=moving_vectors, tps_random_rate=tps_random_rate)
    # # Test 3.
    # warped_image3, _ = image.synthesize_image(cropped_img, bbox=None,
    #                                           pad_ratio=None, moving_vectors=moving_vectors, tps_random_rate=tps_random_rate)

    # cv2.imwrite('test1.jpg', warped_image1 * 255)
    # cv2.imwrite('test2.jpg', warped_image2 * 255)
    #cv2.imwrite('test3.jpg', warped_image3 * 255)


if __name__ == "__main__":
    main()
