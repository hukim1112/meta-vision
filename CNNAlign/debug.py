import tensorflow as tf
import numpy as np
import cv2
import json
from data_loader import load_data
from utils import image


def show_warped(img1, img2, name1, name2):
    cv2.imwrite(name1, img1)
    cv2.imwrite(name2, img2)


def main():
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
    main()
