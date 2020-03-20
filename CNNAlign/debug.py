import tensorflow as tf
import numpy as np
import cv2
import json
from data_loader import load_data


def show_warped(img1, img2, name1, name2):
    cv2.imwrite(name1, img1)
    cv2.imwrite(name2, img2)


def main():
    with open("configs/cnngeo.json") as file:
        config = json.load(file)
    splits = ['train', 'val', 'test']
    datasets = load_data(splits, config)
    split = 'train'
    train_ds = datasets[split].batch(1)
    for i, j, k in train_ds.take(1):
        print(i.shape)
        print(j.shape)
        # a = i.numpy() * 255
        # b = j.numpy() * 255
        # show_warped(a, b, '{}_1.jpg'.format(
        #     split), '{}_2.jpg'.format(split))


if __name__ == "__main__":
    main()
