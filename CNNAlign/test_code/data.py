from geo_transform import tps


import cv2
import numpy as np
from utils import image

src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                       [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                       [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])


def show_warped(img1, img2, name1, name2):
    cv2.imwrite(name1, img1)
    cv2.imwrite(name2, img2)


def test_warping_funciton(image_a, image_b, tps_parameters):
    warped = image.interpolate_with_TPS(
        image_a, src_points, src_points + tps_parameters)
    show_warped(warped, image_b, 'warped1.jpg', 'warped2.jpg')
