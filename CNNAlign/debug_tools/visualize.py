from matplotlib import pyplot as plt
import numpy as np
import cv2


def show_image(images):
    rows, cols = images.shape[:2]
    fig = plt.figure()
    for row in range(rows):
        for col in range(cols):
            image = images[row][col]
            fig.add_subplot(rows, cols, row * col + col + 1).imshow(image)
    plt.show()


def makeBorder(image, bordersize):
    draw_image = image.copy()
    color = [255, 255, 255]
    draw_image = cv2.copyMakeBorder(draw_image,
                                    top=bordersize, bottom=bordersize,
                                    left=bordersize, right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=color)
    return draw_image


def draw_point(image, bordersize, points=None):
    draw_image = image.copy()
    H, W, C = draw_image.shape
    if points is None:
        points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    points = points * (W, H)
    points = points.astype(np.int32)
    for pnt in points:
        draw_image = cv2.circle(draw_image, tuple(
            pnt + bordersize), 1, (0, 1, 0), -1)
    return draw_image


def draw_arrow(image, bordersize, moving_vectors, src_points=None):
    draw_image = image.copy()
    H, W, C = image.shape
    if src_points is None:
        src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                               [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                               [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    src_points = src_points * (W, H)
    src_points = src_points.astype(np.int32)
    moving_vectors = moving_vectors * (W, H)
    dst_points = src_points + moving_vectors
    dst_points = dst_points.astype(np.int32)

    for src, dst in zip(src_points, dst_points):
        draw_image = cv2.arrowedLine(draw_image, tuple(src + bordersize), tuple(dst + bordersize),
                                     (1, 0, 0), 1)
    return draw_image
