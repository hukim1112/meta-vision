from matplotlib import pyplot as plt
import numpy as np
import cv2
def show_image(images):
    if isinstance(images, list):
        rows = len(images)
        shape = images[0].shape
        if len(shape) == 3:
            #list of single images
            cols = 1
            plot_images(images, (rows, cols))
        elif len(shape) == 4:
            #list of multiple images
            cols = shape[0]
            plot_images(images, (rows, cols))
        else:
            ValueError("The shape of images is wrong : {}".format(images[0].shape))
    else:
        TypeError("input must be list, but type is {}".format(type(images)))

def show_TPS_image(images, moving_vectors):
    if isinstance(images, list):
        rows = len(images)
        shape = images[0].shape
        if len(shape) == 3:
            #list of single images
            cols = 1
            plot_images(images, (rows, cols), moving_vectors)
        elif len(shape) == 4:
            #list of multiple images
            cols = shape[0]
            plot_images(images, (rows, cols), moving_vectors)
        else:
            ValueError("The shape of images is wrong : {}".format(images[0].shape))
    else:
        TypeError("input must be list, but type is {}".format(type(images)))

def plot_images(images, plot_shape, moving_vectors=None):
    rows, cols = plot_shape
    fig = plt.figure()
    for row in range(rows):
        for col in range(cols):
            image = images[row][col]
            image = draw_point(image)
            if row > 0:
                if moving_vectors is not None:
                    image = draw_arrow(image, moving_vectors[row][col])
            fig.add_subplot(rows, cols, row*cols+col+1).imshow(image)
    plt.show()

def draw_point(image):
    draw_image = image.copy()
    H,W,C = draw_image.shape
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                               [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                               [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    src_points = src_points*(64, 64)
    src_points = src_points.astype(np.int32)
    bordersize = 20
    color = [255, 255, 255]
    draw_image=cv2.copyMakeBorder(draw_image, 
                                top=bordersize,bottom=bordersize, 
                                left=bordersize, right=bordersize, 
                                borderType= cv2.BORDER_CONSTANT, 
                                value=color )
    for pnts in src_points:
        draw_image = cv2.circle(draw_image, tuple(pnts+20), 1, (0,1,0), -1)
    return draw_image
def draw_arrow(image, moving_vectors):
    draw_image = image.copy()
    H,W,C = image.shape
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                               [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                               [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    src_points = src_points*(64, 64)
    moving_vectors = moving_vectors*(64, 64)
    dst_point = src_points + moving_vectors
    src_points = src_points.astype(np.int32)
    dst_point = dst_point.astype(np.int32)
    bordersize = 20
    for src, dst in zip(src_points, dst_point):
        draw_image = cv2.arrowedLine(draw_image, tuple(src+20), tuple(dst+20), 
                                     (1,0,0), 1)
        #draw_image = cv2.circle(draw_image, tuple(dst+20), 1, (1,0,0), -1)
    return draw_image


