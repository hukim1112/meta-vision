import math
import numpy as np
import cv2
import geo_transform as tps


def interpolate_with_TPS(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)


def crop_image_randomly(image, output_size):
    '''
        input : original image, output size (H, W)
        output : cropped image, bbox
        bbox is a tuple of 4 integers which means (x_min, y_min, x_max, y_max) of a cropping box respectively
    '''
    original_size = (image.shape[0], image.shape[1])
    output_size = (int(output_size[0]), int(output_size[1]))

    crop_range_x = original_size[1] - output_size[1]
    crop_range_y = original_size[0] - output_size[0]

    # Make start and end points (x, y) by integer random sampling from low(inclusive) to high(exclusive)
    x_min, y_min = (np.random.randint(low=0, high=crop_range_x + 1),
                    np.random.randint(low=0, high=crop_range_y + 1))
    x_max, y_max = (x_min + output_size[1],
                    y_min + output_size[0])

    cropped_image = image[y_min:y_max, x_min:x_max]
    bbox = (x_min, y_min, x_max, y_max)

    return cropped_image, bbox


def pad_image(image, pad_ratio):
    '''
        input : original image, padding_ratio( ragne 0~1 )
        output : padded_image(reflected boundary)
    '''
    original_size = (image.shape[0], image.shape[1])
    top = bottom = int(original_size[0] * pad_ratio)
    left = right = int(original_size[1] * pad_ratio)
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_REFLECT)
    return padded_image


def synthesize_image(image, moving_vectors, output_size, bbox=None, pad_ratio=None):
    '''
        input : 
                'original image', 
                'moving_vectors' : None or array of 9 vectors which get source points moved into destination points.
                'bbox' : None or a tuple of 4 integers. A tuple of 4 values which means (x_min, y_min, x_max, y_max) of a cropping box respectively. It exists when the image is cropped.
                'pad_ratio' : None or a float number.
                It exists when the image is padded.
                
        output : padded_image, moving_vectors(sampled randomly)
    '''
    if bbox is None:
        src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                               [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                               [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])
    else:
        # rescaling coordinates of bbox into 0 to 1.0
        nx0, ny0, nx1, ny1 = normalize_bbox(bbox, image.shape[:2])
        src_points = np.array([[nx0, ny0],
                               [(nx0 + nx1) / 2, ny0],
                               [nx1, ny0],
                               [nx0, (ny0 + ny1) / 2],
                               [(nx0 + nx1) / 2, (ny0 + ny1) / 2],
                               [nx1, (ny0 + ny1) / 2],
                               [nx0, ny1],
                               [(nx0 + nx1) / 2, ny1],
                               [nx1, ny1]])
        moving_vectors[:, 0] = moving_vectors[:, 0] * (nx1 - nx0)
        moving_vectors[:, 1] = moving_vectors[:, 1] * (ny1 - ny0)

    if pad_ratio is None:
        nx0, ny0 = src_points[0]
        nx1, ny1 = src_points[-1]
        bbox = denormalize_bbox((nx0, ny0, nx1, ny1), image.shape[:2])
    else:
        image = pad_image(image, pad_ratio)
        def convert_coord_by_pad_ratio(p): return (
            p + pad_ratio) / (1 + 2 * pad_ratio)
        vfunc = np.vectorize(convert_coord_by_pad_ratio)
        src_points = vfunc(src_points)
        nx0, ny0 = src_points[0]
        nx1, ny1 = src_points[-1]
        bbox = denormalize_bbox((nx0, ny0, nx1, ny1), image.shape[:2])
        moving_vectors[:, 0] = moving_vectors[:, 0] / (1+2 * pad_ratio)
        moving_vectors[:, 1] = moving_vectors[:, 1] / (1+2 * pad_ratio)
    dst_points = src_points + moving_vectors
    x_min, y_min, x_max, y_max = bbox
    warped_image = interpolate_with_TPS(image, src_points, dst_points)
    if warped_image.shape[0]<y_min+output_size[0] or warped_image.shape[1]<x_min+output_size[1]:
    	raise ValueError("Index {} is out of bound of image shape {}".format((y_min+output_size[0], x_min+output_size[1]),
    																		  warped_image.shape))
    return warped_image[y_min:y_min+output_size[0], x_min:x_min+output_size[1]], moving_vectors


def normalize_bbox(coord, shape):
    '''
        convert absolute coordinates into normalized coordinates
    '''
    x_min, y_min, x_max, y_max = coord
    x_min, x_max = x_min / shape[1], x_max / shape[1]
    y_min, y_max = y_min / shape[0], y_max / shape[0]
    return x_min, y_min, x_max, y_max


def denormalize_bbox(coord, shape):
    '''
        convert normalized coordinates into absolute coordinates
    '''
    x_min, y_min, x_max, y_max = coord
    x_min, x_max = int(round(x_min * shape[1])), int(round(x_max * shape[1]))
    y_min, y_max = int(round(y_min * shape[0])), int(round(y_max * shape[0]))
    return x_min, y_min, x_max, y_max


def make_synthesized_image_pair(image, moving_vectors, pad_ratio, output_size=(64, 64)):
    image = image.numpy()
    moving_vectors = moving_vectors.numpy()
    cropped_image, bbox = crop_image_randomly(image, output_size)
    warped_image, _ = synthesize_image(
        image.copy(), moving_vectors.copy(), output_size, bbox, pad_ratio)
    return cropped_image, warped_image, moving_vectors