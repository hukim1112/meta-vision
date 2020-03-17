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


def synthesize_image(image, bbox=None, pad_ratio=None, moving_vectors=None):
    '''
        input : 
                'original image', 
                'bbox' : None or a tuple of 4 integers. A tuple of 4 values which means (x_min, y_min, x_max, y_max) of a cropping box respectively. It exists when the image is cropped.
                'pad_ratio' : None or a float number.
                It exists when the image is padded.
                'moving_vectors' : None or array of 9 vectors which get source points moved into destination points.

        output : padded_image, moving_vectors(sampled randomly)
    '''
    if bbox == None:
        src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                               [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                               [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])
    else:
        # rescaling coordinates of bbox into 0 to 1.0
        nx0, ny0, nx1, ny1 = normalize_bbox(bbox, image.shape[:2])
        src_points = np.array([[nx0, ny0], [(nx0 + nx1) / 2, ny0], [nx1, ny0],
                               [nx0, (ny0 + ny1) / 2], [(nx0 + nx1) / 2,
                                                        (ny0 + ny1) / 2], [nx1, (ny0 + ny1) / 2],
                               [nx0, ny1], [(nx0 + nx1) / 2, ny1], [nx1, ny1]])
    if pad_ratio == None:
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
        bbox = denormalize_bbox((nx0, ny0, nx1, ny1), image.shape)

    dst_points = src_points + moving_vectors

    x_min, y_min, x_max, y_max = bbox
    warped_image = interpolate_with_TPS(image, src_points, dst_points)
    return warped_image[y_min:y_max, x_min:x_max], moving_vectors


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


def make_synthesized_image_pair(image, output_size=(64, 64), tps_random_rate=0.4):
    image = image.numpy()
    cropped_image, bbox = crop_image_randomly(image, output_size)
    moving_vectors = (np.random.rand(9, 2) - 0.5) * 2 * tps_random_rate
    pad_ratio = 0.25
    warped_image, moving_vectors = synthesize_image(
        image, bbox, pad_ratio, moving_vectors)
    return cropped_image, warped_image, moving_vectors


def main():
    image = cv2.imread('image.png')[:, :, ::-1]
    output_size = (image.shape[0] * 0.8, image.shape[1] * 0.8)
    cropped_image, bbox = crop_image_randomly(image, output_size)
    tps_random_rate = 0.4
    moving_vectors = (np.random.rand(9, 2) - 0.5) * 2 * tps_random_rate
    # moving_vectors = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    #                            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    #                            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    pad_ratio = 0.25
    warped_image, moving_vectors = synthesize_image(
        image, bbox, pad_ratio, moving_vectors)
    print(image.shape, cropped_image.shape, warped_image.shape)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    plt.imshow(image)
    ax2 = fig.add_subplot(132)
    plt.imshow(cropped_image)
    ax3 = fig.add_subplot(133)
    plt.imshow(warped_image)
    plt.show()


if __name__ == "__main__":
    main()
