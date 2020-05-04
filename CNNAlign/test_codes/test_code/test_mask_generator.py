import geo_transform as tps
import numpy as np
import math


def main():
    np.set_printoptions(precision=3, suppress=True)
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])
    dst_points = np.array([[0.2, 0.2], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [5.0, 1.0], [1.0, 1.0]])

    dshape = (8, 8)
    map_size = (dshape[0] - 1, dshape[1] - 1)
    theta = tps.tps_theta_from_points(src_points, dst_points, reduced=True)
    grid = tps.tps_grid(theta, dst_points, dshape)
    print(grid.shape)
    grid = grid[np.newaxis, :, :, :]
    grid = grid * map_size

    binary_mask = generate_binary_mask(grid3, thresh=1.0)

    for i in range(dshape[0]):
        print("({},{})".format(i, i), "a_x, a_y")
        print(binary_mask[0, i, i])


def generate_binary_mask(grids, thresh):
    num_grid, height, width = grids.shape[:3]
    # coordinates of a_x, a_y, b_x, b_y from feature map A (a_x,a_y) and feature map B (b_x,b_y)
    masks = np.zeros([num_grid, height, width, height, width])
    for d, grid in enumerate(grids):
        for b_y in range(height):
            for b_x in range(width):
                # (a_x, a_y) of feature map A matched (b_x, b_y), coord of feature map B
                a_x, a_y = grid[b_y, b_x]
                range_a_x = range(math.ceil(a_x - thresh),
                                  math.floor(a_x + thresh) + 1)
                range_a_x = list(
                    filter(lambda x: x >= 0 and x <= width - 1, range_a_x))
                range_a_y = range(math.ceil(a_y - thresh),
                                  math.floor(a_y + thresh) + 1)
                range_a_y = list(
                    filter(lambda x: x >= 0 and x <= height - 1, range_a_y))
                for a_x in range_a_x:
                    for a_y in range_a_y:
                        masks[d, a_y, a_x, b_y, b_x] = 1
    return masks


if __name__ == '__main__':
    main()
