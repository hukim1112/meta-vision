import os
import sys
import numpy as np
import geo_transform as tps
# root_path = os.path.abspath('.').split('jupyters')[0]
# sys.path.append(root_path)


def generate_correlations(ijkl, map_size):
    H, W = map_size
    correlations = np.zeros([H, W, H, W], np.float32)
    for i, j, k, l in ijkl:
        correlations[i, j, k, l] = 1.0
    return correlations


def get_tgt_from_src(parameters, map_size, ij):
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    dst_points = src_points + parameters
    theta = tps.tps_theta_from_points(src_points, dst_points, reduced=True)
    grid = tps.tps_grid(theta, dst_points, map_size)
    mapx, mapy = tps.tps_grid_to_remap(grid, map_size)
    map_src2dst = np.stack([mapy, mapx], axis=-1)
    ri, ci = get_corresponding_points(ij, map_src2dst)
    kl = np.stack([ri, ci], axis=-1)
    return kl


def get_corresponding_points(src_points, dst_map):
    dst_map = dst_map[:, :, np.newaxis, :]  # shape = 16,16,1,2
    src_points = src_points[np.newaxis, np.newaxis, :, :]  # shape=1,1,9,2
    distance = np.sum(np.power((dst_map - src_points), 2),
                      axis=-1)  # shape=16,16,9
    distance = np.reshape(distance, [-1, distance.shape[-1]])  # shape 16*16,9
    ri, ci = distance.argmin(
        0) // dst_map.shape[1], distance.argmin(0) % dst_map.shape[1]
    return ri, ci


if __name__ == "__main__":
    pass
