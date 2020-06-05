import os
import sys
import numpy as np
import geo_transform as tps
# root_path = os.path.abspath('.').split('jupyters')[0]
# sys.path.append(root_path)

def get_tgt_from_src(motion_parameters, axay, map_size):
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                           [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                           [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    dst_points = src_points + motion_parameters
    theta = tps.tps_theta_from_points(src_points, dst_points, reduced=True)
    grid = tps.tps_grid(theta, dst_points, map_size)
    mapx, mapy = tps.tps_grid_to_remap(grid, map_size)
    map_src2dst = np.stack([mapx, mapy], axis=-1)
    ri, ci = get_corresponding_points(axay, map_src2dst)
    bxby = np.stack([ci, ri], axis=-1)
    return bxby


def get_corresponding_points(axay, dst_map):
    dst_map = dst_map[:, :, np.newaxis, :]  # shape = 16,16,1,2
    axay = axay[np.newaxis, np.newaxis, :, :]  # shape=1,1,9,2
    distance = np.sum(np.power((dst_map - axay), 2),
                      axis=-1)  # shape=16,16,9
    distance = np.reshape(distance, [-1, distance.shape[-1]])  # shape 16*16,9
    ri, ci = np.argmin(distance, axis = 0) // dst_map.shape[1], np.argmin(distance, axis=0) % dst_map.shape[1]
    return ri, ci

def generate_correlations(axybxy, map_size):
    H, W = map_size
    correlations = np.zeros([H, W, H, W], np.float32)
    axybxy = axybxy.astype(np.int32)
    for ax, ay, bx, by in axybxy:
        correlations[ay, ax, by, bx] = 1.0
    return correlations

if __name__ == "__main__":
    pass
