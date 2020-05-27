import os, sys
import numpy as np
import geo_transform as tps
# root_path = os.path.abspath('.').split('jupyters')[0]
# sys.path.append(root_path)


def synthesize_correlation(parameters, map_size):
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                          [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                          [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])    
    dst_points = src_points + parameters
    theta = tps.tps_theta_from_points(src_points, dst_points, reduced=True)
    grid = tps.tps_grid(theta, dst_points, map_size)
    mapx, mapy = tps.tps_grid_to_remap(grid, map_size)
    #points = np.concatenate([mapy[:,:,np.newaxis], mapx[:,:,np.newaxis]], axis=2)
    print(mapx, mapy)
    print(mapx.shape, mapy.shape)
    map_src2dst = np.stack([mapx, mapy], axis=-1)
    print(map_src2dst.shape)
    print(map_src2dst)

def get_matching_grid_from_B(parameters, center_point):
    src_points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                                   [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                                   [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])

    dst_points = src_points +parameters

    theta = tps.tps_theta_from_points(src_points, dst_points, reduced=True)
    dshape = (64, 64)
    grid = tps.tps_grid(theta, dst_points, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, (64, 64))
    points = np.concatenate([mapy[:,:,np.newaxis], mapx[:,:,np.newaxis]], axis=2)
    #print("points :", points)
    center_point = np.array(center_point)
    #print("grid center : ", center_point)
    center_point = center_point[np.newaxis, np.newaxis, :]
    distance = np.sum(np.power((points - center_point), 2), axis=2)
    #print("distance : ", distance)
    ri, ci = distance.argmin()//distance.shape[1], distance.argmin()%distance.shape[1]
    return (ri, ci)


if __name__ == "__main__":
    parameters = np.zeros([9, 2], dtype=np.float32)
    map_size = (16, 16)
    synthesize_correlation(parameters, map_size)
