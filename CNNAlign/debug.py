import json
import os
import argparse
import numpy as np
import simulator as sim


def main():
    parameters = np.zeros([9, 2], dtype=np.float32)
    map_size = (16, 16)
    ij = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
                   [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
                   [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    ij *= (map_size[0] - 1, map_size[1] - 1)
    kl = sim.get_tgt_from_src(parameters, map_size, ij)
    ijkl = np.concatenate([ij.astype(np.int32), kl], axis=-1)

    correlations = sim.generate_correlations(ijkl, map_size)


if __name__ == "__main__":
    main()
