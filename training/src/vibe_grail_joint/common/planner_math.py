import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def interpolate_poses(poses, times):
    grid = np.arange(len(poses), dtype=float)
    out = np.stack([np.interp(times, grid, poses[:, i]) for i in range(36)], axis=1)
    out[:, 3:7] = Slerp(grid, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))(times).as_quat()[:, [3, 0, 1, 2]]
    return out.astype(np.float32)
