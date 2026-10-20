import gtsam
import numpy as np

def to_pose3(R: np.ndarray, t: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(R), t.reshape(3))

def to_cal3(K: np.ndarray) -> gtsam.Cal3Bundler:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return gtsam.Cal3Bundler(fx, 0.0, 0.0, cx, cy)