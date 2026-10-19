from typing import Optional, List
import cv2
import numpy as np

def recover_pose(
    kp1,
    kp2,
    matches: List[cv2.DMatch],
    E: np.ndarray,
    K: np.ndarray,
    min_inliers: int = 8,
):
    if len(matches) < 5:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    num_inliers, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    if num_inliers < min_inliers:
        return None

    mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, inlier in zip(matches, mask) if inlier]

    return R, t, mask, inlier_matches, num_inliers