from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np

@dataclass
class TriangulationParams:
    min_depth: float = 0.1
    check_cheirality: bool = True

    def __post_init__(self) -> None:
        if self.min_depth <= 0:
            raise ValueError("min_depth must > 0")

class Triangulator:
    def __init__(self, params: Optional[TriangulationParams] = None):
        self.params = params or TriangulationParams()

    def triangulate(self, pts1: np.ndarray, pts2: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray, K: np.ndarray) -> np.ndarray:
        P1 = K @ np.hstack([R1, t1])
        P2 = K @ np.hstack([R2, t2])

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
        w = pts4d[3]
        eps = 1e-9
        safe_w = np.where(np.abs(w) < eps, eps, w)
        pts3d = (pts4d[:3]/safe_w).T

        # Cheirality check
        valid = np.ones(pts3d.shape[0], dtype=bool)
        if self.params.check_cheirality:
            cam1_pts = (R1 @ pts3d.T + t1).T
            cam2_pts = (R2 @ pts3d.T + t2).T
            depth1 = cam1_pts[:, 2]
            depth2 = cam2_pts[:, 2]
            valid = (depth1 > self.params.min_depth) & (depth2 > self.params.min_depth)

        return pts3d, valid
