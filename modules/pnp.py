from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

@dataclass
class PnPParams:
    reprojection_error: float = 4.0
    confidence: float = 0.999
    max_iters: int = 1000
    min_inliers: int = 12
    refine: bool = True
    refine_iters: int = 10
    refine_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.reprojection_error <= 0:
            raise ValueError("reprojection_error must > 0")
        if not(0.0 < self.confidence < 1.0):
            raise ValueError("confidence must be in (0, 1)")
        if self.max_iters <= 0:
            raise ValueError("max_iters must > 0")
        if self.min_inliers <= 4:
            raise ValueError("min_inliers must > 4")

class PnPSolver:
    def __init__(self, params: Optional[PnPParams] = None):
        self.params = params or PnPParams()

    def solve(self, pts3D: np.ndarray, pts2D: np.ndarray, K: np.ndarray, dist_coeffs: Optional[np.ndarray] = None):
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming no distortion

        if len(pts3D) < self.params.min_inliers:
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3D.astype(np.float64),
            pts2D.astype(np.float64),
            K,
            dist_coeffs,
            iterationsCount=self.params.max_iters,
            reprojectionError=self.params.reprojection_error,
            confidence=self.params.confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None:
            return None

        inlier_indices = inliers.ravel()
        if len(inlier_indices) < self.params.min_inliers:
            return None

        if self.params.refine:
            rvec, tvec = cv2.solvePnPRefineLM(
                pts3D[inlier_indices],
                pts2D[inlier_indices],
                K,
                dist_coeffs,
                rvec,
                tvec,
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                    self.params.refine_iters,
                    self.params.refine_eps,
                ),
            )

        R, _ = cv2.Rodrigues(rvec)

        mask = np.zeros(len(pts3D), dtype=bool)
        mask[inlier_indices] = True

        return R, tvec, mask, inlier_indices, len(inlier_indices)
 