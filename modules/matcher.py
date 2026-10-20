from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

@dataclass
class MatcherParams:
    method: str = 'flann'
    norm_type: int = cv2.NORM_L2
    use_ratio_test: bool = True
    ratio_test: float = 0.75
    cross_check: bool = False
    flann_trees: int = 5
    flann_checks: int = 50

    geometric_verification: str = "essential" # "none"|"homography"|"fundamental"|"essential"
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 1000

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.method not in ['bf', 'flann']:
            raise ValueError("method must be 'bf' or 'flann'")
        if self.ratio_test <= 0 or self.ratio_test >= 1:
            raise ValueError("ratio_test must be in (0, 1)")
        if self.geometric_verification not in ["none", "homography", "fundamental", "essential"]:
            raise ValueError("geometric_verification must be one of 'none', 'homography', 'fundamental', 'essential'")
        if self.ransac_threshold <= 0:
            raise ValueError("ransac_threshold must > 0")
        if self.ransac_confidence <= 0 or self.ransac_confidence >= 1:
            raise ValueError("ransac_confidence must be in (0, 1)")
        if self.ransac_max_iters <= 0:
            raise ValueError("ransac_max_iters must > 0")
        if not self.use_ratio_test and not self.cross_check:
            raise ValueError("At least one of use_ratio_test or cross_check must be True")
        if self.use_ratio_test and not (0.0 < self.ratio_test < 1.0):
            raise ValueError("ratio_test must be in (0, 1) when use_ratio_test=True")

    def to_kwargs(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "MatcherParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

class FeatureMatcher:
    def __init__(self, params: Optional[MatcherParams] = None):
        self.params = params or MatcherParams()
        self.matcher = self._create_matcher()

    def _create_matcher(self):
        if self.params.method == 'bf':
            return cv2.BFMatcher(self.params.norm_type, crossCheck=False)
        index_params = dict(
            algorithm=1,
            trees=self.params.flann_trees,
        )
        search_params = dict(checks=self.params.flann_checks)
        return cv2.FlannBasedMatcher(index_params, search_params)

    def _ratio_filter(self, knn_pairs):
        good_matches = []
        for pair in knn_pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.params.ratio_test * n.distance:
                good_matches.append(m)

        return good_matches

    def _top1(self, knn_pairs):
        return [pair[0] for pair in knn_pairs if len(pair) >= 1]

    def _filter_knn(self, knn_pairs):
        if self.params.use_ratio_test:
            return self._ratio_filter(knn_pairs)
        else:
            return self._top1(knn_pairs)

    def _ratio_match(self, desc1, desc2):
        knn = self.matcher.knnMatch(desc1, desc2, k=2)
        return self._filter_knn(knn)

    def _mutual_match(self, desc1, desc2):
        fwd = self.matcher.knnMatch(desc1, desc2, k=2)
        bwd = self.matcher.knnMatch(desc2, desc1, k=2)

        fwd_m = self._filter_knn(fwd)
        bwd_m = self._filter_knn(bwd)
        bwd_map = {m.queryIdx: m.trainIdx for m in bwd_m}
        return [m for m in fwd_m if bwd_map.get(m.trainIdx) == m.queryIdx]

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        if self.params.cross_check:
            return self._mutual_match(desc1, desc2)
        return self._ratio_match(desc1, desc2)

    def _geometric_verification(self, kp1, kp2, matches, K=None):
        method = self.params.geometric_verification

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        if method == "none" or not matches:
            return matches, None
        elif method == "homography":
            model, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.params.ransac_threshold, maxIters=self.params.ransac_max_iters, confidence=self.params.ransac_confidence)
        elif method == "fundamental":
            model, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, self.params.ransac_threshold, self.params.ransac_confidence, self.params.ransac_max_iters)
        elif method == "essential":
            if K is None:
                raise ValueError("Camera intrinsic matrix K must be provided for essential matrix estimation.")
            model, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=self.params.ransac_confidence,
                threshold=self.params.ransac_threshold,
                maxIters=self.params.ransac_max_iters,
            )
        else:
            raise ValueError(f"Unsupported geometric verification method: {method}")

        if mask is None or model is None:
            return [], None
        mask = mask.ravel().astype(bool)
        inliers = [m for m, inlier in zip(matches, mask) if inlier]
        return inliers, model

    def match_image_pair(self, kp1, kp2, desc1, desc2, K=None):
        matches = self.match(desc1, desc2)
        inliers, model = self._geometric_verification(kp1, kp2, matches, K)
        return inliers, model