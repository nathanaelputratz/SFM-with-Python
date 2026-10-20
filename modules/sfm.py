from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np

from modules.ba import bundle_adjustment
from modules.features import FeatureExtractor
from modules.matcher import FeatureMatcher
from modules.pose import recover_pose
from modules.pnp import PnPSolver
from modules.triangulation import Triangulator
from modules.tracks import TrackManager
from modules.points import Points3D
from tqdm.auto import tqdm

@dataclass
class SfMConfig:
    initial_pair: Tuple[int, int] = (0, 1)
    min_pair_inliers: int = 30
    min_init_inliers: int = 100
    min_pnp_inliers: int = 15
    min_essential_inliers: int = 5

@dataclass
class SfMResult:
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]]
    points3d: 'Points3D'
    registered: Set[int]
    failed: Set[int]
    tracks: TrackManager
    keypoints: List[List[cv2.KeyPoint]]
    descriptors: List[Optional[np.ndarray]]

class IncrementalSfM:
    def __init__(self, K: np.ndarray, extractor: FeatureExtractor, matcher: FeatureMatcher, pnp: PnPSolver, triangulator: Triangulator, config: SfMConfig):
        self.K = K
        self.extractor = extractor
        self.matcher = matcher
        self.triangulator = triangulator
        self.pnp = pnp
        self.config = config or SfMConfig()

        self.tm = TrackManager()
        self.points3d = Points3D(self.tm)

        self.keypoints: List[List[cv2.KeyPoint]] = []
        self.descriptors: List[Optional[np.ndarray]] = []
        self.verified_pairs: Dict[Tuple[int, int], List[cv2.DMatch]] = {}
        self.world_poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.registered: Set[int] = set()
        self.failed: Set[int] = set()

    def run(self, images: List[np.ndarray]) -> SfMResult:
        print("[SfM] Starting incremental SfM...")
        self._phase0(images)
        self._phase1()
        self._phase2()
        self._phase3()
        return self._build_result()

    def _phase0(self, images: List[np.ndarray]) -> None:
        print("[Phase 0] Extracting features...")
        for i, img in enumerate(images):
            kp, desc = self.extractor.extract(img)
            self.keypoints.append(kp)
            self.descriptors.append(desc)
            self.tm.add_image(i, len(kp))
            print(f"Image {i}: {len(kp)} keypoints extracted.")

        print("[Phase 0] Matching features between image pairs...")
        for i, j in tqdm(combinations(range(len(images)), 2), desc="Matching pairs", total=len(images) * (len(images) - 1) // 2):
            if self.descriptors[i] is None or self.descriptors[j] is None:
                continue

            inliers, model = self.matcher.match_image_pair(
                self.keypoints[i], self.keypoints[j], self.descriptors[i], self.descriptors[j],
                self.K
            )
            if model is None or len(inliers) < self.config.min_pair_inliers:
                continue
            self.verified_pairs[(i, j)] = inliers
            self.tm.add_matches(i, j, inliers)

        print(f"[Phase 0] Verified {len(self.verified_pairs)} image pairs with sufficient inliers.")

    def _phase1(self) -> None:
        print("[Phase 1] Initializing with the initial image pair...")
        i, j = self.config.initial_pair

        if (i, j) not in self.verified_pairs:
            raise ValueError(f"Initial pair {i, j} does not have enough verified matches.")

        inliers, E = self.matcher.match_image_pair(
            self.keypoints[i], self.keypoints[j],
            self.descriptors[i], self.descriptors[j],
            self.K
        )
        if E is None or len(inliers) < self.config.min_init_inliers:
            raise ValueError(f"Initial pair {i, j} does not have enough inliers for essential matrix estimation.")

        pose = recover_pose(self.keypoints[i], self.keypoints[j], inliers, E, self.K)
        if pose is None:
            raise ValueError(f"Failed to recover pose for initial pair {i, j}.")

        R, t, mask, inlier_matches, num_inliers = pose
        pts1 = np.float32([self.keypoints[i][m.queryIdx].pt for m in inlier_matches])
        pts2 = np.float32([self.keypoints[j][m.trainIdx].pt for m in inlier_matches])

        pts3d, valid = self.triangulator.triangulate(pts1, pts2, np.eye(3), np.zeros((3, 1)), R, t, self.K)
        n_valid = int(valid.sum())

        if n_valid < self.config.min_init_inliers:
            raise ValueError(f"Not enough valid 3D points after triangulation for initial pair {i, j}.")

        self.world_poses[i] = (np.eye(3), np.zeros((3, 1)))
        self.world_poses[j] = (R, t)
        self.registered.update([i, j])

        for idx in np.where(valid)[0]:
            m = inlier_matches[idx]
            root = self.tm.root_of(i, m.queryIdx)
            self.points3d.set(root, pts3d[idx])

        print(f"[Phase 1] Done initializing with images {i} and {j}. Registered {len(self.points3d)} 3D points.")

    def _phase2(self) -> None:
        print("[Phase 2] Incrementally registering images...")
        n_total = len(self.keypoints)

        for new_image in range(n_total):
            if new_image in self.registered:
                continue

            if self._try_register(new_image):
                print(f"[Phase 2] Successfully registered image {new_image}.")
            else:
                self.failed.add(new_image)
                print(f"[Phase 2] Failed registering image {new_image}.")

    def _phase3(self) -> None:
        print("[Phase 3] Finalizing reconstruction...")
        try:
            new_poses, new_points3d = bundle_adjustment(
                world_poses=self.world_poses,
                points3d=self.points3d,
                tm=self.tm,
                keypoints=self.keypoints,
                K=self.K,
                use_schur=True
            )

            self.world_poses = new_poses
            self.points3d = Points3D(self.tm)
            for root, pt in new_points3d.items():
                self.points3d.set(root, pt)
            print(f"[Phase 3] Bundle adjustment completed. Optimized {len(self.world_poses)} camera poses and {len(self.points3d)} 3D points.")

        except Exception as e:
            print(f"[Phase 3] Error during bundle adjustment: {e}")

        print("[Phase 3] Reconstruction finalized.")

    def _try_register(self, new_image: int) -> bool:
        pts3d, pts2d = self._build_2d3d(new_image)
        if len(pts3d) < self.config.min_pnp_inliers:
            return False

        pnp_res = self.pnp.solve(
            np.array(pts3d, dtype=np.float64),
            np.array(pts2d, dtype=np.float64),
            self.K
        )
        if pnp_res is None:
            print(f"[Phase 2] PnP failed for image {new_image}.")
            return False

        R, t, mask, inlier_indices, n_inliers = pnp_res
        if n_inliers < self.config.min_pnp_inliers:
            return False

        self._continue_tracks(new_image)
        self.world_poses[new_image] = (R, t)
        self.registered.add(new_image)
        n_new = self._triangulate_pending(new_image)

        print(f"[Phase 2] Triangulated {n_new} new 3D points for image {new_image}.")
        return True

    def _build_2d3d(self, new_image: int):
        pts3d, pts2d = [], []
        seen = set()

        for root, feat_idx in self.tm.observations_in(new_image):
            if root in seen:
                continue

            pt = self.points3d.get(root)
            if pt is None:
                continue

            seen.add(root)
            pts3d.append(pt)
            pts2d.append(self.keypoints[new_image][feat_idx].pt)

        return pts3d, pts2d

    def _continue_tracks(self, new_image: int) -> None:
        for other in self.registered:
            if other == new_image:
                continue
            matches = self._get_matches(new_image, other)
            if matches:
                self.tm.add_matches(new_image, other, matches)

    def _triangulate_pending(self, new_image: int) -> int:
        candidates = set()
        for root, _ in self.tm.observations_in(new_image):
            if not self.points3d.contains_root(root):
                candidates.add(root)

        n_new = 0
        for root in candidates:
            if self.tm.has_conflict(root):
                continue

            obs = [
                (img, feat) for img, feat in self.tm.members(root)
                if img in self.world_poses
            ]

            if len(obs) < 2:
                continue

            pair = self._select_best_pair(obs)
            if pair is None:
                continue

            _, (i1, f1), (i2, f2) = pair
            R1, t1 = self.world_poses[i1]
            R2, t2 = self.world_poses[i2]
            pt1 = np.array([self.keypoints[i1][f1].pt], dtype=np.float64)
            pt2 = np.array([self.keypoints[i2][f2].pt], dtype=np.float64)
            pts3d, valid = self.triangulator.triangulate(
                pt1, pt2, R1, t1, R2, t2, self.K,
            )
            if valid[0]:
                self.points3d.set(root, pts3d[0])
                n_new += 1
        return n_new

    def _select_best_pair(self, obs):
        if len(obs) < 2:
            return None

        centers = {}
        for img, _ in obs:
            R, t = self.world_poses[img]
            centers[img] = (-R.T @ t.ravel())

        best = None
        for a in range(len(obs)):
            for b in range(a + 1, len(obs)):
                ia, fa = obs[a]
                ib, fb = obs[b]
                if ia == ib:
                    continue
                dist = float(np.linalg.norm(centers[ia] - centers[ib]))
                if best is None or dist > best[0]:
                    best = (dist, (ia, fa), (ib, fb))
        return best

    def _get_matches(self, a: int, b: int) -> List[cv2.DMatch]:
        if (a, b) in self.verified_pairs:
            return self.verified_pairs[(a, b)]
        if (b, a) in self.verified_pairs:
            return [
                cv2.DMatch(
                    _queryIdx=m.trainIdx,
                    _trainIdx=m.queryIdx,
                    _distance=m.distance,
                )
                for m in self.verified_pairs[(b, a)]
            ]
        return []
    
    def _build_result(self) -> SfMResult:
        return SfMResult(
            poses=dict(self.world_poses),
            points3d=self.points3d,
            registered=set(self.registered),
            failed=set(self.failed),
            tracks=self.tm,
            keypoints=self.keypoints,
            descriptors=self.descriptors,
        ) 