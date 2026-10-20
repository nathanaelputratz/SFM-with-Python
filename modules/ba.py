from typing import Dict, Tuple, Optional
import gtsam
import numpy as np

from modules.tracks import TrackManager
from modules.points import Points3D
from modules.utils import to_cal3, to_pose3


def bundle_adjustment(
    world_poses, points3d, tm, keypoints, K,
    use_schur: bool = True,
):
    cal3 = to_cal3(K)

    observed_imgs = set()
    for root, _ in points3d.items():
        if tm.has_conflict(root):
            continue
        members = tm.members(root)
        if len(members) < 2:
            continue
        for img_id, _ in members:
            if img_id in world_poses:
                observed_imgs.add(img_id)

    img_ids = sorted(i for i in world_poses if i in observed_imgs)
    if len(img_ids) < 2:
        print("  [BA] skipped: <2 cameras with observations")
        return dict(world_poses), {r: p for r, p in points3d.items()}

    id_to_idx = {img_id: idx for idx, img_id in enumerate(img_ids)}

    sfm_data = gtsam.SfmData()
    for img_id in img_ids:
        R, t = world_poses[img_id]
        sfm_data.addCamera(gtsam.PinholeCameraCal3Bundler(to_pose3(R, t), cal3))

    root_to_track_idx = {}
    for root, pt3d in points3d.items():
        if tm.has_conflict(root):
            continue
        members = tm.members(root)
        if len(members) < 2:
            continue

        track = gtsam.SfmTrack(pt3d)
        for img_id, feat_idx in members:
            if img_id not in id_to_idx:
                continue
            pt2d = np.array(
                keypoints[img_id][feat_idx].pt, dtype=np.float64
            ).reshape(2, 1)
            track.addMeasurement(id_to_idx[img_id], pt2d)

        if track.numberMeasurements() < 2:
            continue
        sfm_data.addTrack(track)
        root_to_track_idx[root] = sfm_data.numberTracks() - 1

    if sfm_data.numberTracks() == 0:
        print("  [BA] skipped: no valid tracks")
        return dict(world_poses), {r: p for r, p in points3d.items()}

    projection_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    graph = sfm_data.sfmFactorGraph(projection_noise)
    initial = gtsam.initialCamerasAndPointsEstimate(sfm_data)

    params = gtsam.SfmLevenbergMarquardtParams.ceresDefaults()
    if use_schur:
        params.setEliminationMode(gtsam.SfmEliminationMode.Schur)
        params.setLinearSolver(
            gtsam.NonlinearOptimizerParams.LinearSolverType.MULTIFRONTAL_SOLVER
        )
        reduced_ordering = (
            gtsam.SfmLevenbergMarquardtOptimizer.CreateReducedOrdering(graph, initial)
        )
        params.setOrdering(reduced_ordering)

    result = gtsam.SfmLevenbergMarquardtOptimizer(graph, initial, params).optimize()

    new_poses = dict(world_poses)
    for img_id in img_ids:
        cam = result.atPinholeCameraCal3Bundler(id_to_idx[img_id])  # ← INTEGER
        new_poses[img_id] = (
            cam.pose().rotation().matrix(),
            cam.pose().translation(),
        )

    new_points3d = {}
    for root, track_idx in root_to_track_idx.items():
        key = gtsam.Symbol('p', track_idx).key()
        new_points3d[root] = result.atPoint3(key)

    return new_poses, new_points3d