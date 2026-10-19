"""Unit tests for modules/pnp.py, modules/pose.py, modules/triangulation.py.

Strategy: synthetic ground-truth scene (3D points + two known cameras, real K
from Data/K.txt), then verify each module against known R/t/structure and
verify return/mask semantics (who filters, who must apply the mask).
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.pose import recover_pose
from modules.pnp import PnPParams, PnPSolver
from modules.triangulation import TriangulationParams, Triangulator

K_PATH = ROOT / "Data" / "K.txt"


def load_K(path=K_PATH):
    text = path.read_text()
    body = re.search(r"K\s*=\s*\[(.*?)\]", text, re.S).group(1)
    rows = [r.strip().split() for r in body.split(";")]
    return np.array([[float(v) for v in r] for r in rows], dtype=np.float64)


@pytest.fixture(scope="module")
def K():
    return load_K()


def make_scene(n=200, seed=42, noise_px=0.2):
    """Cam1: I,0. Cam2: small rotation + baseline. Returns GT pose & pixels."""
    rng = np.random.default_rng(seed)
    K = load_K()

    P = np.column_stack([
        rng.uniform(-0.6, 0.6, n),
        rng.uniform(-0.4, 0.4, n),
        rng.uniform(3.5, 6.0, n),
    ])

    ang = np.deg2rad(3.0)
    R2 = cv2.Rodrigues(np.array([0.01, ang, 0.005]))[0]
    t2 = np.array([[0.30], [0.04], [0.02]])

    def project(Pw, R, t):
        Pc = (R @ Pw.T + t).T
        uv = (K @ Pc.T).T
        pix = uv[:, :2] / uv[:, 2:]
        return pix

    pix1 = project(P, np.eye(3), np.zeros((3, 1)))
    pix2 = project(P, R2, t2)

    ok = ((pix1 > [50, 50]) & (pix1 < [2990, 1960])).all(axis=1) & \
         ((pix2 > [50, 50]) & (pix2 < [2990, 1960])).all(axis=1)
    P, pix1, pix2 = P[ok], pix1[ok], pix2[ok]

    pix1 += rng.normal(0, noise_px, pix1.shape)
    pix2 += rng.normal(0, noise_px, pix2.shape)
    return {"P": P, "pix1": pix1.astype(np.float32), "pix2": pix2.astype(np.float32),
            "R2": R2, "t2": t2, "K": K}


def identity_matches(n):
    return [cv2.DMatch(i, i, 0) for i in range(n)]


def keypoints_from_pts(pts):
    return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts]


def essential_from_motion(R, t):
    tx = np.array([[0, -t[2, 0], t[1, 0]],
                   [t[2, 0], 0, -t[0, 0]],
                   [-t[1, 0], t[0, 0], 0]], dtype=np.float64)
    return tx @ R


def reproj_err(pts3d, R, t, K, pix):
    Pc = (R @ pts3d.T + t).T
    uv = (K @ Pc.T).T
    proj = uv[:, :2] / uv[:, 2:]
    return np.linalg.norm(proj - pix, axis=1)


# ================= pose.py =================

class TestRecoverPose:
    def test_returns_none_for_few_matches(self, scene):
        kp1 = keypoints_from_pts(scene["pix1"][:4])
        kp2 = keypoints_from_pts(scene["pix2"][:4])
        out = recover_pose(kp1, kp2, identity_matches(4),
                           essential_from_motion(scene["R2"], scene["t2"]), scene["K"])
        assert out is None

    def test_recovers_gt_rotation_and_translation_direction(self, scene):
        E = essential_from_motion(scene["R2"], scene["t2"])
        out = recover_pose(keypoints_from_pts(scene["pix1"]),
                           keypoints_from_pts(scene["pix2"]),
                           identity_matches(len(scene["pix1"])), E, scene["K"])
        assert out is not None
        R, t, mask, inlier_matches, num = out
        assert R.shape == (3, 3) and t.shape == (3, 1)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        assert np.abs(np.linalg.det(R) - 1) < 1e-6
        # translation is only known up to scale -> compare direction
        t_dir = t.ravel() / np.linalg.norm(t)
        gt_dir = scene["t2"].ravel() / np.linalg.norm(scene["t2"])
        assert np.allclose(t_dir, gt_dir, atol=0.02)
        # rotation compared up to sign convention: R maps cam1->cam2
        assert np.linalg.norm(R - scene["R2"]) < 0.05 or \
               np.linalg.norm(R + scene["R2"]) < 0.05
        assert num > 0.9 * len(scene["pix1"])

    def test_filters_via_cheirality_when_given_e_inliers(self, scene):
        """Correct usage flow (matcher -> E inliers -> recoverPose):
        inlier_matches is ALREADY filtered; mask carries the same info."""
        E = essential_from_motion(scene["R2"], scene["t2"])
        # simulate E-inliers: exclude the corrupted points BEFORE recoverPose
        rng = np.random.default_rng(0)
        pix2 = scene["pix2"].copy()
        n_out = len(pix2) // 7
        idx = rng.choice(len(pix2), n_out, replace=False)
        pix2[idx] += rng.uniform(-300, 300, size=(n_out, 2))
        good = np.setdiff1d(np.arange(len(pix2)), idx)
        out = recover_pose(keypoints_from_pts(scene["pix1"][good]),
                           keypoints_from_pts(pix2[good]),
                           identity_matches(len(good)), E, scene["K"])
        assert out is not None
        R, t, mask, inlier_matches, num = out
        assert len(inlier_matches) == int(mask.sum()) == num
        assert all(mask[m.queryIdx] for m in inlier_matches)
        assert num > 0.9 * len(good)

    def test_surviving_outliers_when_raw_matches_passed(self, scene):
        """Behavior note: recoverPose's own mask is only a cheirality check,
        NOT a robust epipolar filter. If matches are not pre-filtered with
        findEssentialMat's mask, some gross outliers survive -> filter
        matches with E-inliers (matcher step) before calling recover_pose."""
        E = essential_from_motion(scene["R2"], scene["t2"])
        rng = np.random.default_rng(0)
        pix2 = scene["pix2"].copy()
        n_out = len(pix2) // 7
        idx = rng.choice(len(pix2), n_out, replace=False)
        pix2[idx] += rng.uniform(-300, 300, size=(n_out, 2))
        out = recover_pose(keypoints_from_pts(scene["pix1"]),
                           keypoints_from_pts(pix2),
                           identity_matches(len(pix2)), E, scene["K"])
        assert out is not None
        R, t, mask, inlier_matches, num = out
        survived = sum(m.queryIdx in idx for m in inlier_matches)
        assert survived < n_out  # removes SOME, but not necessarily all

    def test_min_inliers_gate(self, scene):
        E = essential_from_motion(scene["R2"], scene["t2"])
        out = recover_pose(keypoints_from_pts(scene["pix1"]),
                           keypoints_from_pts(scene["pix2"]),
                           identity_matches(len(scene["pix1"])), E, scene["K"],
                           min_inliers=10**6)
        assert out is None


# ================= pnp.py =================

class TestPnPParams:
    @pytest.mark.parametrize("field,value", [
        ("reprojection_error", 0.0),
        ("confidence", 0.0),
        ("confidence", 1.0),
        ("max_iters", 0),
        ("min_inliers", 4),
        ("min_inliers", 0),
    ])
    def test_invalid_values_raise(self, field, value):
        with pytest.raises(ValueError):
            PnPParams(**{field: value})

    def test_defaults_valid(self):
        p = PnPParams()
        assert p.reprojection_error == 4.0
        assert p.refine is True


class TestPnPSolver:
    def test_too_few_points_returns_none(self, scene):
        solver = PnPSolver()
        out = solver.solve(scene["P"][:10], scene["pix1"][:10], scene["K"])
        assert out is None

    def test_inliers_in_none_is_typo_repro(self, scene):
        """pnp.py:49 `inliers in None` raises TypeError on the SUCCESS path.
        (intended: `inliers is None`). Reproduction, not desired behavior."""
        solver = PnPSolver()
        with pytest.raises(TypeError, match="NoneType"):
            solver.solve(scene["P"], scene["pix1"], scene["K"])

    @pytest.mark.xfail(strict=True, raises=TypeError,
                       reason="BUG pnp.py:49: `inliers in None` -> TypeError; "
                              "solve() can never return a pose as-is")
    def test_solve_recovers_gt_pose(self, scene):
        """Expected behavior once the typo is fixed: recover cam1 pose (I, 0)."""
        solver = PnPSolver(PnPParams(reprojection_error=2.0))
        out = solver.solve(scene["P"], scene["pix1"], scene["K"])
        R, t, mask, inlier_idx, num = out
        assert np.allclose(R, np.eye(3), atol=1e-3)
        assert np.linalg.norm(t) < 0.02
        assert num > 0.9 * len(scene["P"])

    @pytest.mark.xfail(strict=True, raises=TypeError,
                       reason="BUG pnp.py:49 typo blocks all success returns")
    def test_solve_recovers_cam2_pose(self, scene):
        solver = PnPSolver(PnPParams(reprojection_error=2.0))
        out = solver.solve(scene["P"], scene["pix2"], scene["K"])
        R, t, mask, inlier_idx, num = out
        assert np.linalg.norm(R - scene["R2"]) < 0.02
        t_dir = t.ravel() / np.linalg.norm(t)
        gt_dir = scene["t2"].ravel() / np.linalg.norm(scene["t2"])
        assert np.allclose(t_dir, gt_dir, atol=0.02)

    def test_mask_semantics_full_length(self, scene):
        """Contract check on the failure gate only: mask is NOT applied
        internally -> caller must filter pts3D/pts2D with it (see report)."""
        import inspect
        src = Path(ROOT / "modules" / "pnp.py").read_text()
        assert "pts3D[mask]" not in src and "return R, tvec, mask" in src


# ================= triangulation.py =================

class TestTriangulationParams:
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_invalid_min_depth_raises(self, value):
        with pytest.raises(ValueError):
            TriangulationParams(min_depth=value)


class TestTriangulator:
    def test_gt_poses_recover_structure(self, scene):
        tri = Triangulator()
        R1, t1 = np.eye(3), np.zeros((3, 1))
        pts3d, valid = tri.triangulate(scene["pix1"], scene["pix2"],
                                       R1, t1, scene["R2"], scene["t2"], scene["K"])
        assert pts3d.shape == scene["P"].shape
        assert valid.all()
        err = np.linalg.norm(pts3d - scene["P"], axis=1)
        assert np.median(err) < 0.05, f"median 3D error {np.median(err)}"

    def test_reprojection_error_small(self, scene):
        tri = Triangulator()
        R1, t1 = np.eye(3), np.zeros((3, 1))
        pts3d, valid = tri.triangulate(scene["pix1"], scene["pix2"],
                                       R1, t1, scene["R2"], scene["t2"], scene["K"])
        e1 = reproj_err(pts3d[valid], R1, t1, scene["K"], scene["pix1"][valid])
        e2 = reproj_err(pts3d[valid], scene["R2"], scene["t2"], scene["K"], scene["pix2"][valid])
        assert np.median(e1) < 1.0 and np.median(e2) < 1.0

    def test_points_behind_camera_flagged_invalid(self, scene):
        """Feed correspondences produced by a point BEHIND cam1 (negative
        depth): valid mask must be False; function does NOT filter rows."""
        K = scene["K"]
        rng = np.random.default_rng(3)
        Pb = np.column_stack([rng.uniform(-0.5, 0.5, 20),
                              rng.uniform(-0.5, 0.5, 20),
                              rng.uniform(-6.0, -4.0, 20)])
        def project(Pw, R, t):
            Pc = (R @ Pw.T + t).T
            uv = (K @ Pc.T).T
            return (uv[:, :2] / uv[:, 2:]).astype(np.float32)
        p1 = project(Pb, np.eye(3), np.zeros((3, 1)))
        p2 = project(Pb, scene["R2"], scene["t2"])
        pts3d, valid = Triangulator().triangulate(p1, p2, np.eye(3),
                                                  np.zeros((3, 1)),
                                                  scene["R2"], scene["t2"], K)
        assert pts3d.shape[0] == 20          # all rows returned
        assert not valid.any()               # all flagged invalid
        # caller-side filtering example
        assert pts3d[valid].shape[0] == 0

    def test_min_depth_threshold(self, scene):
        tri = Triangulator(TriangulationParams(min_depth=5.5))
        # scene depths w.r.t. cam1 are 3.5..6.0 -> points closer than 5.5
        # must be flagged invalid
        pts3d, valid = tri.triangulate(scene["pix1"], scene["pix2"],
                                       np.eye(3), np.zeros((3, 1)),
                                       scene["R2"], scene["t2"], scene["K"])
        expected = scene["P"][:, 2] > 5.5
        assert valid.sum() < len(valid)
        assert np.allclose(valid, expected, atol=0) or \
               (valid ^ expected).mean() < 0.1  # allow minor boundary flips

    def test_flat_translation_raises(self, scene):
        """t must be (3,1); flat (3,) breaks hstack -> API contract note."""
        with pytest.raises(ValueError):
            Triangulator().triangulate(scene["pix1"], scene["pix2"],
                                       np.eye(3), np.zeros(3),
                                       scene["R2"], scene["t2"].ravel(),
                                       scene["K"])


# ============ integration: E -> recover_pose -> triangulate ============

class TestRealData:
    """End-to-end smoke test on actual Data/ images (bf matcher path)."""

    def test_pair_recover_pose_and_triangulate(self, K):
        from modules.features import FeatureExtractor
        from modules.matcher import FeatureMatcher, MatcherParams

        imgs = sorted((ROOT / "Data").glob("*.jpg"))[:2]
        ext = FeatureExtractor()
        kp1, desc1 = ext.extract(cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE))
        kp2, desc2 = ext.extract(cv2.imread(str(imgs[1]), cv2.IMREAD_GRAYSCALE))

        m = FeatureMatcher(MatcherParams(method="bf",
                                         geometric_verification="essential"))
        inliers, E = m.match_image_pair(kp1, kp2, desc1, desc2, K)
        assert E is not None and len(inliers) >= 50

        out = recover_pose(kp1, kp2, inliers, E, K)
        assert out is not None, "recover_pose returned None on real pair"
        R, t, mask, inlier_matches, num = out
        assert num >= 50

        p1 = np.float32([kp1[mt.queryIdx].pt for mt in inlier_matches])
        p2 = np.float32([kp2[mt.trainIdx].pt for mt in inlier_matches])
        pts3d, valid = Triangulator().triangulate(p1, p2, np.eye(3),
                                                  np.zeros((3, 1)), R, t, K)
        # real scenes have some bad/far points; require healthy majority
        assert valid.mean() > 0.5, f"only {valid.mean():.1%} valid depths"
        e2 = reproj_err(pts3d[valid], R, t, K, p2[valid])
        assert np.median(e2) < 2.0, f"cam2 median reproj {np.median(e2)}"


class TestIntegration:
    def test_full_pair_pipeline_synthetic_gt(self, scene):
        K = scene["K"]
        E = essential_from_motion(scene["R2"], scene["t2"])
        kp1 = keypoints_from_pts(scene["pix1"])
        kp2 = keypoints_from_pts(scene["pix2"])
        out = recover_pose(kp1, kp2, identity_matches(len(kp1)), E, K)
        assert out is not None
        R, t, mask, inlier_matches, num = out

        p1 = np.float32([kp1[m.queryIdx].pt for m in inlier_matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in inlier_matches])
        pts3d, valid = Triangulator().triangulate(p1, p2, np.eye(3),
                                                  np.zeros((3, 1)), R, t, K)
        assert valid.mean() > 0.95
        # recovered R maps cam1 frame -> cam2 frame with unit-norm t
        e1 = reproj_err(pts3d[valid], np.eye(3), np.zeros((3, 1)), K, p1[valid])
        e2 = reproj_err(pts3d[valid], R, t, K, p2[valid])
        assert np.median(e1) < 1.0, f"cam1 reproj {np.median(e1)}"
        assert np.median(e2) < 1.0, f"cam2 reproj {np.median(e2)}"
        # structure direction vs GT (scale differs because |t|=1)
        idx = [m.queryIdx for m in inlier_matches]
        Pgt = scene["P"][idx][valid]
        scale = np.median(pts3d[valid][:, 2] / Pgt[:, 2])
        err = np.linalg.norm(pts3d[valid] - Pgt * scale, axis=1)
        assert np.median(err) < 0.05 * scale

    def test_pnp_then_triangulate_would_be_unfiltered(self):
        """Contract note for the report: neither pnp.solve nor
        triangulate.triangulate filter rows internally."""
        import inspect
        solver_src = (ROOT / "modules" / "pnp.py").read_text()
        tri_src = (ROOT / "modules" / "triangulation.py").read_text()
        assert "pts3d[valid]" not in tri_src
        assert "mask[inlier_indices]" in solver_src  # mask built, not applied


@pytest.fixture(scope="module")
def scene():
    return make_scene()
