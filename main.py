from pathlib import Path
import cv2
import numpy as np

from modules.features import FeatureExtractor, SIFTParams
from modules.matcher import FeatureMatcher, MatcherParams
from modules.pnp import PnPSolver, PnPParams
from modules.triangulation import Triangulator, TriangulationParams
from modules.sfm import IncrementalSfM, SfMConfig

K = np.array([
    [2759.48, 0.0, 1520.69],
    [0.0, 2764.16, 1006.81],
    [0.0, 0.0, 1.0],
])

extractor = FeatureExtractor(
    "sift",
    SIFTParams(
        nfeatures=0,
        contrastThreshold=0.02,
        sigma=1.3
    )
)
matcher = FeatureMatcher(
    MatcherParams(
        method="flann",
        ratio_test=0.75,
        geometric_verification="essential",
        ransac_threshold=1.5,
    )
)
pnp = PnPSolver(PnPParams(reprojection_error=4.0, min_inliers=15))
triangulator = Triangulator(TriangulationParams(min_depth=0.1))

sfm = IncrementalSfM(
    K=K,
    extractor=extractor,
    matcher=matcher,
    pnp=pnp,
    triangulator=triangulator,
    config=SfMConfig(
        initial_pair=(0, 1),
        min_init_inliers=80,
        min_pnp_inliers=15,
    ),
)

paths = sorted(Path("Data").glob("*.jpg"))
images = [cv2.imread(str(p)) for p in paths]
result = sfm.run(images)

print(f"\Final Results:")
print(f"  Registered: {sorted(result.registered)}")
print(f"  Failed:     {sorted(result.failed)}")
print(f"  3D points:  {len(result.points3d)}")

result.points3d.to_ply("sparse.ply", images, sfm.keypoints)