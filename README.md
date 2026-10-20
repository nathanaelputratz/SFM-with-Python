# SFM with Python

Incremental Structure-from-Motion assembled from OpenCV geometry and GTSAM optimization: feature extraction, matching, relative-pose recovery, triangulation, PnP registration, and bundle adjustment.

## Pipeline

`modules/` holds one component per stage:

| Module | Responsibility |
|---|---|
| `features.py` | SIFT keypoint/descriptor extraction |
| `matcher.py` | Descriptor matching with ratio test and geometric verification |
| `pose.py` | Essential matrix decomposition and relative-pose cheirality check |
| `triangulation.py` | Sparse point triangulation from matched rays |
| `tracks.py` | Union-find track merging across image pairs |
| `points.py` | 3D point store keyed by track, PLY export |
| `utils.py` | NumPy to GTSAM `Pose3`/`Cal3Bundler` converters |
| `pnp.py` | Camera re-registration against the existing point cloud |
| `ba.py` | GTSAM Levenberg-Marquardt bundle adjustment |
| `sfm.py` | Incremental SfM controller tying the stages together |

`main.py` runs the incremental pipeline over the image sequence in `Data/` and writes the reconstructed cloud to `sparse.ply`.

## Usage

```bash
source Env/bin/activate
python main.py
```

The camera intrinsic matrix `K` is hardcoded in `main.py`; `Data/K.txt` holds the same values for reference.

## Dependencies

- OpenCV
- NumPy
- GTSAM

## Output

- `sparse.ply` — point cloud with per-point colors sampled from the source images
