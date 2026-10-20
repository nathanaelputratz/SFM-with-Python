from typing import Dict, Iterator, List, Optional, Tuple
import cv2
import numpy as np

from modules.tracks import TrackManager


class Points3D:
    def __init__(self, tm: 'TrackManager'):
        self.tm = tm
        self._data: Dict[Tuple[int, int], np.ndarray] = {}

    def set(self, root: Tuple[int, int], pt: np.ndarray) -> None:
        current_root = self.tm.find(root)
        obs = next(iter(self.tm.members(current_root)))
        self._data[obs] = np.asarray(pt, dtype=np.float64)

    def get(self, root: Tuple[int, int]) -> Optional[np.ndarray]:
        current_root = self.tm.find(root)
        for obs in self.tm.members(current_root):
            if obs in self._data:
                return self._data[obs]
        return None

    def contains_root(self, root: Tuple[int, int]) -> bool:
        return self.get(root) is not None

    def items(self):
        seen = set()
        for obs, pt in self._data.items():
            root = self.tm.root_of(*obs)
            if root in seen:
                continue
            seen.add(root)
            yield root, pt
    
    def __len__(self) -> int:
        return sum(1 for _ in self.items())

    def to_ply(
        self,
        path: str,
        images: List[np.ndarray],
        keypoints: List[List[cv2.KeyPoint]],
    ) -> int:
        pts_items = list(self.items())
        n = len(pts_items)

        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for root, pt in pts_items:
                # Ambil warna dari observasi pertama yang tersedia
                r, g, b = 200, 200, 200   # default abu-abu
                for img_id, feat_idx in self.tm.members(root):
                    if img_id >= len(images):
                        continue
                    kp = keypoints[img_id][feat_idx]
                    u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
                    img = images[img_id]
                    h, w = img.shape[:2]
                    if 0 <= u < w and 0 <= v < h:
                        bgr = img[v, u]
                        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
                        break

                x, y, z = pt[0], pt[1], pt[2]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

        return n