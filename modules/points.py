from typing import Dict, Iterator, Optional, Tuple
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
    