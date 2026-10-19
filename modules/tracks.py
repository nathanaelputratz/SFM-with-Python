from typing import Dict, List, Tuple, Sequence
import cv2


class TrackManager:
    def __init__(self):
        self._parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._rank: Dict[Tuple[int, int], int] = {}
        self._members: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    def find(self, key: Tuple[int, int]) -> Tuple[int, int]:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]

        # Path compression
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]

        return root

    def union(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra

        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra

        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

        self._members[ra].extend(self._members[rb])
        del self._members[rb]

        return ra

    def add_image(self, image_id: int, n_features: int) -> None:
        for i in range(n_features):
            key = (image_id, i)
            self._parent[key] = key
            self._rank[key] = 0
            self._members[key] = [key]

    def add_matches(self, a: int, b: int, matches: Sequence[cv2.DMatch]) -> None:
        for m in matches:
            self.union((a, m.queryIdx), (b, m.trainIdx))

    def members(self, root: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self._members[root]

    def all_roots(self) -> List[Tuple[int, int]]:
        return list(self._members.keys())

    def root_of(self, img_id: int, feat_idx: int) -> Tuple[int, int]:
        return self.find((img_id, feat_idx))

    def observations_in(self, img_id: int) -> List[Tuple[Tuple[int, int], int]]:
        out = []
        for root, members in self._members.items():
            for img, feat in members:
                if img == img_id:
                    out.append((root, feat))
                    break
        return out

    def track_length(self, root: Tuple[int, int]) -> int:
        return len(self._members[root])

    def track_observations(self, root: Tuple[int, int]) -> List[Tuple[int, int]]:
        return self._members[root]

    def images_in_track(self, root: Tuple[int, int]) -> set:
        return {img for img, _ in self._members[root]}

    def has_observation(self, key: Tuple[int, int]) -> bool:
        return key in self._parent

    def has_conflict(self, root: Tuple[int, int]) -> bool:
        seen = set()
        for img, _ in self._members[root]:
            if img in seen:
                return True
            seen.add(img)
        return False