import cv2
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class SIFTParams:
    nfeatures: int = 0
    nOctaveLayers: int = 3
    contrastThreshold: float = 0.04
    edgeThreshold: float = 10.0
    sigma: float = 1.6

    enable_precise_upscale: Optional[bool] = None
    descriptorType: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.nfeatures < 0:
            raise ValueError("nfeatures must >= 0")
        if self.nOctaveLayers < 1:
            raise ValueError("nOctaveLayers must >= 1")
        if self.contrastThreshold <= 0:
            raise ValueError("contrastThreshold must > 0")
        if self.edgeThreshold <= 0:
            raise ValueError("edgeThreshold must > 0")
        if self.sigma <= 0:
            raise ValueError("sigma must > 0")

    def to_kwargs(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "SIFTParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        

class FeatureExtractor:
    def __init__(self, method='sift', params: Optional[SIFTParams] = None):
        self.method = method.lower()
        if self.method == 'sift':
            self.params = params or SIFTParams()
            self.extractor = cv2.SIFT_create(**self.params.to_kwargs())
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.method}")

    def extract(self, image):
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        keypoints, descriptors = self.extractor.detectAndCompute(gray_image, None)
        return keypoints, descriptors
        