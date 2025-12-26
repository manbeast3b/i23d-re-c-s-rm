from dataclasses import dataclass
from typing import Optional, TypeAlias, List
from PIL import Image

from schemas.overridable import OverridableModel


class TrellisParams(OverridableModel):
    """Trellis parameters with automatic fallback to settings."""
    sparse_structure_steps: int
    sparse_structure_cfg_strength: float
    slat_steps: int
    slat_cfg_strength: float
    num_oversamples: int = 1
    # Multi-view parameters
    sparse_structure_cfg_interval: Optional[tuple[float, float]] = None
    sparse_structure_rescale_t: Optional[float] = None
    slat_cfg_interval: Optional[tuple[float, float]] = None
    slat_rescale_t: Optional[float] = None
    mode: Optional[str] = None
    use_multi_view: bool = False
    
    @classmethod
    def from_settings(cls, settings) -> "TrellisParams":
        return cls(
            sparse_structure_steps = settings.trellis_sparse_structure_steps,
            sparse_structure_cfg_strength = settings.trellis_sparse_structure_cfg_strength,
            slat_steps = settings.trellis_slat_steps,
            slat_cfg_strength = settings.trellis_slat_cfg_strength,
            num_oversamples = settings.trellis_num_oversamples,
            sparse_structure_cfg_interval = (
                settings.trellis_sparse_structure_cfg_interval_start,
                settings.trellis_sparse_structure_cfg_interval_end
            ) if hasattr(settings, 'trellis_sparse_structure_cfg_interval_start') else None,
            sparse_structure_rescale_t = getattr(settings, 'trellis_sparse_structure_rescale_t', None),
            slat_cfg_interval = (
                settings.trellis_slat_cfg_interval_start,
                settings.trellis_slat_cfg_interval_end
            ) if hasattr(settings, 'trellis_slat_cfg_interval_start') else None,
            slat_rescale_t = getattr(settings, 'trellis_slat_rescale_t', None),
            mode = getattr(settings, 'trellis_mode', None),
            use_multi_view = getattr(settings, 'trellis_use_multi_view', False),
        )

TrellisParamsOverrides = TrellisParams.Overrides


@dataclass
class TrellisRequest:
    """Request for Trellis 3D generation (internal use only)."""
    image: Image.Image
    seed: int
    params: Optional[TrellisParamsOverrides] = None
    images: Optional[List[Image.Image]] = None  # For multi-view generation


@dataclass(slots=True)
class TrellisResult:
    """Result from Trellis 3D generation."""
    ply_file: bytes | None = None


