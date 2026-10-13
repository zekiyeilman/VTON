"""Pipeline düğümleri: segmentasyon (01), 3B vücut mesh (02), ısı haritası (03)."""
import importlib

_LAZY = {
    "SegFormerNode": "vton.nodes.segmentation",
    "BodyMeshNode": "vton.nodes.body_mesh",
    "FitHeatmapNode": "vton.nodes.heatmap",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'vton.nodes' has no attribute {name!r}")
