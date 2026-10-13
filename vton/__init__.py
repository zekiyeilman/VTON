"""
VTON - Beden duyarlı sanal giydirme (Virtual Try-On) paketi.

Kullanım:
    from vton import VTONEngine, SegFormerNode
    from vton.scraper import vton_otonom_boru_hatti

Ağır bağımlılıklar (torch, transformers, selenium) sadece ilgili sınıf
ilk kullanıldığında yüklenir; `import vton` hızlı kalır.
"""
import importlib

__version__ = "0.2.0"

_LAZY = {
    "VTONEngine": "vton.engine",
    "SegFormerNode": "vton.nodes.segmentation",
    "BodyMeshNode": "vton.nodes.body_mesh",
    "FitHeatmapNode": "vton.nodes.heatmap",
    "Ansur3DTo2DConverter": "vton.models.ansur",
    "render_with_idm_vton": "vton.renderer",
    "ayikla_ust_kiyafet": "vton.extractor",
    "vton_otonom_boru_hatti": "vton.scraper.pipeline",
}

__all__ = list(_LAZY) + ["paths", "__version__"]


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    if name == "paths":
        return importlib.import_module("vton.paths")
    raise AttributeError(f"module 'vton' has no attribute {name!r}")
