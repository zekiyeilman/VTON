"""E-ticaret scraper'ı, beden tablosu önbelleği ve manken kalibrasyonu."""
import importlib

_LAZY = {
    "vton_otonom_boru_hatti": "vton.scraper.pipeline",
    "TrendyolOtonomScraper": "vton.scraper.trendyol",
    "MediaPipeAdaptiveCalibrator": "vton.scraper.calibration",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'vton.scraper' has no attribute {name!r}")
