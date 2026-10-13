"""Geriye donuk uyumluluk katmani. Gercek kod: vton/calibration.py"""
from vton.calibration import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.calibration", run_name="__main__")
