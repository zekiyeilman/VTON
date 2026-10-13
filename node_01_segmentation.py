"""Geriye donuk uyumluluk katmani. Gercek kod: vton/nodes/segmentation.py"""
from vton.nodes.segmentation import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.nodes.segmentation", run_name="__main__")
