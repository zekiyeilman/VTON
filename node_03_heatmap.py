"""Geriye donuk uyumluluk katmani. Gercek kod: vton/nodes/heatmap.py"""
from vton.nodes.heatmap import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.nodes.heatmap", run_name="__main__")
