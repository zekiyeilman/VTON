"""Geriye donuk uyumluluk katmani. Gercek kod: vton/nodes/body_mesh.py"""
from vton.nodes.body_mesh import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.nodes.body_mesh", run_name="__main__")
