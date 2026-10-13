"""Geriye donuk uyumluluk katmani. Gercek kod: vton/renderer.py"""
from vton.renderer import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.renderer", run_name="__main__")
