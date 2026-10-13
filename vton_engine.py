"""Geriye donuk uyumluluk katmani. Gercek kod: vton/engine.py"""
from vton.engine import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.engine", run_name="__main__")
