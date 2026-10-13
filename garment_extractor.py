"""Geriye donuk uyumluluk katmani. Gercek kod: vton/extractor.py"""
from vton.extractor import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.extractor", run_name="__main__")
