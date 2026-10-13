"""Geriye donuk uyumluluk katmani. Gercek kod: vton/models/ansur.py"""
from vton.models.ansur import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.models.ansur", run_name="__main__")
