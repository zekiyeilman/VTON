"""Geriye donuk uyumluluk katmani. Gercek kod: vton/scraper/pipeline.py"""
from vton.scraper.pipeline import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.scraper.pipeline", run_name="__main__")
