"""Geriye donuk uyumluluk katmani. Gercek kod: vton/scraper/trendyol.py"""
from vton.scraper.trendyol import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("vton.scraper.trendyol", run_name="__main__")
