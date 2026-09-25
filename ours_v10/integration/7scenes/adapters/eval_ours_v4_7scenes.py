"""Deployable adapter. Implementation/version lives in the independent ours_v4 repo."""
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'ours_v4'))
from experiments.sevenscenes.runner import main

if __name__=='__main__': main()
