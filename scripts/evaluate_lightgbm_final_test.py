"""Evaluate one validation-selected LightGBM at frozen 2025Q4 origin."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True,help="New artifacts run ID; never overwritten")
    args=parser.parse_args()
    from oinq.modeling.supervised.lightgbm_final_test import run_final_test
    run_final_test(ROOT,args.run_id)

if __name__=="__main__":main()
