import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from saredge.efficiency_cli import summarize

if __name__=='__main__':
    summarize()
