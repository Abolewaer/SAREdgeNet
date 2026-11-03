import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / '十次对比实验'))
from run_ten_seed_comparison import main


if __name__ == '__main__':
    main()
