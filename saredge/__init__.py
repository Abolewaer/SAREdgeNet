
import sys
from pathlib import Path

_runtime = str(Path(__file__).resolve().parents[1] / '.runtime')
if _runtime not in sys.path:
    sys.path.insert(0, _runtime)
