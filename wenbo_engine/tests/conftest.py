import sys
from pathlib import Path

# Ensure repo root is on sys.path so 'wenbo_engine' is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
