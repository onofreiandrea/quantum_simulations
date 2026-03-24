#!/usr/bin/env python3
"""Create wenbo_engine.zip for Spark --py-files distribution."""
import os
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
os.chdir(root)

with zipfile.ZipFile("wenbo_engine.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirs, files in os.walk("wenbo_engine"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".pyc"):
                zf.write(os.path.join(dirpath, f))

n = len(zf.namelist()) if hasattr(zf, "namelist") else "?"
print(f"Created wenbo_engine.zip ({n} files) at {root / 'wenbo_engine.zip'}")
