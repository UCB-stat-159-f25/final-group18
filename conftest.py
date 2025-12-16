"""
This file ensures the repository root is on sys.path so tests can import
project modules regardless of where pytest is run from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
