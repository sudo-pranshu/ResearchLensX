"""Process-level runtime defaults. Import FIRST in experiment entry points (before numpy/torch/faiss).

macOS: torch, faiss-cpu and scikit-learn each ship their own libomp; loading several in one
process intermittently aborts ("OMP: Error #15") or segfaults (exit 139). Allowing duplicate
libomp and single-threaded OpenMP is the standard workaround; MPS/GPU work is unaffected.
"""
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
