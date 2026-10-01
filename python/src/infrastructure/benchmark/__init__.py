"""TinyShakespeare optimizer comparisons and their preserved evidence."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
