"""Optional runtime helpers for backend-specific examples.

This package intentionally avoids importing heavy optional dependencies at
module import time. Import CUDA, PyTorch, or Transformer Engine only inside
the helper functions that need them.
"""

