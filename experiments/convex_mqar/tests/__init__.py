"""CPU-only tests; do not compete with the ongoing CUDA benchmark."""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
