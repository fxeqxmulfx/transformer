import Transformer.GPTMini.Sparsemax.Basic
import Transformer.GPTMini.Sparsemax.Failure
import Transformer.GPTMini.Sparsemax.Certificate.Results

/-!
# Sparsemax training boundaries and measured prediction certificates

Real-valued saturation of the repository's causal projection, motivated by
arXiv:2211.11052v1, §3.1. Convex row inference does not make every outer
training gradient useful. The finite experimental certificate is separate
from a proof of floating-point PyTorch execution.
-/
