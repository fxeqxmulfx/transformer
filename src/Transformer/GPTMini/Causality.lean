import Transformer.GPTMini.Causality.Basic
import Transformer.GPTMini.Causality.Block
import Transformer.GPTMini.Causality.Model

/-!
# Architectural causality of GPTMini

Source: the actual causal-softmax head, pre-norm block and model stack
from archived gpt_mini.py at f11b6e2. Prefix preservation includes every
softmax denominator, RoPE, QKNorm, XSA, residual, FFN and final readout.
It holds for all parameter assignments and across different lengths.
-/
