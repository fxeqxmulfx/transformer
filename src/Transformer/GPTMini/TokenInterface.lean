import Transformer.GPTMini.TokenInterface.Decoding
import Transformer.GPTMini.TokenInterface.Basic
import Transformer.GPTMini.TokenInterface.Control
import Transformer.GPTMini.TokenInterface.Correctness
import Transformer.GPTMini.TokenInterface.Causal
import Transformer.GPTMini.TokenInterface.Specification

/-!
# Actual GPTMini continuations in the Basis integer-list interface

The complete causal-softmax GPTMini forward supplies the final logits;
checked raw encoding, context checks and deterministic greedy decoding
give List Int to List Int. Maximum/tie and strict-margin certificates
connect its discrete prediction to the independent raw Basis semantics.

Source: archived gpt_mini.py at f11b6e2 and the Basis recipes at cbafbe9.
The existing Lean stack shares one epsilon across its normalizations;
the interface documents that distinction from Python's unequal defaults.
Full-stack causality identifies teacher-forced rows with prefix calls.
Vocabulary/context compatibility proves that every Basis prefix reaches
the actual model path. A necessary-and-sufficient logit property states
exactly what given parameters must satisfy on the entire task domain.
Architectural laws are proved for all parameters; the task logit property
is not implied by them, as the concrete failing model demonstrates.
-/
