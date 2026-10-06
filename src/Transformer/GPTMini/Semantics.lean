import Transformer.GPTMini.Semantics.FinalBlock
import Transformer.GPTMini.Semantics.Order
import Transformer.GPTMini.Semantics.ParityState

/-!
# Internal semantic guarantees for the original softmax GPTMini

Source: the archived GPTMini at f11b6e2 and Basis raw-token semantics at
cbafbe9. These results verify original operators, rather than replacing
the model by the task oracle or merely quantifying over possible weights.

Finite-softmax score gaps bound retrieval error, including RoPE, QKNorm,
XSA and approximate key copies. Faithful prefix features give exact
ordered-subsequence presence tests and distinguish the same-bag depth
pair. Raw MQAR's semantic target is independently proved to be its last
adjacent binding in Transformer.Basis.RecallAnswer.

Concrete original-model parameters compute raw ONE counts from embeddings
and the actual fused QKV, retaining the result in the first residual
block. A nonzero attention projection computes this count alongside an
independent completion-phase feature. Local output-matrix kernel equations
prove phase preservation throughout the full stack. Ordered-prefix and
recall-pair encoders for every Basis input remain separate obligations.

The final-block certificate transports internal head and FFN codes to the
answer's separated embedding neighborhood. The complete actual readout
then emits that token through List Int -> List Int; correct final logits
are derived, not assumed. None of these theorems claims full Basis
accuracy for unrestricted parameters, floating-point equivalence, or
optimizer convergence. No new unproved claims are exported.
-/
