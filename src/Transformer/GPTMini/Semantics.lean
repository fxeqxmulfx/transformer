import Transformer.GPTMini.Semantics.FinalBlock
import Transformer.GPTMini.Semantics.Order
import Transformer.GPTMini.Semantics.ParityCorrectness
import Transformer.GPTMini.Semantics.AdjacencyRouting

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
prove phase preservation throughout the full stack. A variable-length
collision is proved for this first count/phase block. A simultaneous BOS
denominator channel repairs it and recovers the raw count from the actual
first hidden state for every parity prompt and supplied-answer prefix,
without an external length input. Explicit original FFN matrices realize
a 68-unit homogeneous bounded-count parity decoder and a simultaneous
69th completion-phase unit. A complete original two-layer parameter family
has distinct tied EVEN/ODD/EOS codes and exactly the same actual count
projection. Its full hidden state is connected to the real decoder FFN,
and exact pre-FFN prompt/answer formulas are proved for all raw bit words.
Uniform finite weights give strict label/EOS margins through final RMSNorm.
The actual checked List Int function solves every legal parity prefix in
both modes and freely generates label then EOS in two calls. Ordered-prefix
and recall-pair encoders remain separate obligations.

An explicit pair of the original 16-dimensional RoPE gives a derived
positive predecessor score gap across all Basis context lengths. The
actual finite softmax/XSA approximately copies that neighbor when the
projected self-value is zero. Realizing its QKV from simultaneous raw
embeddings, excluding post-table false writes and selecting the latest
matching record remain separate recall obligations.

The final-block certificate transports internal head and FFN codes to the
answer's separated embedding neighborhood. The complete actual readout
then emits that token through List Int -> List Int; correct final logits
are derived, not assumed. None of these theorems claims full Basis
accuracy for unrestricted parameters, floating-point equivalence, or
optimizer convergence. No new unproved claims are exported.
-/
