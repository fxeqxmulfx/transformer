import Transformer.GPTMini.Semantics.FinalBlock
import Transformer.GPTMini.Semantics.Order
import Transformer.GPTMini.Semantics.ParityCorrectness
import Transformer.GPTMini.Semantics.AdjacencyRouting
import Transformer.GPTMini.Semantics.RecallMatching
import Transformer.GPTMini.Semantics.RecallLatestGap
import Transformer.GPTMini.Semantics.RecallSlots
import Transformer.GPTMini.Semantics.RecallEmbedding
import Transformer.GPTMini.Semantics.RecallEmbeddingFeatures
import Transformer.GPTMini.Semantics.RecallQKV
import Transformer.GPTMini.Semantics.RecallHeadValues
import Transformer.GPTMini.Semantics.RecallRawHeads
import Transformer.GPTMini.Semantics.RecallRawCopy
import Transformer.GPTMini.Semantics.RecallMarkerWeights
import Transformer.GPTMini.Semantics.RecallRawMarker
import Transformer.GPTMini.Semantics.RecallRawGate
import Transformer.GPTMini.Semantics.RecallRawBinding

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
both modes and freely generates label then EOS in two calls. Complete
gated recall and ordered-prefix depth encoders remain separate obligations.

An explicit pair of the original 16-dimensional RoPE gives a derived
positive predecessor score gap across all Basis context lengths. The
actual finite softmax/XSA approximately copies that neighbor when the
projected self-value is zero. The compact raw projection below realizes
its QKV; excluding post-table false writes and selecting the latest
matching record remain separate recall obligations.

A compact four-digit code distinguishes all 256 recall symbols with eight
coordinates. Placed in four actual slow RoPE pairs, it has exact norm two,
derived QKNorm scores and a content gap of exp(alpha)/50 at context 64.
Among equal keys its actual rotary score strictly prefers the latest
visible record. A uniform normalized latest-write margin
(1-cos(1/100))/4 is derived when raw records are separated by at least
one position; it is also below the categorical content margin. These
simultaneous geometric properties still need copied-key robustness and
complete recall retrieval/readout.

Simultaneous compact raw key/value codes now occupy disjoint ordinary
embedding slots, with protected constant/type coordinates. All 548 entries
have derived norm squared six and a shared actual RMS multiplier. Exact
linear readback and zero cross-channels are proved; copy, position and
gated-key channels are initially zero. These raw coordinates feed the
actual fused QKV/BOS head.

The actual fused QKV and nonzero W_o now realize predecessor and marker
heads simultaneously. Every raw adjacent key/value pair has a derived
finite-softmax/XSA copy error in the true first attention residual;
its own raw value and query codes are retained. At every later position
the actual simultaneous marker head and residual contain exactly
1/(i+1), derived from raw BOS/alphabet IDs and the genuine causal mask,
even with future array entries. Sixteen actual original ReLU2 units now
realize the fixed table cutoff and preserve all raw code/type channels.
The full first block's gated-key slot is exactly zero on keys/BOS and
post-table values, excluding false writes after queries. Table values
retain the genuine compact predecessor copy with its derived positive
position-dependent RMS/gate amplitude. Complete raw-prefix coupling and
robust latest-write retrieval/readout remain.

A finite shared first-head log-temperature now supplies any positive
copy tolerance uniformly over the whole recall context. At a fixed
sixteenth of the derived latest-write margin, the genuine full first
block's table-key error is bounded relative to its actual gate amplitude,
and the stored key has positive norm for positive finite gain. Raw value
and query codes are retained. These are real-arithmetic capacity bounds;
second-block saturation/routing/readout and full validated-parser coupling
remain to be proved, and floating-point or optimization success is not
inferred from the conservative finite temperature.

The final-block certificate transports internal head and FFN codes to the
answer's separated embedding neighborhood. The complete actual readout
then emits that token through List Int -> List Int; correct final logits
are derived, not assumed. None of these theorems claims full Basis
accuracy for unrestricted parameters, floating-point equivalence, or
optimizer convergence. No new unproved claims are exported.
-/
