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
import Transformer.GPTMini.Semantics.RecallRotaryInsert
import Transformer.GPTMini.Semantics.RecallSaturation
import Transformer.GPTMini.Semantics.RecallStateBounds
import Transformer.GPTMini.Semantics.RecallProjectionScale
import Transformer.GPTMini.Semantics.RecallSecondQKV
import Transformer.GPTMini.Semantics.RecallNormalizedInputs
import Transformer.GPTMini.Semantics.RecallScoreError
import Transformer.GPTMini.Semantics.RecallRawScoreGap
import Transformer.GPTMini.Semantics.RecallRawExcluded
import Transformer.GPTMini.Semantics.RecallRawRouting
import Transformer.GPTMini.Semantics.RecallRetrievalAccuracy
import Transformer.GPTMini.Semantics.RecallRetrievalBlock
import Transformer.GPTMini.Semantics.RecallFinalState

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

An explicit original RoPE pair gives a positive predecessor score gap
across all Basis context lengths. Actual finite softmax/XSA copies that
neighbor with derived error when its projected self-value is zero.

A compact four-digit code distinguishes all 256 recall symbols with eight
coordinates. Placed in four actual slow RoPE pairs, it has exact norm two,
derived QKNorm scores and a content gap of exp(alpha)/50 at context 64.
Among equal keys its actual rotary score strictly prefers the latest
visible record. A uniform normalized latest-write margin
(1-cos(1/100))/4 is derived when raw records are separated by at least
one position; it is also below the categorical content margin. The raw
construction below supplies copied-key robustness and actual retrieval.

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
floating-point or optimization success is not inferred from this temperature.

An ordinary shared eight-to-sixteen linear matrix now inserts every real
compact copied key into the original slow rotary pairs. Exact inner
products, norms and copy distances are preserved, including imperfect
copies and position-dependent amplitudes. Its categorical image is the
verified matching code, and its excluded fast coordinates are zero.

Local original-QKNorm laws now cancel independent positive query/key
amplitudes above their actual clipping thresholds. Faithful insertion
and original RoPE transport base copy error to a normalized error at
most twice that error, independently of epsilon or record amplitude.
The lower base norm is derived from the norm-two reference and copy
distance. These operator laws are connected to actual raw clipping below.

Uniform genuine raw-state bounds now give pre-FFN norm in [1,9] and
its actual RMS multiplier in [1/2,8] for epsilon in [0,1]. The lower
norm follows from the protected constant, and the upper norm from both
true heads and W_o. The full block has norm at least one and positive
next-prenorm multiplier at most eight.

The fixed finite table gain 4/tableMargin now makes genuine raw table
amplitudes at least one. Every actual adjacent stored table key has
norm at least one at the fixed copy temperature. The whole raw block
has norm at most M=9+512*gain at every position, with gate decisions
derived from raw BOS/alphabet/table positions. A shared next Q/K gain
(1+epsilon)/(8/sqrt(M^2+64*epsilon)) times each genuine next RMS scale
exceeds epsilon. Raw query self-values stay zero, and prenorm value
code norms are bounded between twice the positive lower scale and
sixteen.

The actual second 64-to-192 fused QKV now evaluates simultaneous raw
query, gated copied-key and independent value projections in the
original sixteen-coordinate head. One ordinary gain scales Q/K and
leaves V independent; all unassigned head rows are zero. Its actual
prenorm retains the genuine position-dependent multiplier, and zero
gated keys or query self-values remain exactly zero.

Actual raw query normalization now equals the verified categorical rotary
direction, and every true adjacent table key has normalized error at most
twice the fixed copy tolerance. These facts use the genuine complete first
block, fused second matrix and actual prenorm; all clipping and norm
conditions are derived from raw tokens and the finite shared gains.
The raw query's own V and K are exactly zero.

At the fixed actual encoder, raw query/table scores now differ from their
true categorical rotary reference by at most twice exp(alpha) times the
copy tolerance. The proof uses the real normalized projections with no
inverse-epsilon amplification. Raw key/query competitors have zero score,
and the genuine own-value stays zero.

Genuine raw table competitors now retain a positive score gap
exp(alpha)*latestMargin/2. Different neighboring keys and earlier writes
of the same key are both covered using their derived rotary comparisons
and both actual copied-key errors. Context displacements follow from
real integer bounds.

Actual BOS and post-table value fillers now have exactly zero matching
score through the complete real gate, second prenorm and K projection.
An actual imperfect matching table record has score at least the positive
retained gap. Its true V is the raw value code with the genuine next RMS
multiplier, and every actual V norm is at most sixteen.

Raw table adjacency and chronological last-write conditions now imply
a complete actual matching-score row gap and finite causal-softmax tail
bound. Original XSA preserves retrieval because the true raw query V is
zero. The genuine head's error from its actual selected value is at most
32*(T-1)*exp(-retainedGap), with value diameter derived. The concrete raw
overwrite control strictly prefers its later, different-valued write.
These raw applicability predicates still require full-parser discharge.

One fixed finite second-head log-temperature now gives any positive
requested retrieval tolerance uniformly over the raw recall context.
Its true gap is log(1+2016/tolerance), evaluated directly in the finite
softmax tail. A positive tolerance at one sixteenth of the true next RMS
lower scale is achieved without an assumed leakage bound, and one shared
output gain has gain*lowerScale=2. No floating-point or AdamW success is
inferred; tied readout and full-parser discharge remain.

The genuine second ordinary block now includes the actual matching fused
matrix, finite shared temperature, nonzero W_o and original zero FFN.
Its exact full state writes compact retrieval into the raw value interval,
preserves query/type coordinates and transports actual head accuracy
through the real merge/output matrix.

Both real blocks now yield a full final-state error at most one eighth
from the first state plus the selected raw value's genuine amplitude,
which is at least two. Query/type/reserved readout axes stay raw. Strict
tied readout, the integer decoder and full validated parser remain.

The final-block certificate transports internal head and FFN codes to the
answer's separated embedding neighborhood. The complete actual readout
then emits that token through List Int -> List Int; correct final logits
are derived, not assumed. None of these theorems claims full Basis
accuracy for unrestricted parameters, floating-point equivalence, or
optimizer convergence. No new unproved claims are exported.
-/
