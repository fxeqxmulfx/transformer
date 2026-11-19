# Certified execution of the actual network

Moved unchanged from [`../flat_spectrum.md`](../flat_spectrum.md) on 2026-11-19: the sections
*Full-stack rounding certificates* through *Trace-dependent execution bounds*. They carry the
exact-arithmetic results of [`sequence_models.md`](sequence_models.md) to executed numbers:
rounding through the full stack (B1–B9), primitive operation errors on a tensor trace (C1–C8),
one certificate from feature recovery to a source margin (J1–J7), checked backend contracts
(K1–K6), one complete captured CUDA invocation (L1–L5) and its trace-dependent bounds (M1–M5).
The Lean is in `Transformer.SequenceModel.Attention` and `Transformer.Precision`.

These labels are this file's. A1–A6, P1–P6 and the other capacity labels are in
[`sequence_models.md`](sequence_models.md); Theorem A and the earlier labels are in
[`../flat_spectrum.md`](../flat_spectrum.md).

## Full-stack rounding certificates

The earlier A6 result compared one rounded aggregate with a real aggregate
on the same stored residuals. B1--B9 close the propagation gap: they bound
the **actual `CRASP.Rotary.run`** against the complete real source `hidden`,
then transfer strict predictions to `CRASP.Rotary.predict` and its existing
uniform weighted-count formula. There is no assumed final residual error,
assumed block Lipschitz constant, or substituted activation.

The source remains diff_gpt `b98c665b`, with the five pinned files listed in
`review/sequence_model_source.json`. Both normalization epsilons are positive.
The exact source RoPE frequencies and per-layer enable flags are unchanged;
the block estimates also allow arbitrary identical phase tables on both
sides. Dimensions are arbitrary admissible finite dimensions. As before,
actual Python execution requires the source FFN size and positions within
`block_size`; the mathematical forward and counting language also admit
longer directly supplied phase tables.

### B1. Pre-normalization, projections and vector rounding

Write W for residual width, d for head dimension, H for the number of heads,
F for FFN width, epsilon for RMS epsilon, and xi for XSA's clipped-L2 epsilon.
For one block let Q, O, G, V and U be its fused-QKV, attention-output,
gate, value and FFN-output linear maps, respectively. All matrix norms below
are Euclidean operator norms. Define

\[
 L_R=2/\sqrt\epsilon,\qquad M=\|Q\|\sqrt W,\qquad C=\|Q\|L_R.
\]

Every actual pre-normalized Q/K/V head has norm at most M, and a residual
change E changes each such head by at most CE. This needs no bound on the
incoming residual itself. Keeping the same exact RoPE phases preserves the
norm of vector differences. Rounding W coordinates without clipping adds
at most `sqrt(W) u`, where `u=2^-s` is the fixed-point quantum.

### B2. Actual SwiGLU, with independent gate and value matrices

The derivative of sigmoid yields a global Lipschitz bound of one. From this
and `abs(silu(t)) <= abs(t)`, the actual SiLU product is bounded on the
pre-normalized input ball. A sufficient FFN constant is

\[
 L_F=\|U\|\sqrt F\left[
   \|V\|\sqrt W(1+\|G\|\sqrt W)\|G\|
   +\|G\|\sqrt W\|V\|\right].
\]

For inputs of norm at most `sqrt(W)`, the real `feedForward` changes by at
most `L_F` times their distance. The two FFN normalizers make its whole
residual map globally Lipschitz with constant

\[
 R=1+L_R L_F L_R.
\]

The proof uses the source SwiGLU formula. It does not reuse the old ReLU
block's constants or assume that the gate and value projections coincide.

### B3. Both the attention distribution and values change

If every input residual changes by at most E, the source Q/K logits change
by at most `2 L_R C E`. Causal softmax row sensitivity and the changing
value vectors therefore give

\[
 A=(16 L_R M+1)C,\qquad
 \|\operatorname{average}(x)-\operatorname{average}(y)\|\le AE.
\]

The reference average has norm at most M by convexity. These constants do
not depend on the number of keys. Unlike A2, this result changes Q, K and V
together as the residual stream moves.

### B4. XSA, all branch normalizers and both residual additions

Define the projection and moving-direction constants

\[
 P=L_R\|O\|\sqrt H,\qquad X=2M(2C/\xi).
\]

If the query residual error is E and each head-average error is at most B,
while the reference averages have norm at most M, the actual local source
update differs by at most

\[
 R\{E+P(2B+XE)\}.
\]

This includes movement of XSA's clipped own-value direction, its explicit
first-position zero, head merging, learned projection, all four branch
normalizers, both residual additions and SwiGLU. Substituting B=AE gives
the full source block constant

\[
 L_{\rm block}=R\{1+P(2A+X)\}.
\]

The bounds hold at the XSA clipping sphere; no differentiability premise
is needed. Small normalization epsilons can make these sufficient global
constants very large. The theorem does not assert that they predict typical
error amplification accurately.

### B5. Actual rounded prefixes and their vector error

One-based natural counting positions are explicitly reindexed to the
zero-based finite source tensor. Every causal prefix term, including the
query's own key/value, is retained. The real ratio in A6 becomes exactly
the coordinate of `Normalized.sourceAverage` on the decoded counting stream.

Let `a=exp(-sqrt(d))` and require `u<a`. Pre-normalization supplies M as the
value-coordinate bound, so the per-coordinate new aggregate error is

\[
 r=\frac{u(1+M)}{a-u}+u.
\]

The head-vector error is at most `sqrt(d) r`. `AggregateInRange` checks
`Fx.NoSaturation` for every actual exponential coefficient, weighted-value
coefficient and pre-rounding quotient. It assumes no error bound or output
correctness. The positive-denominator theorem of A6 applies to these same
premises.

### B6. One actual counted block against the real source

`StepInRange` additionally checks every coordinate of the actual local
update before stored-residual rounding. Combining B3--B5 proves

\[
 E_{\rm next}\le L_{\rm block} E+\rho,\qquad
 \rho=RP\,2\sqrt d\,r+\sqrt W\,u.
\]

The comparison is the existing `Rotary.sourceLayer.step`, with its integer
prefix sums and existing rounding sites, against the complete real source
`block`. It is not a newly defined approximate forward pass. Both the
changed input stream and the new rounding errors are included.

### B7. Full stack with each layer's own weights

`errorBudget` starts from `E_0=sqrt(W) u` and recursively applies B6 using
the actual independently learned block at each layer:

\[
 E_{l+1}=L_l E_l+\rho_l,\qquad 0\le l<D.
\]

`TraceInRange` checks the normalized input and each executed layer's actual
rounding sites, on the actual input word. `stack_error` proves, for every
executed layer n and every source position i,

\[
 \|\operatorname{decode}(\operatorname{counted}_n(i))
       -\operatorname{hidden}_n(i)\|\le E_n.
\]

The normalized embedding, first-token flag, exact source RoPE phases and
entire residual recurrence are connected to the actual implementations.
The constants and recurrence explicitly expose width, head count, FFN
width, depth, learned matrix norms and fractional precision. Total bit
width p controls the range premises. No finite-context cardinality factor
appears in the numerical bound, although whether the range and margin
conditions hold can depend on the input and its length.

### B8. Strict classification and the uniform counting formula

For a target and each competitor v, a sufficient real-source margin is

\[
 L_R E_D(\|e_{\rm target}\|+\|e_v\|)
 <\operatorname{logit}_{\rm target}-\operatorname{logit}_v.
\]

`rounded_predict_true` derives the actual rounded model's strict prediction
from this margin, positive epsilons, `u<a` and `TraceInRange`. `E_D` here is
the computed B7 recurrence, rather than the unproved residual-error premise
of A5. Final RMSNorm and the raw tied embedding norms remain explicit.

`uniform_source_certificate` chooses one formula of counting depth at
most D, with the same exact nonperiodic kernels, before any input word.
It preserves every source decision on every finite word that satisfies
the numerical certificate. This is conditional language agreement with a
uniform formula. It supplies no strict depth lower bound for those kernels,
and does not turn a finite-context periodic replacement into a uniform one.

### B9. Nonconstant concrete witness and limits

The witness has W=4, d=4, H=1, F=11, D=2 and three distinct unit embeddings.
Both residual blocks are identities, and source RoPE is enabled. A synthetic
RMS epsilon 3/4 makes those embeddings fixed points of normalization;
XSA epsilon is one and the fixed-point format is p=8,s=5. These example
epsilons are explicitly different from the source float32 defaults.

Every rounding-range premise holds on every finite three-letter word.
The computed final budget is `3/16`; every correct logit margin is one,
and its sufficient threshold is `sqrt(3)/2 < 1`. The classification theorem
therefore proves that the actual rounded model predicts the token at each
position, for every word. Predictions vary across all three classes. The
actual residual error in this witness is zero: the nonzero computed budget
demonstrates sufficiency and is not claimed sharp. Separate B1 examples
exercise nonzero coordinate rounding, and B2--B4 examples admit arbitrary
learned block matrices at the actual source normalization epsilons.

| Lean module | Result |
| --- | --- |
| `Attention.Stability.Vector`, `Gates` | projection/vector rounding and actual SwiGLU constants |
| `Attention.Stability.Averages`, `Local` | moving logits, values and XSA; full source block sensitivity |
| `Attention.Stability.Aggregate` | actual natural prefix/finite tensor connection and rounded head error |
| `Attention.Stability.Stack` | actual step and full residual-stack error budgets |
| `Attention.Stability.Certificate` | strict prediction and uniform weighted-count formula transfer |
| `Attention.Stability.Witness`, `Certificate` | complete nonconstant witnesses for range and margin hypotheses |

The closed gap is propagation through the specified fixed-point counting
model. Local operations still use exact reals, integer sums are exact, and
the final readout is real. Float32 projections, nonlinearities, reductions,
overflow/underflow, quantized weights, and generated cosine/sine tables need
their own execution-error analysis. A strict hierarchy for source-generated
nonperiodic kernels and any optimizer/generalization guarantee also remain
unproved. No training experiment or Python implementation changes in B1--B9.

Validation of B1--B9: full `lake build` passes (13174 jobs), with no warnings
in the new modules. All 46 new named theorems pass individual `#print axioms`
checks using only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 existing `sorry`, zero dependent results and zero extra axioms.
The regenerated index has 4077 modules, 21050 theorems, zero vacuous results
and zero placeholders. Forbidden checks and `git diff --check` pass, and
all five pinned source hashes match after rereading the actual forward.

The required Python run passes all 418 lab tests with three skips. Its
104-test optimizer suite reproduces the two previously recorded Muon
allocation failures (float32 and float64, 75 allocations of 8519680 bytes
each); Schedule-Free AdamW passes. These are unchanged from A1--A6's
validation. Logs, the individual theorem axiom-check script and source
review are retained in `.kanban/artifacts/129/`.


## Primitive execution certificates

C1--C8 extend B1--B9 from exact local operations in a fixed-point counting
model to a finite tensor trace with explicit operation errors. The reference
is still the pinned `diff_gpt` no-cache forward at `b98c665b`. The conclusion
is conditional on those local errors; it is not a proof that an arbitrary
PyTorch/CUDA backend satisfies them. All widths, block matrices, finite
depths, input words and reference phase tables remain parameters.

### C1. Finite binary32 and explicit arithmetic order

`binary32Value` decodes a sign, exponent field `e<255` and fraction field
`m<2^23`; `binary32_value_mem` proves membership in the existing finite
`Precision.grid 8 23`, including subnormals. For a nearest map Q and two
represented endpoints bracketing x, `IsNearest.error_of_bracket` proves

\[
 |Q(x)-x|\le (b-a)/2.
\]

`binary32_bracket_error` instantiates this with actual binary32 patterns.
Inside a positive normal binade, consecutive fraction fields give absolute
error at most `2^(e-150)/2`, and `binary32_normal_relative_error` gives
`2^-24 |x|`. The normal theorem requires `0<e<255` and `m+1<2^23`;
it does not silently include binade boundaries. Positive subnormal
intervals instead give the absolute bound `2^-149/2`, including intervals
next to zero. Negative values and boundary-crossing intervals can use the
general represented-endpoint theorem. There is no universal relative-error
claim near zero, or an overflow/NaN/flush-to-zero emulator.

`roundedDot mulRound addRound a b n` specifies the sequential recurrence
`z[0]=0`, `z[k+1]=addRound(z[k]+mulRound(a[k]*b[k]))`. The proved error is
at most the sum of all local product and accumulator errors, each evaluated
at the actual rounded partial sum. Uniform bounds P and A give `n(P+A)`.
Represented brackets discharge those premises. Identity product rounding
with nearest accumulator rounding models one correctly rounded FMA per
step; two nearest maps model separate multiply/add. No selected GPU kernel
is asserted to use this order or these formats. `binary32_quarter_ulp`
proves that nearest binary32 rounding of `1+2^-25` is one, giving an actual
nonzero rounding witness.

### C2. Stored RoPE tables need not be exact rotations

`tableRotate` retains the source half-split arithmetic with independently
stored cosine and sine entries. Given coordinate errors at most tau from
`cos(a)` and `sin(a)`, `tableRotate_error` proves

\[
 \|\widehat R x-R_a x\|\le 2\tau\|x\|.
\]

It assumes neither a unit-circle identity for the stored entries nor that
they represent any exact phase. Thus frequency/position computation, table
generation and table-storage errors can be included in tau. Arithmetic
applying those entries is checked separately. An explicit example uses
`c=1+1/1024`, `s=1/1024` at reference phase zero. Odd leftover coordinates
are unchanged in the mathematical helper; the source configuration has
even head width. Disabled RoPE uses the identity phase/table and exact copy.

### C3. Finite traces and local contracts

`AttentionTrace`, `FFNTrace`, `BlockTrace` and `ModelTrace` contain numerical
data only. `AttentionAccurate`, `FFNAccurate`, `BlockAccurate` and
`ModelAccurate` are predicates checking the recorded operations against
their own recorded inputs. They do not assume a final state, block or
prediction error. A common nonnegative delta bounds Euclidean vector errors
and absolute scalar errors; tau bounds table entries independently.

The checks include input RMSNorm, fused QKV, head Q/K RMSNorm, stored table
application, scaled scores, causal weighted averaging, clipped L2
normalization, XSA and the explicit first-position zero, head merge/output
projection, branch RMSNorm, gate/value projections, SiLU, multiplication,
FFN projection, both residual additions, embedding-table error, final
RMSNorm and tied logit dot products. Reshape and slices are exact indexing.
The fused FFN matrix's two halves are represented by their independent
gate/value maps, preserving its exact mathematical function.

`roundedVector_error` turns coordinate brackets of half-width at most eta
into Euclidean error `sqrt(d)*eta`. `roundedMatrix_error` turns explicit
rowwise product/add bounds into `sqrt(m)*n*(P+A)` for an m-by-n reduction.
These supply operation-contract bounds when their rounding/order premises
hold. Weight quantization can be included in the matrix operation error
relative to the reference matrix; the shared embedding-table perturbation
is represented separately and used at both ends of the model.

RMSNorm, L2 normalization and SiLU accuracy remain primitive implementation
contracts. In particular, the causal average check is an explicit SDPA
kernel contract, covering softmax and the weighted reduction at the
recorded scores and values. A fused implementation may use a logical score
trace instead of materializing that tensor. This is not yet a scalar IEEE
proof of its exponential, reciprocal, scaling, summation or reduction
order. Finite real traces also require excluding backend nonfinite results.
A single delta is convenient notation, not a backend-independent error
constant; it may depend on width, context, input range and kernel choice.

### C4. Attention and the approximately normalized XSA direction

Write `L_R=2/sqrt(eps)`, `M=||Q||sqrt(W)` and let H be the head count,
d the head width. `attentionBudget` derives, in execution order,

\[
\begin{aligned}
 e_Q&=\|Q\|\delta+\delta,&
 e_N&=L_R e_Q+\delta,\\
 e_R&=2\tau(\sqrt d+e_N)+e_N+\delta,&
 e_S&=(2\sqrt d+e_R)e_R/\sqrt d+\delta,\\
 e_A&=8e_S M+e_Q+\delta,&
 e_U&=(2/\epsilon_X)e_Q+\delta,\\
 e_H&=[1+(1+e_U)^2]e_A+M(2+e_U)e_U+\delta,\\
 e_P&=\|O\|\sqrt H\,e_H+\delta,&
 e_{\rm attn}&=L_R e_P+\delta.
\end{aligned}
\]

`attention_scores_error` includes the bilinear cross term from perturbing
both Q and K. `attention_error` derives the complete branch bound from
the local checks. The exact clipped direction has norm at most one, but
the computed direction can have norm as large as `1+e_U`.
`xsa_pair_error` proves the corresponding bound, instead of incorrectly
reusing the earlier theorem requiring both directions to have norm at
most one. Its witness has a computed direction of norm two.

### C5. SwiGLU including activation and product execution

Set `e_g=||G||delta+delta`, `e_v=||V||delta+delta`,
`R_g=||G||sqrt(W)` and `R_v=||V||sqrt(W)+e_v`. Then `ffnBudget` gives

\[
 e_{\rm product}=\sqrt F\,[\delta+\delta R_v+
                   R_v(1+R_g)e_g+R_g e_v],\qquad
 e_{\rm ffn}=L_R(\|U\|e_{\rm product}+\delta)+\delta.
\]

`ffn_error` derives this from the actual independent gate and value
projections, the SiLU error, multiplication error, output projection and
post-normalizer. The approximate pre-normalized input need not satisfy
the exact source norm bound; its error is propagated through both matrices.
Examples admit arbitrary block weights at the source float32 epsilons.

### C6. Both residuals and the complete depth recurrence

With the actual B3 FFN residual gain `f` and B4 full-block Lipschitz
constant `L`, the new block radius is

\[
 r=f(e_{\rm attn}+\delta)+e_{\rm ffn}+\delta.
\]

`block_error` and `block_movement_error` prove the same-input and moving-input
bounds. `model_error` derives the complete layer recurrence

\[
 E_0=L_R\delta+\delta,\qquad E_{\ell+1}=L_\ell E_\ell+r_\ell.
\]

The base includes the shared embedding-table perturbation and input
normalizer; each independently weighted block supplies its own constants.
The result bounds every recorded residual state at every depth up to D.
Width affects matrix norms, head/FFN sizes and local arithmetic bounds;
depth affects repeated amplification and accumulated operation error.
This is a sufficient stability bound, not a sharp width/depth lower bound.

### C7. Executed logits and the uniform counting bridge

For raw reference class embedding e, `logitRadius` is

\[
 r_e=(L_R E_D+\delta)\|e\|+(\sqrt W+\delta)\delta+\delta.
\]

`logit_error` proves this bound for the recorded logit, including final
RMSNorm error, the perturbed tied embedding and the executed dot product.
`predict_true` proves a strict computed target prediction when each exact
source target-versus-competitor margin exceeds `r_target+r_competitor`.
Neither an assumed hidden-state error nor an assumed final-logit error
appears among the primitive contracts.

`uniform_execution_certificate` chooses the existing nonperiodic RoPE
counting formula before any input word, with counting depth at most D.
When both B8's fixed-point range/precision/margin certificate and C7's
execution certificate hold against the same exact source, the formula
accepts the target and the executed trace predicts it. These are explicit
conditional target agreements for all finite words. This does not assert
unrestricted equivalence for arbitrary hardware, a context-independent
floating-point error budget, or a strict hierarchy for exact RoPE kernels.

### C8. Witnesses, remaining gap and validation

Every new theorem with hypotheses has a concrete Lean `example`. Exact
source traces witness all operation contracts for arbitrary weights and
phases. `shiftedModel` additionally perturbs each final dot product by
`delta`; `shiftedModel_accurate` proves every local check, and
`shiftedModel_logit_error` proves that its actual logit error equals delta.
This synthetic trace is not attributed to a CUDA implementation.

For B9's W=4, H=1, F=11, D=2, three unit embeddings and zero block matrices,
use `eps=3/4`, `epsilon_X=1`, `delta=1/1024`, `tau=0`, with source RoPE
active. The complete derived budget is `(5 L_R+9)/1024`; each logit bound
is below 1/2, while each correct source margin is one. Thus
`example_executed_prediction` certifies the nonconstant three-class
classifier on every finite word despite its nonzero logit errors.
`example_uniform_execution` also instantiates the counting bridge with
p=8,s=5. The synthetic normalization epsilons are not the source defaults.

| Lean module | Proved connection |
| --- | --- |
| `Precision.Binary32Error`, `ArithmeticError` | finite represented brackets, normal/subnormal bounds, explicit multiply/add or FMA accumulation |
| `Attention.Execution.Tables`, `Rounding` | independent stored RoPE coefficients and scalar-to-tensor operation bounds |
| `Attention.Execution.Trace`, `Basic`, `Budget` | concrete numerical traces, local predicates and derived constants |
| `Attention.Execution.Scores`, `Attention`, `FFN` | complete branch errors, including non-unit computed XSA direction |
| `Attention.Execution.Block`, `Model`, `Readout` | both residuals, arbitrary finite depth, tied logits and strict prediction |
| `Attention.Execution.Witness`, `Certificate` | nonzero-error classifier and conditional uniform exact-RoPE counting transfer |

The closed gap is the propagation of local execution errors, including
independent stored RoPE tables, through the full normalized source stack.
Connecting a selected PyTorch/CUDA kernel to the local contracts remains:
its dtypes and reduction order, transcendental/normalization accuracy,
SDPA error, table-generation bounds and finite-range behavior must still
be established. The scalar/vector adapters are reusable pieces of that
connection, not a completed backend proof. A strict exact-RoPE depth
hierarchy and optimizer/generalization guarantees remain unproved. No
training experiments or Python implementations change in C1--C8.


Validation of C1--C8: full `lake build` passes (13190 jobs); new modules
have no warnings. All 46 new named theorems pass individual `#print axioms`
checks using only `propext`, `Classical.choice` and `Quot.sound`. The audit
remains at 36 pre-existing `sorry`, zero dependent results and zero extra
axioms. The regenerated index has 4093 modules, 21096 theorems, zero vacuous
results and zero placeholders. Forbidden checks and `git diff --check`
pass, and all five pinned source hashes match after reviewing the forward.

All 418 lab tests pass with three skips. The 104-test optimizer suite
reproduces the two existing ScheduleFreeMuon allocation failures, float32
and float64, each with 75 allocations of 8519680 bytes. Schedule-Free AdamW
passes. These failures match B1--B9's baseline. Build/test logs, individual
axiom checks and the source review are retained in `.kanban/artifacts/130/`.

## Unified spectral transformer certificate

`Attention.Spectral.exists_power_certificate` connects feature recovery,
the actual normalized source transformer and exact-RoPE counting logic in
one theorem. It constructs weights from the recovered real score table and
proves a source margin. Under explicit numerical contracts, every class
decision agrees with the task and one fixed family of counting formulae.
The task is (a/b)^K for every prime P, natural K, numerator a and nonzero b.
This is finite representation and conditional execution; the ideal feature
optimization premise is separate from training with Schedule-Free AdamW.
The counting language is the existing exact-RoPE extension, not the periodic
hierarchy.

### J1. Bounded vector outputs in the actual source

P5's prescribed-RoPE construction extends from unit answer vectors to any
finite table t(x) with norm at most one. The actual state after attention is
base + context, where base has norm one and the normalized context has norm
less than one. Thus

\[
 \|t(x)-(\mathrm{base}+\mathrm{context})\|<3<\sqrt W
 \quad\text{when }W>9.
\]

The existing inverse of output RMSNorm reaches the residual displacement.
The source RoPE encoder distinguishes the whole input word projectively,
and the actual bias-free SwiGLU interpolates this finite vector table.
`exists_bounded_block` and `exists_bounded_model` retain attention, Q/K
normalization, XSA, both Peri-RMSNorm branches, embedding normalization and
tied final readout. The original final input token is the readout position.
Extra layers are identity residual blocks; they do not prove an advantage
from active depth.

### J2. Preserve every recovered score

For any real score table S on V^(N+1) by V, let e_v be orthonormal token
coordinates, A the positive raw embedding amplitude, and

\[
 M=1+\sum_x\sum_v |S(x,v)|,\qquad
 t(x)=M^{-1}\sum_v S(x,v)e_v.
\]

The triangle inequality bounds norm t(x) by one. J1 realizes t(x) exactly
before final RMSNorm. Since the tied raw embedding is A e_v, every actual
source logit satisfies

\[
 L_v(x)=g(x)S(x,v),\qquad
 g(x)=\frac{A\sqrt W}{M\sqrt{\|t(x)\|^2+W\,\mathrm{eps}}}
 \ \ge\ g_0=\frac{A\sqrt W}{M\sqrt{1+W\,\mathrm{eps}}}>0.
\]

`exists_score_model` proves this for the complete table; `minimumGain` is
g_0. A spectral score gap m transfers to a source gap at least g_0 m. The
construction uses S, not an assumed table of correct predictions. Positive
row scaling preserves ordering, but need not preserve softmax probabilities.

### J3. Supply the scores by spectral recovery

Let chi_r be a finite bank of multiplicative characters on F_P, with
nonnegative weights w_r summing to one. Require joint separation on the
unit group: each distinct a,b has a positively weighted character with
chi_r(a) != chi_r(b). Individual characters need not each separate answers.
The true feature kernel is

\[
 K_r(a,b)=\chi_r((a/b)^K),\qquad a,b\in\mathbb F_P^\times.
\]

`Fits` requires agreement on an observation pattern Omega. `Minimizes`
requires a global minimum of the sum of coordinate nuclear norms under
those same feature observations. If the bipartite observation graph is
connected and contains a perfect matching, `recovered_bank` proves X_r=K_r
on every pair, including pairs outside Omega. This composes the existing
unique kernel and bank recovery theorems. `recovered_unit_margin` gives a
uniform positive margin for the recovered bank's weighted character readout.

`true_bank_minimizes` supplies witnesses to the ideal premise. A faithful
character exists for every prime, so a singleton bank with weight one
satisfies separation; larger jointly separating banks are also admitted.
Full observation gives explicit witnesses in `RecoveryWitness`; the general
theorem permits any pattern satisfying the two graph conditions. Nothing
here identifies transformer cross-entropy with feature nuclear norm or
asserts that Schedule-Free AdamW finds its minimum.

### J4. Original residues, zero numerators and exponent zero

The source vocabulary is all of F_P, not just its units. On nonzero input
pairs, `powerScores` reads the recovered coordinates and takes their fixed
character-code score for each nonzero candidate; candidate zero receives
score -1. The true unit answer scores one. For a=0, the scores are +1 at
the elementary boundary answer and -1 elsewhere: the boundary answer is
one when K=0 and zero when K>0, using the natural-power convention 0^0=1.

If the unit-bank margin is m>0, the full residue margin is at least
min(1,m)>0. Denominator-zero rows are totalized for interpolation and
excluded from every powered-division correctness claim. No answer oracle
is used on nonzero pairs. `powerLabel_residue` and `tokenUnit_power` prove
that the unit and original-token labels describe the same field computation.

### J5. Finite resources and actual recovered-score computation

`exists_recovered_power_model` composes J2--J4. For two original operands,
its sufficient source family has

\[
 V=P,\quad H=1,\quad W=2(P+1)^2+6P^2+2\,\mathrm{padding},\quad
 F=\lfloor(8W+1)/3\rfloor,\quad D=\mathrm{extraDepth}+1.
\]

It uses the geometric integer-position RoPE grid at the configured source
base, enabled at every layer, and context length two within the configured
block size. It permits eps>0 and 0<xeps<=1. Examples use the source float32
RMS epsilon 2^-23 and XSA epsilon 10^-12. A prime vocabulary can use the
source constructor with `pad_vocab_size_to=1`.

For a general fixed length N+1, `exists_finite_score_model` replaces 6P^2
by 6P^(N+1). This is finite interpolation of recovered scores, not efficient
character arithmetic inside the transformer. These sufficient bounds are
not minimal. Additional depth is identity padding; neither a sharp
width/depth tradeoff nor useful weight-conditioning bounds follow.

### J6. A positive execution allowance exists

Fix the constructed finite weights and positive margin gamma=g_0 m. C6's
recurrence E_n(delta,tau) is continuous along delta=tau=eta and vanishes
at eta=0 for every finite n. The same holds for C7's logit radii. All
budgets and radii are nonnegative for nonnegative local errors. The finite
sum of twice each class radius bounds every pair of class radii. Continuity
therefore yields eta>0 such that

\[
 r_u(\eta,\eta)+r_v(\eta,\eta)<\gamma
 \quad\text{for all classes }u,v.
\]

`exists_execution_tolerance` proves this for any fixed finite model with
positive margin; it does not assume a small final-state error.
`exists_robust_power_source` instantiates all spectral premises for every
prime and K and applies the allowance to C8's synthetic shifted trace.
Its actual logit error equals eta>0, all primitive contracts hold, and
every class decision is correct. Examples include P=3,K=0 and P=97,K=5.
This gives no useful lower bound on eta and does not assert that a selected
hardware format or kernel satisfies the allowance.

### J7. Fixed counting formulae and complete decision agreement

For each format (bits,frac), choose the existing RoPE-aware formula phi_v
for each class before the input word. Each has counting depth at most D
and represents the rounded source classifier. `uniform_margin_certificate`
strengthens the earlier positive-target connection to every class, including
rejection of competitors. `exists_power_certificate` composes it with J5:

\[
 \varphi_v(x)=\widehat{\mathrm{predict}}_v(x)
 =\mathbf 1\{v=(a/b)^K\}.
\]

The conditions remain explicit: positive normalization epsilons;
2^-frac < exp(-sqrt(head_dim)); the rounded trace's `TraceInRange`; its
derived budget below gamma (`CountingBelow`); the executed trace's local
`ModelAccurate` contracts; and its derived pairwise radii below gamma
(`ExecutionBelow`). Both numerical executions are compared through the
same exact source. A format meeting all counting conditions for the
compiled powered-division weights is not constructed here. The formulae
are uniform over words; this task guarantee is for two-token inputs, not
an unbounded arithmetic language or a strict exact-RoPE depth hierarchy.

`example_uniform_decisions` discharges every numerical premise for the
existing nonconstant three-class source: W=4,D=2,eps=3/4,xeps=1,bits=8,frac=5,
delta=1/1024,tau=0. All class decisions agree on equal two-token words
despite nonzero executed logit errors. This witnesses the joint certificate
independently of any unverified backend contract.

| Lean module under `Attention.Spectral` | Proved connection |
| --- | --- |
| `Realization`, `Geometry`, `Scores` | bounded vector interpolation and all positively scaled source logits |
| `Recovery`, `Residues`, `PoweredScores`, `Powered` | ideal bank recovery, zero boundary and source margin |
| `Numerics`, `Decisions`, `Counting` | positive error allowance and every-class numerical/counting agreement |
| `Certificate`, `Existence` | composed theorem and all-prime perturbed execution witnesses |
| `RecoveryWitness`, `CountingWitness` | concrete spectral and numerical premise checks |

Board task #131 covers this composition. Tasks #132--#134 separately track
concrete backend operation contracts, Schedule-Free AdamW reaching the
spectral conditions, and task-specific depth/width lower bounds for exact
nonperiodic RoPE. Those remain questions, not corollaries of this certificate.
No training experiment or Python implementation changes in J1--J7.

Validation of J1--J7: full `lake build` passes (13205 jobs), with no warnings
in the new modules. Individual `#print axioms` checks for all 45 new public
theorems use only `propext`, `Classical.choice` and `Quot.sound`. The global
audit remains at 36 pre-existing `sorry`, zero dependent results and zero
extra axioms. The regenerated index has 4108 modules and 21143 theorems,
with zero vacuous results and zero placeholders. Forbidden checks and
`git diff --check` pass; all five pinned source hashes still match after
reviewing the forward and every new statement against J1--J7.

All 418 lab tests pass with three skips. The 104-test optimizer suite
reproduces C1--C8's two ScheduleFreeMuon allocation failures (float32 and
float64, 75 allocations of 8519680 bytes each). Schedule-Free AdamW passes;
the overall Python command exits unsuccessfully because of those existing
Muon failures. Logs, axiom checks and the source crosswalk are retained
in `.kanban/artifacts/131/`.

## Concrete backend contracts

K1--K6 replace several opaque primitive-error assumptions with algebraic
and polynomial checks and prove those checks for actual captured PyTorch
outputs. This is a finite backend conformance result: the captured CPU and
CUDA numbers have ordinary Lean proofs of their local error bounds. It
does not establish `ModelAccurate` for a complete captured transformer,
every checkpoint, or every future execution of these kernels. That
composition is addressed by task #135 in L1--L5 below.

### K1. Certify normalization scales without trusting sqrt

For a nonnegative scalar t,

\[
 |t-1|\le |t^2-1|.
\]

Let x be an operation's actual input, z its stored output, b>0 its exact
normalizing denominator, and a>=0 a rational certificate scale. Suppose

\[
 |a^2b^2-1|\le\rho,\qquad
 \|z-ax\|\le\beta,\qquad \|x/b\|\le B.
\]

`normalization_residual` proves

\[
 \|z-x/b\|\le\beta+\rho B.
\]

The proof factors the scale error through the normalized vector. The scale
a is a certificate witness; it need not be the hidden intermediate used
by a fused kernel. The product residual is checked at the stored input and
output. No sqrt/rsqrt accuracy claim about a library or device is assumed.

### K2. Source RMSNorm and clipped L2

For RMSNorm at width d and eps>0,

\[
 b^2=(\|x\|^2+d\,\mathrm{eps})/d,\qquad B=\sqrt d.
\]

Thus `rms_residual_error` checks the polynomial residual
`abs(a^2*(norm(x)^2+d*eps)/d-1)` and the observed multiplication residual.
Its local error bound is beta+rho*sqrt(d), retaining the actual epsilon.
It does not multiply this local residual by inverse sqrt(eps). This is a
local execution certificate, not a replacement for the moving-input
Lipschitz bound in C6.

For clipped L2, b=max(norm(x),xeps)>0 and B=1. `l2_residual_error` gives
beta+rho, keeping the clipping boundary. On a known side of the boundary,
b^2 is either the exact sum of coordinate squares or xeps^2, so the scale
check is rational for rational stored inputs. For any width, a coordinate
product-residual bound eta gives beta=sqrt(d)*eta by B1.

### K3. Exponential and SiLU certificates

Define the finite polynomial P_n(x)=sum_{m<n} x^m/m!. For abs(x)<=1 and
n>0, Mathlib's proved exponential-series remainder yields

\[
 |\exp x-P_n(x)|\le |x|^n\frac{n+1}{n!\,n}.
\]

`exponential_error` adds a checked rational residual between a witness e
and P_n(x). For SiLU, check e>=0, an exponential certificate
abs(e-exp(-x))<=E, and the rational division residual
abs(z-x/(1+e))<=beta. `silu_residual_error` gives

\[
 |z-\mathrm{SiLU}(x)|\le\beta+|x|E.
\]

`silu_polynomial_error` composes these checks, so neither exp nor SiLU
accuracy is assumed. This polynomial certificate has an explicit input
range and order; it is not a claim about the accuracy of an arbitrary
CUDA exponential over the whole real line.

### K4. Source causal SDPA from checked masses and a reduction residual

Let a_j be approximate masses with A=sum a_j>0 and b_j the exact
nonnegative masses with B=sum b_j>0. If sum abs(a_j-b_j)<=E,
`normalized_weights_error` proves

\[
 \sum_j\left|a_j/A-b_j/B\right|\le 2E/A.
\]

For stored values with norm at most M, a checked residual beta between
the stored output and the rational weighted sum sum_j (a_j/A)v_j gives
`average_residual_error`:

\[
 \left\|z-\sum_j(b_j/B)v_j\right\|\le\beta+(2E/A)M.
\]

`causalMass` retains exactly the original visible-prefix mask, including
the current token. `causalMass_weight` identifies its normalized masses
with the existing source `causalWeight`. `causal_average_residual` therefore
fills the SDPA-average field of `Execution.AttentionAccurate` directly.
Polynomial witnesses supply mass errors; the reduction residual compares
the actual stored output with an exact rational sum. A fused kernel need
not expose its hidden softmax weights or use the sequential reduction
order of C3's different adapter. These checks still have to be discharged
at each operation in a complete model trace.

### K5. Stored RoPE coefficients

The complex exponential polynomial at ix supplies a common real/imaginary
reference. If abs(x)/(n+1)<=1/2, its proved norm remainder is

\[
 \left\|e^{ix}-\sum_{m<n}(ix)^m/m!\right\|
 \le 2|x|^n/n!.
\]

`rotation_polynomial_error` transfers this to cosine and sine separately.
`stored_rotation_error` adds the rational residual of each stored table
entry. It assumes neither unit-circle identities for those entries nor
correctness of a second floating-point trigonometric implementation.
`head4_phases` proves that the actual base-10000 source frequencies at head
width four are 1 and 1/100. The captured position-one checks therefore
include frequency computation and table generation against exact source
phases, not against reused rounded tables.

### K6. Identified executions and exact-value Lean replay

`./run.py backend-contracts` captures seven fixed operation sites using
Python 3.15.2, PyTorch 2.15.0+cu132 and CUDA 13.2 on the RTX 3050 Laptop GPU:
CPU float32, CPU float64, and CUDA float32 compiled with
`max-autotune/fullgraph/static`. SDPA explicitly selects the MATH backend;
matrix float32 precision is IEEE, with TF32 disabled. Kernel event names
are retained in `review/backend_contract_check.json`. The CUDA run records
fused Triton RMS/L2, softmax, RoPE and SiLU kernels. The GPU reports too few
SMs for GEMM max-autotuning, so the requested compile mode uses its fallback
there. Reduction order remains backend-managed and is not assumed by the
output-residual certificates.

The RMS and L2 input is [1,2,3,4]. All three cases use RMS eps=2^-23 and
L2 eps=10^-12; in particular, the float64 comparison intentionally uses
the same RMS epsilon rather than the source's float64 default. SiLU uses
input 1/2. The last row of the two-token causal SDPA has actual scaled
scores [0,1/2] and distinct unit value vectors. RoPE uses the source
float32 table-construction dtype, even in the float64 case. The matrix
fixture is a nonzero 2-by-4 dyadic product.

`backend_contract_arithmetic.py` decodes every finite recorded binary
value exactly with rational arithmetic. The normalizer scales, order-12
polynomials, coordinate boxes of radius 10^-6 and exact matrix result are
fixed independently of those outputs. It rejects nonfinite values, changed
operation sets, wrong shapes and any violated check. All 21 site checks
pass. The rational exponential witness at -1/2 is
49583642701/81749606400; its analytic error is proved below 10^-10.

`./run.py backend-certificates` exports the actual binary values as rational
literals to `CapturedCPU32`, `CapturedCPU64` and `CapturedCUDA32`. Their
18 named theorems prove the following bounds with ordinary Lean tactics:

| Captured operation | Proved error bound |
| --- | --- |
| RMSNorm | Euclidean error <= 4e-6 |
| Clipped L2 | Euclidean error <= 3e-6 |
| SiLU at 1/2 | absolute error <= 2e-6 |
| Two-token causal SDPA | Euclidean error <= 3e-6 |
| Each source RoPE cosine/sine | absolute error <= 2e-6 |
| Dyadic matrix product | exact equality |

The tests also require the exported Lean literals to equal the frozen
capture, check capture/checker hashes and source hashes, and reject corrupted
coordinates, invalid layouts, missing sites and nonfinite results. The
analytic consequences are proved in Lean rather than inferred from
agreement with a higher-precision floating-point calculation.

Task #132 supplies these concrete primitive certificates. Task #135 below
collects and discharges them throughout one complete, nontrivial transformer
trace and checks the derived model budget against its margin. The fixed
probes above do not prove that a whole model satisfies
`ModelAccurate`, that the bound is useful for the constructed powered-division
weights, or that every future invocation of these kernels is accurate.
Learning conditions (#133) and exact-RoPE depth/width lower bounds (#134)
remain separate. No training or optimizer implementation changes here.

Validation: the full Lean build passes (13217 jobs). Individual axiom checks
cover all 42 new public theorems and use only propext, Classical.choice and
Quot.sound. The global audit reports 36 unchanged sorry, zero results resting
on a sorry and zero extra axioms. The regenerated index has 4120 modules and
21185 theorems, with zero vacuous results and zero placeholders; forbidden
and whitespace checks pass. All new modules build without warnings.

The 424-test lab suite passes with three skips. A final seven-test certificate
run also passes, including the export-consistency test added after full-suite
discovery. The 104-test optimizer suite reproduces the two existing
ScheduleFreeMuon allocation failures (float32/float64, 75 allocations of
8519680 bytes each); Schedule-Free AdamW passes. Consequently the overall
Python command exits unsuccessfully, matching the baseline recorded for
#130 and #131. Logs and the statement/source crosswalk are retained in
`.kanban/artifacts/132/`.

## Complete captured execution

L1--L5 discharge every operation premise for one complete compiled CUDA
invocation of the pinned source transformer. They also prove a limitation
of the current global error budget: it cannot certify this invocation's
decision margin, for any valid uniform operation allowance. This is a
proved obstruction in the sufficient numerical bound, not evidence that
the model computes the wrong decision. Task #135 records this result;
task #136 tracks a replacement bound using the observed norm ranges and
individual operation errors.

### L1. Capture the actual source forward

`./run.py model-trace-capture` imports the unchanged diff_gpt `GPT.forward`
at commit b98c665bd8faa30729fd368d257a13b6d58712a1, checking all five pinned
file hashes first. It uses width four, one head, one block, the source
SwiGLU width eleven, three tied token embeddings, context length two and
the input word [0,1]. The stored weights are deterministic dyadic values;
all five branch matrices are nonzero. Both attention and SwiGLU change the
second residual, as proved by `active_branches`. This is a finite execution
certificate, not a training run or a modular-division accuracy experiment.

The invocation uses Python 3.15.2, PyTorch 2.15.0+cu132, CUDA 13.2 and the
RTX 3050 Laptop GPU. Compilation requests max-autotune, fullgraph and static
shapes. SDPA selects MATH; float32 matrix precision is IEEE. RMS epsilon
is the source float32 value 2^-23, and clipped L2 retains epsilon 10^-12.
The original source constructs its RoPE tables on CPU in float32 before
the model moves to CUDA. Their exact mathematical phases are checked
independently, including the source base-10000 frequency calculation.

Observational hooks and functional wrappers retain 24 tensors from the
actual source call. The wrappers return each original operation's result.
The trace includes all RMS calls, the fused QKV projection, source reshapes,
Q/K rotations, SDPA output, clipped own-value direction, first-position
zero, head merge, both learned output projections, all SwiGLU operations,
both residual additions, final RMSNorm and tied logits. The shared embedding
and cosine/sine tables are also saved. Auxiliary scores are explicitly
computed from the stored Q/K values; they are not claimed to expose the
SDPA kernel's internal scores. The average certificate checks its output
against those auxiliary scores and the captured values.

Returning intermediates can affect compiler fusion. The claim concerns
this instrumented compiled invocation. Its six final logit bits match the
separately compiled uninstrumented forward in this capture; that numerical
comparison is retained as evidence, not used as a Lean premise. Observed
kernel names are retained in `review/model_trace_capture.json`. The GPU's
GEMM autotuning fallback is permitted; no particular reduction order is
assumed by the output-residual checks.

### L2. Check all local premises with exact arithmetic

`model_trace_arithmetic.py` decodes every finite captured value as its exact
binary rational. It checks the complete operation/parameter/table sets,
shapes, source dimensions, input word and epsilons. It rejects nonfinite
values, Boolean or string substitutes, missing sites and corrupt outputs.
There are 52 checked vector/scalar sites, in addition to the exact stored
embedding identification.

Each normalization has its own rational scale witness. Decimal square roots
only propose scales; exact rational squared residuals and coordinate boxes
verify them. Therefore no Decimal sqrt correctness is assumed. Matrix
products, source rotations, scaled dot products, elementwise products, XSA,
residual additions and final logits are checked against their actual stored
inputs using rational sums. Source QKV/head/half-split layouts are retained.

Order-12 polynomial witnesses certify all 22 SiLU coordinates and both
causal average rows. On abs(x)<=1, `exp_checked` bounds the analytic
exponential error by 10^-8 after a rational polynomial residual at most
10^-10. `silu_checked` and `average_checked` transfer those certificates
to the source activation and causal average. The approximate mass sum is
checked above 1/2 and each stored value coordinate below one in magnitude.
The actual source input ranges, rather than a uniform CUDA exp guarantee,
discharge these requirements.

Most rational coordinate boxes have radius 10^-6; widths up to sixteen
convert them to the common Euclidean operation allowance delta=10^-4.
Normalizer and nonlinear analytic remainders are included before that
conversion. The independently checked cosine/sine allowance is tau=2e-6.
These convenient common allowances are upper bounds, not measurements of
each primitive's maximum error.

### L3. A complete ordinary Lean certificate

`./run.py model-trace-certificates` rechecks the rational premises, saves
`review/model_trace_certificates.json`, and exports exact rational literals
and ordinary Lean proof scripts. Lean independently proves all numeric
premises and their analytic consequences. The exporter introduces no
axioms, native decision procedures or unchecked Python-derived equalities.
Tests require its output to equal the committed Lean source exactly.

`Complete.captured` is a full `Execution.ModelTrace`, linked at every
operation boundary to its actual stored input. `attention_accurate`,
`ffn_accurate` and `block_accurate` assemble the checked primitive results.
`captured_accurate` proves

```
ModelAccurate cfg weights (1/8388608) (1/1000000000000)
  angles tokens captured (1/10000) (2/1000000).
```

There is no remaining operation-accuracy hypothesis in this theorem.
`captured_logit_error` applies the existing complete-stack composition
theorem to the actual saved logits. `captured_decisions` proves that the
first token's strict winner is class zero and the second token's strict
winner is class two. These are statements about the captured outputs;
agreement with the exact-real source's decisions requires a useful
comparison bound, which the existing global one does not provide.

The capture process, PyTorch runtime, driver and hardware are not themselves
verified in Lean. The mathematical statements concern the exact encoded
data and the pinned source equations. The scope is one checkpoint and word,
not all weights, future invocations, context lengths or modular tasks.

### L4. An unavoidable obstruction in the existing uniform budget

The final residual addition at token one, coordinate two has exact error

\[
 |\widehat x_{\mathrm{out},1,2}
   -(\widehat x_{\mathrm{mid},1,2}+\widehat f_{1,2})|=2^{-24}.
\]

`residual_error_exact` checks this equality directly on the captured binary
rationals. A coordinate's absolute error is at most its Euclidean error,
so `allowance_floor` proves that **every** `ModelAccurate` certificate for
these same tensors must have delta>=2^-24. Thus the obstruction below cannot
be removed merely by choosing a tighter common allowance than 10^-4.

Let L=2/sqrt(eps), the existing `rmsLip`. `budget_ge_initial` proves that
the old depth recurrence always satisfies

\[
 E_D\ge E_0=L\,\delta+\delta.
\]

Each residual block gain is at least one and its new error charge is
nonnegative. At eps=2^-23, L^2=2^25. Each tied embedding has norm one, so
the existing final logit radius satisfies

\[
 R_v > L^2\delta\ge 2^{25}2^{-24}=2.
\]

This uses only the unavoidable residual error and the old recurrence's
uniform initial/final normalization charge. It holds before considering
any additional inflation from XSA's small clipping epsilon or learned
matrix norms. The true source final RMSNorm has norm at most two; hence
every exact source logit is between -2 and 2 and every class margin is
at most four. Two class radii strictly exceed four.

`no_global_margin` consequently proves that, for these weights and captured
tensors, no valid nonnegative table allowance and uniform operation
allowance can satisfy `MarginAbove`, at any residual vector or target class.
The earlier error theorem remains sound; its sufficient bound is too coarse.
The captured decisions and active branches remain proved.

### L5. Consequence for the flat-spectrum / transformer / C-RASP connection

`no_execution_below` connects the obstruction to the exact
`Spectral.ExecutionBelow` premise used by J6--J7. For every margin gamma<=4,
it is impossible to discharge this premise for the captured trace using
the old uniform budget. Increasing the independent counting precision
does not remove that execution-budget obstruction. No spectral recovery
assumption or optimizer convergence claim has been added.

The required change is numerical: use verified norm intervals for
normalization sensitivity, individual operation errors and exact stored
embedding identification, then propagate those bounds through the actual
source stack. Section M supplies this improved execution certificate for
task #136. The obstruction in L4 remains valid for the old uniform budget;
the new certificate uses a different, operation-dependent bound. Its
counting transfer retains the independent fixed-point conditions.

Validation: the full Lean build passes (13233 jobs), without warnings in the
new modules. Individual axiom checks cover all 56 new public theorems and
use only propext, Classical.choice and Quot.sound. The audit reports 36
unchanged sorry, zero results resting on a sorry and zero extra axioms.
The regenerated index has 4136 modules and 21241 theorems, with zero
vacuous results and zero placeholders. Forbidden and whitespace checks pass.
The certificate command reproduces its saved witnesses and Lean sources
exactly, and all five pinned source hashes still match.

All seven certificate tests pass. The final 432-test lab suite passes with
three skips; the 104-test optimizer suite reproduces the two existing
ScheduleFreeMuon allocation failures (float32/float64, 75 allocations of
8519680 bytes each). Schedule-Free AdamW passes. The overall Python command
therefore exits unsuccessfully for the same baseline failures recorded in
#130--#132. Validation logs and the source/statement review are preserved in
`.kanban/artifacts/135/`.

## Trace-dependent execution bounds

This extension closes the execution-bound gap exhibited in L4 for the same
frozen width-four, one-head, one-block, two-token CUDA invocation. It changes
neither the checkpoint nor the captured tensors, source epsilons or exact
RoPE phases. The new result proves agreement of all captured class decisions
with exact-real source arithmetic. It does not assert that a specific
fixed-point counting execution has already satisfied its separate range and
rounding conditions, or that this checkpoint solves modular division.

### M1. Normalization sensitivity at a recorded input

For dimension d>0, eps>0 and arbitrary real vectors x,y, write

\[
 R_\epsilon(x)=\frac{\sqrt d\,x}{\sqrt{\|x\|^2+d\epsilon}}.
\]

The asymmetric, globally valid secant estimate is

\[
 \|R_\epsilon(x)-R_\epsilon(y)\|
 \le \frac{2\sqrt d}{\sqrt{\|x\|^2+d\epsilon}}\|x-y\|.
\]

Here x is the recorded input. No lower norm bound on the unknown exact
input y is assumed. To prove this, put a=||x||^2+d eps and
b=||y||^2+d eps. The difference of square roots satisfies

\[
 |\sqrt b-\sqrt a|\,\|y\|\le\|x-y\|\sqrt b.
\]

Splitting the normalized difference into the scaled displacement and the
change of scale gives two copies of sqrt(d)/sqrt(a). The old global proof
substituted sqrt(a)>=sqrt(d eps) at this point; the new proof retains a.
`Adaptive.rms_secant` formalizes the estimate. A rational gain L>=0 is valid
whenever 4d<=L^2(||x||^2+d eps), an exact polynomial check on the recorded
coordinates. `rms_step` then gives L E+delta, where E is the previously
proved input error and delta is this normalizer's local residual.

For clipped normalization U_eps(x)=x/max(||x||,eps), eps>0,

\[
 \|U_\epsilon(x)-U_\epsilon(y)\|
 \le\frac{2}{\max(\|x\|,\epsilon)}\|x-y\|.
\]

`l2_secant` retains the first denominator in the original proof and uses
that max with a constant is 1-Lipschitz. The estimate covers zero inputs and
crossings of the clipping boundary. `l2_step` accepts the recorded check
2<=L max(||x||,eps). These are proved analytic estimates, not Jacobians
assumed to remain constant in a neighborhood.

### M2. Composition with distinct residuals and stored weights

Each operation keeps its own error allowance, verified at its own recorded
inputs. For an exact stored matrix A with ||A||<=L, the next error is
L E+delta. `matrix_checked` derives L from the squared Frobenius norm of
its entries by Cauchy--Schwarz. Q, K and V use separate submatrices of the
same fused source matrix, with exact selector identities. The captured
embedding table equals the reference table exactly.

The generic operation rules retain the following terms:

- RMS and clipped L2: the recorded gain from M1 times input error, plus
  the particular normalizer residual.
- Arbitrary RoPE: 2 tau R+E+delta, where R bounds the recorded normalized
  vector, tau bounds both stored coefficients against the exact phase,
  and delta bounds arithmetic applying the stored table.
- Scaled dot product: (2R+E)E/sqrt(d)+delta when both recorded vectors have
  norm at most R and both input errors are at most E.
- Causal average: 8 E_score R_value+E_value+delta_SDPA. The value norm is
  checked on the recorded tensor; the mask includes the current token.
- SiLU: (1+R_gate)E_gate+delta_activation using recorded coordinate bounds.
- Hadamard product: V E_activation+(A+E_activation)E_value+delta_product,
  where V and A bound recorded value and activation coordinates. The cross
  term remains. Squared-coordinate aggregation avoids repeating a
  sqrt(FFN width) loss for every product.
- Residual addition: the two input radii plus its own addition residual.

XSA uses the existing two-input estimate, including the excess norm of a
computed direction; its exact direction has norm at most one. Source head
merging and learned projection are retained. At position zero, the source
and captured attention branches are both exactly zero. That case is proved
separately, so its small post-normalization denominator cannot amplify an
error from another position. The rules hold at arbitrary finite dimensions
and can be composed repeatedly; the fully instantiated certificate below
concerns the actual one-block capture, not every depth or checkpoint.

### M3. Rechecking the frozen operations

`./run.py model-trace-refined` rechecks all 52 original residual sites with
separate allowances, the squared scale residuals at tolerance 10^-12, and
separate Q/K/V arithmetic. Seven generated Lean modules import the original
stored literals unchanged and independently prove the sharper contracts.
The SiLU, softmax and RoPE checks retain their analytic Taylor remainders.
No floating transcendental calculation is accepted as an exact reference.

Representative certified constants are:

| Operation or recorded quantity | Bound |
| --- | ---: |
| Stored Q and K matrix norms, separately | 1/8000 |
| Stored V matrix norm | 101/100 |
| Stored attention projection norm | 13/50 |
| Stored gate, value and FFN output norms | 21/50 each |
| Attention and FFN input RMS gains | 21/10 |
| Q/K RMS gains | 6000 |
| Own-value clipped L2 gain | 2 |
| Active attention output RMS gain | 60 |
| FFN output RMS gain, both tokens | 1500 |
| Final RMS gain | 3/2 |
| Stored RoPE coefficient error | 10^-7 |
| Q/K slice arithmetic error, separately | 4*10^-12 |
| V slice arithmetic error | 4*10^-8 |
| Initial normalized embedding error | 1/4000000 |
| First residual addition error | 0 |
| Second residual addition error | 12*10^-8 |

The actual late addition still has a coordinate error of 2^-24, exactly as
in L4. It is charged at that addition alone. It is not an assumed error in
the stored embedding, either Q/K projection, or an earlier normalizer.
Rejection tests include perturbations small enough to pass the old 10^-6
coordinate checks but fail the new pre-RMS, Q-projection or residual checks.

### M4. Source agreement and a derived numerical margin

`Refined.Source` exposes the exact-real operations of the pinned source.
`source_hidden` proves that this graph is exactly its actual hidden-state
forward, including both residual additions and all normalization placements.
The propagation theorems have no unproved operation, hidden-state error,
source-margin or decision hypotheses. The following conservative radii hold
at both captured token positions:

| Stage | Euclidean error bound |
| --- | ---: |
| Attention input RMS | 53/100000000 |
| Each normalized Q/K | 48/100000000 |
| Each rotated query/key | 6/10000000 |
| Causal average | 31/10000000 |
| Own-value direction | 135/100000000 |
| XSA head | 93/10000000 |
| Attention branch output | 146/1000000 |
| First residual | 147/1000000 |
| FFN pre-RMS | 31/100000 |
| Gate and value, separately | 131/1000000 |
| Activated gate | 164/1000000 |
| Hadamard product | 575/10000000 |
| FFN output projection | 242/10000000 |
| FFN output RMS | 364/10000 |
| Final residual | 366/10000 |
| Final normalized state | 55/1000 |

The tied embeddings have norm one and the recorded final dot products have
zero local residual. Thus `error_logits` proves, for every token i and class v,

\[
 |\widehat\ell_{i,v}-\ell_{i,v}^{\rm source}|\le1/16.
\]

The strict captured winners are 0 and 2. Every competing captured class is
at least 17/20 below its winner, verified directly on the stored rationals.
Subtracting the two logit radii gives at least 17/20-2/16=29/40; in particular
`source_margin` proves the uniform source margin 7/10. `decisions_agree`
proves equality of the complete Boolean class-test vectors, including every
rejected class. The earlier `no_global_margin` and `no_execution_below`
theorems are unchanged: they concern the old sufficient bound, not the
classifier or these new operation-dependent radii.

### M5. Reconnecting to flat spectrum and the counting model

`Spectral.executed_of_logit_bound` accepts any independently proved
classwise execution radii whose pairwise sums fit below the source margin.
`uniform_local_certificate` uses this interface in the existing J7 bridge.
The source margin can therefore come from the same spectral construction;
its hypotheses about recovered, separating characters have not changed.
The formula family is fixed before all input words and traces, and retains
the depth upper bound n_layers and exact source RoPE kernels. This result
asserts no strict nonperiodic depth hierarchy or optimization convergence.

`Refined.source_score` identifies the captured invocation's exact readout
with `Spectral.sourceLogit`. `captured_counting_bridge` supplies its proved
radius 1/16 and source margin 7/10 to the generic theorem. It no longer
requires the impossible old `ExecutionBelow` premise. The formula agrees
with every captured final-token class decision provided the independently
rounded counting execution satisfies `TraceInRange` and `CountingBelow`;
the softmax precision inequality is also explicit. No concrete integer
width/fractional precision for this active checkpoint is certified here.
Those fixed-point premises are the remaining numerical work for an
unconditional captured-to-counting instance. The exact source/captured
agreement in M4 is already unconditional for the recorded tensors.

Validation: the full Lean build passes (13257 jobs), with no warnings in the
new modules. All 119 new public theorems use only propext, Classical.choice
and Quot.sound. The repository audit reports the unchanged 36 sorry, zero
results resting on a sorry and zero extra axioms. The regenerated index has
4160 modules and 21360 theorems, with zero vacuous results and placeholders.
All eleven original/refined trace tests pass, including exact export replay
and the tighter residual rejection cases. All five pinned physical source
hashes still match. Logs and the statement/source review are preserved in
`.kanban/artifacts/136/`.

The final 436-test lab suite passes with three skips. The 104-test optimizer
suite reproduces the two existing ScheduleFreeMuon allocation failures in
float32/float64 (75 allocations of 8519680 bytes each); Schedule-Free AdamW
passes. Consequently the overall `./run.py test` exit status is unsuccessful
for the same baseline failures as L5, while all eleven trace certificate
and rejection tests pass. Task #137 records the remaining fixed-point range
and precision proofs for this active captured checkpoint.
