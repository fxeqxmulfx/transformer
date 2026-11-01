# Residual connections in current DeepSeek, GLM and Kimi models

Checked against official releases and downloaded primary sources on 2026-10-20.
This change formalizes the residual sections and their mathematical properties.
It does not formalize complete model training, attention kernels, benchmark
results, or the other sections of the technical reports.

| Model | Residual mechanism | Distinguishing detail |
| --- | --- | --- |
| DeepSeek V4.1-Flash | Single-Pass mHC with four streams | The read coefficients come from the previous sublayer; current coefficients still determine skip mixing and write-back. |
| GLM-5.3-Flash | mHC with four streams | Current read coefficients, separate attention/MLP connections, and an unweighted mean for the final stream collapse. |
| Kimi K3 | Block Attention Residuals | Learned pseudo-queries attend to completed blocks, the embedding, and the current partial block; keys are RMS-normalized and values remain unnormalized. |

The official GLM repository distinguishes GLM-5.3, which retains the GLM-5.2
base, from GLM-5.3-Flash, which introduces a new base and mHC. The latter is
the relevant new residual architecture. The Flash model card still links the
older GLM-5 report; that report is downloaded for context and is not treated
as evidence for the later Flash architecture. Sources:
[official GLM repository](https://github.com/zai-org/GLM-5),
[Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash).

The DeepSeek version follows the
[official V4.1 announcement](https://deepseek.com/en/news/deepseek-v4-1-flash/).
Kimi K3's residual architecture and report are confirmed by its
[official open-source release](https://www.kimi.com/news/kimi-k3-open-source).
No architecture is inferred from model naming or an unconfirmed API refresh.

## Downloaded manuscripts

All manuscripts are under the root's gitignored `papers/`, shared with the
working checkout. The four arXiv folders are pinned to the versions actually
served during download. Their citations are discoverable by `./run.py papers`.

| Manuscript | Local source | Formalization scope |
| --- | --- | --- |
| [Hyper-Connections](https://arxiv.org/abs/2409.19606v3) | `papers/arXiv-2409.19606v3/iclr2025_conference.tex` | §2.1 residual operator, and the §3.1 standard-residual specialization. |
| [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880v2) | `papers/arXiv-2512.24880v2/main.tex` | §1, Eqs. (1)–(4), and §§4.1–4.2. |
| [Attention Residuals](https://arxiv.org/abs/2603.15031v1) | `papers/arXiv-2603.15031v1/3-attnres.tex`, `4-infra.tex` | §§3–4.2 residual aggregation and exact online merging. |
| [GLM-5](https://arxiv.org/abs/2602.15763v2) | `papers/arXiv-2602.15763v2/0_main.tex` | Downloaded background; no attribution of Flash mHC to this older report. |
| [DeepSeek V4.1-Flash technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/DeepSeek_V41_Tech_Report.pdf) | `papers/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf` | §2.4.1, Eqs. (2)–(6), and the activation traffic model. |
| [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf) | `papers/Kimi-K3/k3_tech_report.pdf` | §2.2, Eqs. (8)–(10), and concrete block partition. |

PDF SHA-256 values:

| PDF | SHA-256 |
| --- | --- |
| DeepSeek V4.1-Flash | `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d` |
| Kimi K3 | `86fb82a63ced501f0c3f4f404c0c6fa88a7a6cfac17aae81fd1a8f455998067c` |

The non-arXiv reports can be restored directly from their pinned URLs:

```bash
mkdir -p papers/DeepSeek-V4.1-Flash papers/Kimi-K3
curl -fL 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/DeepSeek_V41_Tech_Report.pdf' -o papers/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf
curl -fL 'https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf' -o papers/Kimi-K3/k3_tech_report.pdf
```

## Pinned implementation evidence

Only small source/configuration files were downloaded, with no checkpoint
weights. Original files remain in `papers/` and are not vendored into the
implementation or imported into the Python lab.

| Source | Revision | Local evidence |
| --- | --- | --- |
| DeepSeek V4.1-Flash official checkpoint | `314113f8a7dadb388215396ba5fbee2c610a04d6` | `papers/DeepSeek-V4.1-Flash/config.json`, `model.py`, `kernel.py`, `README.md` |
| GLM-5.3-Flash official checkpoint | `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` | `papers/GLM-5.3-Flash/config.json`, `README.md` |
| GLM-5.3 official checkpoint | `ba32f8c19c8738e2c57c71798c601d477a7bf844` | `papers/GLM-5.3/config.json`, `README.md` |
| Kimi K3 official checkpoint | `f831ab66814297da540d832a5235f8e904f29d06` | `papers/Kimi-K3/config.json`, `modeling_kimi_linear.py`, `README.md` |
| Hugging Face Transformers | `21848ed2d35372f6e78d48203df1e6154361e0b6` | `papers/GLM-5.3-Flash/modeling_glm5_next.py`, `modular_glm5_next.py` |
| Kimi K3 report repository | `3cb39dfd32e51c3328e2e4b4af21341247d06c43` | `papers/Kimi-K3/k3_tech_report.pdf` |

Checkpoint source files use
`https://huggingface.co/<owner>/<model>/resolve/<revision>/<path>`.
DeepSeek's two Python files come from `inference/model.py` and
`inference/kernel.py`; the local basenames are shortened. The Transformers
files come from `src/transformers/models/glm5_next/` at the pinned commit.
The Kimi report uses the pinned GitHub raw URL above.

DeepSeek's config gives `hc_mult=4`, `hc_sinkhorn_iters=20`,
`hc_eps=1e-6`, and hidden width 5120. GLM Flash uses the same residual
width, count and epsilon, with hidden width 4096. Kimi has 93 decoder
blocks and `attn_res_block_size=12`, giving eight groups, a final group
of nine decoder blocks, and nine stored output sources including the
embedding. A Kimi decoder block contains attention and MLP sublayers,
each with its own aggregation query and normalization.

## Statement correspondence

| Lean module under `src/Transformer/` | Source and checked content |
| --- | --- |
| `Residual/Basic.lean` | mHC §1, Eqs. (1)–(4): ordinary residual unrolling, weighted vector aggregation, read/skip/write operator, matrix composition, and valid norm-bound witnesses. HC §2.1 uses the transposed `A_r` convention; the common operator names the resulting output-by-input matrix directly. |
| `MHC/Section4_Stability.lean` | mHC §4.1: double stochasticity as an explicit hypothesis; chronological product closure, spectral norm bound in the L2 operator norm, constant-stream and sum conservation, per-stream convex bound, and gated single-stream reduction. |
| `MHC/Section4_Sinkhorn.lean` | mHC §4.2: elementwise exponential initialization, the printed `row(column(M))` iteration order, positivity, exact final row normalization, and a positive finite-step counterexample to exact double stochasticity. |
| `MHC/Section4_Parameterization.lean` | mHC §4.2: affine logits after normalization, separate dynamic scale/static bias, sigmoid read gates, twice-sigmoid write gates, and a branch-dependent complete-block norm bound. |
| `MHC/Section4_Regularized.lean` | Both published kernels: row softmax plus epsilon, one column normalization and nineteen row/column cycles, epsilon in each denominator, transposed skip orientation, GLM current read, and final mean head. |
| `DeepSeekV41/Section2_SinglePass.lean` | Report §2.4.1: original current-read Eq. (2), delayed-read Eq. (6), explicit initial read map, carried read state, a counterexample to unconditional equality with the unshifted block, and traffic arithmetic. |
| `AttentionResiduals/Section3_Attention.lean` | AttnRes §3.1 and Kimi K3 §2.2: learned query against RMS-normalized keys, raw values, positive normalized weights, convex norm bound, zero-query averaging, and logit-shift invariance. |
| `AttentionResiduals/Section3_Blocks.lean` | AttnRes §3.2 and Kimi K3 Eq. (10): embedding always included; no artificial zero source on entry; partial raw sum only after the first branch; exact accumulation and K3's concrete partition. |
| `AttentionResiduals/Section4_OnlineMerge.lean` | AttnRes §4.2, Algorithm 1: unnormalized numerator, shifted exponential mass, rescaling to a common maximum, positive denominator, and equality of two-phase evaluation with attention over the same source union. |

Branch functions include their own input pre-normalization and attention/MLP
transformations as explicit arguments. Their training dynamics and norm bounds
are not hidden in definitions. Coefficient predictors are likewise explicit;
the affine-map model begins with the normalized token vector, while the
implementation models begin with post-projection logits. Kimi's RMSNorm key
formula includes the learned gain and stabilizer directly.

## Limits and corrected readings

The ideal mHC skip product is nonexpansive. It need not be an isometry or
preserve every nonzero signal: averaging the streams `(1,-1)` erases them.
The all-equal stream mode is preserved, and skip-only stream sums are
conserved. Write-back adds a branch-dependent contribution; a complete
block can amplify its input even with an identity skip matrix. Input-dependent
gate predictors also prevent a fixed-map bound from being a full-Jacobian
or global Lipschitz theorem.

The mathematical paper iteration and released kernels have different
normalization orders and regularization. With positive epsilon, the final
column sums in the released real algorithm are strictly below one, including
at the configured twenty steps. Exact double stochasticity cannot be assumed
for that implementation. Infinite Sinkhorn convergence, approximate spectral
bounds for finite iterates, and floating-point error bounds are not proved.

AttnRes bounds the aggregation of its source vectors. A block source is a raw
sum and may grow with block size. The proof therefore bounds complete and
partial block sources explicitly. The sentence in AttnRes §3.2 claiming that
one block reduces to ordinary residual connections is not an unconditional
equality of the printed formulas: two equal unit sources are averaged to one,
whereas a standard residual sum gives two. The formalization keeps the
printed block equations and proves this counterexample.

The online-merge proof defines the statistics unambiguously: `o` is the
unnormalized shifted numerator and `ell` is the shifted exponential mass.
These are the quantities required by Algorithm 1's displayed fractions,
regardless of its loose “Return LSE” comment.

## Validation

The dedicated `scripts/ResidualAxioms.lean` prints the axioms of every new
named theorem. The global audit also covers all definitions and generated
declarations. New proofs contain no `sorry`, assumed paper claims, or
nonstandard axioms; existing unfinished statements are retained.

Commands:

```bash
lake build
./run.py audit
lake env lean scripts/ResidualAxioms.lean
./run.py index
./run.py forbidden
./run.py test
./run.py papers --dry-run
```

Strict lint and the whole-project kernel replay remain optional under the
repository's existing instructions. New module builds must emit no warnings.

## Normalized depth attention (2026-11-01)

The user requested a Q/K-normalized depth-attention variant before changing
the arithmetic transformer. The paper's pseudo-query is an unrestricted
learned vector, and its key RMSNorm has a learned gain. The original
aggregation theorems remain correct. A uniform score bound cannot be
transferred from the supplied token-attention model to that formula.

`AttentionResiduals/Normalized.lean` makes the distinction explicit:

- **ND1:** `source_score_unbounded` fixes width1, epsilon1, gain1 and key1.
  Choosing query `(B+1)*sqrt(2)` makes the score `B+1`, for every real B.
  `source_gain_unbounded` instead fixes the normalized query `1/sqrt(2)`
  and chooses gain `2*(B+1)`. Thus normalizing Q alone does not control an
  unbounded affine gain applied after key normalization.
- **ND2:** `qkDepthScore` applies parameter-free RMSNorm to both operands
  and divides their inner product by `sqrt(d)`. With d>0 and epsilon>0,
  `qkDepthScore_bound` proves absolute score at most `sqrt(d)`, including
  zero operands. A temperature with absolute value at most T gives the
  explicit cap `T*sqrt(d)`; an unbounded temperature is excluded.
- **ND3:** `qkDepthScore_lipschitz` bounds score changes by
  `2/sqrt(epsilon) * (norm(delta_query)+norm(delta_key))`. This global
  estimate becomes large for small epsilon. A bounded score is not a
  guarantee of small gradients or invariance under arbitrarily small scales.
- **ND4:** `qkAttentionResidual` keeps the original source values. Its norm
  stays within a common bound on those values. This does not bound the
  transformations that produced them or prove information preservation.
- **ND5:** a normalized depth weight is at most
  `exp(2*sqrt(d))/N` for N nonempty sources. Preventing arbitrary sharpening
  also imposes a retrieval limit at fixed width; this is a necessary local
  condition, not a construction or optimizer-convergence theorem.

The companion diff_gpt module `domain/model/attention_residual.py` uses
these exact finite-epsilon equations, with no learned gain or temperature.
Its default query has unit RMS, avoiding the zero-norm initialization region.
Independent float64 forward equations, every query/value gradient and
finite-difference gradient checks pass, including zero and extreme scales.
The known-solution routing probe has now completed21 cases; all saved states
replay independently on CPU. The portable summary is in
`experiments/modular_composition/residual_probe_summary.json`; full curves and
replay code are committed in diff_gpt at e13cf13 under
`research/residual_attention/`. Q/K, at unit RMS and the fixed wrong direction,
solves all three data/minibatch seeds. At query RMS1e6, key-only gradients are
exactly zero in FP32. Normalized scores remain bounded by4, but normalized
query gradients are only about8e-9 and all three oversized-query cases remain
stuck after2000 updates. Zero-initialized key-only controls solve the task too.

Thus ND2 does not imply trainability from arbitrary parameter scales. ND3 is
an upper sensitivity bound, not a positive gradient lower bound. Unit-RMS
initialization is the tested correction in the implementation; it is not a
formal optimizer guarantee. This comparison changes query normalization and
score scaling together. It is a query-only known-solution mechanism check,
not evidence that replacing full-transformer residuals improves div97.
The declared full-model residual cohort remains a separate task (#145).

The source repository at b98c665b already has a div97 generalization test:
W128 D1 H4, padded vocabulary128, batch512 with replacement, learning
rate.01, decay1, seeds11/17/23,1000 updates, evaluation every10. Its ledger
records first99% at400/430/360 updates. The new Lean-lab screen uses
different width, rate, decay, vocabulary and sampling, so its P97 result
is not a reproduction failure of that regression test. Q/K score tests
in that repository check finite-epsilon equations and approximate scale
invariance. They do not establish a universal generalization theorem.


## Repeated-input numerical audit (2026-11-01)

The fixed first512 heldout rows of each P97/P193 pair case expose a separate
numerical problem in the original source XSA. For identical input tokens,
all projected values initially coincide. Their attention average is the same
value, regardless of Q/K scores or RoPE. Above the L2 clipping threshold,
XSA should remove it exactly. The original FP32 subtraction instead leaves a
small nonzero vector, which the following Peri-RMSNorm can amplify repeatedly.
This is distinct from the learned pseudo-query saturation in depth attention.

The P97 deep division checkpoint has initial-state gradient RMS2714 on its
eight repeated-token heldout rows, versus.000860 on the504 other rows. The
position-summed gradient RMS on those repeated rows is only.00149, and the
maximum parameter-gradient RMS in the full batch is.0309. The same token
uses one shared embedding; independent position derivatives largely cancel.
Thus the ambient derivative alone is not evidence of exploding optimizer
updates or of a faulty residual sum. Diagnostics use the same512-row mean
loss and do not silently renormalize the smaller stratum.

FP64 replay preserves exactly the averaged FP32 parameter values, rotary
tables and RMS epsilon2^-23. Letting RMSNorm choose its FP64 epsilon would
change the model. In the original deep models, repeated-input logits differ
by5.33--8.09 between precisions, changing seven predictions across the four
fixed batches. Even the original FP64 computation has up to.000943 logit
difference from the centered FP64 evaluation. All differences, gradients and
counts are retained in `experiments/modular_composition/precision_diagnostics.json`.

The correction centers every value at an anchor c before attention. For a
normalized row alpha and own value v, write

```
delta = sum_j alpha_j * (v_j - c) - (v - c)
u = v / max(norm(v), epsilon)
output = delta - inner(delta, u) * u + own_remainder
own_remainder = 0                              if norm(v) >= epsilon
                (1 - norm(v)^2 / epsilon^2)*v  otherwise
```

`SequenceModel/Attention/CenteredXSA.lean` proves: C1 weighted centering;
C2 the exact clipped own-value remainder; C3 equality with the original XSA;
C4 equality for the actual normalized rotary causal head, including the
explicit first-position zero; C5 constant-value annihilation above clipping.
The below-epsilon term is essential: deleting it would change the model.
The formulas agree at the clipping boundary, while autodiff's chosen branch
there also needs testing; the implementation preserves F.normalize's actual
clamp backward convention. These are six proved identities, not a numerical
error bound or a convergence theorem. Concrete examples witness hypotheses.

The diff_gpt implementation centers values without altering cached raw values,
handles shared prefixes and chunked decoding, and retains `centered_xsa=False`
for legacy floating-point replay. Q/K normalization, RoPE, attention weights
and trainable parameter counts are retained. Centering cached prefixes adds
value transforms and temporary storage; no cache-speed improvement is claimed.
The frozen lab learning trajectories use the original implementation throughout.
`centered_replay.py` applies the correction only to diagnostic copies.

After correction, every repeated-input first attention output is exactly zero
in both precisions. Deep repeated-input logit differences fall below5.4e-6;
all eight batches have zero FP32/FP64 prediction disagreements with the centered
formula. This supports the numerical correction on the tested inputs. It does
not establish improved learning, global conditioning or residual superiority.
The unchanged long trajectories still test whether depth needs more updates.

The subsequent **parameter-gradient** check rejected centering alone. In the
lab prototype, its deep FP32 gradient RMS error reached0.827 against FP64,
although the forward was accurate. The actual diff_gpt implementation exposes
the same problem, reaching2.284. Independent position derivatives become large
and cancel poorly when accumulated into shared parameters. These are retained
failures, not a reason to treat the output-only test as sufficient.

The complete correction shares the first-position state after each block on
still-constant token prefixes. C6 says actual attention vanishes for a common
state whose value heads are above clipping. C7 gives the common FFN update.
C8 propagates the common state through every finite layer; C9 proves equal
tied readouts. C10 witnesses the guard at clipping zero. C11 restricts the
head result to visible constant prefixes, allowing arbitrary future values;
C12 lifts it to multihead attention; C13 proves a full block's prefix output
equals its first output. C14 propagates the prefix equality through every finite depth while allowing
arbitrary future tokens, and C15 carries it to the tied readout. These eleven
additional statements are in
`ConstantTokens.lean` and `ConstantPrefixes.lean`. The runtime guard uses the
actual fused-projection values and requires **strictly above** epsilon in every
head. A failed guard permanently disables sharing for that row. The guard
has no extra matrix multiplication. A clipped-value test shows why the
unconditional shortcut would change the function.

These proofs establish exact source-state equalities, with arbitrary rotary
phases. They do not establish floating-point guard reliability, accuracy of
arbitrary independent-input derivatives, or an optimizer convergence theorem.
For the shared-parameter repeated-token map, the common path avoids the
measured cancellation. Cache forwards use centering without this token-prefix
path; no uniform cache-gradient stability guarantee is claimed.

`parameter_precision.py/json` replay **actual diff_gpt**, including its
in-place first-position handling, on all eight frozen checkpoints. Original,
centered-only and guarded versions are each evaluated in FP32 and FP64 with
identical parameters/tables and RMS epsilon. Guarded FP32 parameter-gradient
RMS errors are1.78e-10--2.57e-8; maximum logit differences over all512 rows
are1.14e-5--1.88e-5, with no changed predictions. The corresponding lab
implementation has identical logits and parameter-gradient RMS differences
below5e-9. The earlier prototype audit is retained separately; its different
backward accumulation order must not be mistaken for the actual-source audit.
This validates measured numerical accuracy, not better generalization.

Validation: all17 new Lean theorems build in13262 jobs; audit retains36 sorry,
zero results resting on them and zero extra axioms. INDEX records4165 modules,
21392 theorems, zero vacuous/placeholder statements. Forbidden patterns and
explicit axiom checks pass. The affected diff_gpt CPU suite passes123 tests
with14 skips, including clipping, independent equations/all gradients,
finite differences, arbitrary suffixes, shared-parameter repeated prefixes,
cache shapes and division-protocol checks. The required lab suite passes446
plus104 optimizer tests (13 and14 CPU-only skips). CUDA compile and learning
regression checks remain separately recorded; the long frozen cohort is not
modified by this correction.


## Residual diagnostics after long training (2026-11-01)

All eight pair continuations finish100000 updates with the original residual
and XSA arithmetic. The same first512 train and heldout rows are replayed at
2000 and100000 updates. Dataset fingerprints agree, and manual unrolling
reproduces source logits bitwise in all32 batches. Full measurements and a
compact comparison are in `experiments/modular_composition_long/` as
`residual_diagnostics.json` and `residual_comparison.json`.

| Heldout batch | Final state RMS, early → long | Largest parameter-gradient RMS, early → long |
| --- | --- | --- |
| Division P97 W256D1 | .872 → .890 | .0331 → 2.74e-9 |
| Division P97 W128D4 | 2.083 → 2.649 | .0309 → 9.96e-8 |
| Mixed P97 W256D1 | .983 → .955 | .0290 → .0446 |
| Mixed P97 W128D4 | 2.341 → 1.365 | .0278 → .0420 |
| Division P193 W256D1 | .873 → .794 | 2.60e-6 → 1.17e-8 |
| Division P193 W128D4 | 1.990 → 2.274 | .0335 → 2.98e-8 |
| Mixed P193 W256D1 | 1.146 → 1.013 | .0189 → .0274 |
| Mixed P193 W128D4 | 2.148 → .705 | .0311 → .0447 |

RMS here averages per-row answer-position state RMS; each parameter gradient
is for the same full-batch mean CE, and the table takes the largest parameter
RMS. The source's deep mixed-model ambient state-gradient RMS remains large:
159 for P97 and229 for P193. Shared-parameter gradients are much smaller;
large independent-position derivatives must not be called an equally large
optimizer update. Every repeated/distinct-token stratum is retained.

Ordinary residual addition is compatible with late100% division accuracy in
both depths. Small division endpoint gradients accompany nearly zero loss;
they are not evidence of a failed optimizer. Mixed endpoints fit training
rows but remain near chance on heldout rows, with nonzero heldout parameter
gradients and no observed forward state-norm explosion in these batches.
These observations do not identify the causal reason for failed generalization.
Source-faithful residual alternatives remain task#145; paired retraining with
the numerical correction is task#148. Neither improvement is assumed from
aggregation bounds, normalized scores, or the corrected exact algebra.


## Query coordinate conditioning (2026-11-01)

Task #150 repairs the remaining huge-query failure of the proposed Q/K depth
attention variant. The source AttnRes paper uses key-only normalization; this
is our explicitly separate correction, not a statement attributed to that paper.
The original synthetic probe at diff_gpt `386ae01` found that Q/K normalization
bounds scores but leaves a raw query of RMS 1e6 almost unable to change direction.
Parameter-free RMS alone therefore does not establish useful optimization.

Write RMS_e(v) = v / sqrt(mean(v^2) + e) at positive width. The Lean definition
uses Euclidean norm and totalized real division/square root, including dimension
zero. With positive coordinate scale s, physical query q = s*u, and positive
rechart factor c, the precise claims formalized here are:

- **QC1:** RMS_(s^2 e)(s*u) = RMS_e(u), for any real e and any finite width.
- **QC2:** RMS_(e/s^2)(u) = RMS_e(s*u). Keeping e fixed while shrinking only
  the stored query would change this finite-epsilon function.
- **QC3:** score(e,s,u,k) = inner(RMS_(e/s^2)(u), RMS_e(k))/sqrt(d) equals
  the original Q/K score of s*u and k. The change (s,u) to (s*c,u/c) preserves
  every score and hence softmax aggregation of the original raw values, for
  any finite source set. The key epsilon and score divisor do not change.
- **QC4:** when d>0 and e>0, absolute score is at most sqrt(d), in every
  positive frame. This supplies no positive lower bound on a loss gradient.
- **QC5:** if a scalar loss has Frechet derivative D at s*u, the derivative
  of u -> loss(s*u) is s*D. Differentiability is an explicit hypothesis and
  this statement holds for any real s. No backward is detached or substituted.
- **QC6:** use the actual Schedule-Free reference average x, base z and
  training query y = beta*x + (1-beta)*z. Dividing both x and z by c divides
  y by c. The history stores c^2 times the old raw gradient second moment,
  with unchanged averaging total and maximum learning rate.
- **QC7:** recovering x from y,z and packing x,z into y,z both commute with
  this linear rescaling. These algebraic identities use the existing source
  definitions and totalized inverse, so need no extra beta or c assumptions;
  the runtime uses beta=.9 and positive c.
- **QC8:** for the actual exponential second-moment recurrence, updating
  history c^2*v with gradient c*g gives c^2 times the original next moment.
  No claim equates subsequent AdamW steps at unchanged scalar learning rate.
- **QC9:** let M be the maximum of the three current RMS magnitudes of y,z,x.
  For a threshold T>=1, choose c=M when M>T, otherwise c=1. The factor is
  positive. Each input magnitude divided by c is at most T. For actual
  vectors RMS(v/c)=RMS(v)/c, so all three coordinates satisfy the bound
  immediately after the change. Small and zero states remain unchanged.

`AttentionResiduals/QueryCoordinates.lean` proves QC1--QC5; the dependent
`QueryConditioning.lean` proves QC6--QC9 using the existing Schedule-Free
AdamW definitions. Concrete examples instantiate every hypothesis-bearing
statement. The conditioner is a separate operation before differentiation,
following zero_grad(set_to_none=True), and changes the complete y/z/moment
state and the model's nontrainable scale. Training/evaluation mode, clocks
and sampler history remain intact. Neither RMS has trainable gain or bias.

This preserves the represented function at the change of coordinates. Ordinary
Schedule-Free steps in the new coordinates intentionally change the physical
learning rate. It is an optimization-geometry correction, not equality of the
old and new trajectories, a convergence proof, or a finite-precision guarantee.
The immediate real coordinate bound does not bound future updates or physical
query norms. Overflow/underflow beyond the tested range remains a limitation.

Before new learning, the separate diff_gpt `research/residual_conditioning/`
protocol declares 24 paired fresh cases (scales 0, 1e-8, 1, 1e6; seeds 11,17,23;
2000 updates) and six continuations from all three saved huge-query failures
(10000 more updates with complete history). The original implementation is an
unchanged control. Every curve and endpoint is retained, using Schedule-Free
AdamW and Python3.15/Torch2.15 CUDA max-autotune. The separate 900-second cap
does not reopen the completed two-hour transformer cohort. Full-transformer
residual comparisons remain task #145.


Validation before the retry: all15 new theorems build; explicit axiom lists use
only propext, Classical.choice and Quot.sound. The full build has13264 jobs;
audit retains36 existing sorry, zero dependent results and zero extra axioms.
INDEX has4167 modules and21407 theorems, zero vacuous/placeholder statements;
forbidden checks pass. The affected diff_gpt suite passes166 tests, with one
existing skip. Three compiled FP32 repetitions agree with a separate FP64 execution to
2.67e-7 in output and8.28e-9 in query gradient. The state kernel compiles but
Torch skips CUDA graph capture for its mutated optimizer buffers.

The full repository verification runs452 lab tests successfully (three skips).
Its104-test optimizer stage has the same two ScheduleFreeMuon CUDA allocation
failures already recorded in the earlier self-review, at FP32 and FP64. The
unchanged root source reproduces both failures in the targeted allocation test;
there are no Python/optimizer changes in this Lean worktree. The separate CPU
optimizer check passes104 tests with14 GPU-only skips. This is not a fully green
CUDA verification result; the Schedule-Free AdamW checks used here pass.


The unchanged retry completed all30 declared cases in105.05 seconds; the first
attempt took27.62 seconds before CUDA capture failed. The complete raw outcomes
and failure trace remain in diff_gpt `research/residual_conditioning/`, committed
at `24b3b49`. The fixed query direction is shared; seeds vary data and minibatches.

| Cases, three seeds per variant | Original success | Conditioned success |
| --- | ---: | ---: |
| Zero queries, 2000 updates | 3/3 | 3/3 |
| Tiny queries, 2000 updates | 3/3 | 3/3 |
| Unit queries, 2000 updates | 3/3 | 3/3 |
| RMS1e6 queries, fresh2000 updates | 0/3 | 3/3 |
| All saved RMS1e6 failures, 10000 more updates | 0/3 | 3/3 |

Every corrected huge-query case reaches100% heldout routing, versus0% for each
unchanged counterpart. Fresh corrected heldout MSE is1.81e-9--2.72e-9; resumed
corrected MSE is8.52e-15--9.24e-15. Unchanged huge-query MSE stays4.216--4.226.
The immediate coordinate gradient is about1e6 times larger, as QC5 predicts;
the physical-query gradient is not thereby increased. These are actual ordinary
autodiff gradients, and all original optimizer history is retained. Later chart
changes also occur: unit/fresh-huge/resumed-huge runs have2/3/4 events respectively.

Independent CPU FP64 physical-query equations replay every initial, corrected
initial and final state. Counts and success classifications agree exactly, with
maximum scalar metric error6.28e-7. The largest absolute gradient-norm error is
1.67e-4 at norm1463.36 (relative1.14e-7); all differences remain in the report,
without a near-zero relative-error claim. All source states, data fingerprints,
optimizer clocks and sampled minibatch streams pass. A committed CUDA regression
also checks independent physical equations and both query/value gradients; the
final affected GPT suite passes167 tests with one existing skip.

This closes the demonstrated raw-query conditioning gap in the synthetic task.
It does not prove that normalized scores ensure learnability, nor that these
residuals outperform ordinary addition in the full transformer. The repaired
variant joins task#145 as a separately labeled control, preserving key-only
source AttnRes as a different architecture. No new modular learning or old long
cohort continuation occurred. Compact evidence is `residual_conditioning.json`.


Requested author self-review: the same assistant rechecked submitted transformer
`9ec890da` and diff_gpt `24b3b49`, without delegation. Lean build, audit, regenerated
index comparison, forbidden patterns and all15 axiom lists pass again; QC1--QC9
and the source state mapping were reread. The167-test affected GPT suite passes
again (one existing skip). Replaying all30 saved cases reproduces the entire
committed results JSON byte-for-byte. Task#150 is approved as self-reviewed,
not independently reviewed. The existing Muon CUDA allocation failures and the
full-transformer/learning-theory boundaries above remain explicit.
