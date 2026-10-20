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
