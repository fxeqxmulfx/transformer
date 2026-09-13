# todo

- [ ] `ALM/SAHead.lean` — the lookup head is an ordinary attention head (§1.1)
- [ ] `ALM/PlanarHead.lean` — the hull answers any 2D head, not only the lift (§1.2)
- [ ] `k`-sparse softmax: the truncation bound only, not the `O(k + log n)` (§1.3)
- [ ] QK-norm obstructs the lookup head — state both halves (§1.4)
- [ ] `GPTMini/QKNorm.lean`: `rmsNorm q = √head_dim • normL2 q` (§2.1)
- [ ] `GPTMini/Config.lean`: GQA, `n_kv_heads ∣ n_heads` (§2.1)
- [ ] `XSA.lean`: restate or delete `attention_similarity_bias_observation : True` (§2.2)
- [ ] `GPTMini.lean`: the dangling "Phases 5–6 of `todo.md`" (§3)

The sections below are the reasoning behind these, not further items.

## 1. ALM and ordinary attention

`Transformer.ALM` is complete (333 declarations, no `sorry`, no vacuous statement)
and topologically an island: `src/Transformer/ALM/Defs.lean` imports only
`Transformer.Basic`, and nothing but the root aggregator imports `Transformer.ALM`.
Its head is written out inline at every use site,

    ∑ j, (exp (β * score q (K j)) / ∑ k, exp (β * score q (K k))) • V j

(`ALM/SoftmaxValue.lean`, `ALM/HullHead.lean`), and is nowhere identified with
`Transformer.XSA.SAOutput` (plain causal self-attention through
`Q, K, V : ParamMatrix d`) or with `Transformer.GPTMini.attnOutput`.  "This is a
standard softmax head" is prose in the docstrings, not a theorem.

The authors of the machine (Percepta, *Can LLMs Be Computers?*, 2026-03-11)
state two things the formalization could carry:

> This means the geometric fast path is not limited to our executor construction.
> In principle, it can accelerate any transformer with 2D heads at decoding time,
> replacing full linear scans with efficient geometric retrieval.

> These models could be useful in several modes: as a dedicated fast path paired
> with a slower, more general model; as part of a fast/slow hybrid architecture
> inside a single system; or as a speculative execution model [...]. In such a
> hybrid system the language model would plan and reason, while the execution
> component would run algorithms.

plus: hardmax is not fundamental — `k`-sparse softmax over the top `k` keys,
retrieved from nested convex hulls, costs `O(k + log n)`.

### 1.1 `ALM/SAHead.lean` — the lookup head is an ordinary attention head

Exhibit `Q, K, V : ParamMatrix d` for which `XSA.SAOutput` at position `i` equals
the ALM head over the prefix `j ≤ i`, so that `head_output_at_index`
(`ALM/SoftmaxValue.lean`) becomes a bound on standard attention rather than on a
hand-written sum.  This is what licenses a hybrid layer: a lookup head and a
trained head are then the same operator at different weights.

Note what the theorem must *not* claim.  The paraboloid lift `k ↦ (2k, -‖k‖²)` is
quadratic, so `K` cannot be the lift.  The honest statement is that the lifted
coordinates already sit in the residual stream — written upstream by the FFN, as
`PersistDimension` does in the VM — and that `Q, K` are coordinate projections.
The nonlinearity of lookup lives above the head, not in it.

The causal mask is definitional here: ALM's `Fin n` is the stored prefix, so
`n = i + 1`.

### 1.2 `ALM/PlanarHead.lean` — the fast path for an arbitrary 2D head

For arbitrary planar keys `K : Fin n → ℝ × ℝ` and arbitrary `q` (including
`q.2 < 0` and `q.2 = 0`), the hardmax argmax is answered by one hull query at
`m = q.1 / q.2`, in `log₂ n + 1` comparisons.  Nearly all of it exists:
`ALM/Duality.lean` covers all three sign branches, and
`bsearch_lines_isGreatest` (`ALM/HullLines.lean`) is already stated for arbitrary
lines rather than for lifted keys.  Missing: the one theorem on top.  This is the
"not limited to our executor construction" claim, and the half of the hybrid story
that says a *trained* 2D head gets the same `O(log n)`.

### 1.3 `k`-sparse softmax — the truncation bound only

"Softmax over the top `k` scores approximates the full softmax" reuses
`ALM/Softmax.lean` and `ALM/Lattice.lean` with little new mathematics.  The nested
hulls that retrieve the top `k` in `O(k + log n)` are a new data structure; keep
them out of this file, and do not state the cost as if it were proved.

### 1.4 QK-norm obstructs the lookup head (a cheap negative result)

`GPTMini.preScore` normalizes `q` and `k` and then rotates them by RoPE
(`GPTMini/CausalMHA.lean`), which destroys both the constant second coordinate `1`
of the query lift and the scale of the `-‖k‖²` intercept: the lookup head is *not*
an instance of that head.  On the sphere, however, `argmax ⟪q, k⟫` is still the
nearest neighbour, so what is lost is the unit lattice gap, i.e. a constant, not
the retrieval property.  Worth stating both halves — it is the answer to whether a
real trained architecture can host the fast path.

### Not to be formalized

Differentiability of the trace, speculative decoding, "growing like software":
systems claims, not theorems.  The `β → ∞` limit of `Perspective/Section8_General`
is a false friend — a singular limit of a flow on the sphere, against a single
retrieval step over a discrete key set here.

## 2. Reviewed: `openai/parameter-golf` (2026-09-13)

Train the best language model that fits in a 16 MB artifact in 10 minutes on
8xH100, scored by bits-per-byte on FineWeb.  Nothing in it simplifies a proof:
the content is quantization (int5/int6 QAT, GPTQ, ternary and 1-bit), Muon,
test-time training, depth recurrence, tokenizer tricks.  Three things are still
worth picking up.

### 2.1 The baseline architecture is (almost) `Transformer.GPTMini`

`train_gpt.py` `CausalSelfAttention`: pre-norm RMSNorm, per-head QK norm, RoPE, a
learned per-head query gain, ReLU² MLP, tied embeddings, zero-init output
projection, residual mix with `x0`.  That is the `GPTMini` specification, with two
differences: GQA (`num_kv_heads < num_heads`), which `GPTMini/Config.lean` does not
model, and the gain multiplying `q` rather than the score — the same thing.

One lemma is genuinely missing and cheap: the challenge normalizes `q, k` by
**RMS** norm while `GPTMini/QKNorm.lean` uses `normL2`.  They differ by `√head_dim`,
absorbed into the per-head gain, so `rmsNorm q = √head_dim • normL2 q` closes the
gap and makes the formalized head cover both parameterizations.

### 2.2 XSA is empirically load-bearing, and the `True` placeholder is still debt

Record runs adopt XSA on the deepest layers and then on all of them (1.1307,
1.1271, 1.1147, 1.1122, "XSA-all" at 1.1099).  That is a real citation for the
empirical claim of arXiv:2603.09078 — but it is evidence, not a proof, so it does
not discharge `attention_similarity_bias_observation : True := trivial` in
`XSA.lean`.  That declaration concludes `True` and is forbidden by `CLAUDE.md`:
either state what is actually provable about `⟪y_i, v_i⟫` or delete it.

### 2.3 Differential-Gated Attention (`paper/dg_attention.tex`): a negative result

`payload_t = p_t - α_ℓ p_{t-1}` with `α_ℓ = ℓ/(L-1)`, no learned parameters, and a
documented failure to beat standard attention (1.155 vs 1.152 BPB, within noise);
the learned-gate version collapses at large batch size.  Its only mathematical
content is that the shift-subtract commutes with the value projection.  Nothing to
formalize.  Note for the record that the differential payload is *not* an instance
of `V : ParamMatrix d`: the shift acts across tokens, not within one.

### Also noted

- `zeropower_via_newtonschulz5` (Muon) is a fixed 5-step odd-polynomial iteration
  towards the orthogonal polar factor — formalizable, and unrelated to anything
  this repository proves.  Not a simplification of any existing proof.
- Sliding-window attention appears in most record stacks; `Transformer.Causal`
  (arXiv:2411.04990) formalizes the causal mask only.

## 3. Dangling reference

`GPTMini.lean` points at "Phases 5–6 of `todo.md`" for the clustering/convergence
bridges.  No `todo.md` was ever committed; this file is a new one, and does not
restore those phases.  Either write them out here or drop the reference.
