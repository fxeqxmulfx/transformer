# Complex-linear projections against real ones

Request, 2026-11-20: besides the FFN, DiffGPT's selected model replaces ordinary
projections by complex-linear ones; prove that they approximate better, or refute it.
Lean: `Transformer.ComplexProjection` (`src/Transformer/ComplexProjection/`).

## Sources

DiffGPT, `~/python_projects/gpt` at `5a37c51`:

- `research/main_model/README.txt`: bias-free complex-linear projections `W_Q`, `W_K`,
  `W_V`, `W_O`, `W_g`, `W_u` and `W_F`, stored for real GEMMs as `[[A,-B],[B,A]]`
  (`research/main_model/phase_score_native/src/layout.rs`, `matrix`).
- `research/complex_transformer/README.md`: "The actual restriction is `Wz`, whereas a
  general real projection is `Wz+V conjugate(z)`."

The Lean statements take a real vector space with a complex structure `J`
(`J (J z) = -z`), multiplication by `i`; on the model's stream it is
`(re, im) ↦ (-im, re)` per coordinate pair, an isometry. A map is complex-linear when
it commutes with `J`.

## Results

P1, splitting (`Defs`). Every real-linear `L` is `P + A` with `P = (L - J' L J)/2`
complex-linear and `A = (L + J' L J)/2` conjugate-linear, the `V conjugate(z)` term,
and the splitting is unique (`complexPart_add_antilinearPart`, `complexPart_comm`,
`antilinearPart_anticomm`, `complexPart_unique`). The storage `[[A,-B],[B,A]]` is exactly
the set of real block matrices commuting with `[[0,-1],[1,0]]` (`fromBlocks_comm_iff`).

P2, the exact price at equal width (`Basic`). Let `J'` keep the norm and let `f` be any
map with `f (J z) = J' (f z)`, linear or not. On a set `K` closed under `J`, a uniform
error `δ` of `f` against `L` forces `‖A z‖ ≤ δ` on `K` (`norm_antilinearPart_le`), and the
complex part `P` errs by exactly `‖A z‖` (`norm_complexPart_sub`). So `δ` is reachable
if and only if it bounds `A` on `K` (`exists_comm_approx_iff`). Conjugation on the unit
circle is missed by `1`, as by the zero map (`one_le_of_approx_conj`). The general form
`gap_of_intertwine` also yields the gaps of `ComplexFFN.Basic` (`τ = ±id`).

P3, the branch RMS (`Basic`). `x √d / √(‖x‖² + dε)` commutes with every linear isometry
(`rmsNormEps_comm`), so composing complex maps with it keeps them `J`-equivariant. A
target `R ∘ L` with conjugate-linear `L` is missed by its full norm `‖R (L z)‖`, about
`√d` on the RMS sphere, as by the zero map (`norm_rmsNormEps_le`). Unlike the one-unit
FFN witnesses of `review/complex_ffn.md`, this gap survives the normalization.

P4, equal parameters (`Line`, `EqualParameters`). A complex projection of `ℂ^n` has
`2n²` real parameters (`finrank_complexLinear`), a real one of width `r` has `r²`
(`finrank_realLinear`); equal parameters give `r = √2 n`, between `n` and `2n`
(`finrank_real_euclideanSpace`). With the real stream a subspace `S` of `ℂ^n` and
`n < dim S < 2n` (`complex_real_incomparable_stream`):

- every map landing in `S`, of any form, misses the identity by `1` at a unit vector
  orthogonal to `S` (`one_le_norm_sub_self`, `exists_unit_mem_orthogonal`);
- `S` holds a complex line `u, J u` (`exists_mem_complexLine`, from
  `dim (S ∩ J S) ≥ 2 dim S - 2n`), and the conjugation `lineConj J u` of that line maps
  into `S`, vanishes on `Sᗮ` and shrinks norms (`lineConj_mem`, `lineConj_eq_zero`,
  `norm_lineConj_le`); every map commuting with `i` misses it by `1` on the unit sphere
  (`one_le_of_approx_lineConj`).

Both misses equal the zero map's error. The example in Lean is `n = 2`, `dim S = 3`.

## Verdict

Refuted at equal width: a complex projection is a real projection without its
conjugate-linear half, and it loses exactly that half, before or after the branch RMS.
At equal parameters the classes are incomparable: each misses a projection of the other
by as much as the zero map. So "approximates better" holds in neither comparison.

Not claimed: training dynamics, initialization, or inductive bias. Equivariance under
`z ↦ e^{iθ} z` may help learning phase-structured tasks without adding approximation
power. The results concern single projections and maps commuting with `J`. A whole
block of the model is not such a map: the plain FFN satisfies `f (J z) = -f z`
(`ComplexFFN.plainFFN_anti`), and attention scores read real parts of complex inner
products.
