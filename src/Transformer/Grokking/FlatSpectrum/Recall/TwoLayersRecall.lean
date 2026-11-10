module

public import Transformer.Grokking.FlatSpectrum.Recall.PolynomialCount
public import Transformer.Grokking.FlatSpectrum.Recall.TwoLayers

/-!
# Two gated convolutions recall a number of bindings linear in their dimension

Source: `review/flat_spectrum.md`, *Polynomial readouts and two layers* (added), item PR5, for
Claim 1 of Arora et al., arXiv:2312.04927v1, §4: "Gated-convolution models with two layers
require model dimension to scale at least linearly in sequence length in order to solve
associative recall". The claim is empirical. Here two layers read the context, and a linear
readout reads the second at the query.

Two Coyote layers of dimension `d` are a readout quadratic in `2d` numbers of the context
(`isQuadIn_twoCoyoteLayers`). So for `m` keys:

* exact recall of a non-constant `φ` has `m ≤ 10 d` (`card_le_of_twoCoyoteLayers_recall`): the
  dimension is at least linear in the number of bindings;
* dimension `m` is enough, with a second layer that returns its input
  (`exists_twoCoyoteLayers_recall`), so the least dimension is between `m / 10` and `m`;
* the readout has cut width at most `2d + (2d)²` (`hasCutWidth_twoCoyoteLayers`), so the squared
  error on recall is at least `(m − 2d − 4d²) c^{m−1} ∑_v (φ(v) − μ)²`
  (`twoCoyoteLayers_recall_error`), and a correct argmax over logits `∑_c R_{c v} y_c` has
  `m ≤ 2d + 4d²` (`card_le_of_twoCoyoteLayers_argmax`).

For the error and for the argmax the bound is quadratic in `d`, not linear: item PR6.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Transformer.Zoology

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν] {n d : ℕ}

/-- **Two Coyote layers read at the query have cut width at most `2d + (2d)²`**
(`review/flat_spectrum.md`, PR5). -/
theorem hasCutWidth_twoCoyoteLayers {Q C : Type*} (p₁ p₂ : CoyoteParameters (n + 1) d)
    (w : Fin d → ℝ) (ctx : C → Fin n → Fin d → ℝ) (qry : Q → Fin d → ℝ) :
    HasCutWidth (d + d + (d + d) * (d + d)) fun q x =>
      ∑ c, w c * coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx x) (qry q))) (Fin.last n) c :=
  (isQuadIn_twoCoyoteLayers p₁ p₂ w ctx qry).hasCutWidth

omit [Fintype ν] in
/-- **Two Coyote layers of dimension `d` recall at most `10 d` bindings**
(`review/flat_spectrum.md`, PR5): exact recall of a non-constant `φ` for `m` keys has
`m ≤ 10 d`, for every encoding of the context and of the query and every linear readout. -/
theorem card_le_of_twoCoyoteLayers_recall (p₁ p₂ : CoyoteParameters (n + 1) d) (w : Fin d → ℝ)
    (ctx : (κ → ν) → Fin n → Fin d → ℝ) (qry : κ → Fin d → ℝ) {φ : ν → ℝ} {v v' : ν}
    (hφ : φ v ≠ φ v')
    (h : ∀ q π, ∑ c, w c *
      coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx π) (qry q))) (Fin.last n) c = φ (π q)) :
    Fintype.card κ ≤ 10 * d :=
  (HasPolyCut.card_le_five_mul_of_recall
    ⟨_, (isQuadIn_twoCoyoteLayers p₁ p₂ w ctx qry).isPolyIn⟩ hφ h).trans_eq (by ring)

/-- **The error of two Coyote layers on recall** (`review/flat_spectrum.md`, PR5): at least
`(m − 2d − 4d²) c^{m−1} ∑_v (φ(v) − μ)²`, with `μ` the mean of `φ`. -/
theorem twoCoyoteLayers_recall_error (p₁ p₂ : CoyoteParameters (n + 1) d) (w : Fin d → ℝ)
    (ctx : (κ → ν) → Fin n → Fin d → ℝ) (qry : κ → Fin d → ℝ) {φ : ν → ℝ} {μ : ℝ}
    (hμ : ∑ v, (φ v - μ) = 0) :
    ((Fintype.card κ : ℝ) - ((d + d + (d + d) * (d + d) : ℕ) : ℝ)) *
      ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, (φ v - μ) ^ 2) ≤ ∑ q, ∑ π : κ → ν,
        (φ (π q) - ∑ c, w c *
          coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx π) (qry q))) (Fin.last n) c) ^ 2 :=
  (hasCutWidth_twoCoyoteLayers p₁ p₂ w ctx qry).mul_le_sum_sq hμ

omit [DecidableEq κ] [Fintype ν] in
/-- **Two Coyote layers of dimension `d` classify at most `2d + 4d²` bindings**
(`review/flat_spectrum.md`, PR5): if the largest of the logits `∑_c R_{c v} y_c` at the query
is always that of the bound value, then `m ≤ 2d + (2d)²`. -/
theorem card_le_of_twoCoyoteLayers_argmax (p₁ p₂ : CoyoteParameters (n + 1) d)
    (R : Fin d → ν → ℝ) (ctx : (κ → ν) → Fin n → Fin d → ℝ) (qry : κ → Fin d → ℝ) {v₀ v₁ : ν}
    (hv : v₀ ≠ v₁)
    (h : ∀ π q v, v ≠ π q →
      ∑ c, R c v * coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx π) (qry q))) (Fin.last n) c <
        ∑ c, R c (π q) *
          coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx π) (qry q))) (Fin.last n) c) :
    Fintype.card κ ≤ d + d + (d + d) * (d + d) := by
  refine card_le_of_argmax (L := fun v q π => ∑ c, R c v *
    coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx π) (qry q))) (Fin.last n) c) hv ?_ h
  simpa only [sub_mul, Finset.sum_sub_distrib] using
    hasCutWidth_twoCoyoteLayers p₁ p₂ (fun c => R c v₁ - R c v₀) ctx qry

/-- The layer with `W = 1`, `b₁ = 0`, `h = 0` and `b₂ = 1` returns its input. -/
lemma coyoteLayer_id (u : RealSequence n d) :
    coyoteLayer ⟨fun k c => if k = c then 1 else 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => 1⟩ u =
      u := by
  funext i c
  simp [coyoteLayer, linearProjection, causalConvolution]

omit [Fintype ν] in
/-- **Two Coyote layers of dimension `m` recall `m` bindings** (`review/flat_spectrum.md`,
PR5): the layer of AR6 and then the layer that returns its input. -/
theorem exists_twoCoyoteLayers_recall (m : ℕ) (φ : ν → ℝ) :
    ∃ (p₁ p₂ : CoyoteParameters (m + 1) m) (w : Fin m → ℝ)
      (ctx : (Fin m → ν) → Fin m → Fin m → ℝ) (qry : Fin m → Fin m → ℝ), ∀ q π,
        ∑ c, w c * coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx π) (qry q))) (Fin.last m) c =
          φ (π q) := by
  obtain ⟨p, w, ctx, qry, h⟩ := exists_coyoteLayer_recall m φ
  exact ⟨p, _, w, ctx, qry, fun q π => by rw [coyoteLayer_id]; exact h q π⟩

/-- Two keys with values `±1`: two layers of dimension two recall them, within `m ≤ 10 d`. -/
example : Fintype.card (Fin 2) ≤ 10 * 2 := by
  obtain ⟨p₁, p₂, w, ctx, qry, h⟩ := exists_twoCoyoteLayers_recall 2 (![1, -1] : Fin 2 → ℝ)
  exact card_le_of_twoCoyoteLayers_recall p₁ p₂ w ctx qry (v := 0) (v' := 1) (by norm_num) h

/-- Their logits `φ(v) ∑_c w_c y_c = φ(v) φ(π q)` have the largest one at the bound value. -/
example : Fintype.card (Fin 2) ≤ 2 + 2 + (2 + 2) * (2 + 2) := by
  obtain ⟨p₁, p₂, w, ctx, qry, h⟩ := exists_twoCoyoteLayers_recall 2 (![1, -1] : Fin 2 → ℝ)
  refine card_le_of_twoCoyoteLayers_argmax p₁ p₂ (fun c v => (![1, -1] : Fin 2 → ℝ) v * w c)
    ctx qry (v₀ := 0) (v₁ := 1) (by decide) fun π q v hv => ?_
  simp only [mul_assoc, ← Finset.mul_sum, h]
  generalize π q = u at hv ⊢
  fin_cases v <;> fin_cases u <;> simp_all

example (p₁ p₂ : CoyoteParameters 3 2) (w : Fin 2 → ℝ)
    (ctx : (Fin 30 → Fin 2) → Fin 2 → Fin 2 → ℝ) (qry : Fin 30 → Fin 2 → ℝ) :=
  twoCoyoteLayers_recall_error p₁ p₂ w ctx qry (φ := ![1, -1]) (μ := 0)
    (by simp [Fin.sum_univ_two])

end Transformer.Grokking.FlatSpectrum.Recall
