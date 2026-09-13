/-
# Bridge: QK-norm obstructs the lookup head, and what survives anyway

`Transformer.ALM` builds an exact lookup head out of the paraboloid score
`score q k = 2⟪k,q⟫ - ‖k‖²`, which is `-‖k - q‖²` up to a term independent of
the key: its argmax is the nearest stored key, and on integer keys the gap to
the runner-up is at least `1` (`Transformer.ALM.Lattice`).  `gpt-mini` does not
compute that score.  `GPTMini.preScore` normalizes `q` and `k` first, so the
head sees only directions.

Both halves of the consequence are theorems here, and they point opposite
ways.

**The obstruction.**  `score_smul_key` and `score_smul_query`: the normalized
score is invariant under positive rescaling of either argument.  The lookup
score is not — `-‖k - q‖²` is a statement about position, not direction, and
the whole point of the query lift `q ↦ (q, 1)` is a coordinate that is
constant rather than radial.  `qknorm_indifferent_where_lookup_is_not` makes
this a counterexample rather than a slogan: at *every* temperature `α`, the
QK-normalized head is exactly indifferent between a unit key and twice that
key, while the lookup head strictly prefers the first.  So the ALM head is not
`GPTMini.preScore` at any weights, and no per-head gain repairs it.

**What survives.**  Restrict to keys of one common norm `R` — the sphere the
tokens already live on after `RMSNorm` (`Bridge.SphereResidence`) — and the
obstruction vanishes: `score_sub_eq_const_mul_lookup_sub` shows the two scores
differ by the *positive constant factor* `e^α / (2‖q‖R)` on differences, so
`score_le_iff_lookup_le` gives the same ranking and `score_le_iff_dist_le` the
same nearest-neighbour answer.  What the normalization costs is that constant:
the unit integer gap of the lattice becomes `e^α / (2‖q‖R)`, which is a change
of temperature, not a loss of the retrieval property.

`eps` is `0` throughout: the statements are about the normalization
`F.normalize` computes, of which the implementation's `eps = 1e-6` is the
numerical guard.

Source: `reference/model.py` (`CausalMHA.forward`) against Percepta,
*Can LLMs Be Computers?* (2026-03-11), the paraboloid lookup head.
-/

import Transformer.GPTMini.QKNorm
import Transformer.ALM.Basic

open scoped BigOperators

namespace Transformer
namespace GPTMini
namespace Bridge

variable {d : ℕ}

/-! ### The normalized head sees only directions -/

/-- Exact L2 normalization is invariant under positive rescaling. -/
theorem normL2_smul_of_pos {c : ℝ} (hc : 0 < c) (x : EucSpace d) :
    normL2 0 (c • x) = normL2 0 x := by
  by_cases hx : x = 0
  · simp [normL2, hx]
  · have hxn : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
    unfold normL2
    rw [add_zero, add_zero, norm_smul, Real.norm_eq_abs, abs_of_pos hc, smul_smul]
    congr 1
    field_simp

/-- **The QK-normalized score cannot see the length of a key.** -/
theorem score_smul_key (alpha : ℝ) {c : ℝ} (hc : 0 < c) (q k : EucSpace d) :
    GPTMini.score alpha 0 q (c • k) = GPTMini.score alpha 0 q k := by
  unfold GPTMini.score
  rw [normL2_smul_of_pos hc]

/-- **Nor of a query.**  In particular it cannot see the constant coordinate
of the query lift `q ↦ (q, 1)`, which is what makes the paraboloid score
affine in the key. -/
theorem score_smul_query (alpha : ℝ) {c : ℝ} (hc : 0 < c) (q k : EucSpace d) :
    GPTMini.score alpha 0 (c • q) k = GPTMini.score alpha 0 q k := by
  unfold GPTMini.score
  rw [normL2_smul_of_pos hc]

/-- **The lookup head is not an instance of this one.**  For any unit vector
`v` and *any* temperature `α`: the paraboloid head strictly prefers `v` to
`2v`, and the QK-normalized head is exactly indifferent between them.  No
choice of per-head gain makes the normalized score a monotone function of the
paraboloid score.

Source: `reference/model.py`, `q = F.normalize(q, dim=-1, eps=1e-6)`. -/
theorem qknorm_indifferent_where_lookup_is_not (alpha : ℝ) (v : EucSpace d)
    (hv : ‖v‖ = 1) :
    ALM.score v ((2 : ℝ) • v) < ALM.score v v ∧
      GPTMini.score alpha 0 v ((2 : ℝ) • v) = GPTMini.score alpha 0 v v := by
  refine ⟨?_, score_smul_key alpha (by norm_num) v v⟩
  have hvv : inner (𝕜 := ℝ) v v = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, hv]; norm_num
  have h2 : ‖(2 : ℝ) • v‖ = 2 := by
    rw [norm_smul, Real.norm_eq_abs, hv]; norm_num
  unfold ALM.score
  rw [real_inner_smul_left, hvv, h2, hv]
  norm_num

/-- The hypothesis is satisfiable: the first standard basis vector. -/
example : ‖(WithLp.toLp 2 (fun j => if j = (0 : Fin (d + 1)) then (1 : ℝ) else 0)
    : EucSpace (d + 1))‖ = 1 := by
  rw [EuclideanSpace.norm_eq]
  simp

/-! ### On a common sphere the two heads agree -/

/-- The normalized score in closed form.  At `q = 0` or `k = 0` both sides are
`0`, so no nondegeneracy is needed. -/
theorem score_eq_inner_div (alpha : ℝ) (q k : EucSpace d) :
    GPTMini.score alpha 0 q k
      = Real.exp alpha * (inner (𝕜 := ℝ) q k / (‖q‖ * ‖k‖)) := by
  unfold GPTMini.score normL2
  rw [add_zero, add_zero, real_inner_smul_left, real_inner_smul_right]
  ring

/-- **Only a constant is lost.**  For keys of a common norm `R` — the sphere
`Bridge.SphereResidence` already puts the tokens on — the two heads' score
*differences* are proportional, with the positive factor `e^α / (2‖q‖R)`.

So the unit score gap the integer lattice guarantees the lookup head becomes a
gap of `e^α / (2‖q‖R)` for the normalized head: a rescaling of the inverse
temperature, and nothing else. -/
theorem score_sub_eq_const_mul_lookup_sub (alpha : ℝ) {R : ℝ} (hR : 0 < R)
    (q k₁ k₂ : EucSpace d) (hq : q ≠ 0) (h1 : ‖k₁‖ = R) (h2 : ‖k₂‖ = R) :
    GPTMini.score alpha 0 q k₂ - GPTMini.score alpha 0 q k₁
      = (Real.exp alpha / (2 * ‖q‖ * R)) * (ALM.score q k₂ - ALM.score q k₁) := by
  have hqn : ‖q‖ ≠ 0 := norm_ne_zero_iff.mpr hq
  rw [score_eq_inner_div, score_eq_inner_div, h1, h2]
  unfold ALM.score
  rw [h1, h2, real_inner_comm k₁ q, real_inner_comm k₂ q]
  field_simp
  ring

/-- **And so the ranking is the same.**  On a common sphere the QK-normalized
head orders the keys exactly as the paraboloid head does. -/
theorem score_le_iff_lookup_le (alpha : ℝ) {R : ℝ} (hR : 0 < R)
    (q k₁ k₂ : EucSpace d) (hq : q ≠ 0) (h1 : ‖k₁‖ = R) (h2 : ‖k₂‖ = R) :
    GPTMini.score alpha 0 q k₁ ≤ GPTMini.score alpha 0 q k₂
      ↔ ALM.score q k₁ ≤ ALM.score q k₂ := by
  have hqp : 0 < ‖q‖ := norm_pos_iff.mpr hq
  have hc : 0 < Real.exp alpha / (2 * ‖q‖ * R) := by positivity
  have hd := score_sub_eq_const_mul_lookup_sub alpha hR q k₁ k₂ hq h1 h2
  constructor
  · intro hle
    have h0 : 0 ≤ (Real.exp alpha / (2 * ‖q‖ * R)) * (ALM.score q k₂ - ALM.score q k₁) := by
      rw [← hd]; linarith
    linarith [(mul_nonneg_iff_of_pos_left hc).mp h0]
  · intro hle
    have h0 : 0 ≤ (Real.exp alpha / (2 * ‖q‖ * R)) * (ALM.score q k₂ - ALM.score q k₁) :=
      mul_nonneg hc.le (by linarith)
    linarith [hd ▸ h0]

/-- **Which is nearest-neighbour retrieval.**  Restated through
`ALM.score_gap`: on a common sphere the normalized head's argmax is the key
closest to the query, exactly as the lookup head's is.  The retrieval property
is intact; the lattice gap is what the normalization rescales. -/
theorem score_le_iff_dist_le (alpha : ℝ) {R : ℝ} (hR : 0 < R)
    (q k₁ k₂ : EucSpace d) (hq : q ≠ 0) (h1 : ‖k₁‖ = R) (h2 : ‖k₂‖ = R) :
    GPTMini.score alpha 0 q k₁ ≤ GPTMini.score alpha 0 q k₂
      ↔ ‖k₂ - q‖ ≤ ‖k₁ - q‖ := by
  rw [score_le_iff_lookup_le alpha hR q k₁ k₂ hq h1 h2]
  have g1 := ALM.score_gap q k₁
  have g2 := ALM.score_gap q k₂
  rw [← pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) (two_ne_zero)]
  constructor <;> intro h <;> linarith

/-- The hypotheses are satisfiable, and on the sphere the architecture itself
produces: the query is the first basis vector, both keys are unit vectors, and
`R = 1`. -/
example :
    (0 : ℝ) < 1 ∧
      (WithLp.toLp 2 (fun j => if j = (0 : Fin (d + 1)) then (1 : ℝ) else 0)
        : EucSpace (d + 1)) ≠ 0 ∧
      ‖(WithLp.toLp 2 (fun j => if j = (0 : Fin (d + 1)) then (1 : ℝ) else 0)
        : EucSpace (d + 1))‖ = 1 := by
  have hnorm : ‖(WithLp.toLp 2 (fun j => if j = (0 : Fin (d + 1)) then (1 : ℝ) else 0)
      : EucSpace (d + 1))‖ = 1 := by
    rw [EuclideanSpace.norm_eq]; simp
  exact ⟨by norm_num, fun h => by rw [h, norm_zero] at hnorm; exact zero_ne_one hnorm, hnorm⟩

end Bridge
end GPTMini
end Transformer
