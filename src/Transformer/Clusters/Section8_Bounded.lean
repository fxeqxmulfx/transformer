/-
# The emergence of clusters in self-attention dynamics — the trajectories at
  `V = -I_d` stay bounded

`l:cas1circle` of arXiv:2305.05465v6, §8, `s:c<0`.

**What the source says and what is carried here.**

* `l:cas1circle` is stated as the source states it, as the existence of a
  radius.  The explicit bound of its proof is not carried.

Source: arXiv:2305.05465v6, `l:cas1circle`.
-/

import Transformer.Clusters.Section8_Origin
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Real Filter

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- `e^a a ≥ -1`, the elementary inequality behind `l:cas1circle`. -/
theorem mul_exp_add_one_nonneg (a : ℝ) : 0 ≤ Real.exp a * a + 1 := by
  rcases le_or_gt 0 a with ha | ha
  · positivity
  · have h := Real.add_one_le_exp (-a)
    have he : Real.exp a * Real.exp (-a) = 1 := by rw [← Real.exp_add]; simp
    nlinarith [mul_le_mul_of_nonneg_left h (Real.exp_pos a).le, Real.exp_pos a]

/-- **The drift of `e:-Iddyn` points inward outside the ball of radius `√n`.**
If `‖x_i‖² ≥ n` then `⟨x_i, ẋ_i⟩ < 0`: with `a_j = ⟨x_i, x_j⟩`, each
`e^{a_j} a_j ≥ -1` and `e^{a_i} a_i ≥ a_i ≥ n`, so `Σ_j e^{a_j} a_j > 0`.

Source: arXiv:2305.05465v6, the proof of `l:cas1circle`. -/
theorem inner_negIdDrift_neg (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (Y : Idx n → EucSpace d) (i : Idx n) (hY : (n : ℝ) ≤ ‖Y i‖ ^ 2) :
    inner (𝕜 := ℝ) (Y i) (negIdDrift Q K Y i) < 0 := by
  set a := fun j => inner (𝕜 := ℝ) (Y i) (Y j)
  have hZ : 0 < ∑ k : Idx n, Real.exp (a k) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨i, Finset.mem_univ _⟩
  have hP : ∀ j, attentionMatrix Q K Y i j = (∑ k : Idx n, Real.exp (a k))⁻¹ * Real.exp (a j) := by
    intro j
    simp only [attentionMatrix, Perspective.softmaxWeight,
      show ∀ u v, inner (𝕜 := ℝ) (Q u) (K v) = inner (𝕜 := ℝ) u v from hQK, div_eq_inv_mul, a]
  have hpos : 0 < ∑ j : Idx n, Real.exp (a j) * a j := by
    have hsum : ∑ j : Idx n, (Real.exp (a j) * a j + 1)
        = ∑ j : Idx n, Real.exp (a j) * a j + n := by
      simp [Finset.sum_add_distrib]
    have hi := Finset.single_le_sum (f := fun j => Real.exp (a j) * a j + 1)
      (fun j _ => mul_exp_add_one_nonneg (a j)) (Finset.mem_univ i)
    have hai : a i = ‖Y i‖ ^ 2 := real_inner_self_eq_norm_sq _
    have h1 : a i ≤ Real.exp (a i) * a i := by
      have : 1 ≤ Real.exp (a i) := Real.one_le_exp (by rw [hai]; positivity)
      nlinarith [sq_nonneg ‖Y i‖]
    linarith
  have : inner (𝕜 := ℝ) (Y i) (negIdDrift Q K Y i)
      = -((∑ k : Idx n, Real.exp (a k))⁻¹ * ∑ j : Idx n, Real.exp (a j) * a j) := by
    simp only [negIdDrift, inner_neg_right, inner_sum, inner_smul_right, hP, Finset.mul_sum]
    congr 1
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [this, neg_lt_zero]
  exact mul_pos (inv_pos.2 hZ) hpos

/-- **Lemma (l:cas1circle).**  The trajectories of `e:-Iddyn` are uniformly
bounded in time: there is `R > 0`, depending only on `n` and the initial
configuration, with `‖x_i(t)‖ ≤ R` for every `i ∈ [n]` and every `t ≥ 0`.

As in the source, `‖x_i‖²` decreases whenever it is large, by
`inner_negIdDrift_neg`.  The source's threshold is `2n`, and `n` already
suffices; the radius taken is `√(Σ_j ‖x_j(0)‖² + n) + 1`.

Source: arXiv:2305.05465v6, `l:cas1circle`. -/
theorem exists_bound_negIdDynamics (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : ℝ → Idx n → EucSpace d) (hX : NegIdDynamics Q K X) :
    ∃ R : ℝ, 0 < R ∧ ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖X t i‖ ≤ R := by
  set M := ∑ j : Idx n, ‖X 0 j‖ ^ 2 + n
  refine ⟨Real.sqrt M + 1, by positivity, fun i t ht => ?_⟩
  have hd : ∀ s, HasDerivAt (fun s => ‖X s i‖ ^ 2)
      (2 * inner (𝕜 := ℝ) (X s i) (negIdDrift Q K (X s) i)) s := fun s => (hX s i).norm_sq
  have h0 : ‖X 0 i‖ ^ 2 ≤ M := by
    have := Finset.single_le_sum (f := fun j => ‖X 0 j‖ ^ 2) (fun j _ => sq_nonneg _)
      (Finset.mem_univ i)
    simp only [M]; linarith [(n.cast_nonneg : (0 : ℝ) ≤ n)]
  have hle := image_le_of_deriv_right_lt_deriv_boundary (a := 0) (b := t)
    (f := fun s => ‖X s i‖ ^ 2) (B := fun _ => M) (B' := fun _ => 0)
    (fun s _ => (hd s).continuousAt.continuousWithinAt) (fun s _ => (hd s).hasDerivWithinAt)
    h0 (fun _ => hasDerivAt_const _ _) (fun s _ hs => by
      have hn : (n : ℝ) ≤ ‖X s i‖ ^ 2 := by
        rw [hs]; simp only [M]
        linarith [Finset.sum_nonneg fun j (_ : j ∈ Finset.univ) => sq_nonneg ‖X 0 j‖]
      linarith [inner_negIdDrift_neg Q K hQK (X s) i hn]) ⟨ht, le_rfl⟩
  have hM : 0 ≤ M := by positivity
  have := (Real.le_sqrt (norm_nonneg _) hM).2 hle
  linarith

/-- The hypotheses of `inner_negIdDrift_neg` are satisfiable: one token on the
unit sphere of `ℝ¹`. -/
example : ((1 : ℕ) : ℝ) ≤
    ‖(fun _ : Idx 1 => (EuclideanSpace.single 0 1 : EucSpace 1)) 0‖ ^ 2 := by
  simp

/-- The hypotheses of `exists_bound_negIdDynamics` are satisfiable:
`Q = K = I_d` and the configuration at the origin. -/
example : IsIdentityQK (1 : ParamMatrix d) 1 ∧
    NegIdDynamics (n := n) (1 : ParamMatrix d) 1 (fun _ _ => (0 : EucSpace d)) :=
  ⟨isIdentityQK_one d, (transformerDynamics_neg_one_iff _ _ _).mp (transformerDynamics_zero 1 1 _)⟩

end Clusters
end Transformer
