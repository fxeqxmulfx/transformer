/-
# The emergence of clusters in self-attention dynamics — the stationary
  configurations at `V = -I_d`

`l:stationary` of arXiv:2305.05465v6, §8, `s:c<0`.

**What the source says and what is carried here.**

* `l:stationary` is split in two.  The substantive direction — a stationary
  configuration is the zero one — is `eq_zero_of_isStationaryConfig`; the
  converse is proved, and it is what witnesses the hypotheses.

Source: arXiv:2305.05465v6, `l:stationary`.
-/

import Transformer.Clusters.Section8_Origin

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- A stationary configuration of `e:-Iddyn`: one at which every drift
vanishes.

Source: arXiv:2305.05465v6, `l:stationary`. -/
def IsStationaryConfig (Q K : ParamMatrix d) (X : Idx n → EucSpace d) : Prop :=
  ∀ i : Idx n, ∑ j : Idx n, attentionMatrix Q K X i j • X j = 0

/-- The configuration at the origin is stationary — the converse half of
`l:stationary`, and the witness for its hypotheses. -/
theorem isStationaryConfig_zero (Q K : ParamMatrix d) :
    IsStationaryConfig (n := n) Q K (fun _ => (0 : EucSpace d)) := by
  intro i
  simp

/-- **Lemma (l:stationary).**  The only stationary configuration of `e:-Iddyn`
is `x̄_1 = … = x̄_n = 0`.

The source's proof goes through the convexity of `log Σ_j e^{⟨x, x̄_j⟩}`
(`l:logsumexp`) and its analyticity on the affine hull of the `x̄_i`.  The
proof here stays with `g(x) = Σ_j e^{⟨x, x̄_j⟩}`: stationarity is `∇g(x̄_i) = 0`,
so the tangent bound `e^b ≥ e^a(1 + b - a)` makes every `x̄_i` a global
minimizer of `g`.  At the midpoint of `x̄_i` and `x̄_k`, with
`p_j = e^{⟨x̄_i, x̄_j⟩/2}` and `q_j = e^{⟨x̄_k, x̄_j⟩/2}`, minimality gives
`Σ_j (p_j - q_j)² ≤ 0`, so `⟨x̄_i - x̄_k, x̄_j⟩ = 0` for every `j`, and then
`x̄_i = x̄_k`.  A configuration at one point is stationary only at `0`.

Source: arXiv:2305.05465v6, `l:stationary`. -/
theorem eq_zero_of_isStationaryConfig (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : Idx n → EucSpace d) (hX : IsStationaryConfig Q K X) (i : Idx n) :
    X i = 0 := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  set a := fun i j => inner (𝕜 := ℝ) (X i) (X j)
  set g := fun y : EucSpace d => ∑ j : Idx n, Real.exp (inner (𝕜 := ℝ) y (X j))
  have hS : ∀ i, 0 < ∑ j : Idx n, Real.exp (a i j) := fun i =>
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨i, Finset.mem_univ _⟩
  have hstat : ∀ i, ∑ j : Idx n, Real.exp (a i j) • X j = 0 := by
    intro i
    have h := hX i
    simp only [attentionMatrix, Perspective.softmaxWeight,
      show ∀ u v, inner (𝕜 := ℝ) (Q u) (K v) = inner (𝕜 := ℝ) u v from hQK, div_eq_inv_mul, ← smul_smul,
      ← Finset.smul_sum] at h
    exact (smul_eq_zero.1 h).resolve_left (inv_ne_zero (hS i).ne')
  have htan : ∀ i y, g (X i) ≤ g y := by
    intro i y
    have hsum : ∑ j : Idx n, Real.exp (a i j) * inner (𝕜 := ℝ) (y - X i) (X j) = 0 := by
      have := congrArg (fun v => inner (𝕜 := ℝ) (y - X i) v) (hstat i)
      simpa [inner_sum, inner_smul_right] using this
    have hle : ∀ j, Real.exp (a i j) + Real.exp (a i j) * inner (𝕜 := ℝ) (y - X i) (X j)
        ≤ Real.exp (inner (𝕜 := ℝ) y (X j)) := by
      intro j
      have h1 := Real.add_one_le_exp (inner (𝕜 := ℝ) y (X j) - a i j)
      have h2 : Real.exp (a i j) * Real.exp (inner (𝕜 := ℝ) y (X j) - a i j)
          = Real.exp (inner (𝕜 := ℝ) y (X j)) := by rw [← Real.exp_add]; ring_nf
      have h3 : inner (𝕜 := ℝ) (y - X i) (X j) = inner (𝕜 := ℝ) y (X j) - a i j :=
        inner_sub_left _ _ _
      rw [h3, ← h2]
      nlinarith [Real.exp_pos (a i j)]
    calc g (X i) = ∑ j : Idx n, (Real.exp (a i j) +
          Real.exp (a i j) * inner (𝕜 := ℝ) (y - X i) (X j)) := by
          rw [Finset.sum_add_distrib, hsum, add_zero]
      _ ≤ g y := Finset.sum_le_sum fun j _ => hle j
  have heq : ∀ k, X k = X i := by
    intro k
    set p := fun j => Real.exp (a i j / 2)
    set q := fun j => Real.exp (a k j / 2)
    have hp : ∀ j, Real.exp (a i j) = p j ^ 2 := fun j => by
      simp only [p]; rw [sq, ← Real.exp_add]; ring_nf
    have hq : ∀ j, Real.exp (a k j) = q j ^ 2 := fun j => by
      simp only [q]; rw [sq, ← Real.exp_add]; ring_nf
    have hm : ∀ j, Real.exp (inner (𝕜 := ℝ) ((1 / 2 : ℝ) • (X i + X k)) (X j)) = p j * q j :=
      fun j => by
        simp only [p, q, a]
        rw [← Real.exp_add, inner_smul_left, inner_add_left]
        congr 1; simp; ring
    have h1 := htan i ((1 / 2 : ℝ) • (X i + X k))
    have h2 := htan k ((1 / 2 : ℝ) • (X i + X k))
    simp only [g, hm] at h1 h2
    simp only [show ∀ j, inner (𝕜 := ℝ) (X i) (X j) = a i j from fun _ => rfl,
      show ∀ j, inner (𝕜 := ℝ) (X k) (X j) = a k j from fun _ => rfl, hp, hq] at h1 h2
    have hsq : ∑ j : Idx n, (p j - q j) ^ 2 = 0 := by
      refine le_antisymm ?_ (Finset.sum_nonneg fun j _ => sq_nonneg _)
      have : ∑ j : Idx n, (p j - q j) ^ 2
          = ∑ j, p j ^ 2 + ∑ j, q j ^ 2 - 2 * ∑ j, p j * q j := by
        rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun j _ => by ring
      linarith
    have hpq : ∀ j, a i j = a k j := fun j => by
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg (p j - q j))).1 hsq j
        (Finset.mem_univ _)
      have := Real.exp_injective (sub_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 this))
      linarith
    have hz : inner (𝕜 := ℝ) (X k - X i) (X k - X i) = 0 := by
      rw [inner_sub_left, inner_sub_right, inner_sub_right]
      have e1 := hpq i; have e2 := hpq k
      simp only [a] at e1 e2
      rw [real_inner_comm (X k) (X i)] at *
      linarith
    exact sub_eq_zero.1 (inner_self_eq_zero.1 hz)
  have h := hstat i
  simp only [heq, ← Finset.sum_smul] at h
  exact (smul_eq_zero.1 h).resolve_left (hS i).ne'

/-- The hypotheses of `eq_zero_of_isStationaryConfig` are satisfiable. -/
example : IsIdentityQK (1 : ParamMatrix d) 1 ∧
    IsStationaryConfig (n := n) (1 : ParamMatrix d) 1 (fun _ => (0 : EucSpace d)) :=
  ⟨isIdentityQK_one d, isStationaryConfig_zero 1 1⟩

end Clusters
end Transformer
