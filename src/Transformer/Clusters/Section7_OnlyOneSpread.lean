/-
# The emergence of clusters in self-attention dynamics — a bounded configuration spreads

The first step of the proof of `l:onlyone` (§7 of arXiv:2305.05465v6): a
solution of `e:Idnonresca` with two distinct tokens cannot stay bounded.  The
source argues by compactness — on the compact set `ℐ` of bounded configurations
with separated tokens the log-sum-exp potential is strictly convex, so
`|x_1 - x_n|` has a derivative bounded away from `0`.

Here the bound is explicit.  The monotonicity of the softmax
(`softmax_monotone`) is a sum of terms `(e^a - e^b)(a - b) ≥ min(e^a, e^b)(a - b)²`;
when every score lies in `[-K, K]` every weight is at least
`w = e^{-2K}/n`, and the two terms of the extreme tokens alone give
`ẋ_M - ẋ_L ≥ (w/2)(x_M - x_L)³` (`sub_drift_ge_of_bounded`).

Source: arXiv:2305.05465v6, proof of `l:onlyone`, first paragraph.
-/

import Transformer.Clusters.Section7_OnlyOneCore

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- `(e^a - e^b)(a - b) ≥ min(e^a, e^b)(a - b)²`: the exponential grows at
least at its smaller value. -/
theorem min_exp_mul_sq_le (a b : ℝ) :
    min (Real.exp a) (Real.exp b) * (a - b) ^ 2 ≤ (Real.exp a - Real.exp b) * (a - b) := by
  rcases le_total b a with h | h
  · have h1 : Real.exp b * (a - b) ≤ Real.exp a - Real.exp b := by
      have := Real.add_one_le_exp (a - b)
      have he : Real.exp a = Real.exp b * Real.exp (a - b) := by rw [← Real.exp_add]; ring_nf
      nlinarith [Real.exp_pos b]
    nlinarith [min_le_right (Real.exp a) (Real.exp b), sq_nonneg (a - b),
      mul_le_mul_of_nonneg_right (min_le_right (Real.exp a) (Real.exp b)) (sq_nonneg (a - b))]
  · have h1 : Real.exp a * (b - a) ≤ Real.exp b - Real.exp a := by
      have := Real.add_one_le_exp (b - a)
      have he : Real.exp b = Real.exp a * Real.exp (b - a) := by rw [← Real.exp_add]; ring_nf
      nlinarith [Real.exp_pos a]
    nlinarith [mul_le_mul_of_nonneg_right (min_le_left (Real.exp a) (Real.exp b)) (sq_nonneg (a - b))]

/-- Scores in `[-K, K]` give every weight at least `e^{-2K}/n`. -/
theorem exp_div_le_softmaxWeight (s : Idx (m + 1) → ℝ) {K : ℝ} (hs : ∀ j, |s j| ≤ K)
    (j : Idx (m + 1)) :
    Real.exp (-(2 * K)) / ((m : ℝ) + 1) ≤ Perspective.softmaxWeight s j := by
  have hZ : ∑ k : Idx (m + 1), Real.exp (s k) ≤ ((m : ℝ) + 1) * Real.exp K := by
    have : ∑ k : Idx (m + 1), Real.exp (s k) ≤ ∑ _k : Idx (m + 1), Real.exp K :=
      Finset.sum_le_sum fun k _ => Real.exp_le_exp.2 ((le_abs_self _).trans (hs k))
    simpa using this
  have hZ0 := Perspective.softmaxPartition_pos (Nat.succ_pos m) s
  have hj : Real.exp (-K) ≤ Real.exp (s j) := Real.exp_le_exp.2 (by linarith [neg_abs_le (s j), hs j])
  show Real.exp (-(2 * K)) / ((m : ℝ) + 1) ≤ Real.exp (s j) / ∑ k : Idx (m + 1), Real.exp (s k)
  rw [div_le_div_iff₀ (by positivity) hZ0]
  have he : Real.exp (-(2 * K)) * Real.exp K = Real.exp (-K) := by rw [← Real.exp_add]; ring_nf
  nlinarith [Real.exp_pos (-(2 * K)), mul_le_mul_of_nonneg_left hZ (Real.exp_pos (-(2 * K))).le]

/-- **The monotonicity of the softmax, quantitatively.**  For scores in
`[-K, K]` and two indices `a ≠ b`, with `r_j = s_j - u_j`,
`Σ_j (p_j - q_j) r_j ≥ (e^{-2K}/(2n)) (r_a - r_b)²`.

Source: arXiv:2305.05465v6, proof of `l:onlyone`, the strict convexity of
`e:logsumexpfct` on `ℐ`. -/
theorem sq_le_softmax_monotone (s u : Idx (m + 1) → ℝ) {K : ℝ} (hs : ∀ j, |s j| ≤ K)
    (hu : ∀ j, |u j| ≤ K) {a b : Idx (m + 1)} (hab : a ≠ b) :
    Real.exp (-(2 * K)) / ((m : ℝ) + 1) / 2 * ((s a - u a) - (s b - u b)) ^ 2 ≤
      ∑ j, (Perspective.softmaxWeight s j - Perspective.softmaxWeight u j) * (s j - u j) := by
  set Zs := ∑ k : Idx (m + 1), Real.exp (s k)
  set Zu := ∑ k : Idx (m + 1), Real.exp (u k)
  have hZs : 0 < Zs := Perspective.softmaxPartition_pos (Nat.succ_pos m) s
  have hZu : 0 < Zu := Perspective.softmaxPartition_pos (Nat.succ_pos m) u
  set A : Idx (m + 1) → ℝ := fun j => s j - Real.log Zs
  set B : Idx (m + 1) → ℝ := fun j => u j - Real.log Zu
  have hp : ∀ j, Perspective.softmaxWeight s j = Real.exp (A j) := fun j => by
    simp only [A]; rw [Real.exp_sub, Real.exp_log hZs]; rfl
  have hq : ∀ j, Perspective.softmaxWeight u j = Real.exp (B j) := fun j => by
    simp only [B]; rw [Real.exp_sub, Real.exp_log hZu]; rfl
  have key : ∑ j, (Perspective.softmaxWeight s j - Perspective.softmaxWeight u j) * (s j - u j) =
      ∑ j, (Real.exp (A j) - Real.exp (B j)) * (A j - B j) := by
    have hsum : ∀ v : Idx (m + 1) → ℝ, ∑ j, Perspective.softmaxWeight v j = 1 :=
      Perspective.sum_softmaxWeight (Nat.succ_pos m)
    have h0 : ∑ j, (Perspective.softmaxWeight s j - Perspective.softmaxWeight u j) *
        (Real.log Zs - Real.log Zu) = 0 := by
      rw [← Finset.sum_mul, Finset.sum_sub_distrib, hsum, hsum, sub_self, zero_mul]
    rw [← sub_eq_zero, ← Finset.sum_sub_distrib, ← h0]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [hp, hq]; simp only [A, B]; ring
  set w := Real.exp (-(2 * K)) / ((m : ℝ) + 1)
  have hw : ∀ j, w * (A j - B j) ^ 2 ≤ (Real.exp (A j) - Real.exp (B j)) * (A j - B j) := by
    intro j
    have h1 : w ≤ min (Real.exp (A j)) (Real.exp (B j)) := by
      rw [← hp, ← hq]; exact le_min (exp_div_le_softmaxWeight s hs j) (exp_div_le_softmaxWeight u hu j)
    exact (mul_le_mul_of_nonneg_right h1 (sq_nonneg _)).trans (min_exp_mul_sq_le _ _)
  have hnn : ∀ j, 0 ≤ (Real.exp (A j) - Real.exp (B j)) * (A j - B j) := fun j =>
    (mul_nonneg (by positivity) (sq_nonneg _)).trans (hw j)
  have hpair : (Real.exp (A a) - Real.exp (B a)) * (A a - B a) +
      (Real.exp (A b) - Real.exp (B b)) * (A b - B b) ≤
      ∑ j, (Real.exp (A j) - Real.exp (B j)) * (A j - B j) := by
    rw [← Finset.sum_pair (f := fun j => (Real.exp (A j) - Real.exp (B j)) * (A j - B j)) hab]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) fun j _ _ => hnn j
  have hr : (s a - u a) - (s b - u b) = (A a - B a) - (A b - B b) := by simp only [A, B]; ring
  have hw0 : 0 ≤ w := by positivity
  rw [key, hr]
  nlinarith [hw a, hw b, mul_nonneg hw0 (sq_nonneg ((A a - B a) + (A b - B b)))]

/-- **The extreme tokens separate at a definite speed.**  If every coordinate
lies in `[-R, R]` and `y_L < y_M`, then with `c = e^{-2R²}/(2n)`,
`Σ_j P_Mj y_j - Σ_j P_Lj y_j ≥ c (y_M - y_L)³`.

Source: arXiv:2305.05465v6, proof of `l:onlyone`: `|x_1 - x_n|` increases at
a rate bounded away from `0` on `ℐ`. -/
theorem sub_drift_ge_of_bounded (y : Idx (m + 1) → ℝ) {R : ℝ} (hR : ∀ j, |y j| ≤ R)
    (L M : Idx (m + 1)) (hLM : y L < y M) :
    Real.exp (-(2 * R ^ 2)) / ((m : ℝ) + 1) / 2 * (y M - y L) ^ 3 ≤
      ∑ j, Perspective.softmaxWeight (fun l => y M * y l) j * y j -
        ∑ j, Perspective.softmaxWeight (fun l => y L * y l) j * y j := by
  have hR0 : 0 ≤ R := (abs_nonneg _).trans (hR L)
  have hsc : ∀ k j, |y k * y j| ≤ R ^ 2 := fun k j => by
    rw [abs_mul, sq]; exact mul_le_mul (hR k) (hR j) (abs_nonneg _) hR0
  have hab : M ≠ L := fun h => by rw [h] at hLM; exact lt_irrefl _ hLM
  set c := Real.exp (-(2 * R ^ 2)) / ((m : ℝ) + 1) / 2
  set p := Perspective.softmaxWeight (fun l => y M * y l)
  set q := Perspective.softmaxWeight (fun l => y L * y l)
  have h : c * ((y M * y M - y L * y M) - (y M * y L - y L * y L)) ^ 2 ≤
      ∑ j, (p j - q j) * (y M * y j - y L * y j) :=
    sq_le_softmax_monotone (fun l => y M * y l) (fun l => y L * y l) (hsc M) (hsc L) hab
  have hsum : ∑ j, (p j - q j) * (y M * y j - y L * y j) =
      (y M - y L) * (∑ j, p j * y j - ∑ j, q j * y j) := by
    rw [← Finset.sum_sub_distrib, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hsum] at h
  have hd : 0 < y M - y L := sub_pos.2 hLM
  refine le_of_mul_le_mul_left ?_ hd
  have e : (y M - y L) * (c * (y M - y L) ^ 3) =
      c * ((y M * y M - y L * y M) - (y M * y L - y L * y L)) ^ 2 := by ring
  rw [e]; exact h

/-- **Two bounded extreme tokens cannot stay apart.**  If every token of a
solution of `e:Idnonresca` stays in `[-R, R]`, no two tokens keep a gap
`g > 0`: by `sub_drift_ge_of_bounded` their distance would grow linearly.

Source: arXiv:2305.05465v6, proof of `l:onlyone`, first paragraph. -/
theorem not_gap_of_all_bounded (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (L M : Idx (m + 1)) {R : ℝ} (hR : ∀ t, 0 ≤ t → ∀ j, |x t j| ≤ R) {g : ℝ} (hg : 0 < g) :
    ¬ ∀ t, 0 ≤ t → x t L + g ≤ x t M := by
  intro hgap
  set c := Real.exp (-(2 * R ^ 2)) / ((m : ℝ) + 1) / 2
  have hc : 0 < c := by positivity
  set κ := c * g ^ 3
  have hκ : 0 < κ := by positivity
  have hanti : AntitoneOn (fun t => κ * t - (x t M - x t L)) (Set.Ici 0) := by
    refine antitoneOn_Ici_of_hasDerivAt
      (fun t _ => ((hasDerivAt_id t).const_mul κ).sub ((hder t M).sub (hder t L))) ?_
    intro t ht
    have h1 : c * (x t M - x t L) ^ 3 ≤
        ∑ j, Perspective.softmaxWeight (fun l => x t M * x t l) j * x t j -
          ∑ j, Perspective.softmaxWeight (fun l => x t L * x t l) j * x t j :=
      sub_drift_ge_of_bounded (x t) (hR t ht) L M (by linarith [hgap t ht])
    have h2 : g ^ 3 ≤ (x t M - x t L) ^ 3 :=
      pow_le_pow_left₀ hg.le (by linarith [hgap t ht]) 3
    have h3 := mul_le_mul_of_nonneg_left h2 hc.le
    simp only [mul_one]
    linarith
  set t₂ := (2 * |R| + 1) / κ
  have ht₂ : 0 ≤ t₂ := by positivity
  have h1 := hanti (Set.mem_Ici.2 le_rfl) (Set.mem_Ici.2 ht₂) ht₂
  have h2 : κ * t₂ = 2 * |R| + 1 := by simp only [t₂]; field_simp
  have hM := abs_le.1 ((hR t₂ ht₂ M).trans (le_abs_self R))
  have hL := abs_le.1 ((hR t₂ ht₂ L).trans (le_abs_self R))
  have h0 := hgap 0 le_rfl
  simp only [mul_zero] at h1
  linarith

/-- The hypotheses of `not_gap_of_all_bounded` are satisfiable — by the
constant solution with a single token at `0`, for which only the gap fails. -/
example : ∃ x : ℝ → Idx 1 → ℝ, (∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t) ∧
    ∀ t, 0 ≤ t → ∀ j, |x t j| ≤ 0 :=
  ⟨fun _ _ => 0, fun t k => by simpa using hasDerivAt_const t (0 : ℝ), fun _ _ _ => by simp⟩

/-- The hypotheses of `sq_le_softmax_monotone` and `sub_drift_ge_of_bounded`
are satisfiable: two coordinates `0 < 1` in `[-1, 1]`. -/
example : ∃ y : Idx 2 → ℝ, (∀ j, |y j| ≤ 1) ∧ y 0 < y 1 ∧ (0 : Idx 2) ≠ 1 :=
  ⟨fun j => (j : ℝ), fun j => by fin_cases j <;> simp, by simp, by decide⟩

end Clusters
end Transformer
