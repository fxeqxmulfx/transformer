/-
# Counting heavy atoms from the cluster-mass estimate

The conditional counting step of arXiv:2601.21366v2, §3.2, `cor: bound`.
The estimate of `thm: bound` is passed explicitly in full for every
cluster of the actual atomic configuration; `bound_cluster_mass` now proves it.
The conclusion keeps the source's factor
`M (1 + 2 L sqrt β) (0.5742 + C exp(-β)) / ε`.

The statement is pointwise in `β` and the constant `C`: when the uniform
constants of `bound_cluster_mass` are available, the same constants work
at every temperature above their threshold. The perceptron and SOPD
conditions enter this counting argument only through that estimate.
The unused restriction `L < 2π` is dropped: the argument works for any
finite circular-arc cover. Integer turns in the cover condition allow
arcs to cross angle zero while retaining the original representatives.

A total-mass argument avoids subdividing intervals. With two cover intervals,
or one interval longer than the cluster scale, the displayed factor is
already at least one. For one shorter interval all atoms form a cluster,
and the explicit input supplies the missing lower bound on that factor.
-/

import Transformer.Perceptron.Bound

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- Heavy atoms consume at least their count times `ε` of the unit mass;
arXiv:2601.21366v2, §3.2, proof of `cor: bound`. -/
theorem heavy_atom_count_mul_le_one {N : ℕ} (m : Idx N → ℝ)
    (hm : ∀ i, 0 ≤ m i) (hsum : ∑ i, m i = 1) (ε : ℝ) :
    ε * (({i : Idx N | ε ≤ m i} : Set (Idx N)).ncard : ℝ) ≤ 1 := by
  classical
  let S := Finset.univ.filter fun i : Idx N => ε ≤ m i
  have hset : (S : Set (Idx N)) = {i | ε ≤ m i} := by ext i; simp [S]
  rw [← hset, Set.ncard_coe_finset]
  calc ε * (S.card : ℝ) = ∑ _i ∈ S, ε := by simp [mul_comm]
    _ ≤ ∑ i ∈ S, m i := Finset.sum_le_sum fun i hi => (Finset.mem_filter.mp hi).2
    _ ≤ ∑ i, m i := Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.subset_univ S) (fun i _ _ => hm i)
    _ = 1 := hsum

/-- The unit-mass hypotheses hold for the actual antipodal pair;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
example : (∀ i : Idx 2, (0 : ℝ) ≤ (fun _ : Idx 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Idx 2, (fun _ : Idx 2 => (1 / 2 : ℝ)) i) = 1 := by norm_num

/-- The corresponding count bound for a positive threshold;
arXiv:2601.21366v2, §3.2, the last paragraph of the proof of `cor: bound`.
No geometric or second-order assumption is needed for this part. -/
theorem heavy_atom_count_le_inv {N : ℕ} (m : Idx N → ℝ)
    (hm : ∀ i, 0 ≤ m i) (hsum : ∑ i, m i = 1) (ε : ℝ) (hε : 0 < ε) :
    (({i : Idx N | ε ≤ m i} : Set (Idx N)).ncard : ℝ) ≤ 1 / ε := by
  apply (le_div_iff₀ hε).mpr
  simpa [mul_comm] using heavy_atom_count_mul_le_one m hm hsum ε

/-- Every threshold-bound input holds for two equal positive masses;
arXiv:2601.21366v2, §3.2, `eq: atomic.thm.bound`. -/
example : (∀ i : Idx 2, (0 : ℝ) ≤ (fun _ : Idx 2 => (1 / 2 : ℝ)) i) ∧
    (∑ i : Idx 2, (fun _ : Idx 2 => (1 / 2 : ℝ)) i) = 1 ∧
    (0 : ℝ) < 1 / 2 := by norm_num

/-- **Conditional corollary (`cor: bound`).** The full cluster estimate for
this configuration is `hmass`; no open theorem is invoked. Its constants
are the source's `0.5742`, `C`, and `exp(-β)`. Atomicity supplies `N ≥ 2`,
positive masses summing to one, and the interval cover supplies `L ≥ 0`.
Together they imply the printed bound for every `ε > 0`, including
`ε > 0.5742 + C exp(-β)`, the last case in the manuscript's proof.
Source: arXiv:2601.21366v2, §3.2, `cor: bound` and its proof. -/
theorem bound_atom_count (C β : ℝ) (hC : 0 ≤ C) (hβ : 0 < β)
    (N : ℕ) (m θ : Idx N → ℝ) (μ : Perspective.ProbSphere 2)
    (hatom : IsAtomicOnCircle N m θ μ)
    (hmass : ∀ S : Finset (Idx N), 2 ≤ S.card →
      (∀ i ∈ S, ∀ j ∈ S, ∃ k : ℤ,
        |θ i - θ j + 2 * π * (k : ℝ)| ≤ 1 / (2 * Real.sqrt β)) →
      ∑ i ∈ S, m i ≤ 0.5742 + C * Real.exp (-β))
    (M : ℕ) (hM : 1 ≤ M) (l r : Idx M → ℝ) (L : ℝ)
    (hlength : ∀ j, r j - l j ≤ L)
    (hcover : ∀ i : Idx N, ∃ j : Idx M, ∃ k : ℤ,
      θ i + 2 * π * (k : ℝ) ∈ Set.Icc (l j) (r j))
    (ε : ℝ) (hε : 0 < ε) :
    (({i : Idx N | ε ≤ m i} : Set (Idx N)).ncard : ℝ) ≤
      (M / ε) * (1 + 2 * L * Real.sqrt β) * (0.5742 + C * Real.exp (-β)) := by
  classical
  let Λ := 0.5742 + C * Real.exp (-β)
  have hΛ : (1 / 2 : ℝ) ≤ Λ := by
    have hc := mul_nonneg hC (Real.exp_pos (-β)).le
    dsimp [Λ]
    linarith
  have hΛ0 : 0 ≤ Λ := by linarith
  have hs : 0 < Real.sqrt β := Real.sqrt_pos.mpr hβ
  obtain ⟨j, k, hj⟩ := hcover ⟨0, by have := hatom.two_le; omega⟩
  have hL : 0 ≤ L := by linarith [hlength j, hj.1, hj.2]
  have hx : 0 ≤ 2 * L * Real.sqrt β := by positivity
  have hfactor : 1 ≤ (M : ℝ) * (1 + 2 * L * Real.sqrt β) * Λ := by
    by_cases hM2 : 2 ≤ M
    · have hM2r : (2 : ℝ) ≤ M := by exact_mod_cast hM2
      calc (1 : ℝ) = 2 * 1 * (1 / 2) := by norm_num
        _ ≤ (M : ℝ) * (1 + 2 * L * Real.sqrt β) * Λ := by gcongr; linarith
    have hM1 : M = 1 := by omega
    subst M
    rcases le_or_gt L (1 / (2 * Real.sqrt β)) with hshort | hlong
    · have hone : 1 ≤ Λ := by
        rw [← hatom.mass_sum]
        apply hmass Finset.univ (by simpa using hatom.two_le)
        intro i _ j _
        obtain ⟨a, ki, ha⟩ := hcover i
        obtain ⟨b, kj, hb⟩ := hcover j
        have ha0 : a = 0 := Fin.fin_one_eq_zero a
        have hb0 : b = 0 := Fin.fin_one_eq_zero b
        rw [ha0] at ha
        rw [hb0] at hb
        refine ⟨ki - kj, ?_⟩
        rw [Int.cast_sub]
        have heq : θ i - θ j + 2 * π * ((ki : ℝ) - kj) =
            (θ i + 2 * π * (ki : ℝ)) - (θ j + 2 * π * (kj : ℝ)) := by ring
        rw [heq]
        have hab : |(θ i + 2 * π * (ki : ℝ)) - (θ j + 2 * π * (kj : ℝ))| ≤ L :=
          abs_le.mpr ⟨by linarith [hlength 0, ha.1, hb.2],
            by linarith [hlength 0, ha.2, hb.1]⟩
        exact hab.trans hshort
      simp only [Nat.cast_one, one_mul]
      nlinarith [mul_nonneg hx hΛ0]
    · have hwidth : 1 < L * (2 * Real.sqrt β) :=
        (div_lt_iff₀ (by positivity : 0 < 2 * Real.sqrt β)).mp hlong
      simp only [Nat.cast_one, one_mul]
      nlinarith [mul_le_mul_of_nonneg_right hΛ (by linarith : 0 ≤ 1 + 2 * L * Real.sqrt β)]
  have hcount := heavy_atom_count_mul_le_one m (fun i => (hatom.mass_pos i).le)
    hatom.mass_sum ε
  apply le_of_mul_le_mul_right ?_ hε
  change _ ≤ ((M / ε) * (1 + 2 * L * Real.sqrt β) * Λ) * ε
  have heq : ((M / ε) * (1 + 2 * L * Real.sqrt β) * Λ) * ε =
      (M : ℝ) * (1 + 2 * L * Real.sqrt β) * Λ := by field_simp
  rw [heq]
  have hcount' : (({i : Idx N | ε ≤ m i} : Set (Idx N)).ncard : ℝ) * ε ≤ 1 := by
    simpa [mul_comm] using hcount
  exact hcount'.trans hfactor

/-- Every conditional-corollary hypothesis is satisfied by the actual
antipodal SOPD measure, `β = 1`, `C = exp 1`, two singleton cover intervals,
and `ε = 1/2`. The cluster input is itself proved by the unit-mass bound,
since `C exp(-β) = 1`; no open result enters the witness.
Source: arXiv:2601.21366v2, §3.2, `cor: bound`. -/
example : (0 : ℝ) ≤ Real.exp 1 ∧ (0 : ℝ) < 1 ∧
    IsAtomicOnCircle 2 (fun _ => 1 / 2) antipodalCircleAngles antipodalCircleProb ∧
    IsSOPD 1 0 0 0 0 antipodalCircleProb ∧
    (∀ S : Finset (Idx 2), 2 ≤ S.card →
      (∀ i ∈ S, ∀ j ∈ S, ∃ k : ℤ,
        |antipodalCircleAngles i - antipodalCircleAngles j + 2 * π * (k : ℝ)| ≤
          1 / (2 * Real.sqrt 1)) →
      ∑ i ∈ S, (fun _ : Idx 2 => (1 / 2 : ℝ)) i ≤ 0.5742 + Real.exp 1 * Real.exp (-1)) ∧
    1 ≤ (2 : ℕ) ∧
    (∀ j : Idx 2, antipodalCircleAngles j - antipodalCircleAngles j ≤ 0) ∧
    (∀ i : Idx 2, ∃ j : Idx 2, ∃ k : ℤ,
      antipodalCircleAngles i + 2 * π * (k : ℝ) ∈
        Set.Icc (antipodalCircleAngles j) (antipodalCircleAngles j)) ∧
    (0 : ℝ) < 1 / 2 := by
  refine ⟨(Real.exp_pos 1).le, one_pos, isAtomicOnCircle_antipodalCircleProb,
    isSOPD_antipodalCircleProb 1 one_pos, ?_, by norm_num,
    fun _ => by simp, fun i => ⟨i, 0, by simp⟩, by norm_num⟩
  intro S _ _
  have hsum : ∑ i ∈ S, (fun _ : Idx 2 => (1 / 2 : ℝ)) i ≤
      ∑ i : Idx 2, (fun _ : Idx 2 => (1 / 2 : ℝ)) i :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ S) (fun _ _ _ => by norm_num)
  rw [Real.exp_neg, mul_inv_cancel₀ (Real.exp_pos 1).ne']
  norm_num at hsum ⊢
  linarith

end Transformer.Perceptron
