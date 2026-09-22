/-
# The emergence of clusters in self-attention dynamics — the drift in `d = 1`

The estimates behind `l:auxiliary` of arXiv:2305.05465v6, for the largest
token `x_N` of `e:Idnonresca` on the line: its drift
`Σ_j P_Nj x_j = Σ_j softmax(x_N x_l)_j x_j` is at least `x_N - n/x_N`, and at
least `x_N/n - n e^{-x_N²}/x_N` (`e:pypyly`).  Both come from
`P_Nj ≤ e^{x_N(x_j - x_N)}` and `e^{az} z ≥ -1/a`.

Also the passage from `e:Idnonresca` in `EucSpace 1` to its coordinate: the
scalar equation `ẋ_i = Σ_j softmax(x_i x_l)_j x_j`, and the ordering
`x_1 < … < x_n`, preserved for `t ≥ 0` by `l:distnondec`.

Source: arXiv:2305.05465v6, `l:auxiliary`, `e:infdist`.
-/

import Transformer.Clusters.Section7_Unbounded
import Transformer.Clusters.Section7_DistNonDec

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {n m : ℕ}

/-- `e^{az} z ≥ -1/a` for `a > 0`: `w e^{-w} ≤ 1` for `w = -az`. -/
theorem neg_inv_le_exp_mul_mul {a : ℝ} (ha : 0 < a) (z : ℝ) :
    -(1 / a) ≤ Real.exp (a * z) * z := by
  have h := Real.add_one_le_exp (-(a * z))
  have h1 : Real.exp (a * z) * Real.exp (-(a * z)) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  rw [neg_le, ← neg_mul, le_div_iff₀ ha]
  have he := Real.exp_pos (a * z)
  nlinarith [mul_le_mul_of_nonneg_left h he.le]

/-- The softmax weight of `j` under scores `a y_l`, with `y_N = a` the largest
score, is at most `e^{a(y_j - a)}`. -/
theorem softmaxWeight_le_exp (y : Idx (m + 1) → ℝ) (N j : Idx (m + 1)) :
    Perspective.softmaxWeight (fun l => y N * y l) j ≤ Real.exp (y N * (y j - y N)) := by
  unfold Perspective.softmaxWeight
  have hZ : Real.exp (y N * y N) ≤ ∑ k : Idx (m + 1), Real.exp (y N * y k) :=
    Finset.single_le_sum (f := fun k => Real.exp (y N * y k))
      (fun k _ => (Real.exp_pos _).le) (Finset.mem_univ N)
  rw [div_le_iff₀ ((Real.exp_pos _).trans_le hZ), mul_sub, Real.exp_sub,
    div_mul_eq_mul_div, le_div_iff₀ (Real.exp_pos _)]
  exact mul_le_mul_of_nonneg_left hZ (Real.exp_pos _).le

/-- The drift of a positive token, near its value: with `a = y_N > 0`,
`Σ_j P_Nj y_j ≥ a - n/a`.  The token need not be the largest: the terms with
`y_j > a` only help. -/
theorem sub_le_drift (y : Idx (m + 1) → ℝ) (N : Idx (m + 1)) (ha : 0 < y N) :
    y N - (m + 1) / y N ≤
      ∑ j, Perspective.softmaxWeight (fun l => y N * y l) j * y j := by
  have hs := Perspective.sum_softmaxWeight (Nat.succ_pos m) (fun l => y N * y l)
  have hD : ∑ j, Perspective.softmaxWeight (fun l => y N * y l) j * y j =
      y N + ∑ j, Perspective.softmaxWeight (fun l => y N * y l) j * (y j - y N) := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hs]
    ring
  have hj : ∀ j, -(1 / y N) ≤
      Perspective.softmaxWeight (fun l => y N * y l) j * (y j - y N) := by
    intro j
    rcases le_or_gt (y j) (y N) with h | h
    · exact (neg_inv_le_exp_mul_mul ha (y j - y N)).trans
        (mul_le_mul_of_nonpos_right (softmaxWeight_le_exp y N j) (sub_nonpos.2 h))
    · exact (neg_nonpos.2 (by positivity)).trans
        (mul_nonneg (Perspective.softmaxWeight_nonneg _ _) (sub_nonneg.2 h.le))
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hj j
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
  push_cast at hsum
  rw [hD]
  have : ((m : ℝ) + 1) * -(1 / y N) = -((m + 1) / y N) := by ring
  linarith

/-- The drift of a positive token, far from its value (`e:pypyly`,
`e:lowerboundxi`): with `a = y_N > 0`, `Σ_j P_Nj y_j ≥ a/n - n e^{-a²}/a`.
The weight of the largest coordinate `y_M ≥ a` is at least `1/n`. -/
theorem div_sub_le_drift (y : Idx (m + 1) → ℝ) (N : Idx (m + 1)) (ha : 0 < y N) :
    y N / (m + 1) - (m + 1) * (Real.exp (-y N ^ 2) / y N) ≤
      ∑ j, Perspective.softmaxWeight (fun l => y N * y l) j * y j := by
  set p := Perspective.softmaxWeight (fun l => y N * y l)
  set ε := Real.exp (-y N ^ 2) / y N
  have hε : 0 ≤ ε := by positivity
  -- the largest coordinate `y_M` has `P_NM ≥ 1/n`
  obtain ⟨M, hmax⟩ := Finite.exists_max y
  have hpN : 1 / ((m : ℝ) + 1) ≤ p M := by
    have hZ : ∑ k : Idx (m + 1), Real.exp (y N * y k) ≤ ((m : ℝ) + 1) * Real.exp (y N * y M) := by
      have : ∑ k : Idx (m + 1), Real.exp (y N * y k) ≤ ∑ _k : Idx (m + 1), Real.exp (y N * y M) :=
        Finset.sum_le_sum fun k _ => Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left (hmax k) ha.le)
      simpa using this
    have hZ0 := Perspective.softmaxPartition_pos (Nat.succ_pos m) (fun l => y N * y l)
    show 1 / ((m : ℝ) + 1) ≤ Real.exp (y N * y M) / ∑ k : Idx (m + 1), Real.exp (y N * y k)
    rw [div_le_div_iff₀ (by positivity) hZ0]
    linarith
  -- every other term is at least `-ε`
  have hj : ∀ j, -ε ≤ p j * y j := by
    intro j
    rcases le_or_gt 0 (y j) with h | h
    · exact (neg_nonpos.2 hε).trans (mul_nonneg (Perspective.softmaxWeight_nonneg _ _) h)
    · have h1 : Real.exp (y N * (y j - y N)) = Real.exp (-y N ^ 2) * Real.exp (y N * y j) := by
        rw [← Real.exp_add]; ring_nf
      calc -ε = Real.exp (-y N ^ 2) * -(1 / y N) := by simp only [ε]; ring
        _ ≤ Real.exp (-y N ^ 2) * (Real.exp (y N * y j) * y j) :=
          mul_le_mul_of_nonneg_left (neg_inv_le_exp_mul_mul ha _) (Real.exp_pos _).le
        _ = Real.exp (y N * (y j - y N)) * y j := by rw [h1]; ring
        _ ≤ p j * y j := mul_le_mul_of_nonpos_right (softmaxWeight_le_exp y N j) h.le
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ M)]
  have hsum : -(((m : ℝ) + 1) * ε) ≤ ∑ j ∈ Finset.univ.erase M, p j * y j := by
    have hc : ((Finset.univ.erase M).card : ℝ) ≤ (m : ℝ) + 1 := by
      have := Finset.card_erase_le (s := (Finset.univ : Finset (Idx (m + 1)))) (a := M)
      simp only [Finset.card_univ, Fintype.card_fin] at this
      exact_mod_cast this
    calc -(((m : ℝ) + 1) * ε) ≤ -(((Finset.univ.erase M).card : ℝ) * ε) := by
          have := mul_le_mul_of_nonneg_right hc hε
          linarith
      _ = ∑ _j ∈ Finset.univ.erase M, -ε := by simp
      _ ≤ _ := Finset.sum_le_sum fun j _ => hj j
  have h1 : y N / ((m : ℝ) + 1) ≤ p M * y M := by
    have h2 : y N / ((m : ℝ) + 1) ≤ y M * (1 / ((m : ℝ) + 1)) := by
      rw [mul_one_div]; exact div_le_div_of_nonneg_right (hmax N) (by positivity)
    have h3 := mul_le_mul_of_nonneg_left hpN (ha.le.trans (hmax N))
    linarith [mul_comm (p M) (y M)]
  linarith

/-- The hypothesis of `neg_inv_le_exp_mul_mul` is satisfiable, at `a = 1`. -/
example : -(1 / 1 : ℝ) ≤ Real.exp (1 * 0) * 0 := neg_inv_le_exp_mul_mul one_pos 0

/-- The hypothesis of `sub_le_drift` and `div_sub_le_drift` is satisfiable:
a single positive score. -/
example : ∃ (y : Idx 1 → ℝ) (N : Idx 1), 0 < y N := ⟨fun _ => 1, 0, one_pos⟩

/-! ### From `EucSpace 1` to the line -/

theorem hasDerivAt_coord (X : ℝ → Idx n → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => X s i 0)
      (∑ j, Perspective.softmaxWeight (fun l => X t i 0 * X t l 0) j * X t j 0) t := by
  have h := (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 1)).hasFDerivAt.comp_hasDerivAt t (hX t i)
  have e : (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 1))
      (∑ j, attentionMatrix (1 : ParamMatrix 1) 1 (X t) i j • X t j) =
      ∑ j, Perspective.softmaxWeight (fun l => X t i 0 * X t l 0) j * X t j 0 := by
    simp only [map_sum, map_smul, smul_eq_mul, attentionMatrix, one_apply_eq_self]
    simp [PiLp.inner_apply, mul_comm]
  rw [e] at h
  exact h

theorem norm_eq_abs_coord (v : EucSpace 1) : ‖v‖ = |v 0| := by
  rw [EuclideanSpace.norm_eq, Fin.sum_univ_one, Real.norm_eq_abs, sq_abs, Real.sqrt_sq_eq_abs]

theorem isOrderedConfig_of_nonneg (X : ℝ → Idx n → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (hord : IsOrderedConfig (X 0)) (t : ℝ) (ht : 0 ≤ t) : IsOrderedConfig (X t) := by
  intro i j hij
  set f : ℝ → ℝ := fun s => X s j 0 - X s i 0
  have hf : Continuous f := continuous_iff_continuousAt.2 fun s =>
    ((hasDerivAt_coord X hX s j).continuousAt).sub (hasDerivAt_coord X hX s i).continuousAt
  have hmono := norm_sub_monotone X hX j i
  have hne : ∀ s, 0 ≤ s → f s ≠ 0 := by
    intro s hs h0
    have h1 := hmono hs
    simp only [norm_eq_abs_coord, PiLp.sub_apply] at h1
    have h2 : f 0 ≠ 0 := (sub_pos.2 (hord i j hij)).ne'
    exact h2 (abs_eq_zero.1 (le_antisymm (h1.trans (by simp [f] at h0; simp [h0])) (abs_nonneg _)))
  by_contra hle
  push Not at hle
  obtain ⟨s, hs, hs0⟩ := intermediate_value_Icc' ht hf.continuousOn
    (show (0 : ℝ) ∈ Set.Icc (f t) (f 0) from ⟨by simp only [f]; linarith, (sub_pos.2 (hord i j hij)).le⟩)
  exact hne s hs.1 hs0
/-- The hypotheses of `hasDerivAt_coord` and `isOrderedConfig_of_nonneg` are
satisfiable: the one-token solution `x(t) = e^t z`. -/
example (z : EucSpace 1) :
    IdNonrescaledDynamics (n := 1) (fun t _ => Real.exp t • z) ∧
      IsOrderedConfig (n := 1) (fun _ => Real.exp 0 • z) ∧ (0 : ℝ) ≤ 0 :=
  ⟨idNonrescaledDynamics_single z, isOrderedConfig_subsingleton _, le_rfl⟩

end Clusters
end Transformer
