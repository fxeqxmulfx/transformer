/-
# The optimal learning rate at large batch sizes

arXiv:2405.14578, §2.1, Theorem 4, proved in Appendix E.  As `B → ∞`,
`𝓔_i(B) = erf(√(B/2)μ_i/σ_i)` tends to `sign(μ_i/σ_i) = sign(μ_i)`, eq. (42)
(`tendsto_signMean_atTop`), so the learning rate of eq. (9), continuous in `𝓔`
(`tendsto_lrSign`), tends to its value at `𝓔_i = sign(μ_i)`,
`Σ|μ_i|/(Σ(1 - sign(μ_i)²)H_ii + ΣΣ sign(μ_i)sign(μ_j)H_ij)` (`tendsto_lrSign_atTop`).  Eq. (17)
drops the first sum of the denominator, which vanishes when every `μ_i ≠ 0` (`signDen_sign`), as
the condition `B ≫ πσ_i²/(2μ_i²)` presumes: with that hypothesis eq. (17) holds
(`tendsto_lrSign_atTop_eq`), without it it fails (`exists_not_tendsto_lrSign_atTop`).  "When `B` increases infinitely, the optimal learning rate
will eventually converge to a non-zero value": the limit is positive when `μ ≠ 0` and the
Hessian is positive definite (`sum_abs_div_pos`).

"If we make an (unrealistic) assumption that `μ_i/σ_i ≈ sign(μ_i)`, ... the lower bound of
`ε_max` in Theorem 3 will become the one in Theorem 4, which means that the local peak value of
the optimal learning rate is larger than the final convergence value": at `σ_i = |μ_i|`, every
`μ_i ≠ 0`, the bound of eq. (41) is the limit of eq. (17) (`snrSum_abs_div_eq`), so `ε_max` is
at least that limit (`sum_abs_div_le_lrPeak`).  The paper's next remark, that late in training
the limit "is more likely to exceed the local maximum", is a heuristic, not transcribed.
-/

import Transformer.Surge.Section2_PeakRate
import Transformer.Surge.Section2_SmallBatchLimit

open Filter Topology Real

namespace Transformer.Surge

open Transformer.BatchSize

variable {ι : Type*} [Fintype ι] {μ σ : ι → ℝ} {H : Matrix ι ι ℝ}

/-- `sign(x)x = |x|`. -/
theorem sign_mul_self_eq_abs (x : ℝ) : Real.sign x * x = |x| := by
  rcases lt_trichotomy x 0 with h | rfl | h
  · rw [Real.sign_of_neg h, abs_of_neg h]
    ring
  · simp
  · rw [Real.sign_of_pos h, abs_of_pos h, one_mul]

/-- `sign(x)² = 1`, `x ≠ 0`. -/
theorem sign_sq_eq_one {x : ℝ} (hx : x ≠ 0) : Real.sign x ^ 2 = 1 := by
  rcases Real.sign_apply_eq_of_ne_zero x hx with h | h <;> rw [h] <;> norm_num

/-- The hypothesis of `sign_sq_eq_one` is satisfiable: `x = 1`. -/
example := sign_sq_eq_one one_ne_zero

/-- `x/|x| = sign(x)`, `x ≠ 0`. -/
theorem div_abs_eq_sign {x : ℝ} (hx : x ≠ 0) : x / |x| = Real.sign x := by
  rcases lt_or_gt_of_ne hx with h | h
  · rw [Real.sign_of_neg h, abs_of_neg h, div_neg, div_self hx]
  · rw [Real.sign_of_pos h, abs_of_pos h, div_self hx]

/-- The hypothesis of `div_abs_eq_sign` is satisfiable: `x = 1`. -/
example := div_abs_eq_sign one_ne_zero

/-- **Eq. (42)**: `𝓔(B) = erf(√(B/2)μ/σ) → sign(μ/σ) = sign(μ)` as `B → ∞`, `σ > 0`. -/
theorem tendsto_signMean_atTop (μ : ℝ) {σ : ℝ} (hσ : 0 < σ) :
    Tendsto (fun B => signMean μ σ B) atTop (𝓝 (Real.sign μ)) := by
  have hx : Tendsto (fun B : ℝ => √(B / 2)) atTop atTop :=
    tendsto_sqrt_atTop.comp (tendsto_id.atTop_div_const two_pos)
  rcases lt_trichotomy μ 0 with hμ | rfl | hμ
  · rw [Real.sign_of_neg hμ]
    refine (errorFunction_tendsto_atTop.comp ((hx.atTop_mul_const (neg_pos.2 hμ)).atTop_div_const
      hσ)).neg.congr fun B => ?_
    simp only [Function.comp_apply, signMean, mul_neg, neg_div, errorFunction_neg, neg_neg]
  · simp [signMean, errorFunction_zero]
  · rw [Real.sign_of_pos hμ]
    exact errorFunction_tendsto_atTop.comp ((hx.atTop_mul_const hμ).atTop_div_const hσ)

/-- The hypothesis of `tendsto_signMean_atTop` is satisfiable: `σ = 1`. -/
example (μ : ℝ) := tendsto_signMean_atTop μ one_pos

/-- Eq. (9) is continuous in `𝓔` where its denominator is nonzero. -/
theorem tendsto_lrSign {α : Type*} {l : Filter α} {E : α → ι → ℝ} {e : ι → ℝ}
    (hE : ∀ i, Tendsto (fun a => E a i) l (𝓝 (e i))) (μ : ι → ℝ) (hden : signDen e H ≠ 0) :
    Tendsto (fun a => lrSign (E a) μ H) l (𝓝 (lrSign e μ H)) :=
  (tendsto_finsetSum _ fun i _ => (hE i).mul tendsto_const_nhds).div (tendsto_signDen hE H) hden

/-- The hypotheses of `tendsto_lrSign` are satisfiable: `𝓔 = 0`, `H = 1`. -/
example (μ : Fin 2 → ℝ) := tendsto_lrSign (l := atTop (α := ℝ)) (E := fun _ _ => 0) (e := 0)
  (H := 1) (fun _ => tendsto_const_nhds) μ (by simp [signDen])

/-- **Theorem 4**, Appendix E: as `B → ∞`, the learning rate of eq. (9) tends to
`Σ|μ_i|/(Σ(1 - sign(μ_i)²)H_ii + ΣΣ sign(μ_i)sign(μ_j)H_ij)`, its value at `𝓔_i = sign(μ_i)`,
when every `σ_i > 0` and this denominator is nonzero. -/
theorem tendsto_lrSign_atTop (hσ : ∀ i, 0 < σ i)
    (hden : signDen (fun i => Real.sign (μ i)) H ≠ 0) :
    Tendsto (fun B => lrSign (fun i => signMean (μ i) (σ i) B) μ H) atTop
      (𝓝 ((∑ i, |μ i|) / signDen (fun i => Real.sign (μ i)) H)) := by
  have h := tendsto_lrSign (fun i => tendsto_signMean_atTop (μ i) (hσ i)) μ hden
  simp only [lrSign, sign_mul_self_eq_abs] at h
  exact h

/-- The hypotheses of `tendsto_lrSign_atTop` are satisfiable: `μ = σ = 1`, `H = 1`. -/
example := tendsto_lrSign_atTop (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1) (H := 1)
  (fun _ => one_pos) (by simp [signDen])

/-- The denominator of eq. (9) at `𝓔_i = sign(μ_i)` is `ΣΣ sign(μ_i)sign(μ_j)H_ij` when every
`μ_i ≠ 0`. -/
theorem signDen_sign (hμ : ∀ i, μ i ≠ 0) (H : Matrix ι ι ℝ) :
    signDen (fun i => Real.sign (μ i)) H = ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j := by
  have h (i : ι) : Real.sign (μ i) ^ 2 = 1 := sign_sq_eq_one (hμ i)
  simp [signDen, h]

/-- The hypothesis of `signDen_sign` is satisfiable: `μ = 1`. -/
example (H : Matrix (Fin 2) (Fin 2) ℝ) := signDen_sign (μ := fun _ => 1) (fun _ => one_ne_zero) H

/-- **Theorem 4**, eq. (17): as `B → ∞`, the learning rate of eq. (9) tends to
`Σ|μ_i|/ΣΣ sign(μ_i)sign(μ_j)H_ij`, when every `μ_i ≠ 0`, every `σ_i > 0` and the denominator is
nonzero.  The paper's condition `B ≫ πσ_i²/(2μ_i²)` presumes `μ_i ≠ 0` without stating it; at
`μ_i = 0`, `𝓔_i → 0` leaves `H_ii` in the denominator, and eq. (17) fails
(`exists_not_tendsto_lrSign_atTop`). -/
theorem tendsto_lrSign_atTop_eq (hμ : ∀ i, μ i ≠ 0) (hσ : ∀ i, 0 < σ i)
    (hden : ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j ≠ 0) :
    Tendsto (fun B => lrSign (fun i => signMean (μ i) (σ i) B) μ H) atTop
      (𝓝 ((∑ i, |μ i|) / ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j)) := by
  rw [← signDen_sign hμ] at hden ⊢
  exact tendsto_lrSign_atTop hσ hden

/-- The hypotheses of `tendsto_lrSign_atTop_eq` are satisfiable: `μ = σ = 1`, `H = 1`. -/
example := tendsto_lrSign_atTop_eq (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1) (H := 1)
  (fun _ => one_ne_zero) (fun _ => one_pos) (by simp [Fin.sum_univ_two])

/-- **Theorem 4**, eq. (17), is false without `μ_i ≠ 0`: at `μ = (1, 0)`, `σ = (1, 1)` and the
positive semidefinite `H = 1`, the learning rate of eq. (9) tends to `1/2`, not to
`Σ|μ_i|/ΣΣ sign(μ_i)sign(μ_j)H_ij = 1`. -/
theorem exists_not_tendsto_lrSign_atTop : ∃ (μ σ : Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ),
    (∀ i, 0 < σ i) ∧ H.PosSemidef ∧ ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j ≠ 0 ∧
    ¬Tendsto (fun B => lrSign (fun i => signMean (μ i) (σ i) B) μ H) atTop
      (𝓝 ((∑ i, |μ i|) / ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j)) := by
  refine ⟨![1, 0], fun _ => 1, 1, fun _ => one_pos, Matrix.PosSemidef.one,
    by simp [Fin.sum_univ_two], fun h => ?_⟩
  have h2 := tendsto_lrSign_atTop (μ := ![1, 0]) (σ := fun _ => 1) (H := 1) (fun _ => one_pos)
    (by simp [signDen, Fin.sum_univ_two])
  have := tendsto_nhds_unique h h2
  norm_num [signDen, Fin.sum_univ_two] at this

/-- **Theorem 4**: "the optimal learning rate will eventually converge to a non-zero value": the
limit of eq. (17) is positive when `μ ≠ 0` and the Hessian is positive definite. -/
theorem sum_abs_div_pos (hμ : μ ≠ 0) (hH : H.PosDef) :
    0 < (∑ i, |μ i|) / ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j := by
  obtain ⟨i, hi⟩ := Function.ne_iff.1 hμ
  have hs : (fun i => Real.sign (μ i)) ≠ 0 :=
    Function.ne_iff.2 ⟨i, by simpa [Real.sign_eq_zero_iff] using hi⟩
  rw [sum_sum_mul_eq (fun i => Real.sign (μ i)) H]
  refine div_pos (Finset.sum_pos' (fun j _ => abs_nonneg _) ⟨i, Finset.mem_univ _,
    abs_pos.2 hi⟩) ?_
  simpa using hH.dotProduct_mulVec_pos hs

/-- The hypotheses of `sum_abs_div_pos` are satisfiable: `μ = 1`, `H = 1`. -/
example := sum_abs_div_pos (μ := fun _ : Fin 2 => 1) (H := 1) (Function.ne_iff.2 ⟨0, by simp⟩)
  Matrix.PosDef.one

/-- §2.1, after Theorem 4: at `σ_i = |μ_i|`, where `μ_i/σ_i = sign(μ_i)`, the lower bound of
`ε_max` in eq. (41) is the limit of eq. (17), when every `μ_i ≠ 0`. -/
theorem snrSum_abs_div_eq [DecidableEq ι] (hμ : ∀ i, μ i ≠ 0) (H : Matrix ι ι ℝ) :
    snrSum μ (fun i => |μ i|) /
        ∑ i, ∑ j, (if i = j then 1 else μ i * μ j / (|μ i| * |μ j|)) * H i j =
      (∑ i, |μ i|) / ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j := by
  congr 1
  · exact Finset.sum_congr rfl fun i _ => by rw [← sq_abs, sq, mul_self_div_self]
  · refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    split_ifs with h
    · subst h
      rw [← sq, sign_sq_eq_one (hμ i)]
    · rw [mul_div_mul_comm, div_abs_eq_sign (hμ i), div_abs_eq_sign (hμ j)]

/-- The hypothesis of `snrSum_abs_div_eq` is satisfiable: `μ = 1`. -/
example (H : Matrix (Fin 2) (Fin 2) ℝ) :=
  snrSum_abs_div_eq (μ := fun _ => 1) (fun _ => one_ne_zero) H

/-- §2.1, after Theorem 4: "if we make an (unrealistic) assumption that `μ_i/σ_i ≈ sign(μ_i)`,
... the local peak value of the optimal learning rate is larger than the final convergence
value": at `σ_i = |μ_i|`, `ε_max` is at least the limit of eq. (17), when every `μ_i ≠ 0`,
`ΣH_ii > 0` and `S > 0`. -/
theorem sum_abs_div_le_lrPeak [DecidableEq ι] (hμ : ∀ i, μ i ≠ 0) (hD : 0 < ∑ i, H i i)
    (hS : 0 < offSum μ (fun i => |μ i|) H) :
    (∑ i, |μ i|) / ∑ i, ∑ j, Real.sign (μ i) * Real.sign (μ j) * H i j ≤
      lrPeak μ (fun i => |μ i|) H := by
  rw [← snrSum_abs_div_eq hμ]
  exact div_le_lrPeak hD hS (Finset.sum_nonneg fun i _ => by positivity)

/-- The hypotheses of `sum_abs_div_le_lrPeak` are satisfiable: `μ = 1`, all `H_ij = 1`. -/
example := sum_abs_div_le_lrPeak (μ := fun _ : Fin 2 => 1) (H := Matrix.of fun _ _ => 1)
  (fun _ => one_ne_zero) (by simp) (by simp [offSum, Fin.sum_univ_two])

end Transformer.Surge
