/-
# The peak of the optimal learning rate

arXiv:2405.14578, §2.1, Theorem 3, proved in Appendix D, with `ΣH_ii > 0` and `S > 0`.  The
learning rate `ε(B) = ε_max/(½(√(B_noise/B) + √(B/B_noise)))` of eq. (13) is at most `ε_max`
(`lrSign_signLin_le`), reaches it at `B_peak = B_noise`, eqs. (15) and (16)
(`lrSign_signLin_noiseBatch`), and only there when `M > 0` (`lrSign_signLin_lt`).  It increases
up to `B_noise` and decreases after it (`strictMonoOn_lrSign_signLin`,
`strictAntiOn_lrSign_signLin`): the interval where "the batch size becomes larger and the optimal
learning rate needs to be reduced", the surge of the intro and of the summary's first item.

Eq. (41) writes `ε_max = M/(2√S√ΣH_ii)` (`lrPeak_eq`) and, by the AM-GM inequality, bounds it
below by `M/ΣΣ_{i,j}(μ_iμ_j/(σ_iσ_j) if i ≠ j, 1 if i = j)H_ij` (`div_le_lrPeak`).
-/

import Transformer.Surge.Section2_SmallBatch

open Real

namespace Transformer.Surge

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {μ σ : ι → ℝ} {H : Matrix ι ι ℝ} {B : ℝ}

/-- `B_noise > 0` when `ΣH_ii > 0` and `S > 0`. -/
theorem noiseBatch_pos (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) : 0 < noiseBatch μ σ H :=
  div_pos (mul_pos Real.pi_pos hD) (mul_pos two_pos hS)

/-- `ε_max > 0` when `ΣH_ii > 0`, `S > 0` and `M > 0`. -/
theorem lrPeak_pos (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hM : 0 < snrSum μ σ) :
    0 < lrPeak μ σ H :=
  div_pos (mul_pos (Real.sqrt_pos.2 (div_pos (noiseBatch_pos hD hS) (by positivity))) hM) hD

/-- The hypotheses of `lrPeak_pos` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := lrPeak_pos (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1) (H := Matrix.of fun _ _ => 1)
  (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

/-- **Theorem 3**, eq. (13): `ε(B) ≤ ε_max`, when `M ≥ 0`, `B > 0`. -/
theorem lrSign_signLin_le (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hM : 0 ≤ snrSum μ σ)
    (hB : 0 < B) : lrSign (signLin μ σ B) μ H ≤ lrPeak μ σ H := by
  rw [lrSign_signLin_eq_peak hD hS hB]
  exact div_le_self (div_nonneg (mul_nonneg (Real.sqrt_nonneg _) hM) hD.le)
    (one_le_half_sqrt_add (noiseBatch_pos hD hS) hB)

/-- The hypotheses of `lrSign_signLin_le` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := lrSign_signLin_le (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two])
  (by simp [snrSum]) one_pos

/-- **Theorem 3**, eqs. (15) and (16): at `B = B_noise`, `ε(B) = ε_max`. -/
theorem lrSign_signLin_noiseBatch (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) :
    lrSign (signLin μ σ (noiseBatch μ σ H)) μ H = lrPeak μ σ H := by
  rw [lrSign_signLin_eq_peak hD hS (noiseBatch_pos hD hS), div_self (noiseBatch_pos hD hS).ne']
  norm_num

/-- The hypotheses of `lrSign_signLin_noiseBatch` are satisfiable: `μ = σ = 1`, all
`H_ij = 1`. -/
example := lrSign_signLin_noiseBatch (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two])

/-- **Theorem 3**, eq. (15), `B_peak = B_noise`: when `M > 0`, `ε(B) < ε_max` at every other
`B > 0`. -/
theorem lrSign_signLin_lt (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hM : 0 < snrSum μ σ)
    (hB : 0 < B) (hBN : B ≠ noiseBatch μ σ H) : lrSign (signLin μ σ B) μ H < lrPeak μ σ H := by
  rw [lrSign_signLin_eq_peak hD hS hB]
  exact div_lt_self (lrPeak_pos hD hS hM) (one_lt_half_sqrt_add (noiseBatch_pos hD hS) hB hBN)

/-- The hypotheses of `lrSign_signLin_lt` are satisfiable: `μ = σ = 1`, all `H_ij = 1`, `B = 1`,
where `B_noise = π/2`. -/
example := lrSign_signLin_lt (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two])
  (by simp [snrSum]) one_pos (by
    simp [noiseBatch, offSum, Fin.sum_univ_two]
    linarith [Real.pi_gt_three])

/-- **Eq. (41)**: `ε_max = M/(2√S√ΣH_ii)`, when `ΣH_ii > 0` and `S > 0`. -/
theorem lrPeak_eq (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) :
    lrPeak μ σ H = snrSum μ σ / (2 * √(offSum μ σ H) * √(∑ i, H i i)) := by
  unfold lrPeak noiseBatch
  generalize ∑ i, H i i = D at hD ⊢
  generalize offSum μ σ H = S at hS ⊢
  obtain ⟨a, ha, rfl⟩ : ∃ a, 0 < a ∧ D = a ^ 2 :=
    ⟨√D, Real.sqrt_pos.2 hD, (Real.sq_sqrt hD.le).symm⟩
  obtain ⟨b, hb, rfl⟩ : ∃ b, 0 < b ∧ S = b ^ 2 :=
    ⟨√S, Real.sqrt_pos.2 hS, (Real.sq_sqrt hS.le).symm⟩
  rw [show π * a ^ 2 / (2 * b ^ 2) / (2 * π) = (a / (2 * b)) ^ 2 by field_simp,
    Real.sqrt_sq (by positivity), Real.sqrt_sq hb.le, Real.sqrt_sq ha.le]
  field_simp

/-- The hypotheses of `lrPeak_eq` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := lrPeak_eq (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1) (H := Matrix.of fun _ _ => 1)
  (by simp) (by simp [offSum, Fin.sum_univ_two])

/-- `ΣΣ_{i,j}(μ_iμ_j/(σ_iσ_j) if i ≠ j, 1 if i = j)H_ij = S + ΣH_ii`. -/
theorem sum_sum_ite_one (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) :
    ∑ i, ∑ j, (if i = j then 1 else μ i * μ j / (σ i * σ j)) * H i j =
      offSum μ σ H + ∑ i, H i i := by
  rw [offSum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have h (j : ι) : (if i = j then 1 else μ i * μ j / (σ i * σ j)) * H i j =
      (if i = j then 0 else μ i * μ j / (σ i * σ j)) * H i j + if i = j then H i j else 0 := by
    split_ifs <;> ring
  rw [Finset.sum_congr rfl fun j _ => h j, Finset.sum_add_distrib, Finset.sum_ite_eq]
  simp

/-- **Eq. (41)**, by the AM-GM inequality:
`ε_max ≥ M/ΣΣ_{i,j}(μ_iμ_j/(σ_iσ_j) if i ≠ j, 1 if i = j)H_ij`, when `M ≥ 0`. -/
theorem div_le_lrPeak (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hM : 0 ≤ snrSum μ σ) :
    snrSum μ σ / ∑ i, ∑ j, (if i = j then 1 else μ i * μ j / (σ i * σ j)) * H i j ≤
      lrPeak μ σ H := by
  rw [sum_sum_ite_one, lrPeak_eq hD hS]
  refine div_le_div_of_nonneg_left hM
    (mul_pos (mul_pos two_pos (Real.sqrt_pos.2 hS)) (Real.sqrt_pos.2 hD)) ?_
  nlinarith [sq_nonneg (√(offSum μ σ H) - √(∑ i, H i i)), Real.sq_sqrt hS.le,
    Real.sq_sqrt hD.le]

/-- The hypotheses of `div_le_lrPeak` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := div_le_lrPeak (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

/-- `√B/(1 + B/N) = N√B/(N + B)`, `N > 0`, `B ≥ 0`. -/
theorem sqrt_div_one_add {N : ℝ} (hN : 0 < N) (hB : 0 ≤ B) :
    √B / (1 + B / N) = N * √B / (N + B) := by
  have := hN.ne'
  have : N + B ≠ 0 := by linarith
  field_simp

/-- `N√B/(N + B) < N√B'/(N + B')` for `0 ≤ B < B'` with `√B√B' < N`. -/
theorem mul_sqrt_div_lt {N B' : ℝ} (hN : 0 < N) (hB : 0 ≤ B) (hBB' : B < B')
    (h : √B * √B' < N) : N * √B / (N + B) < N * √B' / (N + B') := by
  have huv : √B < √B' := Real.sqrt_lt_sqrt hB hBB'
  have h1 : √B * √B' ^ 2 = √B * B' := by rw [Real.sq_sqrt (hB.trans hBB'.le)]
  have h2 : √B ^ 2 * √B' = B * √B' := by rw [Real.sq_sqrt hB]
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_pos hN (mul_pos (sub_pos.2 huv) (sub_pos.2 h))]

/-- The hypotheses of `mul_sqrt_div_lt` are satisfiable: `N = 2`, `B = 0`, `B' = 1`. -/
example := mul_sqrt_div_lt (N := 2) two_pos le_rfl one_pos (by simp)

/-- `N√B'/(N + B') < N√B/(N + B)` for `0 ≤ B < B'` with `N < √B√B'`. -/
theorem mul_sqrt_div_lt' {N B' : ℝ} (hN : 0 < N) (hB : 0 ≤ B) (hBB' : B < B')
    (h : N < √B * √B') : N * √B' / (N + B') < N * √B / (N + B) := by
  have huv : √B < √B' := Real.sqrt_lt_sqrt hB hBB'
  have h1 : √B * √B' ^ 2 = √B * B' := by rw [Real.sq_sqrt (hB.trans hBB'.le)]
  have h2 : √B ^ 2 * √B' = B * √B' := by rw [Real.sq_sqrt hB]
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_pos hN (mul_pos (sub_pos.2 huv) (sub_pos.2 h))]

/-- The hypotheses of `mul_sqrt_div_lt'` are satisfiable: `N = 1`, `B = 4`, `B' = 9`. -/
example := mul_sqrt_div_lt' (N := 1) (B := 4) (B' := 9) one_pos (by norm_num) (by norm_num) (by
  rw [show (4 : ℝ) = 2 ^ 2 by norm_num, show (9 : ℝ) = 3 ^ 2 by norm_num,
    Real.sqrt_sq (by norm_num), Real.sqrt_sq (by norm_num)]
  norm_num)

/-- **Theorem 3**: `ε(B)` increases on `(0, B_noise]`, when `M > 0`. -/
theorem strictMonoOn_lrSign_signLin (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H)
    (hM : 0 < snrSum μ σ) :
    StrictMonoOn (fun B => lrSign (signLin μ σ B) μ H) (Set.Ioc 0 (noiseBatch μ σ H)) := by
  intro B hB B' hB' hlt
  have hN := noiseBatch_pos hD hS
  simp only [lrSign_signLin_eq hD.ne' hB.1.le, lrSign_signLin_eq hD.ne' hB'.1.le]
  refine mul_lt_mul_of_pos_right ?_ (div_pos (mul_pos (Real.sqrt_pos.2 (by positivity)) hM) hD)
  rw [sqrt_div_one_add hN hB.1.le, sqrt_div_one_add hN hB'.1.le]
  refine mul_sqrt_div_lt hN hB.1.le hlt ?_
  calc √B * √B' < √B' * √B' := mul_lt_mul_of_pos_right (Real.sqrt_lt_sqrt hB.1.le hlt)
        (Real.sqrt_pos.2 hB'.1)
    _ = B' := Real.mul_self_sqrt hB'.1.le
    _ ≤ noiseBatch μ σ H := hB'.2

/-- The hypotheses of `strictMonoOn_lrSign_signLin` are satisfiable: `μ = σ = 1`, all
`H_ij = 1`. -/
example := strictMonoOn_lrSign_signLin (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

/-- **Theorem 3**: `ε(B)` decreases on `[B_noise, ∞)`, when `M > 0`: past its peak, a larger
batch needs a smaller learning rate. -/
theorem strictAntiOn_lrSign_signLin (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H)
    (hM : 0 < snrSum μ σ) :
    StrictAntiOn (fun B => lrSign (signLin μ σ B) μ H) (Set.Ici (noiseBatch μ σ H)) := by
  intro B hB B' hB' hlt
  have hN := noiseBatch_pos hD hS
  have hB0 : 0 < B := hN.trans_le hB
  simp only [lrSign_signLin_eq hD.ne' hB0.le, lrSign_signLin_eq hD.ne' (hB0.trans hlt).le]
  refine mul_lt_mul_of_pos_right ?_ (div_pos (mul_pos (Real.sqrt_pos.2 (by positivity)) hM) hD)
  rw [sqrt_div_one_add hN hB0.le, sqrt_div_one_add hN (hB0.trans hlt).le]
  refine mul_sqrt_div_lt' hN hB0.le hlt ?_
  calc noiseBatch μ σ H ≤ B := hB
    _ = √B * √B := (Real.mul_self_sqrt hB0.le).symm
    _ < √B * √B' := mul_lt_mul_of_pos_left (Real.sqrt_lt_sqrt hB0.le hlt) (Real.sqrt_pos.2 hB0)

/-- The hypotheses of `strictAntiOn_lrSign_signLin` are satisfiable: `μ = σ = 1`, all
`H_ij = 1`. -/
example := strictAntiOn_lrSign_signLin (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

end Transformer.Surge
