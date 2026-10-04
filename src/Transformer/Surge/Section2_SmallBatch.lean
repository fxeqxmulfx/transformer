/-
# The optimal learning rate at small batch sizes

arXiv:2405.14578, §2.1, Theorem 3, proved in Appendix D.  When `B ≪ πσ_i²/(2μ_i²)`, eq. (39)
replaces `𝓔_i(B)` by `√(2B/π)μ_i/σ_i` (`signLin`).  For any `𝓔`, the diagonal terms of the
denominator of eq. (9) cancel (`signDen_eq`), so eq. (9) becomes
`ε(B) = Σ√(2B/π)μ_i²/σ_i/(ΣH_ii + (2B/π)S)`, with `S = ΣΣ_{i≠j}μ_iμ_jH_ij/(σ_iσ_j)` (`offSum`),
the first line of eq. (40) (`lrSign_signLin`).  With `M = Σμ_i²/σ_i` (`snrSum`),
`B_noise = πΣH_ii/(2S)` of eq. (14) (`noiseBatch`) and `ε_max = √(B_noise/(2π))M/ΣH_ii` of
eq. (16) (`lrPeak`), it is `√B/(1 + B/B_noise)·√(2/π)M/ΣH_ii`, the third line
(`lrSign_signLin_eq`), and `ε_max/(½(√(B_noise/B) + √(B/B_noise)))`, the last line, eq. (13) and
the intro's eq. (2) (`lrSign_signLin_eq_peak`).  The paper notes that `B_noise > 0` and
`ε_max > 0` make `ΣH_ii > 0` and `S > 0`: these are the hypotheses of eq. (13).
-/

import Transformer.Surge.Section2_Theorem2

open Real

namespace Transformer.Surge

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The linearization `√(2B/π)μ_i/σ_i` of `𝓔_i(B)`, eq. (39). -/
noncomputable def signLin (μ σ : ι → ℝ) (B : ℝ) (i : ι) : ℝ := √(2 * B / π) * μ i / σ i

/-- `M = Σμ_i²/σ_i`, the sum of eqs. (13) and (16). -/
noncomputable def snrSum (μ σ : ι → ℝ) : ℝ := ∑ i, μ i ^ 2 / σ i

/-- `S = ΣΣ_{i≠j}μ_iμ_jH_ij/(σ_iσ_j)`, the denominator of eq. (14). -/
noncomputable def offSum (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  ∑ i, ∑ j, (if i = j then 0 else μ i * μ j / (σ i * σ j)) * H i j

/-- `B_noise = πΣH_ii/(2S)`, eq. (14). -/
noncomputable def noiseBatch (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  π * (∑ i, H i i) / (2 * offSum μ σ H)

/-- `ε_max = √(B_noise/(2π))M/ΣH_ii`, eq. (16). -/
noncomputable def lrPeak (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  √(noiseBatch μ σ H / (2 * π)) * snrSum μ σ / ∑ i, H i i

/-- The diagonal terms of the denominator of eqs. (8) and (9) cancel:
`Σ(1 - 𝓔_i²)H_ii + ΣΣ𝓔_i𝓔_jH_ij = ΣH_ii + ΣΣ_{i≠j}𝓔_i𝓔_jH_ij`. -/
theorem signDen_eq (E : ι → ℝ) (H : Matrix ι ι ℝ) :
    signDen E H = ∑ i, H i i + ∑ i, ∑ j, (if i = j then 0 else E i * E j) * H i j := by
  have h (i : ι) : ∑ j, (if i = j then 0 else E i * E j) * H i j =
      ∑ j, E i * E j * H i j - E i ^ 2 * H i i := by
    have h' (j : ι) : (if i = j then 0 else E i * E j) * H i j =
        E i * E j * H i j - if i = j then E i * E j * H i j else 0 := by
      split_ifs <;> ring
    rw [Finset.sum_congr rfl fun j _ => h' j, Finset.sum_sub_distrib, Finset.sum_ite_eq]
    simp [sq]
  simp only [signDen, h, Finset.sum_sub_distrib, sub_mul, one_mul]
  ring

omit [Fintype ι] [DecidableEq ι] in
/-- `(√(2B/π)μ_i/σ_i)(√(2B/π)μ_j/σ_j) = (2B/π)μ_iμ_j/(σ_iσ_j)`, `B ≥ 0`. -/
theorem signLin_mul (μ σ : ι → ℝ) {B : ℝ} (hB : 0 ≤ B) (i j : ι) :
    signLin μ σ B i * signLin μ σ B j = 2 * B / π * (μ i * μ j / (σ i * σ j)) := by
  rw [signLin, signLin]
  linear_combination (μ i * μ j / (σ i * σ j)) *
    Real.mul_self_sqrt (show 0 ≤ 2 * B / π by positivity)

variable {μ σ : ι → ℝ} {H : Matrix ι ι ℝ} {B : ℝ}

/-- **Eq. (40)**, first line: with `𝓔_i = √(2B/π)μ_i/σ_i`, eq. (9) is
`ε(B) = Σ√(2B/π)(μ_i/σ_i)μ_i/(ΣH_ii + (2B/π)S)`, `B ≥ 0`. -/
theorem lrSign_signLin (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) (hB : 0 ≤ B) :
    lrSign (signLin μ σ B) μ H =
      (∑ i, √(2 * B / π) * (μ i / σ i) * μ i) / (∑ i, H i i + 2 * B / π * offSum μ σ H) := by
  rw [lrSign, signDen_eq, offSum, Finset.mul_sum]
  congr 1
  · exact Finset.sum_congr rfl fun i _ => by rw [signLin, mul_div_assoc]
  · congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs
    · ring
    · rw [signLin_mul μ σ hB]
      ring

/-- The hypothesis of `lrSign_signLin` is satisfiable: `B = 1`. -/
example (μ σ : Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ) := lrSign_signLin μ σ H zero_le_one

/-- **Eq. (40)**, third line: `ε(B) = √B/(1 + B/B_noise)·√(2/π)M/ΣH_ii`, when `ΣH_ii ≠ 0`,
`B ≥ 0`. -/
theorem lrSign_signLin_eq (hD : ∑ i, H i i ≠ 0) (hB : 0 ≤ B) :
    lrSign (signLin μ σ B) μ H =
      √B / (1 + B / noiseBatch μ σ H) * (√(2 / π) * snrSum μ σ / ∑ i, H i i) := by
  have hnum : ∑ i, √(2 * B / π) * (μ i / σ i) * μ i = √B * √(2 / π) * snrSum μ σ := by
    rw [show 2 * B / π = B * (2 / π) by ring, Real.sqrt_mul hB, snrSum, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [lrSign_signLin μ σ H hB, hnum, noiseBatch]
  generalize ∑ i, H i i = D at hD ⊢
  generalize offSum μ σ H = S
  rw [div_div_eq_mul_div]
  have hπ := Real.pi_pos.ne'
  rcases eq_or_ne (π * D + 2 * B * S) 0 with h | h
  · have h' : D + 2 * B / π * S = 0 := by
      field_simp
      linarith
    have h'' : 1 + B * (2 * S) / (π * D) = 0 := by
      field_simp
      linarith
    rw [h', h'']
    simp
  · field_simp

/-- The hypotheses of `lrSign_signLin_eq` are satisfiable: `H = 1`, `B = 0`. -/
example (μ σ : Fin 2 → ℝ) := lrSign_signLin_eq (μ := μ) (σ := σ) (H := 1) (by simp) le_rfl

/-- `√(B/N)` and `√(N/B)` are inverse. -/
theorem sqrt_div_inv (N B : ℝ) : √(N / B) = (√(B / N))⁻¹ := by
  rw [← Real.sqrt_inv, inv_div]

/-- `½(√(N/B) + √(B/N)) > 1` when `B ≠ N`: the AM-GM inequality, `N, B > 0`. -/
theorem one_lt_half_sqrt_add {N : ℝ} (hN : 0 < N) (hB : 0 < B) (hBN : B ≠ N) :
    1 < (√(N / B) + √(B / N)) / 2 := by
  have hs : 0 < √(B / N) := Real.sqrt_pos.2 (div_pos hB hN)
  have hs1 : √(B / N) ≠ 1 := fun h => hBN (by
    have h2 := Real.sq_sqrt (div_pos hB hN).le
    rw [h, one_pow, eq_div_iff hN.ne', one_mul] at h2
    exact h2.symm)
  rw [sqrt_div_inv N B]
  generalize √(B / N) = s at hs hs1
  have h : s⁻¹ + s - 2 = (s - 1) ^ 2 / s := by
    field_simp
    ring
  have h2 : 0 < (s - 1) ^ 2 / s :=
    div_pos (lt_of_le_of_ne (sq_nonneg _) (pow_ne_zero 2 (sub_ne_zero.2 hs1)).symm) hs
  linarith

/-- The hypotheses of `one_lt_half_sqrt_add` are satisfiable: `N = 1`, `B = 2`. -/
example := one_lt_half_sqrt_add one_pos two_pos (by norm_num)

/-- `½(√(N/B) + √(B/N)) ≥ 1`: the AM-GM inequality, `N, B > 0`. -/
theorem one_le_half_sqrt_add {N : ℝ} (hN : 0 < N) (hB : 0 < B) :
    1 ≤ (√(N / B) + √(B / N)) / 2 := by
  rcases eq_or_ne B N with rfl | h
  · norm_num [div_self hB.ne']
  · exact (one_lt_half_sqrt_add hN hB h).le

/-- The hypotheses of `one_le_half_sqrt_add` are satisfiable: `N = B = 1`. -/
example := one_le_half_sqrt_add one_pos one_pos

/-- **Theorem 3**, eq. (13), and the intro's eq. (2): when `ΣH_ii > 0` and `S > 0`, eq. (40) is
`ε(B) = ε_max/(½(√(B_noise/B) + √(B/B_noise)))`, `B > 0`. -/
theorem lrSign_signLin_eq_peak (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hB : 0 < B) :
    lrSign (signLin μ σ B) μ H = lrPeak μ σ H /
      ((√(noiseBatch μ σ H / B) + √(B / noiseBatch μ σ H)) / 2) := by
  have hN : 0 < noiseBatch μ σ H := div_pos (mul_pos Real.pi_pos hD) (mul_pos two_pos hS)
  rw [lrSign_signLin_eq hD.ne' hB.le, lrPeak]
  generalize noiseBatch μ σ H = N at hN ⊢
  generalize ∑ i, H i i = D at hD ⊢
  rw [sqrt_div_inv N B]
  have hN0 := hN.ne'
  have key : √B * √(2 / π) = 2 * √(B / N) * √(N / (2 * π)) := by
    rw [← Real.sqrt_mul hB.le, mul_assoc, ← Real.sqrt_mul (div_pos hB hN).le,
      show B / N * (N / (2 * π)) = B * (2 / π) / 4 by field_simp; ring,
      Real.sqrt_div' _ (by norm_num), show (4 : ℝ) = 2 ^ 2 by norm_num,
      Real.sqrt_sq (by norm_num)]
    ring
  have hs : 0 < √(B / N) := Real.sqrt_pos.2 (div_pos hB hN)
  have h1 : 1 + B / N = 1 + √(B / N) ^ 2 := by rw [Real.sq_sqrt (div_pos hB hN).le]
  rw [h1, show √B / (1 + √(B / N) ^ 2) * (√(2 / π) * snrSum μ σ / D) =
    √B * √(2 / π) * snrSum μ σ / (1 + √(B / N) ^ 2) / D by ring, key]
  generalize √(B / N) = s at hs ⊢
  have hD0 := hD.ne'
  have hs0 := hs.ne'
  field_simp

/-- The hypotheses of `lrSign_signLin_eq_peak` are satisfiable: `μ = σ = 1`, all `H_ij = 1`,
`B = 1`. -/
example := lrSign_signLin_eq_peak (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) one_pos

end Transformer.Surge
