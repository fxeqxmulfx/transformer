/-
# The optimal loss improvement at small batch sizes

arXiv:2405.14578, §2.2, Theorem 5, proved in Appendix F.  With `𝓔_i = √(2B/π)μ_i/σ_i` of
eq. (39), the loss improvement of eq. (8) is `½(2B/π)ΣΣμ_i²μ_j²/(σ_iσ_j)/(ΣH_ii + (2B/π)S)`,
the first line of eq. (43) (`lossDropSign_signLin`), and, when `S ≠ 0` and `B > 0`, the last,
`ΔL_max/(1 + B_noise/B)`, eq. (18) (`lossDropSign_signLin_eq`), with
`ΔL_max = ΣΣμ_i²μ_j²/(σ_iσ_j)/(2S)` of eq. (19) (`lossDropSignMax`), which is `M²/(2S)`
(`lossDropSignMax_eq`).  It is at most `ΔL_max` when `ΣH_ii ≥ 0` and `S > 0`
(`lossDropSign_signLin_le`), and tends to it as `B → ∞`
(`tendsto_lossDropSign_signLin_atTop`).  The "≈" of eq. (18), for `B ≪ πσ_i²/(2μ_i²)`, holds as
the one of eq. (13): as `B → 0`, the ratio of eq. (8) to its value at `𝓔_i = √(2B/π)μ_i/σ_i`
tends to `1` when `M ≠ 0` and `ΣH_ii ≠ 0` (`tendsto_lossDropSign_div_lossDropSign_signLin`,
`tendsto_lossDropSign_div_max`): divided by `B/2`, both tend to `(2M/√π)²/(2ΣH_ii)`
(`tendsto_lossDropSign_div`).
-/

import Transformer.Surge.Section2_SmallBatchLimit

open Filter Topology Real

namespace Transformer.Surge

open Transformer.BatchSize

variable {ι : Type*} [Fintype ι]

/-- If `𝓔_i/t → c_i`, `t → 0` and `t ≠ 0`, eq. (8) over `t²` tends to `(Σc_iμ_i)²/(2ΣH_ii)`,
when `ΣH_ii ≠ 0`. -/
theorem tendsto_lossDropSign_div {α : Type*} {l : Filter α} {E : α → ι → ℝ} {t : α → ℝ}
    {c : ι → ℝ} (hE : ∀ i, Tendsto (fun a => E a i / t a) l (𝓝 (c i)))
    (ht : Tendsto t l (𝓝 0)) (ht0 : ∀ᶠ a in l, t a ≠ 0) (μ : ι → ℝ) {H : Matrix ι ι ℝ}
    (hD : ∑ i, H i i ≠ 0) :
    Tendsto (fun a => lossDropSign (E a) μ H / t a ^ 2) l
      (𝓝 ((∑ i, c i * μ i) ^ 2 / (2 * ∑ i, H i i))) := by
  have h1 : Tendsto (fun a => μ ⬝ᵥ E a / t a) l (𝓝 (∑ i, c i * μ i)) := by
    refine (tendsto_finsetSum Finset.univ fun i _ => (hE i).mul_const (μ i)).congr fun a => ?_
    rw [dotProduct, Finset.sum_div]
    exact Finset.sum_congr rfl fun i _ => by ring
  have h := (h1.div_const 2).mul (tendsto_lrSign_div hE ht ht0 μ hD)
  rw [show (∑ i, c i * μ i) / 2 * ((∑ i, c i * μ i) / ∑ i, H i i) =
    (∑ i, c i * μ i) ^ 2 / (2 * ∑ i, H i i) by ring] at h
  refine h.congr fun a => ?_
  rw [← dotProduct_div_two_mul_lrSign]
  ring

variable [DecidableEq ι] {μ σ : ι → ℝ} {H : Matrix ι ι ℝ} {B : ℝ}

/-- `ΔL_max = ΣΣμ_i²μ_j²/(σ_iσ_j)/(2S)`, eq. (19). -/
noncomputable def lossDropSignMax (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  (∑ i, ∑ j, μ i ^ 2 * μ j ^ 2 / (σ i * σ j)) / (2 * offSum μ σ H)

/-- `ΔL_max = M²/(2S)`, with `M = Σμ_i²/σ_i`. -/
theorem lossDropSignMax_eq (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) :
    lossDropSignMax μ σ H = snrSum μ σ ^ 2 / (2 * offSum μ σ H) := by
  have h : ∑ i, ∑ j, μ i ^ 2 * μ j ^ 2 / (σ i * σ j) = snrSum μ σ ^ 2 := by
    rw [snrSum, sq, Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ =>
      (div_mul_div_comm _ _ _ _).symm
  rw [lossDropSignMax, h]

/-- `ΔL_max ≥ 0` when `S > 0`. -/
theorem lossDropSignMax_nonneg (hS : 0 < offSum μ σ H) : 0 ≤ lossDropSignMax μ σ H := by
  rw [lossDropSignMax_eq]
  exact div_nonneg (sq_nonneg _) (by linarith)

/-- The hypothesis of `lossDropSignMax_nonneg` is satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := lossDropSignMax_nonneg (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp [offSum, Fin.sum_univ_two])

/-- `B_noise ≥ 0` when `ΣH_ii ≥ 0` and `S > 0`. -/
theorem noiseBatch_nonneg (hD : 0 ≤ ∑ i, H i i) (hS : 0 < offSum μ σ H) :
    0 ≤ noiseBatch μ σ H :=
  div_nonneg (mul_nonneg Real.pi_pos.le hD) (mul_nonneg zero_le_two hS.le)

/-- The hypotheses of `noiseBatch_nonneg` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := noiseBatch_nonneg (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two])

/-- **Eq. (43)**, first line: with `𝓔_i = √(2B/π)μ_i/σ_i`, eq. (8) is
`ΔL_opt(B) = ½(2B/π)ΣΣμ_i²μ_j²/(σ_iσ_j)/(ΣH_ii + (2B/π)S)`, `B ≥ 0`. -/
theorem lossDropSign_signLin (μ σ : ι → ℝ) (H : Matrix ι ι ℝ) (hB : 0 ≤ B) :
    lossDropSign (signLin μ σ B) μ H = 1 / 2 * (2 * B / π *
      ∑ i, ∑ j, μ i ^ 2 * μ j ^ 2 / (σ i * σ j)) / (∑ i, H i i + 2 * B / π * offSum μ σ H) := by
  rw [lossDropSign, signDen_signLin μ σ H hB, ← mul_div_assoc]
  congr 2
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [signLin_mul μ σ hB]
  ring

/-- The hypothesis of `lossDropSign_signLin` is satisfiable: `B = 1`. -/
example (μ σ : Fin 2 → ℝ) (H : Matrix (Fin 2) (Fin 2) ℝ) :=
  lossDropSign_signLin μ σ H zero_le_one

/-- **Theorem 5**, eq. (18), and the rest of eq. (43): when `S ≠ 0`, eq. (8) at
`𝓔_i = √(2B/π)μ_i/σ_i` is `ΔL_opt(B) = ΔL_max/(1 + B_noise/B)`, `B > 0`. -/
theorem lossDropSign_signLin_eq (hS : offSum μ σ H ≠ 0) (hB : 0 < B) :
    lossDropSign (signLin μ σ B) μ H = lossDropSignMax μ σ H / (1 + noiseBatch μ σ H / B) := by
  rw [lossDropSign_signLin μ σ H hB.le, lossDropSignMax, noiseBatch]
  generalize ∑ i, ∑ j, μ i ^ 2 * μ j ^ 2 / (σ i * σ j) = X
  generalize ∑ i, H i i = D
  generalize offSum μ σ H = S at hS ⊢
  have hπ := Real.pi_pos.ne'
  have hB0 := hB.ne'
  have h1 : D + 2 * B / π * S = (π * D + 2 * B * S) / π := by field_simp
  have h2 : 1 + π * D / (2 * S) / B = (π * D + 2 * B * S) / (2 * S * B) := by
    field_simp
    ring
  rw [h1, h2, div_div_eq_mul_div, div_div_eq_mul_div]
  congr 1
  field_simp

/-- The hypotheses of `lossDropSign_signLin_eq` are satisfiable: `μ = σ = 1`, all `H_ij = 1`,
`B = 1`. -/
example := lossDropSign_signLin_eq (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp [offSum, Fin.sum_univ_two]) one_pos

/-- **Eq. (43)**, last line: `ΔL_opt(B) ≤ ΔL_max`, when `ΣH_ii ≥ 0`, `S > 0`, `B > 0`. -/
theorem lossDropSign_signLin_le (hD : 0 ≤ ∑ i, H i i) (hS : 0 < offSum μ σ H) (hB : 0 < B) :
    lossDropSign (signLin μ σ B) μ H ≤ lossDropSignMax μ σ H := by
  rw [lossDropSign_signLin_eq hS.ne' hB]
  exact div_le_self (lossDropSignMax_nonneg hS)
    (le_add_of_nonneg_right (div_nonneg (noiseBatch_nonneg hD hS) hB.le))

/-- The hypotheses of `lossDropSign_signLin_le` are satisfiable: `μ = σ = 1`, all `H_ij = 1`,
`B = 1`. -/
example := lossDropSign_signLin_le (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) one_pos

/-- `ΔL_opt(B) → ΔL_max` as `B → ∞`, when `S ≠ 0`: with `lossDropSign_signLin_le`, `ΔL_max` is
the supremum of eq. (43) over `B > 0`. -/
theorem tendsto_lossDropSign_signLin_atTop (hS : offSum μ σ H ≠ 0) :
    Tendsto (fun B => lossDropSign (signLin μ σ B) μ H) atTop (𝓝 (lossDropSignMax μ σ H)) := by
  have h : Tendsto (fun B => lossDropSignMax μ σ H / (1 + noiseBatch μ σ H / B)) atTop
      (𝓝 (lossDropSignMax μ σ H / (1 + 0))) :=
    tendsto_const_nhds.div (tendsto_const_nhds.add (tendsto_const_nhds.div_atTop tendsto_id))
      (by norm_num)
  rw [add_zero, div_one] at h
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop 0] with B hB
  rw [lossDropSign_signLin_eq hS hB]

/-- The hypothesis of `tendsto_lossDropSign_signLin_atTop` is satisfiable: `μ = σ = 1`, all
`H_ij = 1`. -/
example := tendsto_lossDropSign_signLin_atTop (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp [offSum, Fin.sum_univ_two])

omit [DecidableEq ι] in
/-- **Theorem 5**, eq. (18): as `B → 0⁺`, the loss improvement of eq. (8) over the one with
`𝓔_i(B)` replaced by `√(2B/π)μ_i/σ_i` tends to `1`, when `M ≠ 0` and `ΣH_ii ≠ 0`. -/
theorem tendsto_lossDropSign_div_lossDropSign_signLin (hM : snrSum μ σ ≠ 0)
    (hD : ∑ i, H i i ≠ 0) :
    Tendsto (fun B => lossDropSign (fun i => signMean (μ i) (σ i) B) μ H /
      lossDropSign (signLin μ σ B) μ H) (𝓝[>] 0) (𝓝 1) := by
  have ht : Tendsto (fun B : ℝ => √(B / 2)) (𝓝[>] 0) (𝓝 0) :=
    tendsto_sqrt_half.mono_right nhdsWithin_le_nhds
  have ht0 : ∀ᶠ B : ℝ in 𝓝[>] 0, √(B / 2) ≠ 0 := by
    filter_upwards [self_mem_nhdsWithin] with B hB
    exact (Real.sqrt_pos.2 (half_pos hB)).ne'
  have h1 := tendsto_lossDropSign_div (fun i => tendsto_signMean_div_sqrt (μ i) (σ i)) ht ht0
    μ hD
  have h2 := tendsto_lossDropSign_div (tendsto_signLin_div_sqrt μ σ) ht ht0 μ hD
  have hK : (∑ i, 2 / √π * (μ i / σ i) * μ i) ^ 2 / (2 * ∑ i, H i i) ≠ 0 := by
    rw [show ∑ i, 2 / √π * (μ i / σ i) * μ i = 2 / √π * snrSum μ σ by
      rw [snrSum, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring]
    exact div_ne_zero (pow_ne_zero 2 (mul_ne_zero (div_ne_zero two_ne_zero
      (Real.sqrt_pos.2 Real.pi_pos).ne') hM)) (mul_ne_zero two_ne_zero hD)
  have h := h1.div h2 hK
  rw [div_self hK] at h
  refine h.congr' ?_
  filter_upwards [ht0] with B hB
  exact div_div_div_cancel_right₀ (pow_ne_zero 2 hB) _ _

/-- The hypotheses of `tendsto_lossDropSign_div_lossDropSign_signLin` are satisfiable:
`μ = σ = 1`, `H = 1`. -/
example := tendsto_lossDropSign_div_lossDropSign_signLin (μ := fun _ : Fin 2 => 1)
  (σ := fun _ => 1) (H := 1) (by simp [snrSum]) (by simp)

/-- **Theorem 5**, eq. (18): as `B → 0⁺`, the loss improvement of eq. (8) over
`ΔL_max/(1 + B_noise/B)` tends to `1`, when `ΣH_ii ≠ 0`, `S ≠ 0` and `M ≠ 0`. -/
theorem tendsto_lossDropSign_div_max (hD : ∑ i, H i i ≠ 0) (hS : offSum μ σ H ≠ 0)
    (hM : snrSum μ σ ≠ 0) :
    Tendsto (fun B => lossDropSign (fun i => signMean (μ i) (σ i) B) μ H /
      (lossDropSignMax μ σ H / (1 + noiseBatch μ σ H / B))) (𝓝[>] 0) (𝓝 1) := by
  refine (tendsto_lossDropSign_div_lossDropSign_signLin hM hD).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with B hB
  rw [lossDropSign_signLin_eq hS hB]

/-- The hypotheses of `tendsto_lossDropSign_div_max` are satisfiable: `μ = σ = 1`, all
`H_ij = 1`. -/
example := tendsto_lossDropSign_div_max (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

end Transformer.Surge
