/-
# The form of the optimal learning rate in the batch size

arXiv:2405.14578, §2.1, eqs. (11) and (12), and §1, eqs. (3) and (4).  When every `𝓔_i(B)` is
a common `f(B)` times `μ_i/σ_i`, the numerator of eq. (9) is of first order in `f(B)` and its
denominator of second order: eq. (9) is `βf(B)/(f(B)² + γ) = β/(f(B) + γ/f(B))`, eq. (11), with
`β = M/S`, `γ = ΣH_ii/S` (`lrSign_mul`, `lrSign_mul_eq_div`).  The linearization of eq. (39) is
such, with `f(B) = √(2B/π)` (`signLin_eq_mul`), and tends to `0` as `B → ∞`
(`tendsto_lrSign_signLin_atTop`).  "Therefore ... `ε(B) ≠ ε_*/(1 + B_noise/B)^α`", eq. (12):
no `ε_*`, `B_*`, `α` make eq. (A.3) of arXiv:1812.06162, which tends to `ε_*`, equal to it at
every `B > 0` (`not_exists_lrCentral`).

When `B ≪ B_noise`, SGD's `ε_max/(1 + B_noise/B)` of eq. (1) is `(ε_max/B_noise)B`, linear
scaling, eq. (3) (`tendsto_stepOpt_div`), and Adam's eq. (2) is `(2ε_max/√B_noise)√B`, square
root scaling, eq. (4) (`tendsto_peak_div_sqrt`): the ratios tend to `1` as `B → 0`.  So does
the ratio of eq. (9) itself to `(2ε_max/√B_noise)√B`, by Theorem 3 (`tendsto_lrSign_div_sqrt`).
-/

import Transformer.NoiseScale.SectionE_Optimization
import Transformer.Surge.Section2_PeakRate
import Transformer.Surge.Section2_SmallBatchLimit

open Filter Topology Real

namespace Transformer.Surge

open Transformer.NoiseScale

variable {ι : Type*} [Fintype ι] [DecidableEq ι] {μ σ : ι → ℝ} {H : Matrix ι ι ℝ}

/-- **Eq. (11)**: when `𝓔_i = fμ_i/σ_i`, eq. (9) is `βf/(f² + γ)`, `β = M/S`, `γ = ΣH_ii/S`:
its numerator is of first order in `f` and its denominator of second order, `S ≠ 0`. -/
theorem lrSign_mul (μ σ : ι → ℝ) (hS : offSum μ σ H ≠ 0) (f : ℝ) :
    lrSign (fun i => f * (μ i / σ i)) μ H =
      snrSum μ σ / offSum μ σ H * f / (f ^ 2 + (∑ i, H i i) / offSum μ σ H) := by
  have hn : ∑ i, f * (μ i / σ i) * μ i = f * snrSum μ σ := by
    rw [snrSum, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  have hd : signDen (fun i => f * (μ i / σ i)) H = ∑ i, H i i + f ^ 2 * offSum μ σ H := by
    rw [signDen_eq, offSum, Finset.mul_sum]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    split_ifs <;> ring
  rw [lrSign, hn, hd]
  generalize ∑ i, H i i = D
  generalize offSum μ σ H = S at hS ⊢
  rw [show f ^ 2 + D / S = (D + f ^ 2 * S) / S by field_simp; ring, div_div_eq_mul_div]
  congr 1
  field_simp

/-- The hypothesis of `lrSign_mul` is satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example (f : ℝ) := lrSign_mul (H := Matrix.of fun _ _ => 1) (fun _ : Fin 2 => 1) (fun _ => 1)
  (by simp [offSum, Fin.sum_univ_two]) f

/-- `βf/(f² + γ) = β/(f + γ/f)`, `f ≠ 0`. -/
theorem mul_div_sq_add (β γ : ℝ) {f : ℝ} (hf : f ≠ 0) : β * f / (f ^ 2 + γ) = β / (f + γ / f) := by
  rw [show f + γ / f = (f ^ 2 + γ) / f by field_simp, div_div_eq_mul_div]

/-- The hypothesis of `mul_div_sq_add` is satisfiable: `f = 1`. -/
example (β γ : ℝ) := mul_div_sq_add β γ one_ne_zero

/-- **Eq. (11)**, its second form: eq. (9) is `β/(f + γ/f)` when `𝓔_i = fμ_i/σ_i`, `f ≠ 0`. -/
theorem lrSign_mul_eq_div (μ σ : ι → ℝ) (hS : offSum μ σ H ≠ 0) {f : ℝ} (hf : f ≠ 0) :
    lrSign (fun i => f * (μ i / σ i)) μ H =
      snrSum μ σ / offSum μ σ H / (f + (∑ i, H i i) / offSum μ σ H / f) := by
  rw [lrSign_mul μ σ hS, mul_div_sq_add _ _ hf]

/-- The hypotheses of `lrSign_mul_eq_div` are satisfiable: `μ = σ = f = 1`, all `H_ij = 1`. -/
example := lrSign_mul_eq_div (H := Matrix.of fun _ _ => 1) (fun _ : Fin 2 => 1) (fun _ => 1)
  (by simp [offSum, Fin.sum_univ_two]) one_ne_zero

omit [Fintype ι] [DecidableEq ι] in
/-- The linearization of eq. (39) is `f(B)μ_i/σ_i`, `f(B) = √(2B/π)`. -/
theorem signLin_eq_mul (μ σ : ι → ℝ) (B : ℝ) :
    signLin μ σ B = fun i => √(2 * B / π) * (μ i / σ i) := by
  funext i
  rw [signLin, mul_div_assoc]

/-- The learning rate of eq. (9) with `𝓔_i(B) = √(2B/π)μ_i/σ_i`, the first line of eq. (40),
tends to `0` as `B → ∞`, `S ≠ 0`. -/
theorem tendsto_lrSign_signLin_atTop (hS : offSum μ σ H ≠ 0) :
    Tendsto (fun B => lrSign (signLin μ σ B) μ H) atTop (𝓝 0) := by
  have hf : Tendsto (fun B : ℝ => √(2 * B / π)) atTop atTop :=
    tendsto_sqrt_atTop.comp ((tendsto_id.const_mul_atTop two_pos).atTop_div_const pi_pos)
  have h := (tendsto_const_nhds (x := snrSum μ σ / offSum μ σ H)).div_atTop
    (hf.atTop_add ((tendsto_const_nhds (x := (∑ i, H i i) / offSum μ σ H)).div_atTop hf))
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop 0] with B hB
  have hf0 : √(2 * B / π) ≠ 0 := (sqrt_pos.2 (by positivity)).ne'
  rw [signLin_eq_mul, lrSign_mul_eq_div μ σ hS hf0]

/-- The hypothesis of `tendsto_lrSign_signLin_atTop` is satisfiable: `μ = σ = 1`, all
`H_ij = 1`. -/
example := tendsto_lrSign_signLin_atTop (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp [offSum, Fin.sum_univ_two])

/-- **Eq. (12)**, `ε(B) ≠ ε_*/(1 + B_noise/B)^α`: no `ε_*`, `B_*`, `α` make eq. (A.3) of
arXiv:1812.06162 the learning rate of eq. (11) with `f(B) = √(2B/π)`, Theorem 3's, at every
`B > 0`, when `ΣH_ii > 0`, `S > 0` and `M > 0`: the latter tends to `0` as `B → ∞`, the former
to `ε_*`, which must then be `0`, while the latter is `ε_max > 0` at `B_noise`.  The paper's
`B_* = B_noise` is a special case. -/
theorem not_exists_lrCentral (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H)
    (hM : 0 < snrSum μ σ) :
    ¬∃ εs Bs α : ℝ, ∀ B > 0, lrSign (signLin μ σ B) μ H = lrCentral εs Bs α B := by
  rintro ⟨εs, Bs, α, h⟩
  have h0 : εs = 0 := tendsto_nhds_unique (tendsto_lrCentral εs Bs α)
    ((tendsto_lrSign_signLin_atTop hS.ne').congr' (by
      filter_upwards [eventually_gt_atTop 0] with B hB
      exact h B hB))
  have h1 := h _ (noiseBatch_pos hD hS)
  rw [lrSign_signLin_noiseBatch hD hS, h0, lrCentral, zero_div] at h1
  exact (lrPeak_pos hD hS hM).ne' h1

/-- The hypotheses of `not_exists_lrCentral` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := not_exists_lrCentral (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

omit [DecidableEq ι] in
/-- **Eq. (3)**: when `B ≪ B_noise`, SGD's `ε_opt(B) = ε_max/(1 + B_noise/B)` of eq. (1), eq.
(2.6) of arXiv:1812.06162, is `(ε_max/B_noise)B`, linear scaling: the ratio tends to `1` as
`B → 0⁺`, when `B_noise > 0` and `ε_max ≠ 0`. -/
theorem tendsto_stepOpt_div {G : ι → ℝ} {S : Matrix ι ι ℝ} (hN : 0 < noiseScale G H S)
    (hm : stepMax G H ≠ 0) :
    Tendsto (fun B => stepOpt G H S B / (stepMax G H / noiseScale G H S * B)) (𝓝[>] 0)
      (𝓝 1) := by
  have h := (tendsto_lrCentral_div_rpow hN (stepMax G H) 1).div_const
    (stepMax G H / noiseScale G H S)
  rw [rpow_one, div_self (div_ne_zero hm hN.ne')] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with B hB
  rw [rpow_one, lrCentral_one, div_div, mul_comm B]

/-- The hypotheses of `tendsto_stepOpt_div` are satisfiable: `G = H = Σ = 1` in one
dimension. -/
example := tendsto_stepOpt_div (G := fun _ : Unit => 1) (H := 1) (S := 1)
  (by simp [noiseScale]) (by simp [stepMax])

/-- **Eq. (4)**: when `B ≪ B_noise`, eq. (2), `ε_max/(½(√(B_noise/B) + √(B/B_noise)))`, is
`(2ε_max/√B_noise)√B`, square root scaling: the ratio tends to `1` as `B → 0⁺`, when
`B_noise > 0` and `ε_max ≠ 0`. -/
theorem tendsto_peak_div_sqrt {εm N : ℝ} (hε : εm ≠ 0) (hN : 0 < N) :
    Tendsto (fun B => εm / ((√(N / B) + √(B / N)) / 2) / (2 * εm / √N * √B)) (𝓝[>] 0)
      (𝓝 1) := by
  obtain ⟨a, ha, rfl⟩ : ∃ a, 0 < a ∧ N = a ^ 2 :=
    ⟨√N, sqrt_pos.2 hN, (sq_sqrt hN.le).symm⟩
  have h : Tendsto (fun B : ℝ => a ^ 2 / (a ^ 2 + B)) (𝓝[>] 0) (𝓝 (a ^ 2 / (a ^ 2 + 0))) :=
    tendsto_nhdsWithin_of_tendsto_nhds ((continuousAt_const.div
      (continuousAt_const.add continuousAt_id) (by simpa using ha.ne')).tendsto)
  rw [add_zero, div_self (by positivity)] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with B (hB : 0 < B)
  obtain ⟨b, hb, rfl⟩ : ∃ b, 0 < b ∧ B = b ^ 2 :=
    ⟨√B, sqrt_pos.2 hB, (sq_sqrt hB.le).symm⟩
  rw [← div_pow, ← div_pow, sqrt_sq (by positivity), sqrt_sq (by positivity), sqrt_sq ha.le,
    sqrt_sq hb.le]
  field_simp

/-- The hypotheses of `tendsto_peak_div_sqrt` are satisfiable: `ε_max = B_noise = 1`. -/
example := tendsto_peak_div_sqrt one_ne_zero one_pos

/-- `ε_max ≠ 0` when `ΣH_ii > 0`, `S > 0` and `M ≠ 0`. -/
theorem lrPeak_ne_zero (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hM : snrSum μ σ ≠ 0) :
    lrPeak μ σ H ≠ 0 :=
  div_ne_zero (mul_ne_zero (sqrt_pos.2 (div_pos (noiseBatch_pos hD hS) (by positivity))).ne'
    hM) hD.ne'

/-- The hypotheses of `lrPeak_ne_zero` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := lrPeak_ne_zero (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

/-- **Eq. (4)** for the learning rate of eq. (9): its ratio to `(2ε_max/√B_noise)√B` tends to
`1` as `B → 0⁺`, when `ΣH_ii > 0`, `S > 0` and `M ≠ 0`. -/
theorem tendsto_lrSign_div_sqrt (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H)
    (hM : snrSum μ σ ≠ 0) :
    Tendsto (fun B => lrSign (fun i => signMean (μ i) (σ i) B) μ H /
      (2 * lrPeak μ σ H / √(noiseBatch μ σ H) * √B)) (𝓝[>] 0) (𝓝 1) := by
  have hN := noiseBatch_pos hD hS
  have h := (tendsto_lrSign_div_peak hD hS hM).mul
    (tendsto_peak_div_sqrt (lrPeak_ne_zero hD hS hM) hN)
  rw [one_mul] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with B (hB : 0 < B)
  have hq : (√(noiseBatch μ σ H / B) + √(B / noiseBatch μ σ H)) / 2 ≠ 0 :=
    (half_pos (add_pos (sqrt_pos.2 (div_pos hN hB)) (sqrt_pos.2 (div_pos hB hN)))).ne'
  exact div_mul_div_cancel₀ (div_ne_zero (lrPeak_ne_zero hD hS hM) hq)

/-- The hypotheses of `tendsto_lrSign_div_sqrt` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := tendsto_lrSign_div_sqrt (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

end Transformer.Surge
