/-
# The small batch approximation

arXiv:2405.14578, §2.1, Theorem 3, proved in Appendix D.  The proof replaces
`𝓔_i(B) = erf(√(B/2)μ_i/σ_i)` by `√(2B/π)μ_i/σ_i` when `B ≪ πσ_i²/(2μ_i²)`, eq. (39).  Their
ratio tends to `1` as `B → 0` (`tendsto_signMean_div_signLin`): both are `(2/√π)(μ_i/σ_i)√(B/2)`
to first order (`tendsto_signMean_div_sqrt`, `signLin_div_sqrt`).  So does the ratio of the
learning rate of eq. (9) to the one with `√(2B/π)μ_i/σ_i`, the first line of eq. (40), when
`M = Σμ_i²/σ_i ≠ 0` and `ΣH_ii ≠ 0` (`tendsto_lrSign_div_lrSign_signLin`): divided by `√(B/2)`,
both tend to `(2/√π)M/ΣH_ii` (`tendsto_lrSign_div`).  With the paper's `ΣH_ii > 0` and `S > 0`,
this is the "≈" of eq. (13) (`tendsto_lrSign_div_peak`).
-/

import Transformer.Surge.Section2_SmallBatch

open Filter Topology Real

namespace Transformer.Surge

open Transformer.BatchSize

variable {ι : Type*} [Fintype ι]

/-- The denominator of eq. (9) tends to `ΣH_ii` when every `𝓔_i` tends to `0`. -/
theorem tendsto_signDen {α : Type*} {l : Filter α} {E : α → ι → ℝ}
    (hE : ∀ i, Tendsto (fun a => E a i) l (𝓝 0)) (H : Matrix ι ι ℝ) :
    Tendsto (fun a => signDen (E a) H) l (𝓝 (∑ i, H i i)) := by
  have h : Tendsto (fun a => signDen (E a) H) l (𝓝 (signDen 0 H)) :=
    (tendsto_finsetSum _ fun i _ => ((tendsto_const_nhds.sub ((hE i).pow 2)).mul
      tendsto_const_nhds)).add (tendsto_finsetSum _ fun i _ => tendsto_finsetSum _ fun j _ =>
      ((hE i).mul (hE j)).mul tendsto_const_nhds)
  simpa [signDen] using h

/-- When `𝓔_i/t → c_i` and `t → 0`, the learning rate of eq. (9) over `t` tends to
`Σc_iμ_i/ΣH_ii`, `ΣH_ii ≠ 0`. -/
theorem tendsto_lrSign_div {α : Type*} {l : Filter α} {E : α → ι → ℝ} {t : α → ℝ}
    {c : ι → ℝ} (hE : ∀ i, Tendsto (fun a => E a i / t a) l (𝓝 (c i)))
    (ht : Tendsto t l (𝓝 0)) (ht0 : ∀ᶠ a in l, t a ≠ 0) (μ : ι → ℝ) {H : Matrix ι ι ℝ}
    (hD : ∑ i, H i i ≠ 0) :
    Tendsto (fun a => lrSign (E a) μ H / t a) l (𝓝 ((∑ i, c i * μ i) / ∑ i, H i i)) := by
  have hE0 (i : ι) : Tendsto (fun a => E a i) l (𝓝 0) := by
    have h := (hE i).mul ht
    rw [mul_zero] at h
    refine h.congr' ?_
    filter_upwards [ht0] with a ha
    field_simp
  have hnum : Tendsto (fun a => ∑ i, E a i / t a * μ i) l (𝓝 (∑ i, c i * μ i)) :=
    tendsto_finsetSum _ fun i _ => (hE i).mul tendsto_const_nhds
  refine (hnum.div (tendsto_signDen hE0 H) hD).congr fun a => ?_
  rw [Pi.div_apply, lrSign, div_right_comm]
  congr 1
  rw [Finset.sum_div]
  exact Finset.sum_congr rfl fun i _ => by ring

/-- `√(B/2) → 0⁺` as `B → 0⁺`. -/
theorem tendsto_sqrt_half : Tendsto (fun B : ℝ => √(B / 2)) (𝓝[>] 0) (𝓝[>] 0) := by
  refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ ?_ ?_
  · have h : Tendsto (fun B : ℝ => √(B / 2)) (𝓝 0) (𝓝 (√(0 / 2))) :=
      (by fun_prop : Continuous fun B : ℝ => √(B / 2)).tendsto 0
    rw [zero_div, Real.sqrt_zero] at h
    exact tendsto_nhdsWithin_of_tendsto_nhds h
  · filter_upwards [self_mem_nhdsWithin] with B hB
    exact Real.sqrt_pos.2 (half_pos hB)

/-- `𝓔(B)/√(B/2) → (2/√π)(μ/σ)` as `B → 0⁺`: the slope of `erf` at `0` is `2/√π`. -/
theorem tendsto_signMean_div_sqrt (μ σ : ℝ) :
    Tendsto (fun B => signMean μ σ B / √(B / 2)) (𝓝[>] 0) (𝓝 (2 / √π * (μ / σ))) := by
  have hd := (errorFunction_hasDerivAt (0 * (μ / σ))).comp 0
    ((hasDerivAt_id 0).mul_const (μ / σ))
  have h := hd.tendsto_slope_zero_right.comp tendsto_sqrt_half
  simp only [zero_mul, neg_zero, zero_pow two_ne_zero, Real.exp_zero, mul_one, id, one_mul,
    zero_add] at h
  refine h.congr fun B => ?_
  simp only [Function.comp_apply, zero_mul, errorFunction_zero, sub_zero, smul_eq_mul, signMean]
  rw [inv_mul_eq_div, mul_div_assoc]

/-- `√(2B/π) = (2/√π)√(B/2)`. -/
theorem sqrt_two_mul_div_pi (B : ℝ) : √(2 * B / π) = 2 / √π * √(B / 2) := by
  rw [show 2 * B / π = (2 / √π) ^ 2 * (B / 2) by
      rw [div_pow, Real.sq_sqrt Real.pi_pos.le]
      ring,
    Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by positivity)]

omit [Fintype ι] in
/-- `√(2B/π)(μ_i/σ_i)/√(B/2) = (2/√π)(μ_i/σ_i)`, `B > 0`. -/
theorem signLin_div_sqrt (μ σ : ι → ℝ) {B : ℝ} (hB : 0 < B) (i : ι) :
    signLin μ σ B i / √(B / 2) = 2 / √π * (μ i / σ i) := by
  have hs := (Real.sqrt_pos.2 (half_pos hB)).ne'
  rw [signLin, sqrt_two_mul_div_pi]
  field_simp

/-- The hypothesis of `signLin_div_sqrt` is satisfiable: `B = 1`. -/
example (μ σ : Fin 2 → ℝ) := signLin_div_sqrt μ σ one_pos 0

omit [Fintype ι] in
/-- **Eq. (39)**: `𝓔_i(B)/(√(2B/π)μ_i/σ_i) → 1` as `B → 0⁺`, when `μ_i ≠ 0`, `σ_i ≠ 0`. -/
theorem tendsto_signMean_div_signLin {μ σ : ι → ℝ} {i : ι} (hμ : μ i ≠ 0) (hσ : σ i ≠ 0) :
    Tendsto (fun B => signMean (μ i) (σ i) B / signLin μ σ B i) (𝓝[>] 0) (𝓝 1) := by
  have hc : 2 / √π * (μ i / σ i) ≠ 0 :=
    mul_ne_zero (div_ne_zero two_ne_zero (Real.sqrt_pos.2 Real.pi_pos).ne') (div_ne_zero hμ hσ)
  have h := (tendsto_signMean_div_sqrt (μ i) (σ i)).div tendsto_const_nhds hc
  rw [div_self hc] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with B hB
  rw [Pi.div_apply, ← signLin_div_sqrt μ σ hB i,
    div_div_div_cancel_right₀ (Real.sqrt_pos.2 (half_pos hB)).ne']

/-- The hypotheses of `tendsto_signMean_div_signLin` are satisfiable: `μ = σ = 1`. -/
example := tendsto_signMean_div_signLin (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1) (i := 0)
  one_ne_zero one_ne_zero

/-- **Theorem 3**, eq. (40): as `B → 0⁺`, the learning rate of eq. (9) over the one with
`𝓔_i(B)` replaced by `√(2B/π)μ_i/σ_i` tends to `1`, when `M ≠ 0` and `ΣH_ii ≠ 0`. -/
theorem tendsto_lrSign_div_lrSign_signLin [DecidableEq ι] {μ σ : ι → ℝ} {H : Matrix ι ι ℝ}
    (hM : snrSum μ σ ≠ 0) (hD : ∑ i, H i i ≠ 0) :
    Tendsto (fun B => lrSign (fun i => signMean (μ i) (σ i) B) μ H /
      lrSign (signLin μ σ B) μ H) (𝓝[>] 0) (𝓝 1) := by
  have ht : Tendsto (fun B : ℝ => √(B / 2)) (𝓝[>] 0) (𝓝 0) :=
    tendsto_sqrt_half.mono_right nhdsWithin_le_nhds
  have ht0 : ∀ᶠ B : ℝ in 𝓝[>] 0, √(B / 2) ≠ 0 := by
    filter_upwards [self_mem_nhdsWithin] with B hB
    exact (Real.sqrt_pos.2 (half_pos hB)).ne'
  have h1 := tendsto_lrSign_div (fun i => tendsto_signMean_div_sqrt (μ i) (σ i)) ht ht0 μ hD
  have h2 := tendsto_lrSign_div (E := signLin μ σ) (c := fun i => 2 / √π * (μ i / σ i))
    (fun i => tendsto_const_nhds.congr' (by
      filter_upwards [self_mem_nhdsWithin] with B hB
      exact (signLin_div_sqrt μ σ hB i).symm)) ht ht0 μ hD
  have hK : (∑ i, 2 / √π * (μ i / σ i) * μ i) / ∑ i, H i i ≠ 0 := by
    rw [show ∑ i, 2 / √π * (μ i / σ i) * μ i = 2 / √π * snrSum μ σ by
      rw [snrSum, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring]
    exact div_ne_zero (mul_ne_zero (div_ne_zero two_ne_zero
      (Real.sqrt_pos.2 Real.pi_pos).ne') hM) hD
  have h := h1.div h2 hK
  rw [div_self hK] at h
  refine h.congr' ?_
  filter_upwards [ht0] with B hB
  exact div_div_div_cancel_right₀ hB _ _

/-- The hypotheses of `tendsto_lrSign_div_lrSign_signLin` are satisfiable: `μ = σ = 1`,
`H = 1`. -/
example := tendsto_lrSign_div_lrSign_signLin (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := 1) (by simp [snrSum]) (by simp)

/-- **Theorem 3**, eq. (13): as `B → 0⁺`, the learning rate of eq. (9) over
`ε_max/(½(√(B_noise/B) + √(B/B_noise)))` tends to `1`, when `ΣH_ii > 0`, `S > 0` and `M ≠ 0`. -/
theorem tendsto_lrSign_div_peak [DecidableEq ι] {μ σ : ι → ℝ} {H : Matrix ι ι ℝ}
    (hD : 0 < ∑ i, H i i) (hS : 0 < offSum μ σ H) (hM : snrSum μ σ ≠ 0) :
    Tendsto (fun B => lrSign (fun i => signMean (μ i) (σ i) B) μ H / (lrPeak μ σ H /
      ((√(noiseBatch μ σ H / B) + √(B / noiseBatch μ σ H)) / 2))) (𝓝[>] 0) (𝓝 1) := by
  refine (tendsto_lrSign_div_lrSign_signLin hM hD.ne').congr' ?_
  filter_upwards [self_mem_nhdsWithin] with B hB
  rw [lrSign_signLin_eq_peak hD hS hB]

/-- The hypotheses of `tendsto_lrSign_div_peak` are satisfiable: `μ = σ = 1`, all `H_ij = 1`. -/
example := tendsto_lrSign_div_peak (μ := fun _ : Fin 2 => 1) (σ := fun _ => 1)
  (H := Matrix.of fun _ _ => 1) (by simp) (by simp [offSum, Fin.sum_univ_two]) (by simp [snrSum])

end Transformer.Surge
