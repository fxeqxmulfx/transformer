module

public import Transformer.AMSGradW.Kernite
public import Mathlib.Topology.Algebra.InfiniteSum.Real

/-! Convergence statements for the clipped and bias-corrected update.
Bounded maximum histories converge for every minibatch sequence. A summable
learning rate gives convergence of all weights from the initialized run. -/

@[expose] public section

namespace Transformer.AMSGradW.Kernite

open Filter
open scoped Topology

variable {ι : Type*} [Fintype ι]

theorem run_maximum_tendsto (rate : ℕ → ℝ) (batchGradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) :
    ∃ V : ι → ℝ, (∀ i, 0 ≤ V i ∧ V i ≤ 1) ∧
      Tendsto (fun t => (run rate batchGradient initial t).maximum) atTop (𝓝 V) := by
  let v (i : ι) (t : ℕ) := (run rate batchGradient initial t).maximum i
  have hm (i : ι) : Monotone (v i) :=
    monotone_nat_of_le_succ fun t => (run_buffer_bounds rate batchGradient initial i t).2.2.2.2.2
  have hb (i : ι) : BddAbove (Set.range (v i)) := by
    refine ⟨1, ?_⟩
    rintro _ ⟨t, rfl⟩
    exact (run_buffer_bounds rate batchGradient initial i t).2.2.2.2.1
  refine ⟨fun i => ⨆ t, v i t, fun i => ⟨le_ciSup (hb i) 0, ?_⟩, tendsto_pi_nhds.mpr fun i =>
    tendsto_atTop_ciSup (hm i) (hb i)⟩
  exact ciSup_le fun t => (run_buffer_bounds rate batchGradient initial i t).2.2.2.2.1

/-- A summable rate has finite total parameter movement. The proof
retains actual clipping, both bias corrections and decoupled decay. -/
theorem run_weights_tendsto_of_summable (rate : ℕ → ℝ) (batchGradient : ℕ → (ι → ℝ) → ι → ℝ)
    (initial : ι → ℝ) (hrate : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (hsum : Summable rate) :
    ∃ w : ι → ℝ, Tendsto (fun t => (run rate batchGradient initial t).weights) atTop (𝓝 w) := by
  let s := run rate batchGradient initial
  have hdist (i : ι) (t : ℕ) : dist ((s t).weights i) ((s (t + 1)).weights i) ≤
      (max |initial i| 10000000000 / 10 + 1000000000) * rate t := by
    let v := direction t ((s (t + 1)).first i) ((s (t + 1)).maximum i)
    have he : (s t).weights i - (s (t + 1)).weights i = rate t * ((s t).weights i / 10 + v) := by
      change (s t).weights i - ((1 - rate t / 10) * (s t).weights i - rate t * v) = _
      ring
    rw [Real.dist_eq, he, abs_mul, abs_of_nonneg (hrate t).1]
    refine (mul_le_mul_of_nonneg_left ((abs_add_le _ _).trans ?_) (hrate t).1).trans_eq (mul_comm _ _)
    rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 10)]
    exact add_le_add (div_le_div_of_nonneg_right (run_abs_weight_le rate batchGradient initial hrate i t)
      (by norm_num)) (abs_direction_le t (run_buffer_bounds rate batchGradient initial i (t + 1)).1 _)
  have hlim (i : ι) : ∃ w : ℝ, Tendsto (fun t => (s t).weights i) atTop (𝓝 w) :=
    cauchySeq_tendsto_of_complete (cauchySeq_of_dist_le_of_summable _
      (hdist i) (hsum.mul_left _))
  choose w hw using hlim
  exact ⟨w, tendsto_pi_nhds.mpr hw⟩

theorem firstBias_tendsto : Tendsto firstBias atTop (𝓝 1) := by
  have hp := tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0 : ℝ) ≤ 9 / 10) (by norm_num)
  have h := (Filter.tendsto_add_atTop_iff_nat 1).2 hp
  change Tendsto (fun t => 1 - (9 / 10 : ℝ) ^ (t + 1)) atTop (𝓝 1)
  simpa only [sub_zero] using h.const_sub 1

theorem denominator_tendsto {v : ℕ → ℝ} {V : ℝ} (h : Tendsto v atTop (𝓝 V)) :
    Tendsto (fun t => denominator t (v t)) atTop (𝓝 (Real.sqrt V + 1 / 100000000)) := by
  have hp := tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0 : ℝ) ≤ 98 / 100) (by norm_num)
  have hcor := ((Filter.tendsto_add_atTop_iff_nat 1).2 hp).const_sub 1
  rw [sub_zero] at hcor
  have hs := Real.continuous_sqrt.continuousAt.tendsto.comp hcor
  have hv := Real.continuous_sqrt.continuousAt.tendsto.comp h
  have hd := hv.div hs (by rw [Real.sqrt_one]; exact one_ne_zero)
  have hlim := hd.add_const (1 / 100000000)
  change Tendsto (fun t => denominator t (v t)) atTop
    (𝓝 (Real.sqrt V / Real.sqrt 1 + 1 / 100000000)) at hlim
  simpa only [Real.sqrt_one, div_one] using hlim

/-- At a positive learning-rate floor, convergence of the parameters
forces the clipped minibatch gradients themselves to converge to one
vector. Bounded adaptive moments alone do not supply this condition. -/
theorem run_clippedGrad_tendsto_of_weights_tendsto (rate : ℕ → ℝ)
    (batchGradient : ℕ → (ι → ℝ) → ι → ℝ) (initial w : ι → ℝ)
    {η : ℝ} (hη : 0 < η) (hηlim : Tendsto rate atTop (𝓝 η))
    (hw : Tendsto (fun t => (run rate batchGradient initial t).weights) atTop (𝓝 w)) :
    ∃ V : ι → ℝ, Tendsto (fun t => (run rate batchGradient initial t).maximum) atTop (𝓝 V) ∧
      Tendsto (fun t i => clip (batchGradient t (run rate batchGradient initial t).weights) i)
        atTop (𝓝 (fun i => -(1 / 10 : ℝ) * (Real.sqrt (V i) + 1 / 100000000) * w i)) := by
  let s := run rate batchGradient initial
  obtain ⟨V, _, hV⟩ := run_maximum_tendsto rate batchGradient initial
  refine ⟨V, hV, tendsto_pi_nhds.mpr fun i => ?_⟩
  have hx : Tendsto (fun t => (s t).weights i) atTop (𝓝 (w i)) := tendsto_pi_nhds.mp hw i
  have hx' := (Filter.tendsto_add_atTop_iff_nat 1).2 hx
  have hmax := (Filter.tendsto_add_atTop_iff_nat 1).2 (tendsto_pi_nhds.mp hV i)
  let D := Real.sqrt (V i) + 1 / 100000000
  let m := -(1 / 10 : ℝ) * D * w i
  have hD : Tendsto (fun t => denominator t ((s (t + 1)).maximum i)) atTop (𝓝 D) :=
    denominator_tendsto hmax
  let scaled (t : ℕ) := ((1 - rate t / 10) * (s t).weights i - (s (t + 1)).weights i) / rate t
  have he : ((1 - η / 10) * w i - w i) / η = -(1 / 10 : ℝ) * w i := by
    field_simp
    ring
  have hscale : Tendsto scaled atTop (𝓝 (-(1 / 10 : ℝ) * w i)) := by
    rw [← he]
    exact ((((hηlim.div_const 10).const_sub 1).mul hx).sub hx').div hηlim hη.ne'
  have hm' : Tendsto (fun t => (s (t + 1)).first i) atTop (𝓝 m) := by
    have h := hscale.mul (firstBias_tendsto.mul hD)
    rw [one_mul, mul_right_comm] at h
    apply h.congr'
    filter_upwards [hηlim.eventually (lt_mem_nhds hη)] with t ht
    have hz : firstBias t * denominator t ((s (t + 1)).maximum i) ≠ 0 :=
      (mul_pos ((by norm_num : (0 : ℝ) < 1 / 10).trans_le (firstBias_lower t))
        ((by norm_num : (0 : ℝ) < 1 / 100000000).trans_le (denominator_lower _ _))).ne'
    change scaled t * (firstBias t * denominator t ((s (t + 1)).maximum i)) = _
    have hstep : (s (t + 1)).weights i =
        (1 - rate t / 10) * (s t).weights i -
          rate t * direction t ((s (t + 1)).first i) ((s (t + 1)).maximum i) := rfl
    rw [show scaled t = direction t ((s (t + 1)).first i) ((s (t + 1)).maximum i) by
      simp only [scaled, hstep, sub_sub_cancel]
      exact mul_div_cancel_left₀ _ (ne_of_gt ht)]
    exact div_mul_cancel₀ _ hz
  have hm : Tendsto (fun t => (s t).first i) atTop (𝓝 m) :=
    (Filter.tendsto_add_atTop_iff_nat 1).1 hm'
  have hg (t : ℕ) : clip (batchGradient t (s t).weights) i =
      ((s (t + 1)).first i - 9 / 10 * (s t).first i) / (1 / 10) := by
    change _ = ((9 / 10 * (s t).first i + (1 / 10) * clip (batchGradient t (s t).weights) i) -
      9 / 10 * (s t).first i) / (1 / 10)
    ring
  have h := (hm'.sub (hm.const_mul (9 / 10))).div_const (1 / 10)
  rw [show (m - 9 / 10 * m) / (1 / 10) = m by ring] at h
  exact h.congr fun t => (hg t).symm

example (g : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (initial : Fin 2 → ℝ) :=
  run_maximum_tendsto (fun _ => 1 / 1000) g initial

example (g : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (initial : Fin 2 → ℝ) :=
  run_weights_tendsto_of_summable (fun t => (1 / 2 : ℝ) ^ t) g initial
    (fun t => ⟨by positivity, (pow_le_one₀ (by norm_num) (by norm_num)).trans (by norm_num)⟩)
    (summable_geometric_of_lt_one (by norm_num) (by norm_num))

example := denominator_tendsto (v := fun _ => 1) tendsto_const_nhds

example :=
  run_clippedGrad_tendsto_of_weights_tendsto (ι := Fin 1) (fun _ => 1 / 1000) (fun _ _ => 0) 0 0
    (by norm_num) tendsto_const_nhds (by
      have hz (t : ℕ) : (run (ι := Fin 1) (fun _ => 1 / 1000) (fun _ _ => 0) 0 t) = ⟨0, 0, 0, 0⟩ := by
        induction t with
        | zero => rfl
        | succ t ih =>
          rw [run, ih]
          simp [step, clip, direction, Pi.zero_def]
      simpa only [hz] using (tendsto_const_nhds (x := (0 : Fin 1 → ℝ))))

end Transformer.AMSGradW.Kernite
