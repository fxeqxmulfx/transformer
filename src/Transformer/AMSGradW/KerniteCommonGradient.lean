module

public import Transformer.AMSGradW.KerniteLimits
public import Transformer.AMSGradW.Perturbation

/-! At a positive learning-rate floor the actual parameter sequence
converges exactly when the clipped minibatch gradients converge. The
history limit determines the equilibrium through decoupled decay. -/

@[expose] public section

namespace Transformer.AMSGradW.Kernite

open Filter
open scoped Topology

variable {ι : Type*} [Fintype ι]

private theorem tendsto_of_affine {x a b : ℕ → ℝ} {q u B : ℝ}
    (hq : |q| < 1) (hx : ∀ t, |x t| ≤ B)
    (ha : Tendsto a atTop (𝓝 q)) (hb : Tendsto b atTop (𝓝 ((1 - q) * u)))
    (hrec : ∀ t, x (t + 1) = a t * x t + b t) :
    Tendsto x atTop (𝓝 u) := by
  let err := fun t => |x t - u|
  let force := fun t => |a t - q| * B + |b t - (1 - q) * u|
  have hf : Tendsto force atTop (𝓝 0) := by
    simpa only [sub_self, abs_zero, zero_mul, add_zero] using
      (((ha.sub_const q).abs.mul_const B).add (hb.sub_const ((1 - q) * u)).abs)
  have he : Tendsto err atTop (𝓝 0) := by
    apply contraction_forcing_tendsto_zero |q| err force (abs_nonneg _) hq (fun t => abs_nonneg _) hf
    intro t
    dsimp only [err, force]
    rw [hrec, show a t * x t + b t - u = q * (x t - u) + ((a t - q) * x t + (b t - (1 - q) * u)) by ring]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul]
    apply add_le_add_right
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul]
    exact add_le_add_left (mul_le_mul_of_nonneg_left (hx t) (abs_nonneg _)) _
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  simpa only [Real.norm_eq_abs] using he

/-- A common clipped gradient supplies the one missing limit in the
momentum recurrence. At a positive rate floor, decay contracts the
remaining affine weight recurrence. -/
theorem run_weights_tendsto_of_clippedGrad (rate : ℕ → ℝ)
    (batchGradient : ℕ → (ι → ℝ) → ι → ℝ) (initial G : ι → ℝ) {η : ℝ}
    (hrate : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (hη : 0 < η) (hηlim : Tendsto rate atTop (𝓝 η))
    (hG : Tendsto (fun t i => clip (batchGradient t (run rate batchGradient initial t).weights) i)
      atTop (𝓝 G)) :
    ∃ V : ι → ℝ, Tendsto (fun t => (run rate batchGradient initial t).maximum) atTop (𝓝 V) ∧
      Tendsto (fun t => (run rate batchGradient initial t).weights) atTop
        (𝓝 (fun i => -10 * G i / (Real.sqrt (V i) + 1 / 100000000))) := by
  let s := run rate batchGradient initial
  obtain ⟨V, _, hV⟩ := run_maximum_tendsto rate batchGradient initial
  refine ⟨V, hV, tendsto_pi_nhds.mpr fun i => ?_⟩
  let g := fun t => clip (batchGradient t (s t).weights) i
  have hg : Tendsto g atTop (𝓝 (G i)) := tendsto_pi_nhds.mp hG i
  have hm : Tendsto (fun t => (s t).first i) atTop (𝓝 (G i)) := by
    refine tendsto_of_affine (a := fun _ => 9 / 10) (b := fun t => 1 / 10 * g t)
      (B := 1) (by norm_num) (fun t => (run_buffer_bounds rate batchGradient initial i t).1)
      tendsto_const_nhds ?_ (fun _ => rfl)
    convert hg.const_mul (1 / 10) using 2; ring
  let D := Real.sqrt (V i) + 1 / 100000000
  have hD : 0 < D := by dsimp only [D]; positivity
  have hv := (Filter.tendsto_add_atTop_iff_nat 1).2 (tendsto_pi_nhds.mp hV i)
  have hd : Tendsto (fun t => denominator t ((s (t + 1)).maximum i)) atTop (𝓝 D) :=
    denominator_tendsto hv
  have hdiv : Tendsto (fun t => direction t ((s (t + 1)).first i) ((s (t + 1)).maximum i))
      atTop (𝓝 (G i / D)) := by
    have h := ((Filter.tendsto_add_atTop_iff_nat 1).2 hm).div (firstBias_tendsto.mul hd)
      (by rw [one_mul]; exact hD.ne')
    rw [one_mul] at h
    exact h
  have hηle : η ≤ 10 := le_of_tendsto hηlim (Eventually.of_forall fun t => (hrate t).2)
  have hq : |1 - η / 10| < 1 := by
    rw [abs_of_nonneg (by linarith)]
    linarith
  refine tendsto_of_affine (a := fun t => 1 - rate t / 10)
    (b := fun t => -rate t * direction t ((s (t + 1)).first i) ((s (t + 1)).maximum i))
    hq (run_abs_weight_le rate batchGradient initial hrate i) ((hηlim.div_const 10).const_sub 1)
    ?_ ?_
  · convert hηlim.neg.mul hdiv using 2; ring
  · intro t
    change (1 - rate t / 10) * (s t).weights i -
      rate t * direction t ((s (t + 1)).first i) ((s (t + 1)).maximum i) = _
    ring

theorem run_weights_tendsto_iff_clippedGrad (rate : ℕ → ℝ)
    (batchGradient : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) {η : ℝ}
    (hrate : ∀ t, 0 ≤ rate t ∧ rate t ≤ 10) (hη : 0 < η) (hηlim : Tendsto rate atTop (𝓝 η)) :
    (∃ w, Tendsto (fun t => (run rate batchGradient initial t).weights) atTop (𝓝 w)) ↔
      ∃ G, Tendsto (fun t i => clip (batchGradient t (run rate batchGradient initial t).weights) i)
        atTop (𝓝 G) := by
  constructor
  · rintro ⟨w, hw⟩
    obtain ⟨V, _, hG⟩ := run_clippedGrad_tendsto_of_weights_tendsto rate batchGradient initial w hη hηlim hw
    exact ⟨fun i => -(1 / 10 : ℝ) * (Real.sqrt (V i) + 1 / 100000000) * w i, hG⟩
  · rintro ⟨G, hG⟩
    obtain ⟨V, _, hw⟩ := run_weights_tendsto_of_clippedGrad rate batchGradient initial G hrate hη hηlim hG
    exact ⟨fun i => -10 * G i / (Real.sqrt (V i) + 1 / 100000000), hw⟩

example (g : ℕ → (Fin 2 → ℝ) → Fin 2 → ℝ) (initial : Fin 2 → ℝ) :=
  run_weights_tendsto_iff_clippedGrad (fun _ => 1 / 100000000) g initial
    (fun _ => by norm_num) (by norm_num) tendsto_const_nhds

example (initial : Fin 2 → ℝ) :=
  run_weights_tendsto_of_clippedGrad (fun _ => 1 / 100000000) (fun _ _ => 0) initial 0
    (fun _ => by norm_num) (by norm_num) tendsto_const_nhds
    (by simpa only [clip, Pi.zero_def, zero_div] using
      (tendsto_const_nhds (x := (0 : Fin 2 → ℝ))))

end Transformer.AMSGradW.Kernite
