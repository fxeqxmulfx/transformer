module

public import Transformer.Grokking.FiniteRecovery.ArgandAdamW
public import Transformer.Grokking.AdamW.MomentLimits

/-! The original WSD schedule holds its final positive rate indefinitely if
called beyond the finite experiment. For ordinary AdamW, a common limit of
the clipped gradients determines the full weight limit explicitly. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.AdamW

open Filter
open scoped Topology

private theorem moving_average_limit {q l : ℝ} {x z : ℕ → ℝ}
    (hq : 0 ≤ q) (hq' : q < 1) (hz : Tendsto z atTop (𝓝 l))
    (h : ∀ t, x (t+1) = q*x t + (1-q)*z t) : Tendsto x atTop (𝓝 l) := by
  have he (t : ℕ) : x t = Grokking.AdamW.momentAt q z (x 0) t := by
    induction t with
    | zero => rfl
    | succ t ih => rw [h, ih]; rfl
  exact (Grokking.AdamW.moment_tendsto_of_input_tendsto q (x 0) l z hq hq' hz).congr
    (fun t => (he t).symm)

variable {ι : Type*} [Fintype ι]

/-- On a positive constant tail, the bias-corrected second moment tends to
the square of the limiting input. Decoupled decay contracts the weight
toward an explicit equilibrium in each coordinate. -/
theorem limit_of_common_clipped_gradient (rate : ℕ → ℝ) (grad : ℕ → (ι → ℝ) → ι → ℝ)
    (initial G : ι → ℝ) {η : ℝ} (hη : 0 < η) (hη' : η ≤ 10)
    (hr : ∀ᶠ t in atTop, rate t = η)
    (hg : Tendsto (fun t => Transformer.AMSGradW.Kernite.clip (grad t (run rate grad initial t).weights))
      atTop (𝓝 G)) :
    Tendsto (fun t => (run rate grad initial t).weights) atTop
      (𝓝 (fun i => -10 * (G i / (|G i| + 1/100000000)))) := by
  let s := run rate grad initial
  let g t := Transformer.AMSGradW.Kernite.clip (grad t (s t).weights)
  obtain ⟨N, hN⟩ := eventually_atTop.mp hr
  apply tendsto_pi_nhds.mpr
  intro i
  have hgi : Tendsto (fun t => g t i) atTop (𝓝 (G i)) := tendsto_pi_nhds.mp hg i
  have hm : Tendsto (fun t => (s t).first i) atTop (𝓝 (G i)) := by
    apply moving_average_limit (q := 9/10) (by norm_num) (by norm_num) hgi
    intro t
    change 9/10 * (s t).first i + 1/10*g t i = _
    ring
  have hv : Tendsto (fun t => (s t).second i) atTop (𝓝 ((G i)^2)) := by
    apply moving_average_limit (q := 95/100) (by norm_num) (by norm_num) (hgi.pow 2)
    intro t
    change 95/100 * (s t).second i + 5/100*(g t i)^2 = _
    ring
  let D := |G i| + 1/100000000
  have hD : 0 < D := by dsimp [D]; positivity
  have hpow := tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0:ℝ) ≤ 95/100) (by norm_num)
  have hc := ((Filter.tendsto_add_atTop_iff_nat 1).2 hpow).const_sub 1
  rw [sub_zero] at hc
  have hscale : Tendsto (fun t => scale t ((s (t+1)).second i)) atTop (𝓝 D) := by
    have hnum := Real.continuous_sqrt.continuousAt.tendsto.comp ((Filter.tendsto_add_atTop_iff_nat 1).2 hv)
    have hden := Real.continuous_sqrt.continuousAt.tendsto.comp hc
    have hh := (hnum.div hden (by simp)).add_const (1/100000000)
    simpa only [scale, D, Pi.div_def, Function.comp_def, Real.sqrt_one, div_one, Real.sqrt_sq_eq_abs] using hh
  let v t := direction t ((s (t+1)).first i) ((s (t+1)).second i)
  have hdir : Tendsto v atTop (𝓝 (G i/D)) := by
    have h := ((Filter.tendsto_add_atTop_iff_nat 1).2 hm).div
      (Transformer.AMSGradW.Kernite.firstBias_tendsto.mul hscale) (by simpa using hD.ne')
    simpa only [one_mul, Pi.div_def, direction, v] using h
  let q := 1-η/10
  apply (Filter.tendsto_add_atTop_iff_nat N).mp
  apply moving_average_limit (q := q) (by dsimp [q]; linarith) (by dsimp [q]; linarith)
    ((Filter.tendsto_add_atTop_iff_nat N).2 (hdir.const_mul (-10)))
  intro t
  change (s ((t+1)+N)).weights i = q * (s (t+N)).weights i + (1-q)*(-10*v (t+N))
  rw [show (t+1)+N = (t+N)+1 by omega]
  change (1-rate (t+N)/10)*(s (t+N)).weights i - rate (t+N)*v (t+N) = _
  rw [hN (t+N) (Nat.le_add_left _ _)]
  dsimp only [q]
  ring

/-- A fixed nonzero gradient is a concrete instance in every finite dimension. -/
example (w c : ι → ℝ) := limit_of_common_clipped_gradient (fun _ => 1/1000) (fun _ _ => c) w
  (Transformer.AMSGradW.Kernite.clip c) (by norm_num) (by norm_num)
  (Eventually.of_forall fun _ => rfl) tendsto_const_nhds

end Transformer.Grokking.FiniteRecovery.AdamW

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter
open scoped Topology

/-- The unmodified native-floor theorem for the entire real-parameter Argand,
including every trainable gate. The stipulated limit concerns the gradients
actually evaluated on this run. -/
theorem adamW_wsd_common_gradient {A : Type*} [Fintype A] {H d L : ℕ}
    (r B : ℕ) (batch : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d L → ℂ)
    (G : Weights A H d L) (hw : B * 2 / 100 ≤ B - 1) (hd : B * 80 / 100 < B - 1)
    (hg : Tendsto (fun t => Transformer.AMSGradW.Kernite.clip
      (batchGradient r (batch t) (adamWRun r batch (wsdRate B) draw t).weights)) atTop (𝓝 G)) :
    Tendsto (fun t => (adamWRun r batch (wsdRate B) draw t).weights) atTop
      (𝓝 (fun i => -10 * (G i / (|G i| + 1/100000000)))) :=
  AdamW.limit_of_common_clipped_gradient _ _ _ G (η := 1/100000000) (by norm_num) (by norm_num)
    ((eventually_ge_atTop (B-1)).mono fun _ ht => wsdRate_eq_floor hw hd ht) hg

end Transformer.Grokking.FiniteRecovery.Argand
