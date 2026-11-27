module

public import Transformer.Optimization.FullMomentumDescent

/-! Compact sublevel bounds and first-order convergence for the clipped
full-gradient momentum update. The local Lipschitz constant is taken on an
expanded ball containing the segments of the actual updates. -/

@[expose] public section

open Filter Metric Set
open scoped Topology NNReal

namespace Transformer.Optimization

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

/-- The small constant step can be selected from the objective and the
initial state, using just a Lipschitz bound on a compact ball. A finite
warmup may supply the initial state for this constant-rate tail. -/
theorem exists_pos_fullMomentum_descent {f : E → ℝ} (hf : Differentiable ℝ f)
    (hLip : LocallyLipschitz (gradient f)) {γ : ℝ} (hγ : 0 < γ)
    (hcoerc : ∀ w, γ * ‖w‖ ^ 2 ≤ f w) (x₀ v₀ : E) (hv₀ : ‖v₀‖ ≤ 10) :
    ∃ R ≥ 0, ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ (x v : ℕ → E),
      x 0 = x₀ → v 0 = v₀ →
      (∀ t, v (t + 1) = (9 / 10 : ℝ) • v t +
        (clipScale (gradient f (x t)))⁻¹ • gradient f (x t)) →
      (∀ t, x (t + 1) = x t - η • v (t + 1)) →
      (∀ t, ‖x t‖ ≤ R) ∧ ∀ t,
        fullMomentumEnergy f η (x (t + 1)) (v (t + 1)) +
          η / 40 * (‖v t‖ ^ 2 + ‖v (t + 1)‖ ^ 2) ≤
        fullMomentumEnergy f η (x t) (v t) := by
  let cap := fullMomentumEnergy f 1 x₀ v₀
  let R := Real.sqrt (cap / γ)
  let B := closedBall (0 : E) (R + 1)
  obtain ⟨L, hL⟩ := hLip.locallyLipschitzOn.exists_lipschitzOnWith_of_compact
    (isCompact_closedBall (0 : E) (R + 1))
  refine ⟨R, Real.sqrt_nonneg _, min (1 / 10) (1 / (240 * ((L : ℝ) + 1))), ?_, ?_⟩
  · positivity
  intro η hη hη₀ x v hx₀ hvinit hvel hpos
  have hη₁ : η ≤ 1 / 10 := hη₀.trans (min_le_left _ _)
  have hηL : (L : ℝ) * η ≤ 1 / 240 := by
    have h := (le_div_iff₀ (by positivity : 0 < 240 * ((L : ℝ) + 1))).mp
      (hη₀.trans (min_le_right _ _))
    nlinarith only [h, hη.le]
  have hv (t : ℕ) : ‖v t‖ ≤ 10 := by
    induction t with
    | zero => rwa [hvinit]
    | succ t ih => exact hvel t ▸ norm_fullMomentum_velocity_le _ _ ih
  have hnorm (t : ℕ) (ht : fullMomentumEnergy f η (x t) (v t) ≤ cap) : ‖x t‖ ≤ R :=
    Real.le_sqrt_of_sq_le ((le_div_iff₀ hγ).mpr (by
      calc
        ‖x t‖ ^ 2 * γ = γ * ‖x t‖ ^ 2 := mul_comm _ _
        _ ≤ f (x t) := hcoerc _
        _ ≤ fullMomentumEnergy f η (x t) (v t) := le_fullMomentumEnergy f hη.le _ _
        _ ≤ cap := ht))
  have hstep (t : ℕ) (ht : fullMomentumEnergy f η (x t) (v t) ≤ cap) :
      fullMomentumEnergy f η (x (t + 1)) (v (t + 1)) +
        η / 40 * (‖v t‖ ^ 2 + ‖v (t + 1)‖ ^ 2) ≤ fullMomentumEnergy f η (x t) (v t) := by
    have hx : x t ∈ B := mem_closedBall_zero_iff.mpr ((hnorm t ht).trans (by linarith))
    have hy : x (t + 1) ∈ B := by
      apply mem_closedBall_zero_iff.mpr
      rw [hpos t]
      refine (norm_sub_le _ _).trans ?_
      rw [norm_smul, Real.norm_of_nonneg hη.le]
      have h := mul_le_mul hη₁ (hv (t + 1)) (norm_nonneg _) (by norm_num)
      linarith [hnorm t ht]
    exact fullMomentumEnergy_step hf hL (convex_closedBall _ _) hη hηL hx hy
      (hv (t + 1)) (hvel t) (hpos t)
  have hcap (t : ℕ) : fullMomentumEnergy f η (x t) (v t) ≤ cap := by
    induction t with
    | zero =>
      rw [hx₀, hvinit]
      unfold cap fullMomentumEnergy
      apply add_le_add (le_refl _)
      have h : η / 2 ≤ 1 / 2 := by linarith
      have hq := one_le_clipScale (gradient f x₀)
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_right h (by linarith [hq])) (sq_nonneg _)
    | succ t ih =>
      exact (le_add_of_nonneg_right (by positivity)).trans ((hstep t ih).trans ih)
  exact ⟨fun t => hnorm t (hcap t), fun t => hstep t (hcap t)⟩

/-- The kinetic budget forces the full gradient to vanish and eventually
turns off clipping. The actual objective values then converge to the
same limit as the energy. -/
theorem fullMomentum_limits {f : E → ℝ} (hf : ∀ x, 0 ≤ f x)
    {η : ℝ} (hη : 0 < η) (x v : ℕ → E)
    (hvel : ∀ t, v (t + 1) = (9 / 10 : ℝ) • v t +
      (clipScale (gradient f (x t)))⁻¹ • gradient f (x t))
    (hdec : ∀ t, fullMomentumEnergy f η (x (t + 1)) (v (t + 1)) +
      η / 40 * (‖v t‖ ^ 2 + ‖v (t + 1)‖ ^ 2) ≤ fullMomentumEnergy f η (x t) (v t)) :
    ∃ value, Tendsto (fun t => f (x t)) atTop (𝓝 value) ∧ Tendsto v atTop (𝓝 0) ∧
      Tendsto (fun t => gradient f (x t)) atTop (𝓝 0) ∧
      ∀ᶠ t in atTop, clipScale (gradient f (x t)) = 1 := by
  let e (t : ℕ) := fullMomentumEnergy f η (x t) (v t)
  have hanti : Antitone e :=
    antitone_nat_of_succ_le fun t => (le_add_of_nonneg_right (by positivity)).trans (hdec t)
  have he : Tendsto e atTop (𝓝 (⨅ t, e t)) := tendsto_atTop_ciInf hanti
    ⟨0, by rintro _ ⟨t, rfl⟩; exact (hf _).trans (le_fullMomentumEnergy f hη.le _ _)⟩
  have hdiff := he.sub ((tendsto_add_atTop_iff_nat 1).2 he)
  rw [sub_self] at hdiff
  have hcap := hdiff.div_const (η / 40)
  rw [zero_div] at hcap
  have hvsq : Tendsto (fun t => ‖v t‖ ^ 2) atTop (𝓝 0) := squeeze_zero
    (fun _ => sq_nonneg _) (fun t => (le_div_iff₀ (by positivity)).mpr (by
      have h := hdec t
      dsimp only [e] at *
      nlinarith only [h, mul_nonneg (show 0 ≤ η / 40 by positivity) (sq_nonneg ‖v (t + 1)‖)])) hcap
  have hvnorm : Tendsto (fun t => ‖v t‖) atTop (𝓝 0) := by
    simpa only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] using
      (Real.continuous_sqrt.tendsto 0).comp hvsq
  have hv := tendsto_zero_iff_norm_tendsto_zero.mpr hvnorm
  have hgc : Tendsto (fun t => (clipScale (gradient f (x t)))⁻¹ • gradient f (x t))
      atTop (𝓝 0) := by
    have h := ((tendsto_add_atTop_iff_nat 1).2 hv).sub (tendsto_const_nhds.smul hv :
      Tendsto (fun t => (9 / 10 : ℝ) • v t) atTop (𝓝 ((9 / 10 : ℝ) • (0 : E))))
    simp only [smul_zero, sub_zero] at h
    exact h.congr fun t => by rw [hvel t, add_sub_cancel_left]
  have hunclip : ∀ᶠ t in atTop, clipScale (gradient f (x t)) = 1 := by
    have hgn : Tendsto (fun t => ‖(clipScale (gradient f (x t)))⁻¹ • gradient f (x t)‖)
        atTop (𝓝 0) := by simpa only [norm_zero] using hgc.norm
    have hh := hgn.eventually (gt_mem_nhds zero_lt_one)
    filter_upwards [hh] with t ht
    have h : ‖gradient f (x t)‖ ≤ 1 := by
      by_contra! hn
      simp only [clipScale, max_eq_left hn.le, norm_smul,
        Real.norm_of_nonneg (inv_nonneg.mpr (norm_nonneg _)),
        inv_mul_cancel₀ (zero_lt_one.trans hn).ne'] at ht
      exact lt_irrefl _ ht
    exact max_eq_right h
  have hg := hgc.congr' (hunclip.mono fun t ht => by rw [ht, inv_one, one_smul])
  have hk : Tendsto (fun t => η / 2 * ‖v t‖ ^ 2) atTop (𝓝 0) := by
    simpa only [mul_zero] using hvsq.const_mul (η / 2)
  have hF := he.sub hk
  rw [sub_zero] at hF
  refine ⟨_, hF.congr' (hunclip.mono fun t ht => ?_), hv, hg, hunclip⟩
  simp [e, fullMomentumEnergy, ht]

end Transformer.Optimization
