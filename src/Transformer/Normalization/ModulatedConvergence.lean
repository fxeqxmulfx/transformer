/-
# The gradient inequality implies convergence of a modulated flow

The desingularizing power potential of Appendix D.1 of arXiv:2510.22026v2
controls displacement, including the finite-time stationary branch.
The gradient inequality is always an explicit hypothesis.
-/

import Transformer.Normalization.ModulatedLength
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity
import Transformer.Normalization.ModulatedCritical

namespace Transformer.Normalization

/-- The power primitive of the gradient inequality in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2 has the expected derivative above the
limiting energy. The strict energy gap is used only for this power rule. -/
theorem desingularized_energy_hasDerivAt (f : ℝ → ℝ) (q L C k alpha t : ℝ)
    (halpha : alpha < 1) (hgap : 0 < f t - L)
    (hf : HasDerivAt f (-q) t) :
    HasDerivAt (fun s => C * k / (1 - alpha) * (f s - L) ^ (1 - alpha))
      (-(C * k * q / (f t - L) ^ alpha)) t := by
  have ha : 1 - alpha ≠ 0 := ne_of_gt (sub_pos.mpr halpha)
  convert ((hf.sub_const L).rpow_const (Or.inl hgap.ne')).const_mul
    (C * k / (1 - alpha)) using 1
  rw [show 1 - alpha - 1 = -alpha by ring, Real.rpow_neg hgap.le]
  field_simp

/-- A linearly decreasing energy above its limit satisfies the power-rule
hypotheses at zero; Appendix D.1 of arXiv:2510.22026v2. -/
example : (0 : ℝ) < 1 ∧ 0 < (1 - (0 : ℝ)) - 0 ∧
    HasDerivAt (fun s : ℝ => 1 - s) (-1) 0 := by
  refine ⟨zero_lt_one, by norm_num, ?_⟩
  convert (hasDerivAt_id (0 : ℝ)).const_sub 1 using 1
  ext s
  rfl

/-- The angle condition and the gradient inequality in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2 bound speed by the decrease of the
desingularizing potential. This cancels both the gradient norm and `λ`. -/
theorem desingularized_speed_bound {N : ℕ} (A : ParamMatrix N)
    (g : EucSpace N) (r C k alpha : ℝ) (hr : 0 < r) (hk : 0 < k)
    (hangle : ‖A g‖ * ‖g‖ ≤ C * inner (𝕜 := ℝ) (A g) g)
    (hKL : r ^ alpha ≤ k * ‖g‖) :
    ‖A g‖ ≤ C * k * inner (𝕜 := ℝ) (A g) g / r ^ alpha := by
  apply (le_div_iff₀ (Real.rpow_pos_of_pos hr alpha)).2
  have h := mul_le_mul_of_nonneg_left hKL (norm_nonneg (A g))
  have ha := mul_le_mul_of_nonneg_left hangle hk.le
  nlinarith

/-- A scalar identity operator and a unit gradient satisfy the speed
estimate hypotheses; Appendix D.1 of arXiv:2510.22026v2. -/
example (g : EucSpace 1) (hg : ‖g‖ = 1) :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    ‖g‖ * ‖g‖ ≤ 1 * inner (𝕜 := ℝ) g g ∧
    (1 : ℝ) ^ (1 / 2 : ℝ) ≤ 1 * ‖g‖ := by
  simp only [hg, one_mul, real_inner_self_eq_norm_sq, Real.one_rpow,
    one_pow, le_refl, zero_lt_one, and_self]

/-- Once the gradient inequality of Appendix D.1 of arXiv:2510.22026v2
holds along a tail whose energy remains above its limit, the modulated
trajectory converges. The gradient inequality is an explicit hypothesis,
so the analytic input is visible in the statement. -/
theorem modulated_converges_of_strict_gradient_inequality {N : ℕ}
    (E : EucSpace N → ℝ) (x : ℝ → EucSpace N) (M : ℝ → ParamMatrix N)
    (a L C k alpha : ℝ) (hC : 0 ≤ C) (hk : 0 < k) (halpha : alpha < 1)
    (hE : ∀ t, a ≤ t → DifferentiableAt ℝ E (x t))
    (hx : ∀ t, a ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t)
    (hangle : ∀ t, a ≤ t →
      ‖M t (gradient E (x t))‖ * ‖gradient E (x t)‖ ≤
        C * inner (𝕜 := ℝ) (M t (gradient E (x t))) (gradient E (x t)))
    (hgap : ∀ t, a ≤ t → L < E (x t))
    (hKL : ∀ t, a ≤ t → (E (x t) - L) ^ alpha ≤ k * ‖gradient E (x t)‖)
    (hlim : Filter.Tendsto (fun t => E (x t)) Filter.atTop (nhds L)) :
    ∃ z : EucSpace N, Filter.Tendsto x Filter.atTop (nhds z) := by
  let p : ℝ → ℝ := fun t => C * k / (1 - alpha) * (E (x t) - L) ^ (1 - alpha)
  let q : ℝ → ℝ := fun t => inner (𝕜 := ℝ)
    (M t (gradient E (x t))) (gradient E (x t))
  let p' : ℝ → ℝ := fun t => -(C * k * q t / (E (x t) - L) ^ alpha)
  have hp t ht : HasDerivAt p (p' t) t :=
    desingularized_energy_hasDerivAt (fun t => E (x t)) (q t) L C k alpha t
      halpha (sub_pos.mpr (hgap t ht))
      (modulated_energy_hasDerivAt E x M t (hE t ht) (hx t ht))
  have hbound t (ht : a ≤ t) : ‖-(M t (gradient E (x t)))‖ ≤ -p' t := by
    rw [norm_neg]
    dsimp [p', q]
    rw [neg_neg]
    exact desingularized_speed_bound (M t) (gradient E (x t))
      (E (x t) - L) C k alpha (sub_pos.mpr (hgap t ht)) hk (hangle t ht) (hKL t ht)
  have hp0 : Filter.Tendsto p Filter.atTop (nhds 0) := by
    have hgap0 : Filter.Tendsto (fun t => E (x t) - L) Filter.atTop (nhds 0) := by
      simpa only [sub_self] using hlim.sub_const L
    have hrpow := (continuousAt_id.rpow_const
      (Or.inr (sub_pos.mpr halpha).le) :
        ContinuousAt (fun r : ℝ => r ^ (1 - alpha)) 0).tendsto.comp hgap0
    simpa only [p, Function.comp_apply, id_eq,
      Real.zero_rpow (sub_pos.mpr halpha).ne', mul_zero] using
      hrpow.const_mul (C * k / (1 - alpha))
  apply converges_of_displacement_bound x p a hp0
  intro s t has hst
  have hd := displacement_le_potential_drop x
    (fun u => -(M u (gradient E (x u)))) p p' s t hst
    (fun u hu => hx u (has.trans hu.1))
    (fun u hu => hp u (has.trans hu.1))
    (fun u hu => hbound u (has.trans hu.1))
  have hpt : 0 ≤ p t := mul_nonneg
    (div_nonneg (mul_nonneg hC hk.le) (sub_pos.mpr halpha).le)
    (Real.rpow_nonneg (sub_pos.mpr (hgap t (has.trans hst))).le _)
  linarith

/-- The gradient inequality from Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2 implies convergence of a modulated gradient trajectory.
This includes trajectories reaching the limiting energy in finite time.
The analytic theorem producing the inequality is not assumed implicitly. -/
theorem modulated_converges_of_gradient_inequality {N : ℕ}
    (E : EucSpace N → ℝ) (x : ℝ → EucSpace N) (M : ℝ → ParamMatrix N)
    (lam : ℝ → ℝ) (a L C k alpha : ℝ) (hC : 0 ≤ C) (hk : 0 < k)
    (halpha : alpha < 1)
    (hE : ∀ t, a ≤ t → DifferentiableAt ℝ E (x t))
    (hx : ∀ t, a ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t)
    (hlam : ∀ t, a ≤ t → 0 < lam t)
    (hM : ∀ t, a ≤ t → ∀ v : EucSpace N,
      lam t * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) (M t v) v)
    (hangle : ∀ t, a ≤ t →
      ‖M t (gradient E (x t))‖ * ‖gradient E (x t)‖ ≤
        C * inner (𝕜 := ℝ) (M t (gradient E (x t))) (gradient E (x t)))
    (hgap : ∀ t, a ≤ t → L ≤ E (x t))
    (hKL : ∀ t, a ≤ t → (E (x t) - L) ^ alpha ≤ k * ‖gradient E (x t)‖)
    (hlim : Filter.Tendsto (fun t => E (x t)) Filter.atTop (nhds L)) :
    ∃ z : EucSpace N, Filter.Tendsto x Filter.atTop (nhds z) := by
  by_cases hhit : ∃ t, a ≤ t ∧ E (x t) = L
  · obtain ⟨b, hab, hb⟩ := hhit
    have hanti := modulated_energy_antitoneOn E x M a hE hx
      (fun t ht => (mul_nonneg (hlam t ht).le (sq_nonneg _)).trans
        (hM t ht (gradient E (x t))))
    have hplateau : ∀ t, b ≤ t → E (x t) = L := by
      intro t hbt
      apply le_antisymm
      · exact (hanti hab (hab.trans hbt) hbt).trans_eq hb
      · exact hgap t (hab.trans hbt)
    have hconst := modulated_constant_of_energy_plateau E x M lam b L
      (fun t ht => hE t (hab.trans ht)) (fun t ht => hx t (hab.trans ht))
      (fun t ht => hlam t (hab.trans ht)) (fun t ht => hM t (hab.trans ht)) hplateau
    refine ⟨x b, tendsto_const_nhds.congr' ?_⟩
    filter_upwards [Filter.eventually_ge_atTop b] with t ht
    exact (hconst t ht).symm
  · apply modulated_converges_of_strict_gradient_inequality E x M a L C k alpha
      hC hk halpha hE hx hangle ?_ hKL hlim
    intro t ht
    exact lt_of_le_of_ne (hgap t ht) (fun heq => hhit ⟨t, ht, heq.symm⟩)

/-- Zero energy and the zero flow satisfy the full conditional convergence
hypotheses, with exponent `1/2`; Appendix D.1 of arXiv:2510.22026v2. -/
example :
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ t : ℝ, 0 ≤ t → DifferentiableAt ℝ (fun _ : EucSpace 1 => (0 : ℝ)) 0) ∧
    (∀ t : ℝ, 0 ≤ t → HasDerivAt (fun _ : ℝ => (0 : EucSpace 1)) 0 t) ∧
    (∀ v : EucSpace 1, (1 : ℝ) * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) v v) ∧
    (∀ t : ℝ, 0 ≤ t → ((0 : ℝ) - 0) ^ (1 / 2 : ℝ) ≤ 1 * ‖(0 : EucSpace 1)‖) ∧
    Filter.Tendsto (fun _ : ℝ => (0 : ℝ)) Filter.atTop (nhds 0) := by
  refine ⟨zero_le_one, zero_lt_one, by norm_num, fun _ _ => differentiableAt_const 0,
    fun t _ => hasDerivAt_const t 0, fun v => by simp, fun _ _ => ?_, tendsto_const_nhds⟩
  norm_num

end Transformer.Normalization
