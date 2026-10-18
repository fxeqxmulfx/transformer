/-
# Smooth cones following the original particle trajectories

Auxiliary to arXiv:2411.04551v3, §1, the Lipschitz well-posedness
argument for `eq: cauchy.pb`, and the characteristic tests in §3.1.

For an actual trajectory `Y`, the regularized distance to `Y(t)` is smooth
in space and `C¹` in time, including for a ReLU vector field. Multiplying
by `L exp(K(T-t))` makes its material derivative nonpositive when the
original field is `K`-Lipschitz on the sphere. Adding a constant records
a terminal test value at the transported center.

The regularized radius is `sqrt(‖x-Y(t)‖²+ε²)`. Its material derivative
is bounded by `K` times the radius, and the exponential factor contributes
the opposite bound. The positive regularization removes the norm kink.

The upper and lower bounds retain an explicit regularization error.
Finite smooth minima of these cones will compare arbitrary weak laws
with their actual transported initial probabilities.
-/

import Transformer.Interpolation.Section1_AutonomousComparison

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A regularized Lipschitz cone centered on an actual particle path,
with the backward exponential slope. Auxiliary for arXiv:2411.04551v3,
§1, the weak transport comparison for `eq: cauchy.pb`. -/
noncomputable def transportCone (K L ε a T : ℝ) (Y : ℝ → SSphere d)
    (q : ℝ × EucSpace d) : ℝ :=
  a + L * exp (K * (T - q.1)) * sqrt (‖q.2 - (Y q.1 : EucSpace d)‖ ^ 2 + ε ^ 2)

/-- Positive regularization makes the cone a valid ambient `C¹` test.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem contDiff_transportCone (K L ε a T : ℝ) (hε : 0 < ε) (Y : ℝ → SSphere d)
    (hY : ContDiff ℝ 1 (fun s => (Y s : EucSpace d))) :
    ContDiff ℝ 1 (transportCone K L ε a T Y) := by
  have hq : ContDiff ℝ 1 (fun q : ℝ × EucSpace d =>
      sqrt (‖q.2 - (Y q.1 : EucSpace d)‖ ^ 2 + ε ^ 2)) :=
    (((contDiff_norm_sq ℝ).comp (contDiff_snd.sub (hY.comp contDiff_fst))).add contDiff_const).sqrt
      (fun q => (add_pos_of_nonneg_of_pos (sq_nonneg _) (sq_pos_of_pos hε)).ne')
  exact contDiff_const.add ((contDiff_const.mul
    (contDiff_const.mul (contDiff_const.sub contDiff_fst)).exp).mul hq)

/-- Each actual cone has nonpositive material derivative on the sphere.
Only the current field and the path's current derivative enter this estimate;
the path is C1 in time. The original Lipschitz estimate cancels its slope decay.
Auxiliary for arXiv:2411.04551v3, §1, the well-posedness argument. -/
theorem fderiv_transportCone_nonpos_at (F : EucSpace d → EucSpace d)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (L ε a T : ℝ)
    (hL : 0 ≤ L) (hε : 0 < ε) (Y : ℝ → SSphere d)
    (hY : ContDiff ℝ 1 (fun r => (Y r : EucSpace d)))
    (t : ℝ) (hYt : HasDerivAt (fun r => (Y r : EucSpace d)) (F (Y t)) t) (x : SSphere d) :
    fderiv ℝ (transportCone K L ε a T Y) (t, (x : EucSpace d)) (1, F x) ≤ 0 := by
  let z (s : ℝ) : EucSpace d := x + (s - t) • F x - Y s
  let R := sqrt (‖(x : EucSpace d) - (Y t : EucSpace d)‖ ^ 2 + ε ^ 2)
  let w := L * exp ((K : ℝ) * (T - t))
  have hR : 0 < R := sqrt_pos.mpr (add_pos_of_nonneg_of_pos (sq_nonneg _) (sq_pos_of_pos hε))
  have hz : z t = (x : EucSpace d) - (Y t : EucSpace d) := by simp [z]
  have hDz : HasDerivAt z (F x - F (Y t)) t := by
    simpa only [Pi.add_apply, Pi.sub_apply, id_eq, one_smul, zero_add] using
      ((((hasDerivAt_id t).sub_const t).smul_const (F x)).const_add (x : EucSpace d)).fun_sub hYt
  have hDq : HasDerivAt (fun s => sqrt (‖z s‖ ^ 2 + ε ^ 2))
      ((2 * inner ℝ ((x : EucSpace d) - (Y t : EucSpace d)) (F x - F (Y t))) / (2 * R)) t := by
    simpa only [hz] using (hDz.norm_sq.add_const (ε ^ 2)).sqrt
      (show ‖z t‖ ^ 2 + ε ^ 2 ≠ 0 from
        (add_pos_of_nonneg_of_pos (sq_nonneg _) (sq_pos_of_pos hε)).ne')
  have hDw : HasDerivAt (fun s => L * exp ((K : ℝ) * (T - s))) (-((K : ℝ) * w)) t := by
    convert (((hasDerivAt_id t).const_sub T).const_mul (K : ℝ)).exp.const_mul L using 1
    · funext s
      simp only [id_eq]
    · dsimp only [w, id_eq]
      ring
  have hDt := (hDw.mul hDq).const_add a
  let γ (s : ℝ) : ℝ × EucSpace d := (s, x + (s - t) • F x)
  have hγ : HasDerivAt γ (1, F x) t := by
    simpa only [Pi.add_apply, id_eq, one_smul, zero_add] using
      (hasDerivAt_id t).prodMk
        ((((hasDerivAt_id t).sub_const t).smul_const (F x)).const_add (x : EucSpace d))
  have hγt : γ t = (t, (x : EucSpace d)) := by simp [γ]
  have hc := ((contDiff_transportCone (K : ℝ) L ε a T hε Y
    hY).differentiable (by norm_num) (t, (x : EucSpace d))).hasFDerivAt
  rw [← hγt] at hc
  have he := (hc.comp_hasDerivAt t hγ).unique hDt
  rw [hγt, hz] at he
  rw [he]
  have hinner : inner ℝ ((x : EucSpace d) - (Y t : EucSpace d)) (F x - F (Y t)) ≤
      (K : ℝ) * ‖(x : EucSpace d) - (Y t : EucSpace d)‖ ^ 2 := by
    calc
      _ ≤ |inner ℝ ((x : EucSpace d) - (Y t : EucSpace d)) (F x - F (Y t))| := le_abs_self _
      _ ≤ ‖(x : EucSpace d) - (Y t : EucSpace d)‖ * ‖F x - F (Y t)‖ := abs_real_inner_le_norm _ _
      _ ≤ ‖(x : EucSpace d) - (Y t : EucSpace d)‖ *
          ((K : ℝ) * ‖(x : EucSpace d) - (Y t : EucSpace d)‖) :=
        mul_le_mul_of_nonneg_left (lipschitzOnWith_iff_norm_sub_le.mp hK x.2 (Y t).2) (norm_nonneg _)
      _ = _ := by ring
  have hRsq : R ^ 2 = ‖(x : EucSpace d) - (Y t : EucSpace d)‖ ^ 2 + ε ^ 2 :=
    sq_sqrt (add_nonneg (sq_nonneg _) (sq_nonneg _))
  have hfrac : (2 * inner ℝ ((x : EucSpace d) - (Y t : EucSpace d)) (F x - F (Y t))) /
      (2 * R) ≤ (K : ℝ) * R := by
    apply (div_le_iff₀ (mul_pos (by norm_num) hR)).mpr
    have hb : (K : ℝ) * ‖(x : EucSpace d) - (Y t : EucSpace d)‖ ^ 2 ≤ (K : ℝ) * R ^ 2 :=
      mul_le_mul_of_nonneg_left (by nlinarith [sq_nonneg ε]) K.coe_nonneg
    nlinarith
  have hw : 0 ≤ w := mul_nonneg hL (exp_pos _).le
  have hb := mul_le_mul_of_nonneg_left hfrac hw
  nlinarith


/-- The pointwise cone comparison applies to every actual autonomous
trajectory. The particle equation supplies its C1 time regularity.
Auxiliary for arXiv:2411.04551v3, §1, the well-posedness argument. -/
theorem fderiv_transportCone_nonpos (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (L ε a T : ℝ)
    (hL : 0 ≤ L) (hε : 0 < ε) (Y : ℝ → SSphere d)
    (hY : ∀ s : ℝ, HasDerivAt (fun r => (Y r : EucSpace d)) (F (Y s)) s)
    (t : ℝ) (x : SSphere d) :
    fderiv ℝ (transportCone K L ε a T Y) (t, (x : EucSpace d)) (1, F x) ≤ 0 :=
  fderiv_transportCone_nonpos_at F K hK L ε a T hL hε Y
    (contDiff_spherical_particle F hF Y hY) t (hY t) x

/-- A regularized cone dominates the corresponding exact distance cone.
Auxiliary for arXiv:2411.04551v3, §1, the weak transport comparison. -/
theorem transportCone_lower (K L ε a T : ℝ) (hL : 0 ≤ L) (Y : ℝ → SSphere d)
    (q : ℝ × EucSpace d) :
    a + L * exp (K * (T - q.1)) * ‖q.2 - (Y q.1 : EucSpace d)‖ ≤ transportCone K L ε a T Y q := by
  have hd : ‖q.2 - (Y q.1 : EucSpace d)‖ ≤ sqrt (‖q.2 - (Y q.1 : EucSpace d)‖ ^ 2 + ε ^ 2) :=
    le_sqrt_of_sq_le (le_add_of_nonneg_right (sq_nonneg _))
  exact add_le_add (le_refl a) (mul_le_mul_of_nonneg_left hd (mul_nonneg hL (exp_pos (K * (T - q.1))).le))

/-- The regularization costs at most the slope times its scale.
Auxiliary for arXiv:2411.04551v3, §1, the weak transport comparison. -/
theorem transportCone_upper (K L ε a T : ℝ) (hL : 0 ≤ L) (hε : 0 ≤ ε) (Y : ℝ → SSphere d)
    (q : ℝ × EucSpace d) :
    transportCone K L ε a T Y q ≤ a + L * exp (K * (T - q.1)) * (‖q.2 - (Y q.1 : EucSpace d)‖ + ε) := by
  have hd : sqrt (‖q.2 - (Y q.1 : EucSpace d)‖ ^ 2 + ε ^ 2) ≤ ‖q.2 - (Y q.1 : EucSpace d)‖ + ε := by
    apply sqrt_le_iff.mpr ⟨add_nonneg (norm_nonneg _) hε, ?_⟩
    nlinarith [mul_nonneg (norm_nonneg (q.2 - (Y q.1 : EucSpace d))) hε]
  exact add_le_add (le_refl a) (mul_le_mul_of_nonneg_left hd (mul_nonneg hL (exp_pos (K * (T - q.1))).le))

/-- An actual projection neuron and its moving transverse atom jointly
realize continuity, compact-sphere Lipschitz regularity and the original
particle equation, at positive regularization and nonnegative slope. -/
example : ∃ (F : EucSpace 2 → EucSpace 2) (K : ℝ≥0) (Y : ℝ → SSphere 2),
    Continuous F ∧ LipschitzOnWith K F (Sphere 2) ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    ContDiff ℝ 1 (fun s => (Y s : EucSpace 2)) ∧
    ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace 2)) (F (Y t)) t := by
  let a : EucSpace 2 := basePoint 1
  let F (x : EucSpace 2) := proj 2 x a
  let Y := ridgeFlow a 0 1 (⟨EuclideanSpace.single 1 1, by simp⟩ : SSphere 2)
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  have hY (t : ℝ) : HasDerivAt (fun s => (Y s : EucSpace 2)) (F (Y t)) t := by
    simpa [F, Y] using hasDerivAt_ridgeFlow a 0 1 (⟨EuclideanSpace.single 1 1, by simp⟩ : SSphere 2) t
  exact ⟨F, K, Y, hF.continuous, hK.mono hsphere, zero_le_one, zero_lt_one,
    contDiff_spherical_particle F hF.continuous Y hY, hY⟩

end Transformer.Interpolation
