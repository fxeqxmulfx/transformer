module

public import Transformer.Interpolation.Section1_SphericalFlowBounds

/-!
# Bounds for time-dependent spherical characteristics

Auxiliary for arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`.
A uniform spatial Lipschitz estimate controls dependence on initial
points in both time directions. A uniform speed bound controls the time
variable. The field retains its explicit time dependence throughout.

For negative times the reversed trajectory solves the field
`(s, x) ↦ -F(-s, x)`, which has the same uniform spatial modulus.
Joint continuity then follows by splitting the distance into a time
increment at fixed initial point and an initial-point increment at
fixed time. Neither estimate assumes that the field is autonomous
or that the flow has spatial derivatives.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Grönwall compares two actual spherical trajectories using only the
original field's spherical Lipschitz bound. Source: arXiv:2411.04551v3,
§1, the well-posedness argument for `eq: ips`. -/
theorem dist_timeDependent_spherical_trajectories_le_exp (F : ℝ → EucSpace d → EucSpace d) (K : ℝ≥0)
    (hK : ∀ s, LipschitzOnWith K (F s) (Sphere d)) (X Y : ℝ → SSphere d)
    (hX : ∀ t, HasDerivAt (fun s => (X s : EucSpace d)) (F t (X t)) t)
    (hY : ∀ t, HasDerivAt (fun s => (Y s : EucSpace d)) (F t (Y t)) t)
    (t : ℝ) (ht : 0 ≤ t) :
    dist (X t) (Y t) ≤ dist (X 0) (Y 0) * exp ((K : ℝ) * t) := by
  have hd (s : ℝ) := (hX s).sub (hY s)
  have hc : Continuous (fun s => (X s : EucSpace d) - (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hd s).continuousAt
  have hb (s : ℝ) : ‖F s (X s) - F s (Y s)‖ ≤ (K : ℝ) * ‖(X s : EucSpace d) - (Y s : EucSpace d)‖ :=
    lipschitzOnWith_iff_norm_sub_le.mp (hK s) (X s).2 (Y s).2
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := ‖(X 0 : EucSpace d) - (Y 0 : EucSpace d)‖) (K := (K : ℝ)) (ε := 0)
    hc.continuousOn (fun s _ => (hd s).hasDerivWithinAt) le_rfl
    (fun s _ => by simpa only [add_zero] using hb s) t ⟨ht, le_rfl⟩
  change dist (X t : EucSpace d) (Y t : EucSpace d) ≤
    dist (X 0 : EucSpace d) (Y 0 : EucSpace d) * exp ((K : ℝ) * t)
  simpa only [gronwallBound_ε0, sub_zero, ← dist_eq_norm] using hG

/-- Every original flow map is Lipschitz in its initial point at all times.
Source: arXiv:2411.04551v3, §1, the original characteristic flow. -/
theorem lipschitzWith_timeDependent_sphericalFlow_initial (F : ℝ → EucSpace d → EucSpace d) (K : ℝ≥0)
    (hK : ∀ s, LipschitzOnWith K (F s) (Sphere d)) (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F t (Φ x t)) t)
    (hΦ₀ : ∀ x, Φ x 0 = x) (t : ℝ) :
    LipschitzWith (Real.toNNReal (exp ((K : ℝ) * |t|))) (fun x => Φ x t) := by
  rw [lipschitzWith_iff_dist_le_mul]
  intro x y
  rw [Real.coe_toNNReal _ (exp_pos _).le, mul_comm]
  by_cases ht : 0 ≤ t
  · simpa only [abs_of_nonneg ht, hΦ₀] using dist_timeDependent_spherical_trajectories_le_exp F K hK
      (Φ x) (Φ y) (hΦ x) (hΦ y) t ht
  · have hrev (z : SSphere d) (r : ℝ) :
        HasDerivAt (fun s => (Φ z (-s) : EucSpace d)) (-F (-r) (Φ z (-r))) r := by
      simpa only [Function.comp_def, neg_smul, one_smul] using (hΦ z (-r)).scomp r ((hasDerivAt_id r).neg)
    simpa only [neg_neg, neg_zero, hΦ₀, abs_of_neg (lt_of_not_ge ht)] using
      dist_timeDependent_spherical_trajectories_le_exp (fun r z => -F (-r) z) K
        (fun r => (hK (-r)).neg) (fun r => Φ x (-r)) (fun r => Φ y (-r))
        (hrev x) (hrev y) (-t) (by linarith)

/-- A finite original spatial modulus exists at every real time.
Source: arXiv:2411.04551v3, §1, the Lipschitz characteristic flow. -/
theorem exists_lipschitzWith_timeDependent_sphericalFlow_initial (F : ℝ → EucSpace d → EucSpace d) (K : ℝ≥0)
    (hK : ∀ s, LipschitzOnWith K (F s) (Sphere d)) (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F t (Φ x t)) t)
    (hΦ₀ : ∀ x, Φ x 0 = x) (t : ℝ) :
    ∃ L : ℝ≥0, LipschitzWith L (fun x => Φ x t) :=
  ⟨_, lipschitzWith_timeDependent_sphericalFlow_initial F K hK Φ hΦ hΦ₀ t⟩

/-- A bound on the original spherical speed gives the actual time modulus.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem lipschitzWith_timeDependent_spherical_trajectory_time (F : ℝ → EucSpace d → EucSpace d) (B : ℝ≥0)
    (hB : ∀ (s : ℝ) (x : SSphere d), ‖F s x‖ ≤ B) (X : ℝ → SSphere d)
    (hX : ∀ t, HasDerivAt (fun s => (X s : EucSpace d)) (F t (X t)) t) : LipschitzWith B X := by
  have hLip : LipschitzWith B (fun s => (X s : EucSpace d)) := by
    apply lipschitzWith_of_nnnorm_deriv_le (𝕜 := ℝ) (fun s => (hX s).differentiableAt)
    intro s
    apply NNReal.coe_le_coe.mp
    rw [coe_nnnorm, (hX s).deriv]
    exact hB s (X s)
  exact hLip.subtype_mk _


/-- The actual original characteristic flow is jointly continuous.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem continuous_timeDependent_sphericalFlow (F : ℝ → EucSpace d → EucSpace d)
    (K C : ℝ≥0) (hK : ∀ t, LipschitzOnWith K (F t) (Sphere d))
    (hB : ∀ (t : ℝ) (x : SSphere d), ‖F t x‖ ≤ C) (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F t (Φ x t)) t)
    (hΦ₀ : ∀ x, Φ x 0 = x) : Continuous (fun q : ℝ × SSphere d => Φ q.2 q.1) := by
  have htime (x : SSphere d) : LipschitzWith C (Φ x) :=
    lipschitzWith_timeDependent_spherical_trajectory_time F C hB (Φ x) (hΦ x)
  have hinitial (t : ℝ) : Continuous (fun x => Φ x t) :=
    (lipschitzWith_timeDependent_sphericalFlow_initial F K hK Φ hΦ hΦ₀ t).continuous
  apply continuous_iff_continuousAt.mpr
  intro q₀
  have ht (q : ℝ × SSphere d) : dist (Φ q.2 q.1) (Φ q.2 q₀.1) ≤ (C : ℝ) * dist q.1 q₀.1 :=
    (htime q.2).dist_le_mul q.1 q₀.1
  have hc : Continuous (fun q : ℝ × SSphere d => (C : ℝ) * dist q.1 q₀.1 + dist (Φ q.2 q₀.1) (Φ q₀.2 q₀.1)) :=
    (continuous_const.mul (continuous_fst.dist continuous_const)).add
      (((hinitial q₀.1).comp continuous_snd).dist continuous_const)
  have hlim : Tendsto (fun q : ℝ × SSphere d => (C : ℝ) * dist q.1 q₀.1 + dist (Φ q.2 q₀.1) (Φ q₀.2 q₀.1))
      (𝓝 q₀) (𝓝 0) := by
    simpa only [dist_self, mul_zero, add_zero] using hc.continuousAt.tendsto (x := q₀)
  apply tendsto_iff_dist_tendsto_zero.mpr
  exact squeeze_zero (fun _ => dist_nonneg) (fun q =>
    (dist_triangle _ (Φ q.2 q₀.1) _).trans (add_le_add (ht q) le_rfl)) hlim


/-- A projection neuron driven by `cos(t)` and its actual sinusoidally
reparametrized flow jointly realize every trajectory, initial-value,
uniform spatial bound, speed bound and nonnegative-time hypothesis. -/
example : ∃ (F : ℝ → EucSpace 2 → EucSpace 2) (Φ : SSphere 2 → ℝ → SSphere 2),
    (∀ t, LipschitzOnWith 2 (F t) (Sphere 2)) ∧
    (∀ (t : ℝ) (x : SSphere 2), ‖F t x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace 2)) (F t (Φ x t)) t) ∧
    (∀ x, Φ x 0 = x) ∧ (0 : ℝ) ≤ 1 := by
  let p : EucSpace 2 := basePoint 1
  let F (t : ℝ) (x : EucSpace 2) := cos t • proj 2 x p
  let Φ (x : SSphere 2) (t : ℝ) := ridgeFlow p 0 1 x (sin t)
  have hp : ‖p‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 1).2
  refine ⟨F, Φ, ?_, ?_, ?_, ?_, zero_le_one⟩
  · intro t
    apply lipschitzOnWith_iff_norm_sub_le.mpr
    intro x hx y hy
    change ‖cos t • proj 2 x p - cos t • proj 2 y p‖ ≤ (2 : ℝ) * ‖x - y‖
    rw [← smul_sub, norm_smul, Real.norm_eq_abs]
    have he := norm_proj_sub_proj_le (mem_sphere_zero_iff_norm.mp hx)
      (mem_sphere_zero_iff_norm.mp hy) hp.le (le_refl ‖x - y‖)
      (show ‖p - p‖ ≤ (0 : ℝ) by simp)
    simpa only [zero_add, one_mul] using
      mul_le_mul (abs_cos_le_one t) he (norm_nonneg _) zero_le_one
  · intro t x
    rw [norm_smul, Real.norm_eq_abs]
    change |cos t| * ‖proj 2 (x : EucSpace 2) p‖ ≤ (1 : ℝ)
    have he := (norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) p).trans hp.le
    simpa only [mul_one] using mul_le_mul (abs_cos_le_one t) he (norm_nonneg _) zero_le_one
  · intro x t
    have hd := (hasDerivAt_ridgeFlow p 0 1 x (sin t)).scomp t (hasDerivAt_sin t)
    simpa [Function.comp_def, Φ, F] using hd
  · intro x
    simp only [Φ, sin_zero, ridgeFlow_zero]

end Transformer.Interpolation
