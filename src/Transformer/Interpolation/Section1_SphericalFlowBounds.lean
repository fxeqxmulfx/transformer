module

public import Transformer.Interpolation.Section3_RidgeFlowContinuity

/-!
# Regularity of actual spherical characteristic flows

ArXiv:2411.04551v3, §1, `eq: ips`, `eq: neural.ode.sphere`,
and the Lipschitz well-posedness argument for `eq: cauchy.pb`.

Any actual flow of a field Lipschitz on the sphere has the original
Grönwall dependence on its initial point. Reversing time retains the
estimate for negative times. Compactness bounds a continuous field,
so all trajectories share a Lipschitz bound in time. Combining these
two estimates proves joint continuity of the actual characteristic flow.

The bounds use the original field on the sphere itself. They supply
the regularity needed to push arbitrary probabilities along any such
proved particle flow, including fields with several ReLU activations.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Grönwall compares two actual spherical trajectories using only the
original field's spherical Lipschitz bound. Source: arXiv:2411.04551v3,
§1, the well-posedness argument for `eq: ips`. -/
theorem dist_spherical_trajectories_le_exp (F : EucSpace d → EucSpace d) (K : ℝ≥0)
    (hK : LipschitzOnWith K F (Sphere d)) (X Y : ℝ → SSphere d)
    (hX : ∀ t, HasDerivAt (fun s => (X s : EucSpace d)) (F (X t)) t)
    (hY : ∀ t, HasDerivAt (fun s => (Y s : EucSpace d)) (F (Y t)) t)
    (t : ℝ) (ht : 0 ≤ t) :
    dist (X t) (Y t) ≤ dist (X 0) (Y 0) * exp ((K : ℝ) * t) := by
  have hd (s : ℝ) := (hX s).sub (hY s)
  have hc : Continuous (fun s => (X s : EucSpace d) - (Y s : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun s => (hd s).continuousAt
  have hb (s : ℝ) : ‖F (X s) - F (Y s)‖ ≤ (K : ℝ) * ‖(X s : EucSpace d) - (Y s : EucSpace d)‖ :=
    lipschitzOnWith_iff_norm_sub_le.mp hK (X s).2 (Y s).2
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := ‖(X 0 : EucSpace d) - (Y 0 : EucSpace d)‖) (K := (K : ℝ)) (ε := 0)
    hc.continuousOn (fun s _ => (hd s).hasDerivWithinAt) le_rfl
    (fun s _ => by simpa only [add_zero] using hb s) t ⟨ht, le_rfl⟩
  change dist (X t : EucSpace d) (Y t : EucSpace d) ≤
    dist (X 0 : EucSpace d) (Y 0 : EucSpace d) * exp ((K : ℝ) * t)
  simpa only [gronwallBound_ε0, sub_zero, ← dist_eq_norm] using hG

/-- Every original flow map is Lipschitz in its initial point at all times.
Source: arXiv:2411.04551v3, §1, the original characteristic flow. -/
theorem lipschitzWith_sphericalFlow_initial (F : EucSpace d → EucSpace d) (K : ℝ≥0)
    (hK : LipschitzOnWith K F (Sphere d)) (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F (Φ x t)) t)
    (hΦ₀ : ∀ x, Φ x 0 = x) (t : ℝ) :
    LipschitzWith (Real.toNNReal (exp ((K : ℝ) * |t|))) (fun x => Φ x t) := by
  rw [lipschitzWith_iff_dist_le_mul]
  intro x y
  rw [Real.coe_toNNReal _ (exp_pos _).le, mul_comm]
  by_cases ht : 0 ≤ t
  · simpa only [abs_of_nonneg ht, hΦ₀] using dist_spherical_trajectories_le_exp F K hK
      (Φ x) (Φ y) (hΦ x) (hΦ y) t ht
  · have hrev (z : SSphere d) (r : ℝ) :
        HasDerivAt (fun s => (Φ z (-s) : EucSpace d)) (-F (Φ z (-r))) r := by
      simpa only [Function.comp_def, neg_smul, one_smul] using (hΦ z (-r)).scomp r ((hasDerivAt_id r).neg)
    simpa only [neg_neg, neg_zero, hΦ₀, abs_of_neg (lt_of_not_ge ht)] using
      dist_spherical_trajectories_le_exp (fun z => -F z) K hK.neg (fun r => Φ x (-r)) (fun r => Φ y (-r))
        (hrev x) (hrev y) (-t) (by linarith)

/-- A finite original spatial modulus exists at every real time.
Source: arXiv:2411.04551v3, §1, the Lipschitz characteristic flow. -/
theorem exists_lipschitzWith_sphericalFlow_initial (F : EucSpace d → EucSpace d) (K : ℝ≥0)
    (hK : LipschitzOnWith K F (Sphere d)) (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F (Φ x t)) t)
    (hΦ₀ : ∀ x, Φ x 0 = x) (t : ℝ) :
    ∃ L : ℝ≥0, LipschitzWith L (fun x => Φ x t) :=
  ⟨_, lipschitzWith_sphericalFlow_initial F K hK Φ hΦ hΦ₀ t⟩

/-- A bound on the original spherical speed gives the actual time modulus.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem lipschitzWith_spherical_trajectory_time (F : EucSpace d → EucSpace d) (B : ℝ≥0)
    (hB : ∀ x : SSphere d, ‖F x‖ ≤ B) (X : ℝ → SSphere d)
    (hX : ∀ t, HasDerivAt (fun s => (X s : EucSpace d)) (F (X t)) t) : LipschitzWith B X := by
  have hLip : LipschitzWith B (fun s => (X s : EucSpace d)) := by
    apply lipschitzWith_of_nnnorm_deriv_le (𝕜 := ℝ) (fun s => (hX s).differentiableAt)
    intro s
    apply NNReal.coe_le_coe.mp
    rw [coe_nnnorm, (hX s).deriv]
    exact hB (X s)
  exact hLip.subtype_mk _


/-- Compactness supplies a common time modulus for a continuous field.
Source: arXiv:2411.04551v3, §1, the characteristic flow. -/
theorem exists_lipschitzWith_sphericalFlow_time (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F (Φ x t)) t) :
    ∃ C : ℝ≥0, ∀ x, LipschitzWith C (Φ x) := by
  obtain ⟨B, hB⟩ := (isCompact_univ : IsCompact (univ : Set (SSphere d))).exists_bound_of_continuousOn
    (hF.comp continuous_subtype_val).continuousOn
  let C := Real.toNNReal |B|
  have hcB (x : SSphere d) : ‖F x‖ ≤ C := by
    rw [Real.coe_toNNReal _ (abs_nonneg _)]
    exact (hB x (mem_univ x)).trans (le_abs_self _)
  exact ⟨C, fun x => lipschitzWith_spherical_trajectory_time F C hcB _ (hΦ x)⟩

/-- The actual original characteristic flow is jointly continuous.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem continuous_sphericalFlow (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (Φ : SSphere d → ℝ → SSphere d)
    (hΦ : ∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace d)) (F (Φ x t)) t)
    (hΦ₀ : ∀ x, Φ x 0 = x) : Continuous (fun q : ℝ × SSphere d => Φ q.2 q.1) := by
  obtain ⟨C, htime⟩ := exists_lipschitzWith_sphericalFlow_time F hF Φ hΦ
  have hinitial (t : ℝ) : Continuous (fun x => Φ x t) := (lipschitzWith_sphericalFlow_initial F K hK Φ hΦ hΦ₀ t).continuous
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

/-- A genuine projection neuron and its complete original flow jointly
realize the continuity, spherical Lipschitz, speed, trajectory, initial-value
and nonnegative-time hypotheses. The field is nonzero away from its axis. -/
example : ∃ (F : EucSpace 2 → EucSpace 2) (K : ℝ≥0) (Φ : SSphere 2 → ℝ → SSphere 2),
    Continuous F ∧ LipschitzOnWith K F (Sphere 2) ∧
    (∀ x : SSphere 2, ‖F x‖ ≤ (1 : ℝ≥0)) ∧
    (∀ x t, HasDerivAt (fun s => (Φ x s : EucSpace 2)) (F (Φ x t)) t) ∧
    (∀ x, Φ x 0 = x) ∧ (0 : ℝ) ≤ 1 := by
  let a : EucSpace 2 := basePoint 1
  let F (x : EucSpace 2) := proj 2 x a
  let Φ := ridgeFlow a 0 1
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  refine ⟨F, K, Φ, hF.continuous, hK.mono hsphere, ?_, ?_, ridgeFlow_zero _ _ _, zero_le_one⟩
  · intro x
    exact (norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) a).trans
      (mem_sphere_zero_iff_norm.mp (basePoint 1).2).le
  · intro x t
    simpa [F, Φ] using hasDerivAt_ridgeFlow a 0 1 x t

end Transformer.Interpolation
