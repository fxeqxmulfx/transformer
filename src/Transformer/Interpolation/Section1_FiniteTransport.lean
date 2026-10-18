/-
# Finite smooth transport comparisons for arbitrary original weak laws

Auxiliary to arXiv:2411.04551v3, §1, the Lipschitz well-posedness
argument for `eq: cauchy.pb`, and the characteristic tests in §3.1.

Attach a terminal scalar test value to each actual particle trajectory.
The normalized smooth minimum of their backward distance cones is an
ambient `C¹` test with nonpositive material derivative. Its integral
therefore decreases along every actual weak solution, without any
finite-support condition on that solution.

At the terminal time it dominates the chosen Lipschitz scalar test.
Each initial entry gives an explicit upper bound, retaining both the
regularized-distance error and the finite-minimum approximation error.
These estimates allow the finite centers to approach an arbitrary law.
-/

import Transformer.Interpolation.Section1_TransportCones
import Transformer.Interpolation.Section1_SoftMinimum

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ} {ι : Type*} [Fintype ι] [Nonempty ι]

/-- A finite smooth minimum of cones following actual particle paths.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
noncomputable def finiteTransportTest (K L ε β T : ℝ) (Y : ι → ℝ → SSphere d)
    (f : SSphere d → ℝ) (q : ℝ × EucSpace d) : ℝ :=
  finiteSoftMin β (fun i => transportCone K L ε (f (Y i T)) T (Y i) q)

/-- Time regularity of the actual paths makes this an admissible test.
The terminal scalar test itself needs no differentiability.
Auxiliary for arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem contDiff_finiteTransportTest (K L ε β T : ℝ) (hε : 0 < ε)
    (Y : ι → ℝ → SSphere d) (hY : ∀ i, ContDiff ℝ 1 (fun s => (Y i s : EucSpace d)))
    (f : SSphere d → ℝ) : ContDiff ℝ 1 (finiteTransportTest K L ε β T Y f) :=
  contDiff_finiteSoftMin β _ fun i => contDiff_transportCone K L ε _ T hε (Y i) (hY i)

/-- The actual finite transport test has nonpositive material derivative
using only the current field and the current derivatives of its C1 centers.
Auxiliary for arXiv:2411.04551v3, §1, the weak comparison argument. -/
theorem fderiv_finiteTransportTest_nonpos_at (F : EucSpace d → EucSpace d)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (L ε β T : ℝ)
    (hL : 0 ≤ L) (hε : 0 < ε) (hβ : 0 < β) (Y : ι → ℝ → SSphere d)
    (hY : ∀ i, ContDiff ℝ 1 (fun r => (Y i r : EucSpace d)))
    (f : SSphere d → ℝ) (t : ℝ)
    (hYt : ∀ i, HasDerivAt (fun r => (Y i r : EucSpace d)) (F (Y i t)) t) (x : SSphere d) :
    fderiv ℝ (finiteTransportTest K L ε β T Y f) (t, (x : EucSpace d)) (1, F x) ≤ 0 := by
  apply fderiv_finiteSoftMin_nonpos β hβ
    (fun i => transportCone K L ε (f (Y i T)) T (Y i))
    (fun i => contDiff_transportCone _ _ _ _ _ hε _ (hY i))
  exact fun i => fderiv_transportCone_nonpos_at F K hK L ε _ T hL hε (Y i) (hY i) t (hYt i) x


/-- The pointwise finite comparison applies along the original complete
autonomous particle paths.
Auxiliary for arXiv:2411.04551v3, §1, the weak comparison argument. -/
theorem fderiv_finiteTransportTest_nonpos (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (L ε β T : ℝ)
    (hL : 0 ≤ L) (hε : 0 < ε) (hβ : 0 < β) (Y : ι → ℝ → SSphere d)
    (hY : ∀ i s, HasDerivAt (fun r => (Y i r : EucSpace d)) (F (Y i s)) s)
    (f : SSphere d → ℝ) (t : ℝ) (x : SSphere d) :
    fderiv ℝ (finiteTransportTest K L ε β T Y f) (t, (x : EucSpace d)) (1, F x) ≤ 0 :=
  fderiv_finiteTransportTest_nonpos_at F K hK L ε β T hL hε hβ Y
    (fun i => contDiff_spherical_particle F hF _ (hY i)) f t (fun i => hY i t) x

/-- At the terminal time the finite comparison dominates the original
Lipschitz scalar test. Auxiliary for arXiv:2411.04551v3, §1,
the characteristic proof of weak well-posedness. -/
theorem le_finiteTransportTest_terminal (K : ℝ) (L : ℝ≥0) (ε β T : ℝ) (hβ : 0 < β)
    (Y : ι → ℝ → SSphere d) (f : SSphere d → ℝ) (hf : LipschitzWith L f) (x : SSphere d) :
    f x ≤ finiteTransportTest K L ε β T Y f (T, (x : EucSpace d)) := by
  apply le_finiteSoftMin β hβ _ (f x)
  intro i
  have hd := hf.dist_le_mul x (Y i T)
  rw [Real.dist_eq, Subtype.dist_eq, dist_eq_norm] at hd
  have hh : f x ≤ f (Y i T) + (L : ℝ) * ‖(x : EucSpace d) - (Y i T : EucSpace d)‖ := by
    have h := (le_abs_self (f x - f (Y i T))).trans hd
    linarith
  have hc := transportCone_lower K (L : ℝ) ε (f (Y i T)) T L.coe_nonneg (Y i) (T, (x : EucSpace d))
  simp only [sub_self, mul_zero, exp_zero, mul_one] at hc
  exact hh.trans hc

omit [Nonempty ι] in
/-- Any actual transported center gives the explicit upper estimate.
Auxiliary for arXiv:2411.04551v3, §1, the weak comparison argument. -/
theorem finiteTransportTest_le_entry (K L ε β T : ℝ) (hL : 0 ≤ L) (hε : 0 ≤ ε) (hβ : 0 < β)
    (Y : ι → ℝ → SSphere d) (f : SSphere d → ℝ) (q : ℝ × EucSpace d) (i : ι) :
    finiteTransportTest K L ε β T Y f q ≤ f (Y i T) +
      L * exp (K * (T - q.1)) * (‖q.2 - (Y i q.1 : EucSpace d)‖ + ε) + log (Fintype.card ι) / β := by
  exact (finiteSoftMin_le β hβ _ i).trans
    (add_le_add (transportCone_upper K L ε (f (Y i T)) T hL hε (Y i) q) le_rfl)

/-- The comparison applies to every original weak solution, including
arbitrary initial probabilities. Auxiliary for arXiv:2411.04551v3,
§1, the Lipschitz well-posedness argument for `eq: cauchy.pb`. -/
theorem integral_finiteTransportTest_le (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (L ε β T : ℝ)
    (hL : 0 ≤ L) (hε : 0 < ε) (hβ : 0 < β) (Y : ι → ℝ → SSphere d)
    (hY : ∀ i s, HasDerivAt (fun r => (Y i r : EucSpace d)) (F (Y i s)) s)
    (f : SSphere d → ℝ) (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ)
    (s t : ℝ) (hst : s ≤ t) :
    (∫ x : SSphere d, finiteTransportTest K L ε β T Y f (t, (x : EucSpace d)) ∂(μ t : Measure (SSphere d))) ≤
      ∫ x : SSphere d, finiteTransportTest K L ε β T Y f (s, (x : EucSpace d)) ∂(μ s : Measure (SSphere d)) := by
  exact cauchyPB_spacetime_test_le θ F hF hfield μ hμ
    (fun r x => finiteTransportTest K L ε β T Y f (r, x))
    (contDiff_finiteTransportTest _ _ _ _ _ hε Y (fun i => contDiff_spherical_particle F hF _ (hY i)) f)
    (fun r x => fderiv_finiteTransportTest_nonpos F hF K hK L ε β T hL hε hβ Y hY f r x) s t hst

/-- An arbitrary original weak law is bounded by the initial comparison
integral for every terminal Lipschitz test. Auxiliary for arXiv:2411.04551v3,
§1, the characteristic proof of well-posedness of `eq: cauchy.pb`. -/
theorem integral_test_le_finiteTransport_initial (θ : TimeParams d)
    (F : EucSpace d → EucSpace d) (hF : Continuous F)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d), fullVF d θ ρ s x = F x)
    (K : ℝ≥0) (hK : LipschitzOnWith K F (Sphere d)) (L : ℝ≥0) (ε β T : ℝ)
    (hε : 0 < ε) (hβ : 0 < β) (hT : 0 ≤ T) (Y : ι → ℝ → SSphere d)
    (hY : ∀ i s, HasDerivAt (fun r => (Y i r : EucSpace d)) (F (Y i s)) s)
    (f : SSphere d → ℝ) (hf : LipschitzWith L f)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    (∫ x : SSphere d, f x ∂(μ T : Measure (SSphere d))) ≤
      ∫ x : SSphere d, finiteTransportTest K L ε β T Y f (0, (x : EucSpace d)) ∂(μ 0 : Measure (SSphere d)) := by
  have hc := (contDiff_finiteTransportTest K L ε β T hε Y
    (fun i => contDiff_spherical_particle F hF _ (hY i)) f).continuous
  have hi := integral_mono_ae (integrable_spherical_continuous (μ T) f hf.continuous)
    (integrable_spherical_continuous (μ T) _ (hc.comp (continuous_const.prodMk continuous_subtype_val)))
    (ae_of_all _ fun x => le_finiteTransportTest_terminal K L ε β T hβ Y f hf x)
  exact hi.trans (integral_finiteTransportTest_le θ F hF hfield K hK L ε β T L.coe_nonneg hε hβ Y hY f μ hμ 0 T hT)

/-- A real projection neuron, two actual particle paths, a nonzero weak
law, and the nonsmooth distance test jointly realize all hypotheses.
The comparison does not require a differentiable terminal test. -/
example : ∃ (θ : TimeParams 2) (F : EucSpace 2 → EucSpace 2) (K : ℝ≥0)
    (Y : Fin 2 → ℝ → SSphere 2) (f : SSphere 2 → ℝ) (μ : ℝ → Perspective.ProbSphere 2),
    Continuous F ∧ LipschitzOnWith K F (Sphere 2) ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ) (x : SSphere 2), fullVF 2 θ ρ s x = F x) ∧
    (∀ i, ContDiff ℝ 1 (fun s => (Y i s : EucSpace 2))) ∧
    (∀ i s, HasDerivAt (fun r => (Y i r : EucSpace 2)) (F (Y i s)) s) ∧
    LipschitzWith 1 f ∧ cauchyPB 2 θ μ ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 := by
  let a : EucSpace 2 := basePoint 1
  let θ : TimeParams 2 := fun _ => reluRidgeParams (0 : Fin 2) a 0 1
  let F (x : EucSpace 2) := proj 2 x a
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let Y (i : Fin 2) := ridgeFlow a 0 1 (if i = 0 then z else antipode 2 z)
  let f (x : SSphere 2) := dist x (basePoint 1)
  let μ := ridgeLaw a 0 1 (Perspective.diracProb 2 z)
  have hF : ContDiff ℝ 1 F := contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  obtain ⟨K, hK⟩ := Landscape.GlobalFlow.exists_lipschitzOnWith_closedBall hF.locallyLipschitz 1
  have hsphere : Sphere 2 ⊆ Metric.closedBall 0 1 := fun x hx =>
    mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp hx).le
  have hY (i : Fin 2) (t : ℝ) : HasDerivAt (fun s => (Y i s : EucSpace 2)) (F (Y i t)) t := by
    simpa [F, Y] using hasDerivAt_ridgeFlow a 0 1 (if i = 0 then z else antipode 2 z) t
  refine ⟨θ, F, K, Y, f, μ, hF.continuous, hK.mono hsphere, ?_,
    fun i => contDiff_spherical_particle F hF.continuous _ (hY i), hY,
    LipschitzWith.dist_left _, cauchyPB_ridgeLaw _ _ _ _ _, zero_le_one, zero_lt_one⟩
  intro ρ s x
  rw [fullVF_reluRidgeParams]
  simp [F]

end Transformer.Interpolation
