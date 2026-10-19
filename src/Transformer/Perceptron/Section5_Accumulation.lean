module

public import Transformer.Perceptron.Section5_CriticalSupport
public import Transformer.Perceptron.Section5_CircleGeometry
public import Mathlib.Topology.ClusterPt
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# Critical support accumulation and local curvature control

ArXiv:2601.21366v2, §5.2, the accumulation-point argument in
the proof of `thm: circle.gelu`. At an accumulation point of
zeros, an actual first derivative must be zero. This local fact
requires differentiability at the point, before any analytic
continuation or global cancellation is considered.

The circle chart is injective on a short interval around any
representative. Accumulating support angles therefore supply two
distinct actual sphere points in every open neighborhood. Continuity
then makes a vanishing curvature coefficient and the oscillation
of the actual pair kernel small on the same neighborhood.

The accumulation hypotheses are realized on the full real line.
The local estimates have a joint constant-zero example on the
actual circle, with the positive tolerance fixed to one.
-/

@[expose] public section

open Real Set Filter Metric
open scoped Topology

namespace Transformer.Perceptron

/-- Every real angle is an accumulation point of the full angle line.
Source: arXiv:2601.21366v2, §5.2, the real angular coordinates
used in the proof of `thm: circle.gelu`. -/
theorem accPt_univ_real (p : ℝ) : AccPt p (principal (univ : Set ℝ)) := by
  apply accPt_principal_iff_clusterPt.mpr
  apply clusterPt_principal_iff.mpr
  intro U hU
  obtain ⟨r, hr, hball⟩ := Metric.mem_nhds_iff.mp hU
  refine ⟨p + r / 2, hball ?_, mem_univ _, ?_⟩
  · rw [Metric.mem_ball, Real.dist_eq,
      show p + r / 2 - p = r / 2 by ring, abs_of_pos (by positivity)]
    linarith
  · simp only [mem_singleton_iff]
    linarith

/-- Every open neighborhood of an accumulation point contains another
point of the accumulating set. Source: arXiv:2601.21366v2, §5.2,
choosing nearby support angles in the proof of `thm: circle.gelu`. -/
theorem exists_ne_of_accPt_open (S : Set ℝ) (p : ℝ) (hp : AccPt p (principal S))
    (U : Set ℝ) (hU : IsOpen U) (hpU : p ∈ U) :
    ∃ q ∈ S, q ∈ U ∧ q ≠ p := by
  obtain ⟨q, hqU, hqS, hqne⟩ :=
    (clusterPt_principal_iff.mp (accPt_principal_iff_clusterPt.mp hp)) U (hU.mem_nhds hpU)
  exact ⟨q, hqS, hqU, by simpa only [mem_singleton_iff] using hqne⟩

/-- At an accumulation point of critical angles the actual force
derivative is zero. Source: arXiv:2601.21366v2, §5.2,
the local consequence of the support-zero condition in the proof
of `thm: circle.gelu`. Global analyticity is not required for this fact. -/
theorem derivative_zero_at_accPt_of_zeros (g : ℝ → ℝ) (S : Set ℝ) (p d : ℝ)
    (hp : AccPt p (principal S)) (hzero : ∀ q ∈ S, g q = 0)
    (hd : HasDerivAt g d p) : d = 0 := by
  by_contra hd0
  obtain ⟨U, hU, hpU, huniq⟩ := zero_isolated_of_hasDerivAt g p d hd hd0
  obtain ⟨q, hqS, hqU, hqp⟩ := exists_ne_of_accPt_open S p hp U hU hpU
  exact hqp (huniq q hqU (hzero q hqS))

/-- The actual chart is injective on a fixed short interval about every
angle. Source: arXiv:2601.21366v2, §5.2, the small angular neighborhoods
in the proof of `thm: circle.gelu`. Periodic duplicate representatives
cannot occur inside this interval of length `1/2`. -/
theorem circlePoint_injOn_short_interval (p : ℝ) :
    InjOn circlePoint (Ioo (p - 1 / 4) (p + 1 / 4)) := by
  intro θ hθ φ hφ heq
  by_contra hne
  have hsmall : |θ - φ| < (1 / 2 : ℝ) := abs_lt.mpr
    ⟨by linarith [hθ.1, hφ.2], by linarith [hθ.2, hφ.1]⟩
  have hpos : 0 < |θ - φ| := abs_pos.mpr (sub_ne_zero.mpr hne)
  have hπ : |θ - φ| ≤ π := by linarith [Real.pi_gt_three]
  have hcos := Real.cos_lt_cos_of_nonneg_of_le_pi (show (0 : ℝ) ≤ 0 from le_rfl) hπ hpos
  rw [Real.cos_zero, Real.cos_abs] at hcos
  have hcosone : Real.cos (θ - φ) = 1 := by
    rw [← inner_circlePoint_circlePoint, heq, real_inner_self_eq_norm_sq,
      mem_sphere_zero_iff_norm.mp (circlePoint φ).2, one_pow]
  linarith

/-- Accumulating angles give two distinct actual circle points in every
open neighborhood of their limit. Source: arXiv:2601.21366v2, §5.2,
choosing the two localized directions in the proof of `thm: circle.gelu`. -/
theorem exists_two_circlePoints_of_accPt (S : Set ℝ) (p : ℝ)
    (hp : AccPt p (principal S)) (U : Set (SSphere 2)) (hU : IsOpen U)
    (hpU : circlePoint p ∈ U) :
    ∃ θ φ : ℝ, θ ∈ S ∧ φ ∈ S ∧ circlePoint θ ≠ circlePoint φ ∧
      circlePoint θ ∈ U ∧ circlePoint φ ∈ U := by
  let V : Set ℝ := Ioo (p - 1 / 4) (p + 1 / 4) ∩ circlePoint ⁻¹' U
  have hV : IsOpen V := isOpen_Ioo.inter (hU.preimage continuous_circlePoint)
  have hpV : p ∈ V := ⟨⟨by linarith, by linarith⟩, hpU⟩
  obtain ⟨θ, hθS, hθV, hθp⟩ := exists_ne_of_accPt_open S p hp V hV hpV
  have hV' : IsOpen (V ∩ ({θ}ᶜ : Set ℝ)) := hV.inter isOpen_compl_singleton
  have hpV' : p ∈ V ∩ ({θ}ᶜ : Set ℝ) :=
    ⟨hpV, by simpa only [mem_compl_iff, mem_singleton_iff] using hθp.symm⟩
  obtain ⟨φ, hφS, hφV, -⟩ := exists_ne_of_accPt_open S p hp _ hV' hpV'
  refine ⟨θ, φ, hθS, hφS, ?_, hθV.2, hφV.1.2⟩
  intro heq
  have hθφ := circlePoint_injOn_short_interval p hθV.1 hφV.1.1 heq
  exact hφV.2 (mem_singleton_iff.mpr hθφ.symm)

/-- Continuity gives one open neighborhood with small curvature and
small pair-kernel oscillation. Source: arXiv:2601.21366v2, §5.2,
the local kernel-modulus estimate in the proof of `thm: circle.gelu`.
The curvature need only vanish at the chosen limit point. -/
theorem exists_open_local_curvature_bounds (A : SSphere 2 → ℝ) (hA : Continuous A)
    (K : SSphere 2 × SSphere 2 → ℝ) (hK : Continuous K)
    (p : SSphere 2) (hAp : A p = 0) (ε : ℝ) (hε : 0 < ε) :
    ∃ U : Set (SSphere 2), IsOpen U ∧ p ∈ U ∧ (∀ x ∈ U, A x ≤ ε) ∧
      ∀ x ∈ U, ∀ y ∈ U, |K (x, y) - K (p, p)| ≤ ε := by
  let T : Set (SSphere 2 × SSphere 2) := {q | |K q - K (p, p)| < ε}
  have hT : IsOpen T := isOpen_lt (hK.sub continuous_const).abs continuous_const
  have hpT : (p, p) ∈ T := by
    change |K (p, p) - K (p, p)| < ε
    simpa only [sub_self, abs_zero] using hε
  obtain ⟨V, W, hV, hW, hpV, hpW, hVW⟩ := isOpen_prod_iff.mp hT p p hpT
  refine ⟨{x | A x < ε} ∩ (V ∩ W),
    (isOpen_lt hA continuous_const).inter (hV.inter hW), ?_, ?_, ?_⟩
  · refine ⟨?_, hpV, hpW⟩
    change A p < ε
    rwa [hAp]
  · intro x hx
    exact hx.1.le
  · intro x hx y hy
    exact (hVW ⟨hx.2.1, hy.2.2⟩).le

/-- All accumulation, zero-set, actual-derivative and neighborhood
inputs are jointly realized by the full real line, constant zero
force and its actual zero derivative at angle zero. -/
example : AccPt (0 : ℝ) (principal (univ : Set ℝ)) ∧
    (∀ q ∈ (univ : Set ℝ), (0 : ℝ → ℝ) q = 0) ∧
    HasDerivAt (0 : ℝ → ℝ) 0 0 ∧ IsOpen (univ : Set ℝ) ∧
    (0 : ℝ) ∈ (univ : Set ℝ) ∧ IsOpen (univ : Set (SSphere 2)) ∧
    circlePoint 0 ∈ (univ : Set (SSphere 2)) :=
  ⟨accPt_univ_real 0, fun _ _ => rfl, hasDerivAt_const 0 0,
    isOpen_univ, mem_univ _, isOpen_univ, mem_univ _⟩

/-- All local-bound hypotheses hold for zero coefficient and kernel
at an actual circle point and positive bound one. -/
example : Continuous (0 : SSphere 2 → ℝ) ∧
    Continuous (0 : SSphere 2 × SSphere 2 → ℝ) ∧
    (0 : SSphere 2 → ℝ) (circlePoint 0) = 0 ∧ (0 : ℝ) < 1 :=
  ⟨continuous_const, continuous_const, rfl, one_pos⟩

end Transformer.Perceptron
