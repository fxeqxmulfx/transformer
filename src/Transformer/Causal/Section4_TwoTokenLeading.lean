module

public import Transformer.Causal.Section4_SubspaceCovectors
public import Transformer.Causal.Section4_ScalarLimits
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Exact leading-component equations for two original causal tokens

ArXiv:2411.04990v2, §4, `thm2`, restricted to two tokens.
The leading covectors eliminate the value matrix from the leading
components, while retaining the actual nonlinear radial coefficient.
A covector orthogonal to the leader's initial value remains orthogonal
to that leader. Its actual follower component solves a homogeneous
scalar equation and preserves a strictly positive initial sign.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Causal

variable {d n : ℕ}

/-- The exact integrating-factor identity preserves the initial sign of
a homogeneous scalar component. Source: arXiv:2411.04990v2, §§3–4,
the linear representation in `lemma1` and the leading splitting of `thm2`. -/
theorem scalar_eq_initial_mul_exp_integral (h r : ℝ → ℝ) (hh : Continuous h)
    (hr : ∀ t, HasDerivAt r (h t * r t) t) (t : ℝ) :
    r t = r 0 * Real.exp (∫ s in (0 : ℝ)..t, h s) := by
  let H : ℝ → ℝ := fun u => ∫ s in (0 : ℝ)..u, h s
  have hH (u : ℝ) : HasDerivAt H (h u) u :=
    intervalIntegral.integral_hasDerivAt_right (hh.intervalIntegrable _ _)
      (hh.stronglyMeasurableAtFilter _ _) hh.continuousAt
  have hd (u : ℝ) : HasDerivAt (fun s => r s * Real.exp (-H s)) 0 u := by
    convert (hr u).mul ((hH u).neg.exp) using 1
    ring
  have hc := is_const_of_deriv_eq_zero (fun u => (hd u).differentiableAt)
    (fun u => (hd u).deriv) t 0
  have he : r t * Real.exp (-H t) = r 0 := by
    simpa only [H, intervalIntegral.integral_same, neg_zero, Real.exp_zero, mul_one] using hc
  have hm := congrArg (· * Real.exp (H t)) he
  simpa only [mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, mul_one] using hm

/-- A positive initial homogeneous component stays strictly positive at
every real time. Source: arXiv:2411.04990v2, §4, `thm2`, the two-token leading components. -/
theorem scalar_pos_of_homogeneous (h r : ℝ → ℝ) (hh : Continuous h)
    (hr : ∀ t, HasDerivAt r (h t * r t) t) (h0 : 0 < r 0) (t : ℝ) : 0 < r t := by
  rw [scalar_eq_initial_mul_exp_integral h r hh hr t]
  exact mul_pos h0 (Real.exp_pos _)

/-- The actual radial coefficient removed by the original tangent projection.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, `thm2`. -/
noncomputable def csaRadialCoefficient (β : ℝ) (Q K V : ParamMatrix d)
    (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) : ℝ :=
  inner ℝ (X t i : EucSpace d)
    (csaAverage β Q K V (fun j => (X t j : EucSpace d)) i)

/-- The shared scalar coefficient of the actual follower's leading components.
Source: arXiv:2411.04990v2, §4, `thm2`, the actual second row of `eqn:csa`. -/
noncomputable def csaLeadingCoefficient (β : ℝ) (Q K V : ParamMatrix d) (lam : ℝ)
    (X : ℝ → SphereTuple d 2) (t : ℝ) : ℝ :=
  lam * csaTokenWeight β Q K (fun j => (X t j : EucSpace d)) 1 1 -
    csaRadialCoefficient β Q K V X t 1

/-- The original follower's shared leading coefficient is continuous along
every actual flow. Source: arXiv:2411.04990v2, §4, the smooth causal field in `thm2`. -/
theorem continuous_csaLeadingCoefficient (β : ℝ) (Q K V : ParamMatrix d) (lam : ℝ)
    (X : ℝ → SphereTuple d 2) (hX : CSA d 2 β Q K V X) :
    Continuous (csaLeadingCoefficient β Q K V lam X) := by
  have hx (i : Idx 2) : Continuous (fun t => (X t i : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun t =>
      ((CSA_iff_csaAmbientField β Q K V X).mp hX t i).continuousAt
  have hall : Continuous (fun t => fun i => (X t i : EucSpace d)) := continuous_pi hx
  exact (continuous_const.mul ((contDiff_csaTokenWeight β Q K 1 1).continuous.comp hall)).sub
    ((hx 1).inner ((contDiff_csaAverage β Q K V 1).continuous.comp hall))

/-- The leader's leading scalar component times its explicit positive
radial scale is exactly its initial component.
Source: arXiv:2411.04990v2, §§3–4, `lemma1` and `thm2`. -/
theorem csaLeader_leading_component_scale (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (t : ℝ) :
    singleTokenScale V lam (fun s => (X s 0 : EucSpace d)) t *
      inner ℝ v (X t 0 : EucSpace d) = inner ℝ v (X 0 0 : EucSpace d) := by
  have h := congrArg (fun z => inner ℝ v z)
    (starProjection_singleTokenLift V L lam heig hinv (fun s => (X s 0 : EucSpace d))
      (csaLeader_singleTokenODE β Q K V X hX) t)
  simpa only [← L.inner_starProjection_left_eq_right,
    L.starProjection_eq_self_iff.mpr hv, singleTokenLift, real_inner_smul_right] using h

/-- A leading covector initially orthogonal to the leader remains exactly
orthogonal to it. Source: arXiv:2411.04990v2, §4, `thm2`, using §3, `lemma1`. -/
theorem csaLeader_leading_orthogonal (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X)
    (hzero : inner ℝ v (X 0 0 : EucSpace d) = 0) (t : ℝ) :
    inner ℝ v (X t 0 : EucSpace d) = 0 := by
  have h := csaLeader_leading_component_scale β Q K V L lam heig hinv v hv X hX t
  rw [hzero] at h
  exact (mul_eq_zero.mp h).resolve_left (singleTokenScale_pos V lam _ t).ne'

/-- The actual second leading component has the original positive leader
forcing and the shared original radial coefficient.
Source: arXiv:2411.04990v2, §4, `thm2`, specialized to two tokens. -/
theorem hasDerivAt_csaSecond_leading (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L)
    (X : ℝ → SphereTuple d 2) (hX : CSA d 2 β Q K V X) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ v (X s 1 : EucSpace d))
      (lam * csaTokenWeight β Q K (fun j => (X t j : EucSpace d)) 1 0 *
          inner ℝ v (X t 0 : EucSpace d) +
        csaLeadingCoefficient β Q K V lam X t * inner ℝ v (X t 1 : EucSpace d)) t := by
  have h := hasDerivAt_csaLeadingOverlap β Q K V L lam heig hinv v hv X hX 1 t
  apply h.congr_deriv
  simp only [Fin.sum_univ_two, csaLeadingCoefficient, csaRadialCoefficient]
  ring

/-- A positive follower component along a leading direction orthogonal to
the leader stays positive at every real time. The homogeneous equation is
derived from the actual original row, not assumed.
Source: arXiv:2411.04990v2, §4, `thm2`, two-token case. -/
theorem csaSecond_pos_transverse_leading (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L)
    (X : ℝ → SphereTuple d 2) (hX : CSA d 2 β Q K V X)
    (hzero : inner ℝ v (X 0 0 : EucSpace d) = 0)
    (hpos : 0 < inner ℝ v (X 0 1 : EucSpace d)) (t : ℝ) :
    0 < inner ℝ v (X t 1 : EucSpace d) := by
  apply scalar_pos_of_homogeneous (csaLeadingCoefficient β Q K V lam X)
    (fun s => inner ℝ v (X s 1 : EucSpace d))
    (continuous_csaLeadingCoefficient β Q K V lam X hX) ?_ hpos t
  intro s
  simpa only [csaLeader_leading_orthogonal β Q K V L lam heig hinv v hv X hX hzero s,
    mul_zero, zero_add] using hasDerivAt_csaSecond_leading β Q K V L lam heig hinv v hv X hX s

/-- The genuine identity-value two-token flow from two orthogonal unit
vectors realizes the transverse sign assumptions and all actual-flow
hypotheses. Source: arXiv:2411.04990v2, §4, `thm2`. -/
example : ∃ (v : EucSpace 2) (X : ℝ → SphereTuple 2 2),
    (∀ w ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) w = (1 : ℝ) • w) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    v ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    CSA 2 2 1 0 0 (ContinuousLinearMap.id ℝ _) X ∧
    inner ℝ v (X 0 0 : EucSpace 2) = 0 ∧ 0 < inner ℝ v (X 0 1 : EucSpace 2) := by
  let u : SSphere 2 := basePoint 1
  let v : SSphere 2 := ⟨EuclideanSpace.single (1 : Fin 2) 1, by simp⟩
  let X₀ : SphereTuple 2 2 := ![u, v]
  let X := csaFlow 1 0 0 (ContinuousLinearMap.id ℝ _) X₀
  refine ⟨v, X, fun w _ => by simp, fun z hz => hz,
    Submodule.mem_top, (csaFlow_spec _ _ _ _ _).2, ?_, ?_⟩
  · simp [X, csaFlow_zero, X₀, u, v, basePoint, EuclideanSpace.inner_single_left]
  · simp [X, csaFlow_zero, X₀, v]

end Transformer.Causal
