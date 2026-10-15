/-
# Almost-everywhere convergence of the original two-token causal system

ArXiv:2411.04990v2, §4, proof of `thm1`, base case `k=2`.
The exact original overlap estimate gives actual spherical convergence
for every original solution whose second initial token is not the
antipode of the first. The leader is fixed, so both original tokens
converge to the original first initial token.

For the original uniform product initialization the antipodal relation
is null. The law of two distinct original coordinates is the product
of their original common marginal; each antipodal fiber is a singleton
of zero marginal mass in dimension at least two. This establishes the
actual almost-everywhere two-token assertion, with its original
universal quantifier over all solutions and arbitrary query and key
matrices. The estimates also allow every real temperature.
-/

import Transformer.Causal.Section4_SecondOverlap
import Transformer.Causal.Section4_TwoClusterRefutation
import Transformer.Perspective.UniformAtomless

open Real MeasureTheory Filter Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- Vanishing actual spherical overlap defect implies actual spherical convergence.
Source: arXiv:2411.04990v2, §4, the overlap argument concluding the proof of `thm1`. -/
theorem csaSphere_tendsto_of_overlap (x : ℝ → SSphere d) (a : SSphere d)
    (h : Tendsto (fun t => 1 - inner ℝ (x t : EucSpace d) (a : EucSpace d)) atTop (𝓝 0)) :
    Tendsto x atTop (𝓝 a) := by
  apply tendsto_subtype_rng.mpr
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  have hs : Tendsto (fun t => ‖(x t : EucSpace d) - (a : EucSpace d)‖ ^ 2) atTop (𝓝 0) := by
    simpa only [csaSphere_dist_sq, mul_zero] using h.const_mul 2
  have hn := (Real.continuous_sqrt.tendsto 0).comp hs
  simpa only [Function.comp_def, Real.sqrt_zero, Real.sqrt_sq_eq_abs,
    abs_of_nonneg (norm_nonneg _)] using hn

/-- Every actual original second token converges to the leader outside its original initial antipode.
Source: arXiv:2411.04990v2, §4, the base case `k=2` of `thm1`. -/
theorem csaSecond_tendsto (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d 2)
    (hX : CSA d 2 β Q K (ContinuousLinearMap.id ℝ _) X)
    (hanti : X 0 1 ≠ Perspective.antipode d (X 0 0)) :
    Tendsto (fun t => X t 1) atTop (𝓝 (X 0 0)) := by
  let lam := csaWeightFloor β Q K 2 * (1 + inner ℝ (X 0 1 : EucSpace d) (X 0 0 : EucSpace d))
  have hlam : 0 < lam := csaSecondOverlap_rate_pos β Q K (X 0) hanti
  have he : Tendsto (fun t : ℝ => Real.exp (-lam * t)) atTop (𝓝 0) := by
    simpa only [Function.comp_def, id_eq, neg_mul] using Real.tendsto_exp_neg_atTop_nhds_zero.comp
      (Filter.Tendsto.const_mul_atTop hlam (tendsto_id : Tendsto (fun t : ℝ => t) atTop atTop))
  refine csaSphere_tendsto_of_overlap (fun t => X t 1) (X 0 0) ?_
  refine squeeze_zero' (g := fun t =>
    (1 - inner ℝ (X 0 1 : EucSpace d) (X 0 0 : EucSpace d)) * Real.exp (-lam * t))
    (Eventually.of_forall (fun t => sub_nonneg.mpr
    (csaSphereOverlap_bounds (X t 1) (X 0 0)).2)) ?_ ?_
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    exact csaSecondOverlap_decay β Q K X hX t ht
  · simpa only [mul_zero] using
      he.const_mul (1 - inner ℝ (X 0 1 : EucSpace d) (X 0 0 : EucSpace d))

/-- Both original tokens of every actual solution converge to the original leader off the initial antipode.
Source: arXiv:2411.04990v2, §4, the complete two-token base case of `thm1`. -/
theorem CSA_twoToken_tendsto (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d 2)
    (hX : CSA d 2 β Q K (ContinuousLinearMap.id ℝ _) X)
    (hanti : X 0 1 ≠ Perspective.antipode d (X 0 0)) (k : Idx 2) :
    Tendsto (fun t => X t k) atTop (𝓝 (X 0 0)) := by
  fin_cases k
  · exact (tendsto_const_nhds : Tendsto (fun _ : ℝ => X 0 0) atTop (𝓝 (X 0 0))).congr'
      (Eventually.of_forall (fun t => (csaLeader_constant β Q K X hX t).symm))
  · exact csaSecond_tendsto β Q K X hX hanti

/-- The actual constructed original two-token flow has the same convergence outside the initial antipode.
Source: arXiv:2411.04990v2, §4, the two-token flow in `thm1`. -/
theorem csaFlow_twoToken_tendsto (β : ℝ) (Q K : ParamMatrix d) (X₀ : SphereTuple d 2)
    (hanti : X₀ 1 ≠ Perspective.antipode d (X₀ 0)) (k : Idx 2) :
    Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t k) atTop (𝓝 (X₀ 0)) := by
  have h0 := csaFlow_zero β Q K (ContinuousLinearMap.id ℝ _) X₀
  simpa only [h0] using CSA_twoToken_tendsto β Q K _ (csaFlow_spec β Q K _ X₀).2
    (by simpa only [h0] using hanti) k

/-- The actual spherical antipode used in the original exceptional set is continuous.
Source: arXiv:2411.04990v2, §4, the antipodal limiting configurations in `thm1`. -/
theorem continuous_csaAntipode : Continuous (Perspective.antipode d) :=
  continuous_induced_rng.mpr continuous_subtype_val.neg

/-- The original uniform tuple law assigns zero mass to an antipodal pair of distinct original coordinates.
Source: arXiv:2411.04990v2, §4, the exceptional two-token set in `thm1`.
This uses the original rotation-invariant product law, rather than assuming atomlessness. -/
theorem measure_csaAntipodalPair_eq_zero (hd : 2 ≤ d) (P : Measure (SphereTuple d n))
    (hP : Perspective.UniformTuple d n P) {i j : Idx n} (hij : i ≠ j) :
    P {X | X j = Perspective.antipode d (X i)} = 0 := by
  obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
  have := hσprob
  let B : Set (SSphere d × SSphere d) := {p | p.2 = Perspective.antipode d p.1}
  have hB : MeasurableSet B := measurableSet_eq_fun measurable_snd
    (continuous_csaAntipode.measurable.comp measurable_fst)
  have hm : Measurable (fun X : SphereTuple d n => (X i, X j)) :=
    (measurable_pi_apply i).prodMk (measurable_pi_apply j)
  have hp : {X : SphereTuple d n | X j = Perspective.antipode d (X i)} =
      (fun X => (X i, X j)) ⁻¹' B := rfl
  rw [hp, ← Measure.map_apply hm hB, Perspective.map_pair_pi d n σ hij, Measure.prod_apply hB]
  have hf : ∀ x : SSphere d, σ (Prod.mk x ⁻¹' B) = 0 := by
    intro x
    have hpre : Prod.mk x ⁻¹' B = {Perspective.antipode d x} := by ext y; rfl
    rw [hpre]
    exact Perspective.measure_singleton_eq_zero_of_invariant d hd σ hσinv _
  simp only [hf, lintegral_zero]

/-- Under the original actual uniform initialization, the full two-token convergence assertion holds almost surely.
Source: arXiv:2411.04990v2, §4, `thm1`, base case `k=2`; arbitrary original matrices are retained. -/
theorem single_cluster_twoToken (hd : 2 ≤ d) (β : ℝ) (Q K : ParamMatrix d)
    (P : Measure (SphereTuple d 2)) (hP : Perspective.UniformTuple d 2 P) :
    ∀ᵐ X₀ ∂P, ∀ X : ℝ → SphereTuple d 2, X 0 = X₀ →
      CSA d 2 β Q K (ContinuousLinearMap.id ℝ _) X →
        ∀ k, Tendsto (fun t => X t k) atTop (𝓝 (X₀ 0)) := by
  have ha : ∀ᵐ X₀ ∂P, X₀ 1 ≠ Perspective.antipode d (X₀ 0) := by
    rw [ae_iff]
    simpa only [not_not] using measure_csaAntipodalPair_eq_zero hd P hP (i := 0) (j := 1) (by decide)
  filter_upwards [ha] with X₀ hanti
  intro X h0 hX k
  simpa only [h0] using CSA_twoToken_tendsto β Q K X hX (by simpa only [h0] using hanti) k

/-- The original one-token identity-value assertion holds for every initial configuration and every solution.
Source: arXiv:2411.04990v2, §4, the fixed leader in the proof of `thm1`. -/
theorem CSA_oneToken_tendsto (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d 1)
    (hX : CSA d 1 β Q K (ContinuousLinearMap.id ℝ _) X) (k : Idx 1) :
    Tendsto (fun t => X t k) atTop (𝓝 (X 0 0)) := by
  have hk : k = 0 := Subsingleton.elim k 0
  subst k
  exact tendsto_const_nhds.congr'
    (Eventually.of_forall (fun t => (csaLeader_constant β Q K X hX t).symm))

/-- The original dimension, uniform product law, and distinct-coordinate premises are jointly realized on the circle.
Source: arXiv:2411.04990v2, §4, the independent uniform initialization of `thm1`. -/
example : 2 ≤ 2 ∧ Perspective.UniformTuple 2 2 anisotropicUniformLaw ∧ (0 : Idx 2) ≠ 1 :=
  ⟨le_rfl, uniformTuple_anisotropicUniformLaw, by decide⟩

/-- The original solution, initial antipodal margin, and overlap-convergence hypotheses hold at actual nonzero consensus.
Source: arXiv:2411.04990v2, §4, the consensus equilibria of `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
      (basePoint 1 : SSphere 2) ≠ Perspective.antipode 2 (basePoint 1) ∧
      Tendsto (fun _ : ℝ => 1 - inner ℝ (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2))
        atTop (𝓝 0) := by
  have hi : inner ℝ (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (basePoint 1).2, one_pow]
  refine ⟨CSA_const_consensus β Q K (basePoint 1), ?_, by simpa only [hi, sub_self] using
    (tendsto_const_nhds : Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (𝓝 0))⟩
  intro h
  have hp := congrArg (fun z : SSphere 2 => inner ℝ (z : EucSpace 2) (basePoint 1 : EucSpace 2)) h
  change inner ℝ (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2) =
    inner ℝ (-(basePoint 1 : EucSpace 2)) (basePoint 1 : EucSpace 2) at hp
  rw [inner_neg_left, hi] at hp
  norm_num at hp

/-- The original one-token solution premise is realized by the same nonzero circle point.
Source: arXiv:2411.04990v2, §4, the fixed leader of `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 1 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) :=
  CSA_const_consensus β Q K (basePoint 1)

end Transformer.Causal
