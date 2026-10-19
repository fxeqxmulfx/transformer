module

public import Transformer.Interpolation.Section5_PureAttention
public import Mathlib.MeasureTheory.Constructions.BorelSpace.Metrizable
public import Mathlib.Analysis.Real.Sqrt

/-!
# From empirical weak limits to the measurable cluster map

ArXiv:2411.04551v3, §5, `prop:generic-discrete`;
arXiv:2312.10794v5, §6, full clustering of the attention particles.

For a fixed nonempty finite tuple, weak convergence of its empirical
probability measure to a Dirac mass forces every token to converge to
that point. The bounded continuous squared-distance test has integral
tending to zero; each squared token distance is bounded by `n` times
that integral. The converse follows by testing finite averages.

The cluster map is therefore the almost-everywhere limit of the first
token of the actual flow at integer times. Those maps are continuous,
so the limit is almost-everywhere measurable. Measurability of the
cluster map is derived, rather than added to the source's assumptions.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory Filter Topology

namespace Transformer.Interpolation

variable {d n : ℕ}

/-- The squared ambient distance to a fixed unit token is a bounded
continuous test on the compact sphere.
Source: arXiv:2411.04551v3, §5, the weak Dirac limit in `prop:generic-discrete`. -/
noncomputable def squaredDistanceTest (z : SSphere d) : BoundedContinuousFunction (SSphere d) ℝ :=
  BoundedContinuousFunction.mkOfCompact
    ⟨fun x => ‖(x : EucSpace d) - (z : EucSpace d)‖ ^ 2, by fun_prop⟩

/-- The distance test vanishes at the proposed cluster point.
Source: arXiv:2411.04551v3, §5, the Dirac limit in `prop:generic-discrete`. -/
theorem squaredDistanceTest_self (z : SSphere d) : squaredDistanceTest z z = 0 := by
  change ‖(z : EucSpace d) - (z : EucSpace d)‖ ^ 2 = 0
  simp only [sub_self, norm_zero, zero_pow (by decide : 2 ≠ 0)]

/-- **An empirical weak Dirac limit forces every token to that point.**
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`, and
arXiv:2312.10794v5, §6, the particle interpretation of full clustering. -/
theorem tokens_tendsto_of_empiricalProb_tendsto_dirac {ι : Type*} {F : Filter ι}
    (hn : 0 < n) (X : ι → SphereTuple d n) (z : SSphere d)
    (hμ : Tendsto (fun t => Perspective.empiricalProb hn (X t)) F
      (nhds (Perspective.diracProb d z))) (i : Idx n) :
    Tendsto (fun t => X t i) F (nhds z) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hmean := ProbabilityMeasure.tendsto_iff_forall_integral_tendsto.mp hμ
    (squaredDistanceTest z)
  have hz : (∫ x, squaredDistanceTest z x
      ∂(Perspective.diracProb d z : Measure (SSphere d))) = 0 := by
    change (∫ x, squaredDistanceTest z x ∂Measure.dirac z) = 0
    rw [integral_dirac, squaredDistanceTest_self]
  rw [hz] at hmean
  have hsum (t : ι) :
      (n : ℝ) * (∫ x, squaredDistanceTest z x
        ∂(Perspective.empiricalProb hn (X t) : Measure (SSphere d))) =
      ∑ j : Idx n, ‖(X t j : EucSpace d) - (z : EucSpace d)‖ ^ 2 := by
    rw [Perspective.integral_empiricalProb, smul_eq_mul, ← mul_assoc,
      mul_inv_cancel₀ hn', one_mul]
    rfl
  have hbound (t : ι) : ‖(X t i : EucSpace d) - (z : EucSpace d)‖ ^ 2 ≤
      (n : ℝ) * (∫ x, squaredDistanceTest z x
        ∂(Perspective.empiricalProb hn (X t) : Measure (SSphere d))) := by
    rw [hsum]
    exact Finset.single_le_sum
      (f := fun j : Idx n => ‖(X t j : EucSpace d) - (z : EucSpace d)‖ ^ 2)
      (fun j _ => sq_nonneg _) (Finset.mem_univ i)
  have hsq := squeeze_zero (fun t => sq_nonneg ‖(X t i : EucSpace d) - (z : EucSpace d)‖)
    hbound (by simpa only [mul_zero] using hmean.const_mul (n : ℝ))
  have hnorm := Real.continuous_sqrt.continuousAt.tendsto.comp hsq
  apply tendsto_iff_dist_tendsto_zero.mpr
  simpa only [Function.comp_def, Subtype.dist_eq, dist_eq_norm, Real.sqrt_sq_eq_abs,
    abs_of_nonneg (norm_nonneg _), Real.sqrt_zero] using hnorm

/-- The nonempty-tuple and actual weak-limit hypotheses hold for a constant
nonzero token; its empirical probability measure is a unit Dirac mass. -/
example (z : SSphere 3) : (0 : ℕ) < 1 ∧
    Tendsto (fun _ : ℝ => Perspective.empiricalProb one_pos (fun _ : Idx 1 => z))
      atTop (nhds (Perspective.diracProb 3 z)) := by
  refine ⟨one_pos, ?_⟩
  simpa only [Perspective.empiricalProb_one] using
    (tendsto_const_nhds : Tendsto (fun _ : ℝ => Perspective.diracProb 3 z) atTop _)

/-- Token convergence implies empirical weak convergence to the same Dirac
mass. Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`. -/
theorem empiricalProb_tendsto_dirac_of_tokens_tendsto {ι : Type*} {F : Filter ι}
    (hn : 0 < n) (X : ι → SphereTuple d n) (z : SSphere d)
    (hX : ∀ i : Idx n, Tendsto (fun t => X t i) F (nhds z)) :
    Tendsto (fun t => Perspective.empiricalProb hn (X t)) F
      (nhds (Perspective.diracProb d z)) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  apply ProbabilityMeasure.tendsto_iff_forall_integral_tendsto.mpr
  intro φ
  change Tendsto (fun t => ∫ x, φ x
    ∂(Perspective.empiricalProb hn (X t) : Measure (SSphere d))) F
    (nhds (∫ x, φ x ∂Measure.dirac z))
  have hs := (tendsto_finsetSum Finset.univ
    (fun i _ => φ.continuous.continuousAt.tendsto.comp (hX i))).const_mul ((n : ℝ)⁻¹)
  simpa only [Perspective.integral_empiricalProb, smul_eq_mul, Function.comp_def,
    integral_dirac, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ← mul_assoc, inv_mul_cancel₀ hn', one_mul] using hs

example (z : SSphere 3) : (0 : ℕ) < 1 ∧
    ∀ i : Idx 1, Tendsto (fun _ : ℝ => (fun _ : Idx 1 => z) i) atTop (nhds z) :=
  ⟨one_pos, fun _ => tendsto_const_nhds⟩

/-- The actual one-token attention trajectory stays at its initial token.
Source: arXiv:2312.10794v5, §2, `SA`, and arXiv:2411.04551v3,
§5, the one-token case of `prop:generic-discrete`. -/
theorem saFlow_one (β : ℝ) (X₀ : SphereTuple d 1) :
    Perspective.saFlow β X₀ = (fun _ : ℝ => X₀) := by
  have hX : X₀ = (fun _ : Idx 1 => X₀ 0) :=
    funext fun i => congrArg X₀ (Subsingleton.elim i 0)
  calc Perspective.saFlow β X₀ = Perspective.saFlow β (fun _ : Idx 1 => X₀ 0) :=
      congrArg (Perspective.saFlow β) hX
    _ = (fun _ : ℝ => fun _ : Idx 1 => X₀ 0) := Perspective.saFlow_consensus β one_pos _
    _ = (fun _ : ℝ => X₀) := congrArg (fun Y => fun _ : ℝ => Y) hX.symm

/-- **The actual cluster map is almost-everywhere measurable.** It is the
limit of the continuous first-token flow maps at integer times.
Source: arXiv:2411.04551v3, §5, `prop:generic-discrete`.
Only the actual empirical clustering conclusion is assumed here. -/
theorem aemeasurable_of_empirical_saFlow_cluster (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n)
    (ref : Measure (SphereTuple d n)) (f : SphereTuple d n → SSphere d)
    (hf : ∀ᵐ X ∂ref, Tendsto (fun t => Perspective.empiricalProb hn
      (Perspective.saFlow β X t)) atTop (nhds (Perspective.diracProb d (f X)))) :
    AEMeasurable f ref := by
  let i : Idx n := ⟨0, hn⟩
  apply aemeasurable_of_tendsto_metrizable_ae'
    (f := fun k : ℕ => fun X : SphereTuple d n => Perspective.saFlow β X (k : ℝ) i)
  · intro k
    exact ((continuous_apply i).comp
      (Perspective.continuous_saFlow β hβ hn (k : ℝ) (Nat.cast_nonneg k))).measurable.aemeasurable
  · filter_upwards [hf] with X hX
    exact (tokens_tendsto_of_empiricalProb_tendsto_dirac hn _ (f X) hX i).comp
      tendsto_natCast_atTop_atTop

/-- All sign, nonempty-tuple and almost-sure clustering hypotheses hold
for an actual one-token flow. This exhibits the full limit hypothesis,
including its probability-measure content. -/
example (z : SSphere 3) : (0 : ℝ) ≤ 0 ∧ (0 : ℕ) < 1 ∧
    ∀ᵐ X ∂(Measure.dirac (fun _ : Idx 1 => z)),
      Tendsto (fun t : ℝ => Perspective.empiricalProb one_pos (Perspective.saFlow 0 X t))
        atTop (nhds (Perspective.diracProb 3 (X 0))) := by
  refine ⟨le_rfl, one_pos, ae_of_all _ ?_⟩
  intro X
  simpa only [saFlow_one, Perspective.empiricalProb_one] using
    (tendsto_const_nhds : Tendsto (fun _ : ℝ => Perspective.diracProb 3 (X 0)) atTop _)

end Transformer.Interpolation
