/-
# Almost orthogonality at the source's exact distance radius

ArXiv:2312.10794v5, Appendix D, Step 2, `eq: almost.ortho.vec`.
The already constructed measurable normalized-residual frame is used with
projection radius `sqrt(log(d)/d)/2`. The sharper moment-degree estimate in
`AppendixD_ProjectionScale` supplies the same dimension-power failure bound
at this smaller radius. Fubini and induction sum the actual failure events.

On the success event, the proved normalization bound gives actual pairwise
orthogonal unit points within `sqrt(log(d)/d)` of the original sample.
No independence of a frame from its own input is assumed: conditioning is
on the original sample prefix, exactly as in the previous construction.
The resulting probability bound is stronger than the displayed source bound.
-/

import Transformer.Perspective.AppendixD_ProjectionScale
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Metastability.Section4_IndependentPrefix

open Real MeasureTheory Set
open Transformer.Metastability

namespace Transformer.Perspective

/-- A successful prefix leaves next-step failure controlled by the proved
tail at half the distance radius. Source: arXiv:2312.10794v5, Appendix D,
Step 2, the original independent-sample conditioning argument. -/
theorem iidSphere_half_projection_compl_succ_le {d m n : ℕ}
    (hd : halfProjectionThreshold n ≤ d) (hmn : m ≤ n)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    (iidSphere d (m + 1) ν).real
        (projectionSuccess d (uniformProjectionRadius d / 2) (m + 1))ᶜ ≤
      (iidSphere d m ν).real (projectionSuccess d (uniformProjectionRadius d / 2) m)ᶜ +
        (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  have : IsProbabilityMeasure ν := hν.1
  have hdone := ((halfProjectionThreshold_spec n).2 d hd).1
  have hr : uniformProjectionRadius d / 2 < 1 := by
    have h := uniformProjectionRadius_lt_one hdone
    linarith
  apply iidSphere_real_event_le_prefix_compl_add d m ν _
    (measurableSet_projectionSuccess _ _ _) _ (measurableSet_projectionSuccess _ _ _).compl _
    (Real.rpow_nonneg (by positivity) _)
  intro X hX
  have hsub : {x : SSphere d | Fin.snoc X x ∈
        (projectionSuccess d (uniformProjectionRadius d / 2) (m + 1))ᶜ} ⊆
      {x : SSphere d | uniformProjectionRadius d / 2 ≤
        ‖frameProjection (sequentialFrame d m X) (x : EucSpace d)‖} := by
    intro x hx
    change ¬ Fin.snoc X x ∈ projectionSuccess d (uniformProjectionRadius d / 2) (m + 1) at hx
    change uniformProjectionRadius d / 2 ≤ ‖frameProjection (sequentialFrame d m X) (x : EucSpace d)‖
    apply le_of_not_gt
    intro hp
    exact hx ((mem_projectionSuccess_snoc_iff d m _ X x).mpr ⟨hX, hp⟩)
  apply (measureReal_mono hsub (measure_ne_top ν _)).trans
  exact uniform_frameProjection_tail_half_scale hd hmn ν hν _
    (orthonormal_sequentialFrame_of_success d m _ hr X hX)

/-- The actual total failure probability at half the comparison radius
is at most the sample length times the original rate. Source:
arXiv:2312.10794v5, Appendix D, Step 2, the union bound over actual prefixes. -/
theorem iidSphere_half_projection_compl_le {d m n : ℕ}
    (hd : halfProjectionThreshold n ≤ d) (hmn : m ≤ n)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    (iidSphere d m ν).real (projectionSuccess d (uniformProjectionRadius d / 2) m)ᶜ ≤
      (m : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  revert hmn
  induction m with
  | zero => intro hmn; simp [projectionSuccess]
  | succ m ih =>
    intro hmn
    calc
      _ ≤ (iidSphere d m ν).real (projectionSuccess d (uniformProjectionRadius d / 2) m)ᶜ +
          (d : ℝ) ^ (-(1 : ℝ) / 64) :=
        iidSphere_half_projection_compl_succ_le hd (by omega) ν hν
      _ ≤ (m : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) + (d : ℝ) ^ (-(1 : ℝ) / 64) :=
        add_le_add (ih (by omega)) le_rfl
      _ = _ := by rw [Nat.cast_add, Nat.cast_one]; ring

/-- Every actual projection succeeds with probability at least
`1 - n d^(-1/64)`. Source: arXiv:2312.10794v5, Appendix D, Step 2.
All events here are measurable events of the original independent sample. -/
theorem iidSphere_half_projection_ge_union_bound {d n : ℕ}
    (hd : halfProjectionThreshold n ≤ d) (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    1 - (n : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) ≤
      (iidSphere d n ν).real (projectionSuccess d (uniformProjectionRadius d / 2) n) := by
  have : IsProbabilityMeasure ν := hν.1
  have : IsProbabilityMeasure (iidSphere d n ν) := isProbabilityMeasure_iidSphere d n ν
  have hfail := iidSphere_half_projection_compl_le hd le_rfl ν hν
  rw [probReal_compl_eq_one_sub (measurableSet_projectionSuccess _ _ _)] at hfail
  linarith

/-- The actual unit frame is within the displayed source distance, after
normalization. Source: arXiv:2312.10794v5, Appendix D, Step 2,
`eq: almost.ortho.vec`. The smaller projection threshold accounts for the
normalization of the residual; the source's distance constant is retained. -/
theorem iidSphere_orthogonal_approximation_exact_radius {d n : ℕ}
    (hd : halfProjectionThreshold n ≤ d) (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    1 - (n : ℝ) * (d : ℝ) ^ (-(1 : ℝ) / 64) ≤
      (iidSphere d n ν).real {X : SphereTuple d n |
        ∃ Y : SphereTuple d n,
          (∀ i j : Idx n, i ≠ j →
            inner (𝕜 := ℝ) (Y i : EucSpace d) (Y j : EucSpace d) = 0) ∧
          ∀ i : Idx n, ‖(X i : EucSpace d) - (Y i : EucSpace d)‖ ≤
            Real.sqrt (Real.log d / (d : ℝ))} := by
  have : IsProbabilityMeasure ν := hν.1
  have : IsProbabilityMeasure (iidSphere d n ν) := isProbabilityMeasure_iidSphere d n ν
  apply (iidSphere_half_projection_ge_union_bound hd ν hν).trans
  apply measureReal_mono _ (measure_ne_top _ _)
  intro X hX
  have hdone := ((halfProjectionThreshold_spec n).2 d hd).1
  have hr : uniformProjectionRadius d / 2 < 1 := by
    have h := uniformProjectionRadius_lt_one hdone
    linarith
  obtain ⟨Y, hY, hclose⟩ := exists_orthogonal_approximation_of_success d n _ hr X hX
  refine ⟨Y, hY, fun i => ?_⟩
  simpa only [uniformProjectionRadius, mul_div_cancel₀ _ (by norm_num : (2 : ℝ) ≠ 0)] using hclose i

/-- Every concentration hypothesis is realized by the actual uniform law
in an actual dimension above the threshold, with a nonempty sample space. -/
example : ∃ d : ℕ, halfProjectionThreshold 2 ≤ d ∧ (1 : ℕ) ≤ 2 ∧
    IsUniformOn d (MeanField.uniformLaw d) ∧ Nonempty (SphereTuple d 2) := by
  let d := halfProjectionThreshold 2
  have hd := ((halfProjectionThreshold_spec 2).2 d le_rfl).1
  let x : SSphere d := ⟨EuclideanSpace.single ⟨0, by omega⟩ 1,
    mem_sphere_zero_iff_norm.mpr (by simp)⟩
  exact ⟨d, le_rfl, by decide, isUniformOn_uniformLaw (by omega), ⟨fun _ => x⟩⟩

/-- **Equation (eq: almost.ortho.vec).** For fixed `n ≥ 2`, above a dimension
threshold depending only on `n`, an i.i.d. uniform sample admits pairwise
orthogonal unit points within `sqrt(log(d)/d)` with probability at least
`1 - 2 n² d^(-1/64)`.

The standalone statement previously omitted the standing large-dimension
threshold of `thm: phase.transition.curve`, inside whose proof this equation
appears. That threshold is restored here; the distance radius and probability
constant are unchanged. Orthogonal residuals are normalized explicitly, and
the projection tail at half the distance radius pays for normalization.
The argument uses proved spherical moments instead of assuming Lévy's bound.

Source: arXiv:2312.10794v5, §6.2, `thm: phase.transition.curve`, and
Appendix D, Step 2, `eq: almost.ortho.vec`. -/
theorem almost_orthogonal (n : ℕ) (hn : 2 ≤ n) :
    ∃ d_star : ℕ, n ≤ d_star ∧ ∀ d : ℕ, d_star ≤ d →
      ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
        1 - 2 * (n : ℝ) ^ 2 * (d : ℝ) ^ (-(1 / 64 : ℝ)) ≤
          (P {X : SphereTuple d n | ∃ Y : SphereTuple d n,
            (∀ i j : Idx n, i ≠ j →
              inner (𝕜 := ℝ) (Y i : EucSpace d) (Y j : EucSpace d) = 0) ∧
            ∀ i : Idx n, ‖(X i : EucSpace d) - (Y i : EucSpace d)‖ ≤
              Real.sqrt (Real.log d / d)}).toReal := by
  refine ⟨halfProjectionThreshold n, (halfProjectionThreshold_spec n).1.le, ?_⟩
  intro d hd P hP
  rcases hP with ⟨ν, hνprob, hνinv, rfl⟩
  have hbound := iidSphere_orthogonal_approximation_exact_radius hd ν ⟨hνprob, hνinv⟩
  have hnreal : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hcoeff : (n : ℝ) ≤ 2 * (n : ℝ) ^ 2 := by nlinarith
  have hscaled := mul_le_mul_of_nonneg_right hcoeff
    (Real.rpow_nonneg (by positivity : (0 : ℝ) ≤ d) (-(1 : ℝ) / 64))
  simp only [neg_div] at hbound hscaled
  change 1 - 2 * (n : ℝ) ^ 2 * (d : ℝ) ^ (-(1 / 64 : ℝ)) ≤
    (iidSphere d n ν).real _
  exact (show 1 - 2 * (n : ℝ) ^ 2 * (d : ℝ) ^ (-(1 / 64 : ℝ)) ≤
    1 - (n : ℝ) * (d : ℝ) ^ (-(1 / 64 : ℝ)) by linarith).trans hbound

example : (2 : ℕ) ≤ 2 := le_rfl

end Transformer.Perspective
