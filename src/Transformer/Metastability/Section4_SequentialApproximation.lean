/-
# Genuine orthogonal approximation on the original success event

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
Every successful actual sample prefix gives an actual orthonormal
frame, and every original point lies within twice the projection
radius of its actual normalized residual. Both statements are proved
by induction on the original sample length using the actual recursion.

At radius `sqrt(log(d)/d)`, this produces genuine sphere points with
the exact distance bound `sqrt(4 log(d)/d)` in the original theorem.
The measurable projection-success event is thus contained in the
original `nearOrthogonal` event. Monotonicity transfers its actual
independent-sample probability lower bound to the original event.
No orthogonality, distance estimate or probability estimate is
assumed on failed prefixes, or built into the success predicate.
-/

import Transformer.Metastability.Section4_SequentialFrame

open Real MeasureTheory Set

namespace Transformer.Metastability

/-- Actual successful prefixes give actual orthonormal frames.
Source: arXiv:2410.06833v1, §4, the original inductive construction;
arXiv:2312.10794v5, Appendix D, Step 2. Radius below one suffices. -/
theorem orthonormal_sequentialFrame_of_success (d m : ℕ) (r : ℝ) (hr : r < 1)
    (X : Fin m → SSphere d) (hX : X ∈ projectionSuccess d r m) :
    Orthonormal ℝ (sequentialFrame d m X) := by
  induction m with
  | zero => exact ⟨fun i => i.elim0, fun i => i.elim0⟩
  | succ m ih =>
    have hprev := ih (Fin.init X) hX.1
    have hp : ‖frameProjection (sequentialFrame d m (Fin.init X))
        (X (Fin.last m) : EucSpace d)‖ < 1 := hX.2.trans hr
    exact orthonormal_snoc_normalizedFrameResidual _ hprev _
      (norm_eq_of_mem_sphere (X (Fin.last m))) hp

/-- Each actual successful frame vector has genuine unit norm.
Source: arXiv:2410.06833v1, §4, the original sphere-valued approximation. -/
theorem norm_sequentialFrame_of_success (d m : ℕ) (r : ℝ) (hr : r < 1)
    (X : Fin m → SSphere d) (hX : X ∈ projectionSuccess d r m) (i : Fin m) :
    ‖sequentialFrame d m X i‖ = 1 :=
  (orthonormal_sequentialFrame_of_success d m r hr X hX).norm_eq_one i

/-- Every actual successful frame vector is nonzero.
Source: arXiv:2410.06833v1, §4, the original normalized-residual
construction and its genuine unit sphere points. -/
theorem sequentialFrame_ne_zero_of_success (d m : ℕ) (r : ℝ) (hr : r < 1)
    (X : Fin m → SSphere d) (hX : X ∈ projectionSuccess d r m) (i : Fin m) :
    sequentialFrame d m X i ≠ 0 := by
  apply norm_ne_zero_iff.mp
  rw [norm_sequentialFrame_of_success d m r hr X hX i]
  norm_num

/-- Actual successful frame coordinates have the original Kronecker
inner products. Source: arXiv:2410.06833v1, §4, the pairwise orthogonal
unit points required by the original concentration statement. -/
theorem inner_sequentialFrame_of_success (d m : ℕ) (r : ℝ) (hr : r < 1)
    (X : Fin m → SSphere d) (hX : X ∈ projectionSuccess d r m) (i j : Fin m) :
    inner (𝕜 := ℝ) (sequentialFrame d m X i) (sequentialFrame d m X j) =
      if i = j then 1 else 0 :=
  orthonormal_iff_ite.mp (orthonormal_sequentialFrame_of_success d m r hr X hX) i j

/-- The actual successful frame is nondegenerate.
Source: arXiv:2410.06833v1, §4, the successive actual orthogonal
subspaces in the original independent-sample construction. -/
theorem linearIndependent_sequentialFrame_of_success (d m : ℕ) (r : ℝ) (hr : r < 1)
    (X : Fin m → SSphere d) (hX : X ∈ projectionSuccess d r m) :
    LinearIndependent ℝ (sequentialFrame d m X) :=
  (orthonormal_sequentialFrame_of_success d m r hr X hX).linearIndependent

/-- Every actual successful frame vector is within twice the original
projection radius of its own original sample point.
Source: arXiv:2410.06833v1, §4, `eq: upto-t`, including normalization;
arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem norm_sub_sequentialFrame_of_success (d m : ℕ) (r : ℝ) (hr : r < 1)
    (X : Fin m → SSphere d) (hX : X ∈ projectionSuccess d r m) :
    ∀ i : Fin m, ‖(X i : EucSpace d) - sequentialFrame d m X i‖ ≤ 2 * r := by
  induction m with
  | zero => intro i; exact i.elim0
  | succ m ih =>
    intro i
    cases i using Fin.lastCases with
    | cast i => simpa only [sequentialFrame_castSucc, Fin.init] using ih (Fin.init X) hX.1 i
    | last =>
      rw [sequentialFrame_last]
      have hp : ‖frameProjection (sequentialFrame d m (Fin.init X))
          (X (Fin.last m) : EucSpace d)‖ < 1 := hX.2.trans hr
      apply (norm_sub_normalizedFrameResidual_le _ _ (norm_eq_of_mem_sphere _) hp).trans
      exact mul_le_mul_of_nonneg_left hX.2.le (by norm_num)

/-- Actual successful ambient frame vectors give genuine original
sphere points with the proved original orthogonality and distance bounds.
Source: arXiv:2410.06833v1, §4, the existential approximation in
`prop: concentration unif`. No sphere-valued frame is assumed. -/
theorem exists_orthogonal_approximation_of_success (d m : ℕ) (r : ℝ) (hr : r < 1)
    (X : Fin m → SSphere d) (hX : X ∈ projectionSuccess d r m) :
    ∃ w : Fin m → SSphere d,
      (∀ i j : Fin m, i ≠ j → inner (𝕜 := ℝ) (w i : EucSpace d) (w j : EucSpace d) = 0) ∧
      ∀ i : Fin m, ‖(X i : EucSpace d) - (w i : EucSpace d)‖ ≤ 2 * r := by
  have hframe := orthonormal_sequentialFrame_of_success d m r hr X hX
  let w : Fin m → SSphere d := fun i => ⟨sequentialFrame d m X i,
    mem_sphere_zero_iff_norm.mpr (hframe.norm_eq_one i)⟩
  refine ⟨w, fun i j hij => hframe.inner_eq_zero hij, ?_⟩
  exact norm_sub_sequentialFrame_of_success d m r hr X hX

/-- The actual success event is contained in the original orthogonal
approximation event whenever its proved distance radius fits the original
one. Source: arXiv:2410.06833v1, §4, `eq: upto-t`. -/
theorem projectionSuccess_subset_nearOrthogonal (d n : ℕ) (r : ℝ) (hr : r < 1)
    (hbound : 2 * r ≤ Real.sqrt (4 * Real.log d / (d : ℝ))) :
    projectionSuccess d r n ⊆ nearOrthogonal d n := by
  intro X hX
  obtain ⟨w, hw, hclose⟩ := exists_orthogonal_approximation_of_success d n r hr X hX
  exact ⟨w, hw, fun i => (hclose i).trans hbound⟩

/-- At the original radius, every actual successful configuration is
in the original event. Source: arXiv:2410.06833v1, §4,
`prop: concentration unif`, with its original distance constant. -/
theorem projectionSuccess_original_scale_subset {d n : ℕ}
    (hd : uniformProjectionThreshold n ≤ d) :
    projectionSuccess d (uniformProjectionRadius d) n ⊆ nearOrthogonal d n := by
  have hdone := ((uniformProjectionThreshold_spec n).2 d hd).1
  exact projectionSuccess_subset_nearOrthogonal d n _
    (uniformProjectionRadius_lt_one hdone) (two_uniformProjectionRadius d).le

/-- The actual independent-sample success probability is a lower bound
for the probability of the original approximation event.
Source: arXiv:2410.06833v1, §4, the last event inclusion in the original
concentration proof. The probability estimate itself is not assumed here. -/
theorem measure_nearOrthogonal_ge_projectionSuccess {d n : ℕ}
    (hd : uniformProjectionThreshold n ≤ d) (ν : Measure (SSphere d)) [IsProbabilityMeasure ν] :
    (iidSphere d n ν).real (projectionSuccess d (uniformProjectionRadius d) n) ≤
      (iidSphere d n ν).real (nearOrthogonal d n) := by
  have : IsProbabilityMeasure (iidSphere d n ν) := by unfold iidSphere; infer_instance
  exact measureReal_mono (projectionSuccess_original_scale_subset hd) (measure_ne_top _ _)

/-- Two actual orthogonal points satisfy success at the original radius
and its exact distance factor, jointly witnessing the geometric hypotheses. -/
example : ∃ X : Fin 2 → SSphere 2,
    X ∈ projectionSuccess 2 (uniformProjectionRadius 2) 2 ∧ uniformProjectionRadius 2 < 1 ∧
    2 * uniformProjectionRadius 2 = Real.sqrt (4 * Real.log (2 : ℕ) / ((2 : ℕ) : ℝ)) := by
  let a : SSphere 2 := ⟨EuclideanSpace.single 0 1, mem_sphere_zero_iff_norm.mpr (by simp)⟩
  let b : SSphere 2 := ⟨EuclideanSpace.single 1 1, mem_sphere_zero_iff_norm.mpr (by simp)⟩
  let X : Fin 2 → SSphere 2 := Fin.snoc (Fin.snoc Fin.elim0 a) b
  refine ⟨X, ?_, uniformProjectionRadius_lt_one (by decide), two_uniformProjectionRadius 2⟩
  have hr := uniformProjectionRadius_pos (by decide : 1 < 2)
  simpa [X, projectionSuccess, sequentialProjection, sequentialFrame, normalizedFrameResidual,
    frameProjection, frameSynthesis, frameCoordinates, a, b, Fin.snoc,
    EuclideanSpace.inner_single_left] using And.intro hr hr

end Transformer.Metastability
