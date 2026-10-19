module

public import Transformer.Metastability.Section4_NormalizedResidual

/-!
# The actual measurable frame from successive sample prefixes

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The original construction uses the previous sample prefix, projects
the next actual point against its actual frame, and appends the
normalized residual. This recursion is total and measurable even
when a failed prefix produces zero residuals. Its success event
requires every actual projection to be smaller than the given radius.

The frame is independent of this radius. Its success event records
the original projection thresholds; it contains no orthogonality,
distance estimate, probability estimate or hidden conclusion.
Those geometric consequences follow from the normalized-residual
lemmas when the original radius is below one. The measurable total
recursion allows conditioning on an actual prefix without imposing
orthonormality on prefixes for which success has already failed.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Metastability

/-- The actual successive normalized-residual frame of the original
sample prefix. Source: arXiv:2410.06833v1, §4, `prop: concentration unif`;
arXiv:2312.10794v5, Appendix D, Step 2. Failures remain ambient vectors. -/
noncomputable def sequentialFrame (d : ℕ) :
    (m : ℕ) → (Fin m → SSphere d) → Fin m → EucSpace d
  | 0, X => fun i => (X i : EucSpace d)
  | m + 1, X => Fin.snoc (sequentialFrame d m (Fin.init X))
      (normalizedFrameResidual (sequentialFrame d m (Fin.init X)) (X (Fin.last m) : EucSpace d))

/-- The actual projection of the newest sample point onto the frame
of its original prefix. Source: arXiv:2410.06833v1, §4, the original
projection threshold; arXiv:2312.10794v5, Appendix D, Step 2. -/
noncomputable def sequentialProjection (d m : ℕ) (X : Fin (m + 1) → SSphere d) : EucSpace d :=
  frameProjection (sequentialFrame d m (Fin.init X)) (X (Fin.last m) : EucSpace d)

/-- The actual success event requires every original projection
threshold, recursively through the actual sample prefix.
Source: arXiv:2410.06833v1, §4, the original inductive approximation. -/
noncomputable def projectionSuccess (d : ℕ) (r : ℝ) : (m : ℕ) → Set (Fin m → SSphere d)
  | 0 => univ
  | m + 1 => {X | Fin.init X ∈ projectionSuccess d r m ∧ ‖sequentialProjection d m X‖ < r}

/-- Appending the next actual point preserves every earlier frame
vector. Source: arXiv:2410.06833v1, §4, using only the original prefix. -/
@[simp] theorem sequentialFrame_castSucc (d m : ℕ) (X : Fin (m + 1) → SSphere d) (i : Fin m) :
    sequentialFrame d (m + 1) X i.castSucc = sequentialFrame d m (Fin.init X) i := by
  simp only [sequentialFrame, Fin.snoc_castSucc]

/-- The last actual frame vector is the original normalized residual.
Source: arXiv:2410.06833v1, §4, the original successive approximation. -/
@[simp] theorem sequentialFrame_last (d m : ℕ) (X : Fin (m + 1) → SSphere d) :
    sequentialFrame d (m + 1) X (Fin.last m) =
      normalizedFrameResidual (sequentialFrame d m (Fin.init X)) (X (Fin.last m) : EucSpace d) := by
  simp only [sequentialFrame, Fin.snoc_last]

/-- Appending an original point computes the actual next frame from
the actual previous prefix alone. Source: arXiv:2410.06833v1, §4,
the original independent-sample construction. -/
theorem sequentialFrame_snoc (d m : ℕ) (X : Fin m → SSphere d) (x : SSphere d) :
    sequentialFrame d (m + 1) (Fin.snoc X x) =
      Fin.snoc (sequentialFrame d m X)
        (normalizedFrameResidual (sequentialFrame d m X) (x : EucSpace d)) := by
  simp only [sequentialFrame, Fin.init_snoc, Fin.snoc_last]

/-- The original prefix map is measurable.
Source: arXiv:2410.06833v1, §4, the original independent-sample recursion. -/
theorem measurable_sphere_init (d m : ℕ) :
    Measurable (fun X : Fin (m + 1) → SSphere d => Fin.init X) := by
  apply measurable_pi_iff.mpr
  intro i
  exact measurable_pi_apply i.castSucc

/-- The actual original frame is measurable on all sample prefixes,
including failed prefixes. Source: arXiv:2410.06833v1, §4, conditioning
on earlier original sample points in `prop: concentration unif`. -/
theorem measurable_sequentialFrame (d m : ℕ) : Measurable (sequentialFrame d m) := by
  induction m with
  | zero =>
    apply measurable_pi_iff.mpr
    intro i
    exact i.elim0
  | succ m ih =>
    have hframe := ih.comp (measurable_sphere_init d m)
    have hpoint : Measurable (fun X : Fin (m + 1) → SSphere d => (X (Fin.last m) : EucSpace d)) :=
      continuous_subtype_val.measurable.comp (measurable_pi_apply (Fin.last m))
    have hnew := (measurable_normalizedFrameResidual (d := d) (m := m)).comp (hframe.prodMk hpoint)
    apply measurable_pi_iff.mpr
    intro i
    cases i using Fin.lastCases with
    | last => simpa only [sequentialFrame_last, Function.comp_def] using hnew
    | cast i =>
      simpa only [sequentialFrame_castSucc, Function.comp_def] using (measurable_pi_apply i).comp hframe

/-- The actual newest projection is measurable on all original sample
prefixes. Source: arXiv:2410.06833v1, §4, the original conditional
projection estimate in `prop: concentration unif`. -/
theorem measurable_sequentialProjection (d m : ℕ) : Measurable (sequentialProjection d m) := by
  have hframe := (measurable_sequentialFrame d m).comp (measurable_sphere_init d m)
  have hpoint : Measurable (fun X : Fin (m + 1) → SSphere d => (X (Fin.last m) : EucSpace d)) :=
    continuous_subtype_val.measurable.comp (measurable_pi_apply (Fin.last m))
  unfold sequentialProjection frameProjection frameSynthesis frameCoordinates
  fun_prop

/-- The actual event that every original projection succeeds is
measurable. Source: arXiv:2410.06833v1, §4, the original probability
event constructed from successive sample prefixes. -/
theorem measurableSet_projectionSuccess (d : ℕ) (r : ℝ) (m : ℕ) :
    MeasurableSet (projectionSuccess d r m) := by
  induction m with
  | zero => exact MeasurableSet.univ
  | succ m ih =>
    exact (ih.preimage (measurable_sphere_init d m)).inter
      (measurableSet_lt (measurable_sequentialProjection d m).norm measurable_const)

/-- An actual successful prefix and an actual small newest projection
are exactly the success condition after appending the original point.
Source: arXiv:2410.06833v1, §4, the original independent-sample induction. -/
theorem mem_projectionSuccess_snoc_iff (d m : ℕ) (r : ℝ)
    (X : Fin m → SSphere d) (x : SSphere d) :
    Fin.snoc X x ∈ projectionSuccess d r (m + 1) ↔
      X ∈ projectionSuccess d r m ∧ ‖frameProjection (sequentialFrame d m X) (x : EucSpace d)‖ < r := by
  simp only [projectionSuccess, mem_ofPred_eq, sequentialProjection, Fin.init_snoc, Fin.snoc_last]

/-- Enlarging the actual radius enlarges the original success event.
Source: arXiv:2410.06833v1, §4, the original projection thresholds. -/
theorem projectionSuccess_mono (d m : ℕ) {r s : ℝ} (hrs : r ≤ s) :
    projectionSuccess d r m ⊆ projectionSuccess d s m := by
  induction m with
  | zero => exact Subset.rfl
  | succ m ih =>
    intro X hX
    exact ⟨ih hX.1, hX.2.trans_le hrs⟩

/-- Two actual orthogonal points satisfy the original success event
at a positive radius below one, witnessing its geometric hypotheses. -/
example : ∃ X : Fin 2 → SSphere 2,
    X ∈ projectionSuccess 2 (1 / 2) 2 ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by
  let a : SSphere 2 := ⟨EuclideanSpace.single 0 1, mem_sphere_zero_iff_norm.mpr (by simp)⟩
  let b : SSphere 2 := ⟨EuclideanSpace.single 1 1, mem_sphere_zero_iff_norm.mpr (by simp)⟩
  let X : Fin 2 → SSphere 2 := Fin.snoc (Fin.snoc Fin.elim0 a) b
  refine ⟨X, ?_, by norm_num, by norm_num⟩
  simp [X, projectionSuccess, sequentialProjection, sequentialFrame, normalizedFrameResidual,
    frameProjection, frameSynthesis, frameCoordinates, a, b, Fin.snoc,
    EuclideanSpace.inner_single_left]

end Transformer.Metastability
