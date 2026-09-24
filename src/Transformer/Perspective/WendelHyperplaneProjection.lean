/-
# General position after orthogonal projection

If a finite family together with a new vector is linearly independent, then
the old vectors remain independent after projection onto the new vector's
orthogonal hyperplane. This is the linear-algebraic step of the dimension
reduction in Wendel's sign-count recurrence.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelCountBridge
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional

namespace Transformer.Perspective

/-- Independence of the old vectors together with `x` implies independence
of their projections onto `xᗮ`. A projected dependence lifts to a dependence
among the old vectors and `x` itself.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem linearIndependent_orthogonalProjection_of_option (d : ℕ)
    (ι : Type*) [Fintype ι] (x : EucSpace d) (v : ι → EucSpace d)
    (h : LinearIndependent ℝ (fun o : Option ι => o.elim x v)) :
    LinearIndependent ℝ
      (fun i : ι => ((ℝ ∙ x)ᗮ).orthogonalProjectionOnto (v i)) := by
  classical
  let H : Submodule ℝ (EucSpace d) := (ℝ ∙ x)ᗮ
  rw [Fintype.linearIndependent_iff]
  intro a ha
  have hsumproj : H.orthogonalProjectionOnto
      (∑ i, a i • v i) = 0 := by
    rw [map_sum]
    simpa only [map_smul] using ha
  have hspan : (∑ i, a i • v i) ∈ (ℝ ∙ x) := by
    have hm : (∑ i, a i • v i) ∈ (H.orthogonalProjectionOnto :
        EucSpace d →ₗ[ℝ] H).ker := LinearMap.mem_ker.mpr hsumproj
    rw [Submodule.ker_orthogonalProjectionOnto, Submodule.orthogonal_orthogonal] at hm
    exact hm
  obtain ⟨t, ht⟩ := Submodule.mem_span_singleton.mp hspan
  let b : Option ι → ℝ := fun o => o.elim (-t) a
  have hrel : ∑ o : Option ι, b o • (o.elim x v) = 0 := by
    rw [Fintype.sum_option]
    simp only [b, Option.elim_none, Option.elim_some]
    rw [← ht]
    module
  have hall := (Fintype.linearIndependent_iff.mp h) b hrel
  intro i
  exact hall (some i)

/-- Strict sign feasibility in an arbitrary real inner-product space. This
agrees with `StrictSignPattern` when the space is `EucSpace d`.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
def StrictSignPatternIn (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (n : ℕ) (v : Idx n → E) (mask : Idx n → Bool) : Prop :=
  ∃ w : E, ∀ i : Idx n,
    0 < inner (𝕜 := ℝ) (if mask i then -v i else v i) w

/-- The sign cones counted by the slice term are precisely the strict sign
cones of the projected vectors in the orthogonal hyperplane.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem slicePattern_iff_strictSignPatternIn_projection (d n : ℕ)
    (v : Idx n → EucSpace d) (x : EucSpace d) (mask : Idx n → Bool) :
    (∃ w : EucSpace d,
      (∀ i : Idx n, 0 < inner (𝕜 := ℝ) (if mask i then -v i else v i) w) ∧
        inner (𝕜 := ℝ) x w = 0) ↔
      StrictSignPatternIn ((ℝ ∙ x)ᗮ) n
        (fun i => ((ℝ ∙ x)ᗮ).orthogonalProjectionOnto (v i)) mask := by
  let H : Submodule ℝ (EucSpace d) := (ℝ ∙ x)ᗮ
  have hinner (i : Idx n) (u : H) :
      inner (𝕜 := ℝ)
        (if mask i then -(H.orthogonalProjectionOnto (v i))
          else H.orthogonalProjectionOnto (v i)) u =
      inner (𝕜 := ℝ) (if mask i then -v i else v i) (u : EucSpace d) := by
    by_cases hm : mask i
    · simp only [hm, ite_true, inner_neg_left]
      exact congrArg Neg.neg (H.inner_orthogonalProjectionOnto_eq_of_mem_right u (v i))
    · simp only [hm]
      exact H.inner_orthogonalProjectionOnto_eq_of_mem_right u (v i)
  constructor
  · rintro ⟨w, hw, hxw⟩
    have hwH : w ∈ H :=
      Submodule.mem_orthogonal_singleton_iff_inner_right.mpr hxw
    refine ⟨⟨w, hwH⟩, fun i => ?_⟩
    rw [hinner]
    exact hw i
  · rintro ⟨u, hu⟩
    refine ⟨(u : EucSpace d), fun i => ?_, ?_⟩
    · rw [← hinner]
      exact hu i
    · exact Submodule.mem_orthogonal_singleton_iff_inner_right.mp u.2

/-- The orthogonal hyperplane of a nonzero vector in `ℝ^(d+1)` has dimension
`d`, as needed to apply the next lower-dimensional sign count.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem finrank_orthogonal_singleton_eucSpace (d : ℕ)
    (x : EucSpace (d + 1)) (hx : x ≠ 0) :
    Module.finrank ℝ ((ℝ ∙ x)ᗮ) = d := by
  let _ : Fact (Module.finrank ℝ (EucSpace (d + 1)) = d + 1) :=
    ⟨by simp⟩
  exact Submodule.finrank_orthogonal_span_singleton hx

/-- Count strict sign patterns of vectors in any real inner-product space.
The hyperplane in the recurrence is itself such a space.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def strictSignCountIn (E : Type*) [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] (n : ℕ) (v : Idx n → E) : ℕ := by
  classical
  exact ∑ mask : Idx n → Bool, if StrictSignPatternIn E n v mask then 1 else 0

/-- The slice correction term of the Euclidean recurrence is exactly the
strict sign count of the projected vectors in the orthogonal hyperplane.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem sliceSignCount_eq_strictSignCountIn_projection (d n : ℕ)
    (v : Idx n → EucSpace d) (x : EucSpace d) :
    sliceSignCount d n v x =
      strictSignCountIn ((ℝ ∙ x)ᗮ) n
        (fun i => ((ℝ ∙ x)ᗮ).orthogonalProjectionOnto (v i)) := by
  classical
  simp only [sliceSignCount, strictSignCountIn]
  simp_rw [slicePattern_iff_strictSignPatternIn_projection d n v x]

/-- The polymorphic count specializes to the Euclidean count used in
`strictSignCount_snoc`. Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignCountIn_eucSpace (d n : ℕ) (v : Idx n → EucSpace d) :
    strictSignCountIn (EucSpace d) n v = strictSignCount d n v := by
  rfl

end Transformer.Perspective
