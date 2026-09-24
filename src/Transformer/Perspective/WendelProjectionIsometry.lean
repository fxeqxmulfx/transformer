/-
# Euclidean coordinates for the projected Wendel sample

An orthogonal hyperplane of dimension `d` is linearly isometric to `ℝ^d`.
Strict sign patterns are invariant under that isometry, so the slice term in
Wendel's recurrence becomes an ordinary lower-dimensional sign count.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelHyperplaneProjection
import Mathlib.Analysis.InnerProductSpace.GramSchmidtOrtho

namespace Transformer.Perspective

/-- A finite-dimensional real inner-product space with dimension `d` is
linearly isometric to the Euclidean space used by the Wendel count.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def euclideanIsometryOfFinrank (E : Type*) [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] (d : ℕ)
    (hd : Module.finrank ℝ E = d) : E ≃ₗᵢ[ℝ] EucSpace d := by
  let b : OrthonormalBasis (Fin d) ℝ E :=
    InnerProductSpace.gramSchmidtOrthonormalBasis (by simpa using hd)
      (fun _ => (0 : E))
  exact b.repr

/-- Linear isometries preserve realizability of strict sign patterns.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignPatternIn_isometry_iff (E F : Type*)
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    (e : E ≃ₗᵢ[ℝ] F) (n : ℕ) (v : Idx n → E) (mask : Idx n → Bool) :
    StrictSignPatternIn F n (fun i => e (v i)) mask ↔
      StrictSignPatternIn E n v mask := by
  have hinner (i : Idx n) (w : E) :
      inner (𝕜 := ℝ) (if mask i then -e (v i) else e (v i)) (e w) =
        inner (𝕜 := ℝ) (if mask i then -v i else v i) w := by
    by_cases hi : mask i <;> simp [hi, e.inner_map_map]
  constructor
  · rintro ⟨w, hw⟩
    refine ⟨e.symm w, fun i => ?_⟩
    rw [← hinner i (e.symm w), e.apply_symm_apply]
    exact hw i
  · rintro ⟨w, hw⟩
    refine ⟨e w, fun i => ?_⟩
    rw [hinner]
    exact hw i

/-- The generic strict sign count is invariant under linear isometry.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignCountIn_isometry (E F : Type*)
    [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    (e : E ≃ₗᵢ[ℝ] F) (n : ℕ) (v : Idx n → E) :
    strictSignCountIn F n (fun i => e (v i)) = strictSignCountIn E n v := by
  classical
  simp only [strictSignCountIn]
  simp_rw [strictSignPatternIn_isometry_iff E F e n v]

/-- The hyperplane-slice correction is an ordinary strict sign count in
Euclidean dimension `d`. This is the dimension-reduction identity in the
Wendel recurrence.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem sliceSignCount_eq_lowerDimCount (d n : ℕ)
    (v : Idx n → EucSpace (d + 1)) (x : EucSpace (d + 1)) (hx : x ≠ 0) :
    sliceSignCount (d + 1) n v x =
      strictSignCount d n (fun i =>
        euclideanIsometryOfFinrank ((ℝ ∙ x)ᗮ) d
          (finrank_orthogonal_singleton_eucSpace d x hx)
          (((ℝ ∙ x)ᗮ).orthogonalProjectionOnto (v i))) := by
  let H : Submodule ℝ (EucSpace (d + 1)) := (ℝ ∙ x)ᗮ
  let e : H ≃ₗᵢ[ℝ] EucSpace d :=
    euclideanIsometryOfFinrank H d (finrank_orthogonal_singleton_eucSpace d x hx)
  calc
    sliceSignCount (d + 1) n v x =
        strictSignCountIn H n (fun i => H.orthogonalProjectionOnto (v i)) :=
      sliceSignCount_eq_strictSignCountIn_projection (d + 1) n v x
    _ = strictSignCountIn (EucSpace d) n
        (fun i => e (H.orthogonalProjectionOnto (v i))) :=
      (strictSignCountIn_isometry H (EucSpace d) e n _).symm
    _ = _ := strictSignCountIn_eucSpace d n _

/-- Wendel's strict-sign recurrence, with the slice term replaced by a count
in one lower Euclidean dimension.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem strictSignCount_snoc_dimRecurrence (d n : ℕ)
    (v : Idx n → EucSpace (d + 1)) (x : EucSpace (d + 1)) (hx : x ≠ 0) :
    strictSignCount (d + 1) (n + 1) (Fin.snoc v x) =
      strictSignCount (d + 1) n v +
        strictSignCount d n (fun i =>
          euclideanIsometryOfFinrank ((ℝ ∙ x)ᗮ) d
            (finrank_orthogonal_singleton_eucSpace d x hx)
            (((ℝ ∙ x)ᗮ).orthogonalProjectionOnto (v i))) := by
  rw [strictSignCount_snoc (d + 1) n v x hx,
    sliceSignCount_eq_lowerDimCount d n v x hx]

/-- The new-vector hypothesis in the dimension recurrence is realizable. -/
example : (eOne : EucSpace 1) ≠ 0 := by
  intro h
  have hh := inner_eOne_eOne
  simp [h] at hh

end Transformer.Perspective
