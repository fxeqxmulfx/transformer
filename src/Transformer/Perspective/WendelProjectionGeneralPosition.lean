/-
# General position in the projected Wendel sample

Appending one vector to a general-position sample and projecting the old
vectors onto its orthogonal hyperplane leaves every small old subfamily
independent. This is the nondegeneracy transfer in the dimension reduction.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelProjectionIsometry

namespace Transformer.Perspective

/-- General position of the extended family descends to the projected old
family, for every subset of at most `d` old indices.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem linearGeneralPosition_orthogonalProjection (d n : ℕ)
    (v : Idx n → EucSpace (d + 1)) (x : EucSpace (d + 1))
    (hgen : ∀ J : Finset (Idx (n + 1)), J.card ≤ d + 1 →
      LinearIndependent ℝ (fun j : J =>
        (Fin.snoc v x : Idx (n + 1) → EucSpace (d + 1)) j.1)) :
    ∀ I : Finset (Idx n), I.card ≤ d →
      LinearIndependent ℝ
        (fun i : I => ((ℝ ∙ x)ᗮ).orthogonalProjectionOnto (v i)) := by
  classical
  intro I hI
  let f : Option I → Idx (n + 1) := fun o =>
    o.elim (Fin.last n) (fun i => Fin.castSucc (i : Idx n))
  have hf : Function.Injective f := by
    intro a b hab
    cases a with
    | none =>
      cases b with
      | none => rfl
      | some b =>
        have h := hab
        simp [f] at h
        exact ((Fin.castSucc_ne_last (b : Idx n)) h.symm).elim
    | some a =>
      cases b with
      | none =>
        have h := hab
        simp [f] at h
      | some b =>
        have h := hab
        simp [f] at h
        simp [h]
  let J : Finset (Idx (n + 1)) := Finset.univ.image f
  have hJcard : J.card ≤ d + 1 := by
    have hcard : J.card = I.card + 1 := by
      rw [Finset.card_image_of_injective _ hf]
      simp
    omega
  let g : Option I → J := fun o =>
    ⟨f o, Finset.mem_image.mpr ⟨o, Finset.mem_univ _, rfl⟩⟩
  have hg : Function.Injective g := by
    intro a b hab
    exact hf (congrArg Subtype.val hab)
  have hopt : LinearIndependent ℝ (fun o : Option I => o.elim x (fun i => v i)) := by
    have h := (hgen J hJcard).comp g hg
    convert h using 1
    funext o
    cases o <;> simp [g, f]
  exact linearIndependent_orthogonalProjection_of_option (d + 1) I x
    (fun i => v i) hopt

/-- The lower-dimensional Euclidean sample produced by the hyperplane
isometry remains in linear general position.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem linearGeneralPosition_projectedEuclidean (d n : ℕ)
    (v : Idx n → EucSpace (d + 1)) (x : EucSpace (d + 1)) (hx : x ≠ 0)
    (hgen : ∀ J : Finset (Idx (n + 1)), J.card ≤ d + 1 →
      LinearIndependent ℝ (fun j : J =>
        (Fin.snoc v x : Idx (n + 1) → EucSpace (d + 1)) j.1)) :
    ∀ I : Finset (Idx n), I.card ≤ d →
      LinearIndependent ℝ (fun i : I =>
        euclideanIsometryOfFinrank ((ℝ ∙ x)ᗮ) d
          (finrank_orthogonal_singleton_eucSpace d x hx)
          (((ℝ ∙ x)ᗮ).orthogonalProjectionOnto (v i))) := by
  intro I hI
  let H : Submodule ℝ (EucSpace (d + 1)) := (ℝ ∙ x)ᗮ
  let e : H ≃ₗᵢ[ℝ] EucSpace d :=
    euclideanIsometryOfFinrank H d (finrank_orthogonal_singleton_eucSpace d x hx)
  have hproj := linearGeneralPosition_orthogonalProjection d n v x hgen I hI
  have hmap := hproj.map' e.toLinearEquiv.toLinearMap
    (LinearMap.ker_eq_bot.mpr e.injective)
  exact hmap

/-- Removing the last point preserves linear general position.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem linearGeneralPosition_init (d n : ℕ)
    (v : Idx (n + 1) → EucSpace d)
    (hgen : ∀ J : Finset (Idx (n + 1)), J.card ≤ d →
      LinearIndependent ℝ (fun j : J => v j)) :
    ∀ I : Finset (Idx n), I.card ≤ d →
      LinearIndependent ℝ (fun i : I => Fin.init v i) := by
  classical
  intro I hI
  let f : Idx n → Idx (n + 1) := Fin.castSucc
  let J : Finset (Idx (n + 1)) := I.image f
  have hJcard : J.card ≤ d := by
    rw [Finset.card_image_of_injective _ (Fin.castSucc_injective n)]
    exact hI
  let g : I → J := fun i =>
    ⟨f i, Finset.mem_image.mpr ⟨i, i.2, rfl⟩⟩
  have hg : Function.Injective g := by
    intro a b hab
    apply Subtype.ext
    exact (Fin.castSucc_injective n) (congrArg Subtype.val hab)
  have h := (hgen J hJcard).comp g hg
  convert h using 1
  funext i
  rfl

/-- In a positive-dimensional general-position family, the appended point
is nonzero. Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem last_ne_zero_of_linearGeneralPosition (d n : ℕ) (hd : 1 ≤ d)
    (v : Idx (n + 1) → EucSpace d)
    (hgen : ∀ J : Finset (Idx (n + 1)), J.card ≤ d →
      LinearIndependent ℝ (fun j : J => v j)) :
    v (Fin.last n) ≠ 0 := by
  classical
  let J : Finset (Idx (n + 1)) := {Fin.last n}
  have hJcard : J.card ≤ d := by simp [J, hd]
  have h := LinearIndependent.ne_zero (⟨Fin.last n, by simp [J]⟩ : J)
    (hgen J hJcard)
  exact h

end Transformer.Perspective
