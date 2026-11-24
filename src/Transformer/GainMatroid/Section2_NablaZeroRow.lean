module

public import Transformer.GainMatroid.Section2_ExchangedMatrices

/-!
# `∇_Y A` needs the zero row

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, citing Brylawski: "Similarly if `M(A′)`
contains a triad `Y`, then `A′` is projectively equivalent to a matrix `A` in which the columns
corresponding to `Y` are as the first three columns of `I(K₄)`, possibly with a row omitted or
additional zero rows. Let `∇_Y A` denote the matrix `A` obtained by replacing the columns of `Y`
with those of the last three columns of `I(K₄)`. Then `M(∇_Y A) = ∇_Y M(A)`."

This fails for some matrices in that form, which is why `IsTriadForm` also asks that the row `v`
of `I(K₄)` vanish off `Y`. A `Δ`-`Y` exchange keeps the nonloops outside the triangle
(`IsDeltaMatroid.indep_singleton`), so `∇_Y M` keeps the elements outside `Y` that are not
coloops (`IsNablaMatroid.dual_indep_singleton`). Over `𝔽₃`, `zeroRowMatrix` has the first three
columns of `I(K₄)` on its triad `{0, 1, 2}` (`zeroRowMatrix_triad`). Its column `3` is the sum of
the columns `0` and `2`, so it is not a coloop (`zeroRowMatrix_dual_indep`). But in
`∇_Y zeroRowMatrix` it is the only column with an entry in the row `v`, so there it is a coloop
(`nablaMatrix_zeroRowMatrix_not_dual_indep`). Hence `nablaMatrix_fails`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open Submodule

variable {E : Type*} [DecidableEq E]

/-- `M(K₄)` has no loops: the empty set is a flat. -/
theorem k4Matroid_isFlat_empty : k4Matroid.IsFlat ∅ := by
  have h : ∀ C ∈ k4Triangles, ∀ g ∈ C, ∃ f ∈ C, f ≠ g := by decide
  refine (k4Matroid_isFlat_iff ∅).2 fun C hC g hg hC' => ?_
  obtain ⟨f, hf, hfg⟩ := h C hC g hg
  exact hC' f hf hfg

/-- **A `Δ`-`Y` exchange keeps the nonloops outside the triangle.** The loops of `M` form a flat of
the generalized parallel connection, as the triangle has no loops. -/
theorem IsDeltaMatroid.indep_singleton {M N : Matroid E} {a b c e : E}
    (h : IsDeltaMatroid M a b c N) (he : M.Indep {e}) (hea : e ≠ a) (heb : e ≠ b) (hec : e ≠ c) :
    N.Indep {e} := by
  obtain ⟨hab, -, -, htri, P, hPE, hflat, rfl⟩ := h
  have hY (i : Fin 3) : ![a, b, c] i ∉ M.loops := by
    refine (htri.isNonloop_of_mem (Set.nontrivial_of_mem_mem_ne (x := a) (y := b) (by simp)
      (by simp) hab) ?_).not_isLoop
    fin_cases i <;> simp
  have hK : deltaEmbed a b c ⁻¹' (Sum.inl '' M.loops) = ∅ := by
    ext g
    rcases g with i | k
    · simp [deltaEmbed, hY i]
    · simp [deltaEmbed]
  have hS : P.IsFlat (Sum.inl '' M.loops) := by
    rw [hflat, Set.preimage_image_eq _ Sum.inl_injective, hK]
    exact ⟨Matroid.isFlat_closure ∅, k4Matroid_isFlat_empty⟩
  have hrel : deltaRelabel a b c e = .inl e := by simp [deltaRelabel, hea, heb, hec]
  rw [Matroid.comap_indep_iff, Set.image_singleton, hrel, Matroid.indep_singleton]
  refine ⟨?_, Set.injOn_singleton _ _⟩
  have heE : Sum.inl e ∈ P.E := by
    rw [hPE]
    exact .inl ⟨e, he.subset_ground (Set.mem_singleton e), rfl⟩
  rw [Matroid.isNonloop_iff_notMem_loops heE]
  intro hl
  obtain ⟨f, hf, hfe⟩ := hS.loops_subset hl
  rw [Sum.inl_injective hfe] at hf
  exact (Matroid.indep_singleton.1 he).not_isLoop hf

/-- **`∇_Y` keeps the elements outside `Y` that are not coloops.** -/
theorem IsNablaMatroid.dual_indep_singleton {M M' : Matroid E} {a b c e : E}
    (h : IsNablaMatroid M a b c M') (he : M✶.Indep {e}) (hea : e ≠ a) (heb : e ≠ b)
    (hec : e ≠ c) : M'✶.Indep {e} := by
  obtain ⟨N, hN, rfl⟩ := h
  rw [Matroid.dual_dual]
  exact hN.indep_singleton he hea heb hec

/-- A ternary matrix with the first three columns of `I(K₄)` on `{0, 1, 2}`, whose row `v = 3` is
nonzero on the column `3`. -/
def zeroRowMatrix : Matrix (Fin 4) (Fin 5) (ZMod 3) :=
  !![1, 0, 0, 1, 1;
     0, 1, 0, 0, -1;
     0, 0, 1, 1, 0;
     -1, -1, -1, 1, 0]

/-- With one element `e` of `{0, 1, 2}` left out, the columns `e`, `3` and `4` span every column. -/
theorem zeroRowMatrix_col_mem_span {e : Fin 5} (he : e = 0 ∨ e = 1 ∨ e = 2) (f : Fin 5) :
    (fun r => zeroRowMatrix r f) ∈
      span (ZMod 3) ((fun g r => zeroRowMatrix r g) '' ({0, 1, 2} \ {e})ᶜ) := by
  have hmem (g : Fin 5) (hg : g = e ∨ g = 3 ∨ g = 4) : (fun r => zeroRowMatrix r g) ∈
      span (ZMod 3) ((fun g r => zeroRowMatrix r g) '' ({0, 1, 2} \ {e})ᶜ) := by
    refine subset_span ⟨g, ?_, rfl⟩
    rcases hg with rfl | rfl | rfl
    · simp
    · rcases he with rfl | rfl | rfl <;> simp
    · rcases he with rfl | rfl | rfl <;> simp
  have he' := hmem e (.inl rfl)
  have h3 := hmem 3 (.inr (.inl rfl))
  have h4 := hmem 4 (.inr (.inr rfl))
  rcases he with rfl | rfl | rfl <;> fin_cases f
  all_goals first
    | exact he' | exact h3 | exact h4
    | (convert sub_mem he' h4 using 1; funext r; fin_cases r <;> decide)
    | (convert sub_mem h3 he' using 1; funext r; fin_cases r <;> decide)
    | (convert add_mem he' h4 using 1; funext r; fin_cases r <;> decide)
    | (convert sub_mem (sub_mem h3 he') h4 using 1; funext r; fin_cases r <;> decide)

/-- **`{0, 1, 2}` is a triad of `M(zeroRowMatrix)`**: the functional `x + y − z` vanishes on the
columns `3` and `4` only. -/
theorem zeroRowMatrix_triad : (matrixMatroid zeroRowMatrix)✶.IsCircuit {0, 1, 2} := by
  have hA := matrixMatroid_isMatrixRepresentation zeroRowMatrix
  rw [Matroid.isCircuit_iff_dep_forall_sdiff_singleton_indep, Matroid.Dep]
  refine ⟨⟨fun h => ?_, ?_⟩, fun e he => ?_⟩
  · rw [hA.dual_indep_iff] at h
    have hgen : ∀ f : Fin 5, f ≠ 0 → f ≠ 1 → f ≠ 2 →
        zeroRowMatrix 0 f + zeroRowMatrix 1 f - zeroRowMatrix 2 f = 0 := by decide
    have hle : span (ZMod 3) ((fun f r => zeroRowMatrix r f) '' {0, 1, 2}ᶜ) ≤
        LinearMap.ker ((LinearMap.proj 0 : (Fin 4 → ZMod 3) →ₗ[ZMod 3] ZMod 3) +
          LinearMap.proj 1 - LinearMap.proj 2) := by
      rw [span_le]
      rintro _ ⟨f, hf, rfl⟩
      simp only [Set.mem_compl_iff, Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at hf
      exact LinearMap.mem_ker.2 (hgen f hf.1 hf.2.1 hf.2.2)
    have h0 : zeroRowMatrix 0 0 + zeroRowMatrix 1 0 - zeroRowMatrix 2 0 ≠ 0 := by decide
    exact h0 (LinearMap.mem_ker.1 (hle (h 0)))
  · rw [Matroid.dual_ground, hA.1]
    exact Set.subset_univ _
  · rw [hA.dual_indep_iff]
    exact zeroRowMatrix_col_mem_span (by simpa using he)

/-- The column `3` of `zeroRowMatrix` is the sum of the columns `0` and `2`: it is not a coloop. -/
theorem zeroRowMatrix_dual_indep : (matrixMatroid zeroRowMatrix)✶.Indep {3} := by
  rw [(matrixMatroid_isMatrixRepresentation _).dual_indep_iff]
  intro e
  by_cases he : e = 3
  · subst he
    have h : (fun r => zeroRowMatrix r 3) =
        (fun r => zeroRowMatrix r 0) + fun r => zeroRowMatrix r 2 := by
      funext r
      fin_cases r <;> decide
    rw [h]
    exact add_mem (subset_span ⟨0, Set.mem_compl_singleton_iff.2 (by decide), rfl⟩)
      (subset_span ⟨2, Set.mem_compl_singleton_iff.2 (by decide), rfl⟩)
  · exact subset_span ⟨e, he, rfl⟩

/-- In `∇_Y zeroRowMatrix`, the column `3` is the only one with an entry in the row `v`: it is a
coloop. -/
theorem nablaMatrix_zeroRowMatrix_not_dual_indep :
    ¬(matrixMatroid (nablaMatrix zeroRowMatrix 0 1 2 0 1 2))✶.Indep {3} := by
  rw [(matrixMatroid_isMatrixRepresentation _).dual_indep_iff]
  intro h
  have hgen : ∀ f : Fin 5, f ≠ 3 → nablaMatrix zeroRowMatrix 0 1 2 0 1 2 3 f = 0 := by decide
  have hle : span (ZMod 3) ((fun f r => nablaMatrix zeroRowMatrix 0 1 2 0 1 2 r f) '' {3}ᶜ) ≤
      LinearMap.ker (LinearMap.proj 3 : (Fin 4 → ZMod 3) →ₗ[ZMod 3] ZMod 3) := by
    rw [span_le]
    rintro _ ⟨f, hf, rfl⟩
    exact LinearMap.mem_ker.2 (hgen f hf)
  have h3 : nablaMatrix zeroRowMatrix 0 1 2 0 1 2 3 3 ≠ 0 := by decide
  exact h3 (LinearMap.mem_ker.1 (hle (h 3)))

/-- **`M(∇_Y A) = ∇_Y M(A)` (§2.4) fails as printed.** `zeroRowMatrix` has the first three columns
of `I(K₄)` on its triad `Y = {0, 1, 2}`, but `M(∇_Y zeroRowMatrix)` is not `∇_Y M(zeroRowMatrix)`:
the column `3` is a coloop of the first and not of the second. -/
theorem nablaMatrix_fails :
    (∀ r, zeroRowMatrix r 0 = (iK4 : Matrix (Fin 4) (Fin 6) (ZMod 3)) r 0) ∧
      (∀ r, zeroRowMatrix r 1 = (iK4 : Matrix (Fin 4) (Fin 6) (ZMod 3)) r 1) ∧
      (∀ r, zeroRowMatrix r 2 = (iK4 : Matrix (Fin 4) (Fin 6) (ZMod 3)) r 2) ∧
      (matrixMatroid zeroRowMatrix)✶.IsCircuit {0, 1, 2} ∧
      ¬IsNablaMatroid (matrixMatroid zeroRowMatrix) 0 1 2
        (matrixMatroid (nablaMatrix zeroRowMatrix 0 1 2 0 1 2)) :=
  ⟨by decide, by decide, by decide, zeroRowMatrix_triad, fun h =>
    nablaMatrix_zeroRowMatrix_not_dual_indep
      (h.dual_indep_singleton zeroRowMatrix_dual_indep (by decide) (by decide) (by decide))⟩

/-- The edge `3` of `clawGraph` is not a bridge. -/
theorem clawMatroid_dual_indep : clawMatroid✶.Indep {3} := by
  have h : clawMatroid✶.Indep (({3} : Finset (Fin 5)) : Set (Fin 5)) :=
    (matroidOfFinset_dual_indep_iff _).2 ⟨{0, 1, 2}, by decide +kernel, by decide⟩
  simpa using h

example : nablaMatroid✶.Indep {3} := by
  obtain ⟨N, hN, h⟩ := isNablaMatroid_of_isTriadForm clawMatrix_represents clawMatrix_isTriadForm
    clawMatroid_triad nablaMatrix_clawMatrix_represents
  rw [h, Matroid.dual_dual]
  exact hN.indep_singleton clawMatroid_dual_indep (by decide) (by decide) (by decide)

example := (isNablaMatroid_of_isTriadForm clawMatrix_represents clawMatrix_isTriadForm
  clawMatroid_triad nablaMatrix_clawMatrix_represents).dual_indep_singleton clawMatroid_dual_indep
  (by decide) (by decide) (by decide)

example := zeroRowMatrix_col_mem_span (.inl rfl) 1

end Transformer.GainMatroid
