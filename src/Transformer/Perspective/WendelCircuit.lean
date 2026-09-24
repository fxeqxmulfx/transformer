/-
# The unique circuit at one point beyond full dimension

For `d + 1` points in `d` dimensions in linear general position, there is a
linear dependence whose every coefficient is nonzero. This is the first
nontrivial case of the deterministic sign count in Wendel's theorem.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelGeneralPosition
import Transformer.Perspective.WendelOneDimBasic
import Mathlib.LinearAlgebra.Dimension.Finite

namespace Transformer.Perspective

/-- A dependence of `d + 1` vectors in general position cannot have even one
zero coefficient unless it is the zero dependence.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem relation_eq_zero_of_coeff_eq_zero (d : ℕ) (X : SphereTuple d (d + 1))
    (hgen : ∀ I : Finset (Idx (d + 1)), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d))
    (c : Idx (d + 1) → ℝ)
    (hrel : ∑ i, c i • (X i : EucSpace d) = 0)
    (j : Idx (d + 1)) (hj : c j = 0) : c = 0 := by
  classical
  let I : Finset (Idx (d + 1)) := Finset.univ.erase j
  have hIcard : I.card ≤ d := by simp [I]
  have hLI := hgen I hIcard
  have hrelI : ∑ i ∈ I, c i • (X i : EucSpace d) = 0 := by
    have hjterm : c j • (X j : EucSpace d) = 0 := by simp [hj]
    have hs := Finset.sum_erase
      (s := (Finset.univ : Finset (Idx (d + 1))))
      (f := fun i : Idx (d + 1) => c i • (X i : EucSpace d)) hjterm
    simpa only [I] using hs.trans hrel
  have hrelI' : ∑ i : I, c i • (X i : EucSpace d) = 0 := by
    rw [Finset.sum_subtype I (by intro x; rfl)] at hrelI
    exact hrelI
  have hzero : ∀ i : I, c i = 0 :=
    (Fintype.linearIndependent_iff.mp hLI) (fun i : I => c i) hrelI'
  funext i
  by_cases hij : i = j
  · simpa [hij] using hj
  · exact hzero ⟨i, Finset.mem_erase.mpr ⟨hij, Finset.mem_univ i⟩⟩

/-- In a minimally dependent sample, every coefficient of a nontrivial
relation is nonzero: if one vanished, the remaining `d` vectors would already
be dependent.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem exists_full_support_relation (d : ℕ) (X : SphereTuple d (d + 1))
    (hgen : ∀ I : Finset (Idx (d + 1)), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d)) :
    ∃ c : Idx (d + 1) → ℝ, (∀ i, c i ≠ 0) ∧
      ∑ i, c i • (X i : EucSpace d) = 0 := by
  classical
  have hdep : ¬LinearIndependent ℝ (fun i : Idx (d + 1) => (X i : EucSpace d)) := by
    intro h
    have hcard := h.fintype_card_le_finrank
    simp only [Fintype.card_fin, finrank_euclideanSpace_fin] at hcard
    omega
  obtain ⟨c, hrel, i₀, hi₀⟩ := Fintype.not_linearIndependent_iff.mp hdep
  refine ⟨c, ?_, hrel⟩
  intro j hj
  have hzero := relation_eq_zero_of_coeff_eq_zero d X hgen c hrel j hj
  exact hi₀ (congrFun hzero i₀)

/-- All relations among `d + 1` vectors in general position are multiples
of any one full-support relation.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem relation_eq_smul_of_full_support (d : ℕ) (X : SphereTuple d (d + 1))
    (hgen : ∀ I : Finset (Idx (d + 1)), I.card ≤ d →
      LinearIndependent ℝ fun i : I => (X i : EucSpace d))
    (c : Idx (d + 1) → ℝ) (hc : ∀ i, c i ≠ 0)
    (hrel : ∑ i, c i • (X i : EucSpace d) = 0)
    (c' : Idx (d + 1) → ℝ)
    (hrel' : ∑ i, c' i • (X i : EucSpace d) = 0) :
    ∃ t : ℝ, c' = fun i => t * c i := by
  classical
  let j : Idx (d + 1) := ⟨0, by omega⟩
  let t : ℝ := c' j / c j
  have hj : c' j - t * c j = 0 := by
    dsimp [t]
    field_simp [hc j]
    ring
  have hsubrel : ∑ i, (c' i - t * c i) • (X i : EucSpace d) = 0 := by
    simp_rw [sub_smul, mul_smul]
    rw [Finset.sum_sub_distrib, ← Finset.smul_sum, hrel', hrel]
    simp
  have hzero := relation_eq_zero_of_coeff_eq_zero d X hgen
    (fun i => c' i - t * c i) hsubrel j hj
  refine ⟨t, ?_⟩
  funext i
  have hi := congrFun hzero i
  simpa using sub_eq_zero.mp hi

/-- The general-position and nonzero-relation hypotheses above occur already
for two identical points on the one-dimensional sphere. -/
example :
    (∀ I : Finset (Idx 2), I.card ≤ 1 →
      LinearIndependent ℝ fun i : I =>
        (((fun _ : Idx 2 => eOne) i : SSphere 1) : EucSpace 1)) ∧
    (∃ c : Idx 2 → ℝ, (∀ i, c i ≠ 0) ∧
      ∑ i, c i • (eOne : EucSpace 1) = 0) := by
  have hone : (eOne : EucSpace 1) ≠ 0 := by
    intro h
    have hh := inner_eOne_eOne
    simp [h] at hh
  constructor
  · intro I hI
    let _ : Subsingleton I := Finset.card_le_one_iff_subsingleton_coe.mp hI
    rw [linearIndependent_subsingleton_index_iff]
    intro i
    exact hone
  · refine ⟨fun i => if i = (0 : Idx 2) then 1 else -1, ?_, ?_⟩
    · intro i
      fin_cases i <;> norm_num
    · rw [Fin.sum_univ_two]
      simp

end Transformer.Perspective
