/-
# Sparse optimal convex parameters

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.2 and Appendix A.4.
The appendix invokes Carathéodory's theorem to bound the number of heads at
an optimum by `N + 1`.  In the convex row model this corresponds to an
optimum with at most `N + 1` nonzero token rows, provided an optimum exists.
-/

import Transformer.Convexifying.Section3_Caratheodory
import Transformer.Convexifying.Section3_ScalarCounterexample

open scoped BigOperators

namespace Transformer.Convexifying

/-- The nonzero token rows of a convex scalar parameter matrix. -/
noncomputable def nonzeroRows {n d : ℕ} (Z : Fin n → Vec d) : Finset (Fin n) :=
  Finset.univ.filter (fun k => normSq (Z k) ≠ 0)

/-- The sample response of a normalized token direction (Appendix A.4). -/
private noncomputable def sampleAtom {N n d : ℕ} (X : Data N n d)
    (Z : Fin n → Vec d) (k : Fin n) : Vec N :=
  fun i => ∑ q, unitDirection (Z k) q * X i k q
/-- Decompose scalar predictions into weighted token responses (Appendix A.4). -/
private theorem convexPrediction_eq_atom_sum {N n d : ℕ}
    (X : Data N n d) (Z : Fin n → Vec d) (i : Fin N) :
    convexPrediction (X i) Z = ∑ k, norm₂ (Z k) * sampleAtom X Z k i := by
  unfold convexPrediction sampleAtom
  apply Finset.sum_congr rfl
  intro k _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro q _
  have hprod := (unitDirection_spec (Z k)).1 q
  rw [← hprod]
  ring
/-- Reconstruct token rows from sparse convex weights (Appendix A.4). -/
private noncomputable def sparseReconstruction {n d : ℕ}
    (Z : Fin n → Vec d) (R : ℝ) (w : Fin n → ℝ) : Fin n → Vec d :=
  fun k q => (R * w k) * unitDirection (Z k) q
/-- The reconstructed rows have the prescribed predictions (Appendix A.4). -/
private theorem sparseReconstruction_prediction {N n d : ℕ}
    (X : Data N n d) (Z : Fin n → Vec d) (R : ℝ)
    (w : Fin n → ℝ) (i : Fin N) :
    convexPrediction (X i) (sparseReconstruction Z R w) =
      R * ∑ k, w k * sampleAtom X Z k i := by
  unfold convexPrediction sparseReconstruction sampleAtom
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  rw [Finset.mul_sum]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro q _
  ring
/-- Reconstruction does not exceed its assigned row mass (Appendix A.4). -/
private theorem sparseReconstruction_norm_le {n d : ℕ}
    (Z : Fin n → Vec d) (R : ℝ) (w : Fin n → ℝ)
    (hR : 0 ≤ R) (hw : ∀ k, 0 ≤ w k) (k : Fin n) :
    norm₂ (sparseReconstruction Z R w k) ≤ R * w k := by
  change norm₂ (fun q => (R * w k) * unitDirection (Z k) q) ≤ _
  rw [norm₂_smul, abs_of_nonneg (mul_nonneg hR (hw k))]
  have hunit := (unitDirection_spec (Z k)).2
  nlinarith [mul_nonneg (sub_nonneg.mpr hunit) (mul_nonneg hR (hw k))]
/-- Reconstruction cannot introduce support outside its nonzero weights (Appendix A.4). -/
private theorem sparseReconstruction_support_le {n d : ℕ}
    (Z : Fin n → Vec d) (R : ℝ) (w : Fin n → ℝ) :
    (nonzeroRows (sparseReconstruction Z R w)).card ≤
      (Finset.univ.filter fun k => w k ≠ 0).card := by
  apply Finset.card_le_card
  intro k hk
  have hnonzero : normSq (sparseReconstruction Z R w k) ≠ 0 :=
    (Finset.mem_filter.mp hk).2
  apply Finset.mem_filter.mpr
  refine ⟨Finset.mem_univ _, ?_⟩
  intro hwk
  have hzero : sparseReconstruction Z R w k = fun _ => 0 := by
    funext q
    simp [sparseReconstruction, hwk]
  apply hnonzero
  rw [hzero]
  simp [normSq]
/-- The appendix's sparse optimum, with attainment made explicit for arbitrary
losses.  An optimal parameter can be chosen with at most `N + 1` nonzero
token rows.  The source does not state an attainment hypothesis; it is needed
when the loss is left arbitrary here.  The proof applies Carathéodory to the
normalized sample responses and reconstructs rows without increasing the
regularizer.

Source: arXiv:2211.11052v1, Appendix A.4,
`eq:dual_cons_multihead_bidual` and the paragraph following it. -/
theorem sparse_convex_optimum {N n d : ℕ}
    (X : Data N n d) (y : Vec N) (L : ℝ → ℝ → ℝ) (β : ℝ)
    (hβ : 0 ≤ β)
    (hopt : ∃ Z : Fin n → Vec d,
      ∀ Z', correctedConvexObjective X y L β Z ≤
        correctedConvexObjective X y L β Z') :
    ∃ Z : Fin n → Vec d,
      (∀ Z', correctedConvexObjective X y L β Z ≤
        correctedConvexObjective X y L β Z') ∧
      (nonzeroRows Z).card ≤ N + 1 := by
  obtain ⟨Z, hZopt⟩ := hopt
  let R : ℝ := ∑ k, norm₂ (Z k)
  have hRnonneg : 0 ≤ R := Finset.sum_nonneg (fun k _ => norm₂_nonneg _)
  by_cases hRzero : R = 0
  · have hzero (k : Fin n) : norm₂ (Z k) = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg (fun k _ => norm₂_nonneg (Z k))).mp hRzero k
        (Finset.mem_univ _)
    have hsupport : nonzeroRows Z = ∅ := by
      ext k
      simp [nonzeroRows, ← norm₂_sq, hzero k]
    refine ⟨Z, hZopt, ?_⟩
    rw [hsupport]
    simp
  · have hRpos : 0 < R := lt_of_le_of_ne hRnonneg (Ne.symm hRzero)
    let a : Fin n → Vec N := sampleAtom X Z
    let w : Fin n → ℝ := fun k => norm₂ (Z k) / R
    have hw (k : Fin n) : 0 ≤ w k := div_nonneg (norm₂_nonneg _) hRnonneg
    have hsum : ∑ k, w k = 1 := by
      change (∑ k, norm₂ (Z k) / R) = 1
      rw [← Finset.sum_div]
      exact div_self hRzero
    obtain ⟨w', hw', hsum', hpred', hsupp'⟩ := sparse_convex_weights a w hw hsum
    let Z' : Fin n → Vec d := sparseReconstruction Z R w'
    have hpred (i : Fin N) :
        convexPrediction (X i) Z' = convexPrediction (X i) Z := by
      rw [sparseReconstruction_prediction, convexPrediction_eq_atom_sum]
      have heq := hpred' i
      change (∑ k, w' k * sampleAtom X Z k i) =
        ∑ k, w k * sampleAtom X Z k i at heq
      rw [heq, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro k _
      change R * (norm₂ (Z k) / R * sampleAtom X Z k i) = _
      field_simp
    have hreg : (∑ k, norm₂ (Z' k)) ≤ R := by
      calc
        (∑ k, norm₂ (Z' k)) ≤ ∑ k, R * w' k :=
          Finset.sum_le_sum (fun k _ => sparseReconstruction_norm_le Z R w' hRnonneg hw' k)
        _ = R := by rw [← Finset.mul_sum, hsum']; ring
    have hobj : correctedConvexObjective X y L β Z' ≤
        correctedConvexObjective X y L β Z := by
      unfold correctedConvexObjective
      simp only [hpred]
      exact add_le_add_right (mul_le_mul_of_nonneg_left hreg hβ) _
    refine ⟨Z', ?_, ?_⟩
    · intro Z''
      exact hobj.trans (hZopt Z'')
    · exact (sparseReconstruction_support_le Z R w').trans hsupp'

/-- The positivity and attainment hypotheses are satisfiable for one zero
datum with zero target: `Z = 0` is a minimizer. -/
example : (0 : ℝ) ≤ 1 ∧
    (∃ Z : Fin 1 → Vec 1,
      ∀ Z', correctedConvexObjective zeroData (fun _ => 0) squareLoss 1 Z ≤
        correctedConvexObjective zeroData (fun _ => 0) squareLoss 1 Z') := by
  constructor
  · norm_num
  · refine ⟨fun _ _ => 0, ?_⟩
    intro Z
    simp [correctedConvexObjective, convexPrediction, zeroData,
      squareLoss, norm₂, normSq]

end Transformer.Convexifying
