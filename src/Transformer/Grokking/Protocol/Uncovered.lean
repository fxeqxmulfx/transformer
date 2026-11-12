module

public import Transformer.Grokking.Protocol.DivisionTarget
public import Mathlib.Logic.Equiv.Fintype

/-!
# Splits that keep two residues out of training have positive probability

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; `experiments/grok_division/PROTOCOL.md`, *The split*. The uniform law on the shuffles
is `uniformPerm` of `Transformer.Grokking.Protocol.Target`.

A uniformly random half of the equations can miss a residue altogether. Here the residues `1`
and `2` (`hidden`) are missed: a split is `uncovered` when no training equation has one of
them as its numerator, its denominator or its quotient (`Avoids`). A residue is the quotient
of `96` equations, the denominator of `97` and the numerator of `96`, so all but `289`
equations for each residue of a set avoid the set (`card_filter_avoid_ge`). So at least
`8734` of the `9312` equations avoid both hidden residues (`card_avoids_ge`), more than the
`4656` of a training set. Hence some shuffle is uncovered (`exists_uncovered`), and the
uncovered shuffles have positive probability (`uniformPerm_uncovered_pos`). It is tiny,
roughly `2⁻⁵⁹⁴` by a count that is not formalized here; what matters is that it is not `0`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- The residues kept out of training: `1` and `2`. -/
def hidden : Finset (Fin 97) := {1, 2}

/-- An equation that shows no hidden residue: not as numerator, denominator or quotient. -/
def Avoids (x : Equation) : Prop :=
  numerator x ∉ hidden ∧ denominator x ∉ hidden ∧ x.1 ∉ hidden

instance : DecidablePred Avoids := fun x => by
  unfold Avoids
  infer_instance

/-- **The uncovered splits**: no training equation shows a hidden residue. -/
def uncovered : Set (Equiv.Perm (Fin (97 * 96))) :=
  {π | ∀ i : Fin 4656, Avoids (trainAt equations (by norm_num : 4656 ≤ 97 * 96) π i)}

/-- A shuffle can put any `k` of at least `k` marked items in the first `k` slots. -/
lemma exists_perm_forall_castLE {N k : ℕ} (hk : k ≤ N) (g : Fin N → Prop) [DecidablePred g]
    (hg : k ≤ (Finset.univ.filter g).card) :
    ∃ π : Equiv.Perm (Fin N), ∀ i : Fin k, g (π (Fin.castLE hk i)) := by
  obtain ⟨Q, hQ, hcard⟩ := Finset.exists_subset_card_eq hg
  let e : {j : Fin N // (j : ℕ) < k} ≃ {j : Fin N // j ∈ Q} :=
    (Fin.castLEquiv hk).symm.trans (Q.equivFinOfCardEq hcard).symm
  refine ⟨e.extendSubtype, fun i => ?_⟩
  have hi : ((Fin.castLE hk i : Fin N) : ℕ) < k := i.isLt
  rw [Equiv.extendSubtype_apply_of_mem e _ hi]
  exact (Finset.mem_filter.1 (hQ (e ⟨_, hi⟩).2)).2

/-- The quotient is fixed by the numerator and the denominator. -/
lemma quotient_eq_of_numerator_eq {x y : Equation} (ha : numerator x = numerator y)
    (hd : x.2 = y.2) : x.1 = y.1 := by
  have hx := (inDivisionCell_operands x).2
  have hy := (inDivisionCell_operands y).2
  have hden : denominator x = denominator y := by simp only [denominator, hd]
  rw [ha, hden, hy] at hx
  have h := (ZMod.natCast_eq_natCast_iff' _ _ _).1 hx
  rw [Nat.mod_eq_of_lt y.1.isLt, Nat.mod_eq_of_lt x.1.isLt] at h
  exact Fin.ext h.symm

/-- **All but `289` equations for each residue of a set avoid the set**: a residue is the
quotient of `96` equations, the denominator of `97` and the numerator of `96`. -/
theorem card_filter_avoid_ge (H : Finset (Fin 97)) :
    97 * 96 ≤ (Finset.univ.filter fun j : Fin (97 * 96) => numerator (equations j) ∉ H ∧
      denominator (equations j) ∉ H ∧ (equations j).1 ∉ H).card + 289 * H.card := by
  have hinj := equations_bijective.injective
  have hprod : ∀ n : ℕ, (H ×ˢ (Finset.univ : Finset (Fin n))).card = H.card * n := fun n => by
    rw [Finset.card_product, Finset.card_univ, Fintype.card_fin]
  have h1 : (Finset.univ.filter fun j : Fin (97 * 96) => (equations j).1 ∈ H).card ≤
      H.card * 96 := by
    rw [← hprod 96]
    refine Finset.card_le_card_of_injOn equations (fun j hj => ?_) hinj.injOn
    exact Finset.mem_product.2 ⟨(Finset.mem_filter.1 hj).2, Finset.mem_univ _⟩
  have h2 : (Finset.univ.filter fun j : Fin (97 * 96) =>
      denominator (equations j) ∈ H).card ≤ H.card * 97 := by
    rw [← hprod 97]
    refine Finset.card_le_card_of_injOn (fun j => (denominator (equations j), (equations j).1))
      (fun j hj => Finset.mem_product.2 ⟨(Finset.mem_filter.1 hj).2, Finset.mem_univ _⟩)
      fun i _ j _ hij => ?_
    obtain ⟨hden, hquo⟩ := Prod.mk.inj hij
    have := congrArg Fin.val hden
    simp only [denominator] at this
    exact hinj (Prod.ext hquo (Fin.ext (by omega)))
  have h3 : (Finset.univ.filter fun j : Fin (97 * 96) =>
      numerator (equations j) ∈ H).card ≤ H.card * 96 := by
    rw [← hprod 96]
    refine Finset.card_le_card_of_injOn (fun j => (numerator (equations j), (equations j).2))
      (fun j hj => Finset.mem_product.2 ⟨(Finset.mem_filter.1 hj).2, Finset.mem_univ _⟩)
      fun i _ j _ hij => ?_
    obtain ⟨hnum, hden⟩ := Prod.mk.inj hij
    exact hinj (Prod.ext (quotient_eq_of_numerator_eq hnum hden) hden)
  have hcover : (Finset.univ : Finset (Fin (97 * 96))) ⊆
      (Finset.univ.filter fun j => numerator (equations j) ∉ H ∧
        denominator (equations j) ∉ H ∧ (equations j).1 ∉ H) ∪
        (Finset.univ.filter fun j => (equations j).1 ∈ H) ∪
        (Finset.univ.filter fun j => denominator (equations j) ∈ H) ∪
        (Finset.univ.filter fun j => numerator (equations j) ∈ H) := fun j _ => by
    by_cases h3 : numerator (equations j) ∈ H
    · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h3⟩)
    by_cases h2 : denominator (equations j) ∈ H
    · exact Finset.mem_union_left _ (Finset.mem_union_right _
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, h2⟩))
    by_cases h1 : (equations j).1 ∈ H
    · exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_right _
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, h1⟩)))
    · exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_left _
        (Finset.mem_filter.2 ⟨Finset.mem_univ _, h3, h2, h1⟩)))
  have hle := (Finset.card_le_card hcover).trans
    ((Finset.card_union_le _ _).trans (Nat.add_le_add_right
      ((Finset.card_union_le _ _).trans (Nat.add_le_add_right (Finset.card_union_le _ _) _)) _))
  rw [Finset.card_univ, Fintype.card_fin] at hle
  omega

/-- **At least `8734` equations avoid the hidden residues.** -/
theorem card_avoids_ge :
    8734 ≤ (Finset.univ.filter fun j : Fin (97 * 96) => Avoids (equations j)).card := by
  have h := card_filter_avoid_ge hidden
  have hsub : (Finset.univ.filter fun j : Fin (97 * 96) => numerator (equations j) ∉ hidden ∧
      denominator (equations j) ∉ hidden ∧ (equations j).1 ∉ hidden) ⊆
      Finset.univ.filter fun j => Avoids (equations j) := fun j hj =>
    Finset.mem_filter.2 ⟨Finset.mem_univ j, (Finset.mem_filter.1 hj).2⟩
  have hle := Finset.card_le_card hsub
  have hc : hidden.card = 2 := rfl
  omega

/-- **Some shuffle is uncovered.** -/
theorem exists_uncovered : ∃ π, π ∈ uncovered :=
  exists_perm_forall_castLE (by norm_num : 4656 ≤ 97 * 96) (fun j => Avoids (equations j))
    (le_trans (by norm_num) card_avoids_ge)

/-- A set of shuffles that contains one has positive probability. -/
lemma uniformPerm_pos_of_mem {n : ℕ} {A : Set (Equiv.Perm (Fin n))} {π : Equiv.Perm (Fin n)}
    (hπ : π ∈ A) : 0 < uniformPerm n A := by
  have hle : ((Finset.univ : Finset (Equiv.Perm (Fin n))).card : ENNReal)⁻¹ * Measure.dirac π A ≤
      uniformPerm n A := by
    simp only [uniformPerm, Measure.uniformOfFinset, Measure.coe_finsetSum, Measure.coe_smul,
      Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    exact Finset.single_le_sum (f := fun a => ((Finset.univ : Finset (Equiv.Perm (Fin n))).card :
      ENNReal)⁻¹ * Measure.dirac a A) (fun _ _ => zero_le) (Finset.mem_univ π)
  refine lt_of_lt_of_le ?_ hle
  rw [Measure.dirac_apply_of_mem hπ, mul_one]
  exact ENNReal.inv_pos.2 (ENNReal.natCast_ne_top _)

/-- **The uncovered splits have positive probability.** -/
theorem uniformPerm_uncovered_pos : 0 < uniformPerm (97 * 96) uncovered := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  exact uniformPerm_pos_of_mem hπ

/-! The witnesses: `6 / 4 = 50` avoids the hidden residues and `1 / 3 = 65` does not; three
items of which two are marked, and two slots. -/

example : Avoids (50, 3) ∧ ¬ Avoids (65, 2) := by decide

example := exists_perm_forall_castLE (by norm_num : 2 ≤ 3) (fun j : Fin 3 => j ≠ 1) (by decide)

example := quotient_eq_of_numerator_eq (x := (50, 3)) (y := (50, 3)) rfl rfl

example := uniformPerm_pos_of_mem (n := 2) (A := {1}) (π := 1) rfl

end Transformer.Grokking.Protocol
