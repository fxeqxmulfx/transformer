module

public import Transformer.Grokking.Protocol.QuotientTransfer
public import Transformer.Grokking.Protocol.Uncovered

/-!
# Splits that give a training anchor to each orbit of the scalings

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; `experiments/grok_division/PROTOCOL.md`, *The split*. The uniform law on the shuffles
is `uniformPerm` of `Transformer.Grokking.Protocol.Target`.

The scalings of the operands have two orbits: the 9216 equations with a nonzero quotient and
the 96 with the quotient zero. The certificate of `HeldOutCertificate` takes the anchor of a
held-out equation in the orbit of that equation. A split that trains on an equation of each
orbit is `anchored`. This is not the complement of `uncovered` of `Uncovered`, which asks that no
training equation shows the residue `1` or `2`: the two names are about different things.

* The split that trains on the first 4656 equations of the list is anchored
  (`one_mem_anchored`), so a uniformly random split is anchored with positive probability
  (`uniformPerm_anchored_pos`).
* At most 96 equations have the quotient zero (`card_quotient_zero_le`), fewer than the 4656
  that are held out: some split holds all of them out (`exists_not_mem_anchored`), and a
  uniformly random split is not anchored with positive probability
  (`uniformPerm_compl_anchored_pos`).

So to be anchored is a condition on the split that is neither sure nor impossible. Its
probability is computed in `LabelBound` (`uniformPerm_anchored_eq`).

The condition is the exact one for the anchors: every equation is a training equation or is
held out (`mem_heldout_or_exists_trainEq`), and a split is anchored if and only if every
held-out equation has a training equation whose quotient a unit carries to its own
(`mem_anchored_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- The training equation in the slot `i` of the split `π`. -/
abbrev trainEq (π : Equiv.Perm (Fin (97 * 96))) (i : Fin 4656) : Equation :=
  trainAt equations (by norm_num : 4656 ≤ 97 * 96) π i

/-- **The anchored splits**: among the training equations there is one with a nonzero quotient
and one with the quotient zero, an anchor for each orbit of the scalings. -/
def anchored : Set (Equiv.Perm (Fin (97 * 96))) :=
  {π | (∃ i, quotient (trainEq π i) ≠ 0) ∧ ∃ i, quotient (trainEq π i) = 0}

/-- The split that trains on the first 4656 equations of the list is anchored:
`0 / 1 = 0` is in its slot `0` and `1 / 1 = 1` in its slot `96`. -/
theorem one_mem_anchored : (1 : Equiv.Perm (Fin (97 * 96))) ∈ anchored :=
  ⟨⟨⟨96, by norm_num⟩, by decide⟩, ⟨⟨0, by norm_num⟩, by decide⟩⟩

/-- **A uniformly random split is anchored with positive probability.** -/
theorem uniformPerm_anchored_pos : 0 < uniformPerm (97 * 96) anchored :=
  uniformPerm_pos_of_mem one_mem_anchored

/-- At most 96 equations have the quotient zero. -/
lemma card_quotient_zero_le :
    (Finset.univ.filter fun j : Fin (97 * 96) => quotient (equations j) = 0).card ≤ 96 := by
  have h := Finset.card_le_card_of_injOn
    (s := Finset.univ.filter fun j : Fin (97 * 96) => quotient (equations j) = 0)
    (t := (Finset.univ : Finset (Fin 96))) (fun j => (equations j).2)
    (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _)) fun i hi j hj hij =>
      equations_bijective.injective (Prod.ext
        (((Finset.mem_filter.1 hi).2 : quotient (equations i) = 0).trans
          ((Finset.mem_filter.1 hj).2 : quotient (equations j) = 0).symm) hij)
  rwa [Finset.card_univ, Fintype.card_fin] at h

/-- **Some split holds out every equation with numerator zero**: its training equations all
have a nonzero quotient. -/
theorem exists_not_mem_anchored : ∃ π : Equiv.Perm (Fin (97 * 96)), π ∉ anchored := by
  have hcard : 4656 ≤
      (Finset.univ.filter fun j : Fin (97 * 96) => ¬quotient (equations j) = 0).card := by
    have h := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin (97 * 96))))
      fun j => quotient (equations j) = 0
    rw [Finset.card_univ, Fintype.card_fin] at h
    have h96 := card_quotient_zero_le
    omega
  obtain ⟨π, hπ⟩ := exists_perm_forall_castLE (by norm_num : 4656 ≤ 97 * 96)
    (fun j => ¬quotient (equations j) = 0) hcard
  exact ⟨π, fun h => h.2.elim fun i hi => hπ i hi⟩

/-- **A uniformly random split is not anchored with positive probability.** -/
theorem uniformPerm_compl_anchored_pos : 0 < uniformPerm (97 * 96) anchoredᶜ := by
  obtain ⟨π, hπ⟩ := exists_not_mem_anchored
  exact uniformPerm_pos_of_mem (Set.mem_compl hπ)

/-- Every equation is a training equation of the split or is held out. -/
lemma mem_heldout_or_exists_trainEq (π : Equiv.Perm (Fin (97 * 96))) (x : Equation) :
    x ∈ heldout equations 4656 π ∨ ∃ i, trainEq π i = x := by
  obtain ⟨j₀, rfl⟩ := equations_bijective.surjective x
  by_cases h : 4656 ≤ (π.symm j₀ : ℕ)
  · exact Or.inl (Finset.mem_image.2 ⟨π.symm j₀, Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩,
      by rw [Equiv.apply_symm_apply]⟩)
  · refine Or.inr ⟨⟨(π.symm j₀ : ℕ), not_le.1 h⟩, ?_⟩
    have hcast : Fin.castLE (by norm_num : 4656 ≤ 97 * 96) ⟨(π.symm j₀ : ℕ), not_le.1 h⟩ =
        π.symm j₀ := Fin.ext rfl
    rw [trainEq, trainAt, hcast, Equiv.apply_symm_apply]

/-- **The anchored splits are those that give every held-out equation an anchor in its orbit**:
a training equation whose quotient a unit carries to the quotient of the equation. -/
theorem mem_anchored_iff (π : Equiv.Perm (Fin (97 * 96))) :
    π ∈ anchored ↔ ∀ x ∈ heldout equations 4656 π, ∃ (i : Fin 4656) (c : (ZMod 97)ˣ),
      (c : ZMod 97) * quotient (trainEq π i) = quotient x := by
  constructor
  · rintro ⟨⟨i₁, h₁⟩, i₀, h₀⟩ x -
    by_cases hx : quotient x = 0
    · exact ⟨i₀, 1, by rw [h₀, hx, mul_zero]⟩
    · exact ⟨i₁, Units.mk0 (quotient x) hx * (Units.mk0 _ h₁)⁻¹, by
        rw [Units.val_mul, Units.val_mk0, Units.val_inv_eq_inv_val, Units.val_mk0,
          inv_mul_cancel_right₀ h₁]⟩
  · intro h
    have key : ∀ x : Equation, ∃ i, (quotient (trainEq π i) = 0 ↔ quotient x = 0) := fun x => by
      rcases mem_heldout_or_exists_trainEq π x with hx | ⟨i, rfl⟩
      · obtain ⟨i, c, hc⟩ := h x hx
        exact ⟨i, by rw [← hc, mul_eq_zero, or_iff_right c.ne_zero]⟩
      · exact ⟨i, Iff.rfl⟩
    obtain ⟨i₁, h₁⟩ := key (1, 0)
    obtain ⟨i₀, h₀⟩ := key (0, 0)
    exact ⟨⟨i₁, fun hq => absurd (h₁.1 hq) (by decide)⟩, i₀, h₀.2 (by decide)⟩

/-! The witnesses: the equation in the slot `96` of the split of the first 4656 equations, and
that split, whose held-out equations all have a training equation in their orbit. -/

example : trainEq 1 ⟨96, by norm_num⟩ = (1, 0) := by decide

example := (mem_anchored_iff 1).1 one_mem_anchored

end Transformer.Grokking.Protocol
