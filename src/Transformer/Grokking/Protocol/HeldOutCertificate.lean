module

public import Transformer.Grokking.Protocol.SolverScale
public import Transformer.Grokking.Protocol.QuotientWitness
public import Transformer.Grokking.Protocol.Refutation

/-!
# Held-out answers of the decoder from two training answers and the structure of its logits

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: the
decoder, the operation `x / y (mod 97)`, the random half of the equations held out and the
validation accuracy; `experiments/grok_division/README.md`, *The decoders that answer every
equation are not quotient-equivariant*, for the certificate.

`heldout_accuracy_eq_one_of_anchors` is the certificate on the decoder's own logits. Its
premises are: two training equations, one of each orbit of the scalings; the margins of their
answers; and, for every held-out equation, how far its logits are from those of the anchor of
its orbit, read through the relabelling that the scaling of the operands induces. The
relabelling is fixed by the operands of the two equations: `c = (a'/a) / (b'/b)`. No held-out
answer is a premise, and the structure alone decides nothing: the decoder with every
parameter zero has it exactly and answers no equation (`predict_answerLogits_zero_ne`).

* The premises are consistent on the decoder: `solver` satisfies them on the split that
  trains on the first 4656 equations, with a positive error (`heldout_accuracy_solver`).
* A split must give an anchor to each orbit (`covered`). A uniformly random split does with
  positive probability (`uniformPerm_covered_pos`) and fails to with positive probability
  (`uniformPerm_compl_covered_pos`): the 96 equations with numerator zero can all be held out.
* The margins of the training answers do not suffice: there are parameters and a split with
  every training equation answered and a held-out one not
  (`exists_fit_training_wrong_heldout`).

That a run of the optimizer reaches parameters with this structure and these margins is not
proved here. At the trained checkpoints measured in the experiment's README no training
equation is such an anchor for the equations with a nonzero numerator.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- The training equation in the slot `i` of the split `π`. -/
abbrev trainEq (π : Equiv.Perm (Fin (97 * 96))) (i : Fin 4656) : Equation :=
  trainAt equations (by norm_num : 4656 ≤ 97 * 96) π i

/-- **The splits that cover both orbits**: among the training equations there is one with a
nonzero quotient and one with the quotient zero. -/
def covered : Set (Equiv.Perm (Fin (97 * 96))) :=
  {π | (∃ i, quotient (trainEq π i) ≠ 0) ∧ ∃ i, quotient (trainEq π i) = 0}

/-- The split that trains on the first 4656 equations of the list covers both orbits:
`0 / 1 = 0` is in its slot `0` and `1 / 1 = 1` in its slot `96`. -/
theorem one_mem_covered : (1 : Equiv.Perm (Fin (97 * 96))) ∈ covered :=
  ⟨⟨⟨96, by norm_num⟩, by decide⟩, ⟨⟨0, by norm_num⟩, by decide⟩⟩

/-- **A uniformly random split covers both orbits with positive probability.** -/
theorem uniformPerm_covered_pos : 0 < uniformPerm (97 * 96) covered :=
  uniformPerm_pos_of_mem one_mem_covered

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
theorem exists_not_mem_covered : ∃ π : Equiv.Perm (Fin (97 * 96)), π ∉ covered := by
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

/-- **A uniformly random split fails to cover both orbits with positive probability.** -/
theorem uniformPerm_compl_covered_pos : 0 < uniformPerm (97 * 96) coveredᶜ := by
  obtain ⟨π, hπ⟩ := exists_not_mem_covered
  exact uniformPerm_pos_of_mem (Set.mem_compl hπ)

/-- **Two training answers and the structure of the decoder's logits give every held-out
answer.** Let the logits of the decoder at `θ`, for every held-out equation, be within the
error of its orbit of those of a training equation of that orbit, after the relabelling that
carries the anchor's quotient to its own and one constant; and let the margin of each of the
two training answers be above twice the error of its orbit. Then the held-out accuracy is
`1`. No answer of a held-out equation is assumed. -/
theorem heldout_accuracy_eq_one_of_anchors (θ : Param paperSizes → ℝ)
    (π : Equiv.Perm (Fin (97 * 96))) {i₁ i₀ : Fin 4656} {ε₁ ε₀ : ℝ}
    (ha₀ : quotient (trainEq π i₀) = 0)
    (h₁ : ∀ x ∈ heldout equations 4656 π, quotient x ≠ 0 → ∃ (c : (ZMod 97)ˣ) (κ : ℝ),
      (c : ZMod 97) * quotient (trainEq π i₁) = quotient x ∧
        ∀ t, |answerLogits θ x (scaleToken c t) - answerLogits θ (trainEq π i₁) t - κ| ≤ ε₁)
    (h₀ : ∀ x ∈ heldout equations 4656 π, quotient x = 0 → ∃ (c : (ZMod 97)ˣ) (κ : ℝ),
      ∀ t, |answerLogits θ x (scaleToken c t) - answerLogits θ (trainEq π i₀) t - κ| ≤ ε₀)
    (hm₁ : 2 * ε₁ < margin (answerLogits θ (trainEq π i₁)) (residueToken (trainEq π i₁).1))
    (hm₀ : 2 * ε₀ < margin (answerLogits θ (trainEq π i₀)) (residueToken (trainEq π i₀).1)) :
    accuracy (answerLogits θ) (fun x => residueToken x.1) (heldout equations 4656 π) = 1 := by
  refine (accuracy_eq_one_iff (heldout_nonempty equations (by norm_num) π)).2 fun x hx => ?_
  by_cases hq : quotient x = 0
  · obtain ⟨c, κ, h⟩ := h₀ x hx hq
    exact predict_eq_residueToken_of_anchor (by rw [ha₀, hq, mul_zero]) h hm₀
  · obtain ⟨c, κ, hc, h⟩ := h₁ x hx hq
    exact predict_eq_residueToken_of_anchor hc h hm₁

/-- **The joint witness**: the solver on the split that trains on the first 4656 equations,
with the anchors `1 / 1 = 1` and `0 / 1 = 0` and the error `1/52200` in both orbits. -/
theorem heldout_accuracy_solver :
    accuracy (answerLogits solver) (fun x => residueToken x.1) (heldout equations 4656 1) = 1 := by
  have h96 : quotient (trainEq 1 ⟨96, by norm_num⟩) = 1 := by decide
  refine heldout_accuracy_eq_one_of_anchors solver 1 (i₁ := ⟨96, by norm_num⟩)
    (i₀ := ⟨0, by norm_num⟩) (ε₁ := 1 / 52200) (ε₀ := 1 / 52200) (by decide)
    (fun x _ hx => ⟨Units.mk0 (quotient x) hx, _, ?_, abs_answerLogits_solver_scaleToken ?_⟩)
    (fun x _ hx => ⟨1, _, abs_answerLogits_solver_scaleToken (c := 1) ?_⟩) ?_ ?_
  · rw [Units.val_mk0, h96, mul_one]
  · rw [Units.val_mk0, h96, mul_one]
  · rw [hx]
    exact mul_zero _
  · linarith [solverScale_le_margin (trainEq 1 ⟨96, by norm_num⟩),
      (solverScale_mem (trainEq 1 ⟨96, by norm_num⟩)).1]
  · linarith [solverScale_le_margin (trainEq 1 ⟨0, by norm_num⟩),
      (solverScale_mem (trainEq 1 ⟨0, by norm_num⟩)).1]

/-- At exchanged parameters, the logits of an equation that shows no hidden residue are its
logits with the two hidden tokens renamed. -/
lemma answerLogits_exchange_of_avoids (θ : Param paperSizes → ℝ) {x : Equation} (hx : Avoids x)
    (c : Fin 239) : answerLogits (θ ∘ exchange) x c = answerLogits θ x (hiddenSwap c) := by
  have h := logits_comp_paramPerm (S := paperSizes) (τ := hiddenSwap) (θ := θ)
    (act := relu paperSizes) (read x) (4 : Fin 6) c
  have htok : hiddenSwap ∘ read x = read x := funext (fixed_supervised hx).1
  rw [htok] at h
  exact h

/-- **Fitting every training answer does not give the held-out answers.** There are parameters
of the decoder and a split with all 4656 training equations answered and a held-out equation
not: the solver with the tokens of the residues `1` and `2` exchanged, on a split that shows
neither residue in training. -/
theorem exists_fit_training_wrong_heldout :
    ∃ (θ : Param paperSizes → ℝ) (π : Equiv.Perm (Fin (97 * 96))),
      (∀ i : Fin 4656, predict (answerLogits θ (trainEq π i)) = residueToken (trainEq π i).1) ∧
        ∃ x ∈ heldout equations 4656 π, predict (answerLogits θ x) ≠ residueToken x.1 := by
  obtain ⟨π, hπ⟩ := exists_uncovered
  refine ⟨solver ∘ exchange, π, fun i => ?_, hiddenOne,
    mem_heldout_of_not_avoids hπ (by decide), fun h1 => ?_⟩
  · have hx : Avoids (trainEq π i) := hπ i
    have hfix : hiddenSwap (residueToken (trainEq π i).1) = residueToken (trainEq π i).1 :=
      hiddenSwap_residueToken hx.2.2
    have e : answerLogits (solver ∘ exchange) (trainEq π i) =
        answerLogits solver (trainEq π i) := funext fun c => by
      rw [answerLogits_exchange_of_avoids solver hx, answerLogits_solver, answerLogits_solver,
        markLogits, markLogits]
      exact congrArg _ (if_congr ⟨fun h => hiddenSwap.injective (h.trans hfix.symm),
        fun h => by rw [h, hfix]⟩ rfl rfl)
    rw [e]
    exact predict_answerLogits_solver _
  · exact not_both_predicted solver ⟨predict_answerLogits_solver hiddenTwo, h1⟩

/-! The witnesses: the solver on the split of the first 4656 equations, and `5 / 4 = 50`,
which shows no hidden residue. -/

example := heldout_accuracy_solver

example := answerLogits_exchange_of_avoids solver (x := (50, 3)) (by decide) eosToken

end Transformer.Grokking.Protocol
