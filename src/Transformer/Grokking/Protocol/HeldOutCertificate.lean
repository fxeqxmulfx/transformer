module

public import Transformer.Grokking.Protocol.SolverScale
public import Transformer.Grokking.Protocol.QuotientWitness
public import Transformer.Grokking.Protocol.Refutation
public import Transformer.Grokking.Protocol.OrbitAnchors

/-!
# Held-out answers of the decoder from training answers and the structure of its logits

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: the
decoder, the operation `x / y (mod 97)`, the random half of the equations held out and the
validation accuracy; `experiments/grok_division/README.md`, *The decoders that answer every
equation are not quotient-equivariant*, for the certificate and its measurement.

`heldout_accuracy_eq_one_of_training_anchors` is the certificate on the decoder's own logits.
Its premise gives every held-out equation a training equation, its anchor: the logits of the
equation, read through the relabelling that the scaling of the operands induces, are within
`ε` of the anchor's after one constant, and the margin of the anchor's answer is above `2 ε`.
The relabelling is by a unit `c` with `c · quotient a = quotient x`, which the operands of the
two equations fix when the numerators are not zero: `c = (a'/a) / (b'/b)`. The conclusion is
the held-out accuracy `1`. No held-out answer is a premise, and the structure alone decides
nothing: the decoder with every parameter zero has it exactly and answers no equation
(`predict_answerLogits_zero_ne`).

* With one anchor for each orbit and one error for each, this is
  `heldout_accuracy_eq_one_of_anchors`; the split is then `anchored` (`OrbitAnchors`).
* The premises are consistent on the decoder: `solver` satisfies them on the split that
  trains on the first 4656 equations, with a positive error (`heldout_accuracy_solver`).
* The margins of the training answers do not suffice: there are parameters and a split with
  every training equation answered and a held-out one not
  (`exists_fit_training_wrong_heldout`).

That a run of the optimizer reaches parameters with this structure and these margins is not
proved here. The experiment's README measures both forms on the trained checkpoints.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- **Training answers and the structure of the decoder's logits give every held-out answer.**
Let every held-out equation have a training equation, its anchor, such that the logits of the
decoder at `θ` for the equation, read through a relabelling that carries the anchor's quotient
to its own, are within `ε` of the anchor's after one constant, and the margin of the anchor's
answer is above `2 ε`. Then the held-out accuracy is `1`. The anchor and the error may change
with the equation, and no answer of a held-out equation is assumed. -/
theorem heldout_accuracy_eq_one_of_training_anchors (θ : Param paperSizes → ℝ)
    (π : Equiv.Perm (Fin (97 * 96)))
    (h : ∀ x ∈ heldout equations 4656 π, ∃ (i : Fin 4656) (c : (ZMod 97)ˣ) (κ ε : ℝ),
      (c : ZMod 97) * quotient (trainEq π i) = quotient x ∧
        (∀ t, |answerLogits θ x (scaleToken c t) - answerLogits θ (trainEq π i) t - κ| ≤ ε) ∧
        2 * ε < margin (answerLogits θ (trainEq π i)) (residueToken (trainEq π i).1)) :
    accuracy (answerLogits θ) (fun x => residueToken x.1) (heldout equations 4656 π) = 1 :=
  (accuracy_eq_one_iff (heldout_nonempty equations (by norm_num) π)).2 fun x hx => by
    obtain ⟨i, c, κ, ε, hc, hε, hm⟩ := h x hx
    exact predict_eq_residueToken_of_anchor hc hε hm

/-- **Two training answers suffice when the logits move with the scalings**: one anchor for
the held-out equations with a nonzero quotient and one, with the quotient zero, for the
others, each with a margin above twice the error of its orbit. -/
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
  refine heldout_accuracy_eq_one_of_training_anchors θ π fun x hx => ?_
  by_cases hq : quotient x = 0
  · obtain ⟨c, κ, h⟩ := h₀ x hx hq
    exact ⟨i₀, c, κ, ε₀, by rw [ha₀, hq, mul_zero], h, hm₀⟩
  · obtain ⟨c, κ, hc, h⟩ := h₁ x hx hq
    exact ⟨i₁, c, κ, ε₁, hc, h, hm₁⟩

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

/-! The witnesses: the solver on the split of the first 4656 equations, where every held-out
equation has the anchor `1 / 1 = 1` or `0 / 1 = 0`; and `5 / 4 = 50`, which shows no hidden
residue. -/

example : accuracy (answerLogits solver) (fun x => residueToken x.1)
    (heldout equations 4656 1) = 1 := by
  have h96 : quotient (trainEq 1 ⟨96, by norm_num⟩) = 1 := by decide
  refine heldout_accuracy_eq_one_of_training_anchors solver 1 fun x _ => ?_
  by_cases hx : quotient x = 0
  · have h0 : ((1 : (ZMod 97)ˣ) : ZMod 97) * quotient (trainEq 1 ⟨0, by norm_num⟩) =
        quotient x := by
      rw [hx]
      exact mul_zero _
    exact ⟨⟨0, by norm_num⟩, 1, _, 1 / 52200, h0, abs_answerLogits_solver_scaleToken h0,
      by linarith [solverScale_le_margin (trainEq 1 ⟨0, by norm_num⟩),
        (solverScale_mem (trainEq 1 ⟨0, by norm_num⟩)).1]⟩
  · exact ⟨⟨96, by norm_num⟩, Units.mk0 (quotient x) hx, _, 1 / 52200,
      by rw [Units.val_mk0, h96, mul_one],
      abs_answerLogits_solver_scaleToken (by rw [Units.val_mk0, h96, mul_one]),
      by linarith [solverScale_le_margin (trainEq 1 ⟨96, by norm_num⟩),
        (solverScale_mem (trainEq 1 ⟨96, by norm_num⟩)).1]⟩

example := heldout_accuracy_solver

example := answerLogits_exchange_of_avoids solver (x := (50, 3)) (by decide) eosToken

end Transformer.Grokking.Protocol
