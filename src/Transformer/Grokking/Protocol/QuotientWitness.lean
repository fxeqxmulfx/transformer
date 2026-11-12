module

public import Transformer.Grokking.Protocol.QuotientTransfer

/-!
# Witnesses of the two-anchor certificate: a positive error, and the symmetry without an answer

Source: `experiments/grok_division/README.md`, *The decoders that answer every equation are not
quotient-equivariant*; `experiments/grok_division/src/bin/quotient.rs`, the test
`equivariant_logits_are_certified_and_a_bend_is_measured`; arXiv:2201.02177v1 (Power et al.),
appendix *Additional experimental details*, for the decoder and the division.

* `bentLogits` marks the answer and raises the token `<|eos|>` by `1/4` in the one equation
  `86 / 7 = 40`. Every equation is within `1/8` of an anchor of its orbit that is not the bent
  one, after the relabelling and a constant (`abs_bentLogits_scaleToken`). The anchors
  `3 / 5 = 20` and `0 / 5 = 0` have a margin of at least `1 > 2 · 1/8`, so every answer is
  certified (`predict_bentLogits`), although the bent equation has a margin of `3/4` only
  (`margin_bentLogits_le`): the certificate holds with a positive error.
* The decoder with every parameter zero gives every token the logit `0`
  (`answerLogits_zero`). Its logits move with every scaling exactly
  (`abs_answerLogits_zero_scaleToken`), the answers have the margin `0`
  (`margin_answerLogits_zero`), and it answers no equation (`predict_answerLogits_zero_ne`):
  the symmetry of the logits gives nothing without the margins of the anchors.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

lemma scaleToken_eosToken (c : (ZMod 97)ˣ) : scaleToken c eosToken = eosToken :=
  scaleToken_of_not_residue c residueToken_ne_eosToken

lemma scaleToken_eq_eosToken_iff (c : (ZMod 97)ˣ) (t : Fin 239) :
    scaleToken c t = eosToken ↔ t = eosToken := by
  conv_lhs => rw [← scaleToken_eosToken c]
  exact (scaleToken c).apply_eq_iff_eq

/-- The marked logits, with the token `<|eos|>` raised by `1/4` in the equation
`86 / 7 = 40`. -/
noncomputable def bentLogits (x : Equation) (t : Fin 239) : ℝ :=
  markLogits x t + if x = (40, 6) ∧ t = eosToken then 1 / 4 else 0

lemma bentLogits_of_ne {a : Equation} (ha : a ≠ (40, 6)) : bentLogits a = markLogits a := by
  funext t
  simp [bentLogits, ha]

/-- **Every equation is within `1/8` of an anchor of its orbit that is not bent**, after the
relabelling and a constant: `1/8` for the bent equation and `0` for the others. -/
lemma abs_bentLogits_scaleToken {a x : Equation} {c : (ZMod 97)ˣ} (ha : a ≠ (40, 6))
    (hc : (c : ZMod 97) * quotient a = quotient x) :
    ∃ κ : ℝ, ∀ t, |bentLogits x (scaleToken c t) - bentLogits a t - κ| ≤ 1 / 8 := by
  have hm : ∀ t, markLogits x (scaleToken c t) = markLogits a t := fun t => by
    have := abs_nonpos_iff.1 (markLogits_scaleToken hc t)
    linarith
  by_cases hx : x = (40, 6)
  · refine ⟨1 / 8, fun t => ?_⟩
    rw [bentLogits_of_ne ha, bentLogits, hm, if_congr (and_iff_right hx) rfl rfl,
      if_congr (scaleToken_eq_eosToken_iff c t) rfl rfl]
    by_cases ht : t = eosToken
    · rw [ite_eq_left ht]
      norm_num [abs_le]
    · rw [ite_eq_right ht]
      norm_num [abs_le]
  · refine ⟨0, fun t => ?_⟩
    rw [bentLogits_of_ne ha, bentLogits_of_ne hx, hm]
    norm_num

lemma one_le_margin_bentLogits {a : Equation} (ha : a ≠ (40, 6)) :
    1 ≤ margin (bentLogits a) (residueToken a.1) := by
  rw [bentLogits_of_ne ha]
  exact one_le_margin_markLogits a

/-- The bent equation has a margin of at most `3/4`: below the margin `1` of the anchors. -/
lemma margin_bentLogits_le : margin (bentLogits (40, 6)) (residueToken 40) ≤ 3 / 4 := by
  have h := margin_le (bentLogits (40, 6)) (y := residueToken 40) (c := eosToken)
    (residueToken_ne_eosToken 40).symm
  have h1 : bentLogits (40, 6) (residueToken 40) = 1 := by
    simp [bentLogits, markLogits, residueToken_ne_eosToken]
  have h2 : bentLogits (40, 6) eosToken = 1 / 4 := by
    simp [bentLogits, markLogits, (residueToken_ne_eosToken _).symm]
  rw [h1, h2] at h
  linarith

/-- **Every answer of the bent logits is certified** from the anchors `3 / 5 = 20` and
`0 / 5 = 0` with the error `1/8` (`experiments/grok_division/src/bin/quotient.rs`, the test
named above). -/
theorem predict_bentLogits (x : Equation) : predict (bentLogits x) = residueToken x.1 :=
  predict_eq_residueToken_of_two_anchors (L := bentLogits) (a₁ := (20, 4)) (a₀ := (0, 4))
    (ε₁ := 1 / 8) (ε₀ := 1 / 8) rfl
    (fun _ hx => (exists_unit_mul_twenty hx).elim fun c hc =>
      (abs_bentLogits_scaleToken (by decide) hc).elim fun κ h => ⟨c, κ, hc, h⟩)
    (fun _ hx => (abs_bentLogits_scaleToken (a := (0, 4)) (c := 1) (by decide)
      (by rw [hx]; exact mul_zero _)).elim fun κ h => ⟨1, κ, h⟩)
    (by linarith [one_le_margin_bentLogits (a := (20, 4)) (by decide)])
    (by linarith [one_le_margin_bentLogits (a := (0, 4)) (by decide)]) x

/-- The decoder with every parameter zero gives every token the logit `0`. -/
lemma answerLogits_zero (x : Equation) (t : Fin 239) : answerLogits (fun _ => 0) x t = 0 := by
  simp [answerLogits, logits]

/-- The logits of the decoder with every parameter zero move with every scaling exactly: the
hypothesis of `predict_eq_residueToken_of_anchor` holds for every anchor, every unit and
`ε = 0`. -/
lemma abs_answerLogits_zero_scaleToken (a x : Equation) (c : (ZMod 97)ˣ) (t : Fin 239) :
    |answerLogits (fun _ => 0) x (scaleToken c t) - answerLogits (fun _ => 0) a t - 0| ≤ 0 := by
  simp [answerLogits_zero]

/-- No answer of the decoder with every parameter zero has a positive margin. -/
lemma margin_answerLogits_zero (x : Equation) :
    margin (answerLogits (fun _ => 0) x) (residueToken x.1) = 0 := by
  have : Nonempty {c : Fin 239 // c ≠ residueToken x.1} :=
    ⟨⟨eosToken, (residueToken_ne_eosToken x.1).symm⟩⟩
  simp [margin, answerLogits_zero]

/-- **The symmetry without a margin answers nothing**: the decoder with every parameter zero
has logits that move with every scaling, and it predicts the token `<|eos|>` for every
equation. -/
theorem predict_answerLogits_zero_ne (x : Equation) :
    predict (answerLogits (fun _ => 0) x) ≠ residueToken x.1 := by
  have h0 : answerLogits (fun _ => 0) x = fun _ => 0 := funext (answerLogits_zero x)
  have h : predict (fun _ : Fin 239 => (0 : ℝ)) = eosToken :=
    predict_eq_iff.2 ⟨fun _ => le_rfl, fun c hc => absurd hc (by simp [eosToken])⟩
  rw [h0, h]
  exact (residueToken_ne_eosToken x.1).symm

example := scaleToken_eq_eosToken_iff 1 eosToken

example := bentLogits_of_ne (a := (20, 4)) (by decide)

example := abs_bentLogits_scaleToken (a := (0, 4)) (x := (0, 9)) (c := 1) (by decide)
  (by decide)

example := one_le_margin_bentLogits (a := (20, 4)) (by decide)

example := predict_bentLogits (40, 6)

example := predict_answerLogits_zero_ne (20, 4)

end Transformer.Grokking.Protocol
