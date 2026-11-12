module

public import Transformer.Grokking.Protocol.Division
public import Transformer.Grokking.Protocol.MarginTransfer

/-!
# Two training answers and the structure of the logits give every answer of the division

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, the
operation `x / y (mod p)` with `p = 97` and the validation accuracy;
`experiments/grok_division/README.md`, *The decoders that answer every equation are not
quotient-equivariant*; `experiments/grok_division/src/bin/quotient.rs`.

Scaling the numerator by `s` and the denominator by `t`, both nonzero, sends the equation
`a / b = c` to `(s a) / (t b) = (s / t) c`. The logits of a model move with the scalings when
those of the second equation are those of the first with every residue `r` relabelled
`(s / t) r` and the other tokens kept (`scaleToken`). The equations with a nonzero numerator,
which are those with a nonzero quotient (`numerator_eq_zero_iff`), are one orbit of the
scalings, and those with numerator zero are the other.

* If the logits of the equation `x`, read through the relabelling by a unit `c` with
  `c · quotient a = quotient x`, are within `ε` of those of the anchor `a` in every token after
  one constant is taken off, and the margin of the anchor's answer is above `2 ε`, then `x` is
  answered (`predict_eq_residueToken_of_anchor`).
* So one anchor with a nonzero quotient and one with the quotient zero answer all 9312
  equations when every equation is within the error of its orbit, `ε₁` or `ε₀`, of the anchor
  of its orbit, and the margin of each anchor is above twice the error of its orbit
  (`predict_eq_residueToken_of_two_anchors`). For the quotient zero any unit serves, since
  `c · 0 = 0`.

The hypotheses are on the logits of the equations and on the answers of the two anchors. No
answer of another equation is assumed, and the classes are all 239 tokens: a token that is not
a residue competes too. The logits are any function of the equation, so the theorems apply to
`answerLogits θ` of the decoder; they do not say that a trained `θ` satisfies them, and at the
checkpoints measured in the experiment's README no training equation with a nonzero quotient
is an anchor that does.

The witness here is `markLogits`, which marks the answer; `QuotientWitness` has one with a
positive error and the decoder with all parameters zero, which has the symmetry and no answer.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Grokking.DivisionOrbits

open scoped Classical in
/-- **The relabelling of the tokens by a unit** `c` of the field of 97 elements: the token of
the residue `r` goes to that of `c r`, and the other 142 tokens stay
(`experiments/grok_division/src/bin/quotient.rs`, `relabel`). -/
noncomputable def scaleToken (c : (ZMod 97)ˣ) : Equiv.Perm (Fin 239) :=
  (Units.mulLeft c : Equiv.Perm (ZMod 97)).extendDomain
    (Equiv.ofInjective residueToken residueToken_injective)

lemma scaleToken_residueToken (c : (ZMod 97)ˣ) (r : ZMod 97) :
    scaleToken c (residueToken r) = residueToken ((c : ZMod 97) * r) := by
  classical
  exact Equiv.Perm.extendDomain_apply_image (Units.mulLeft c)
    (Equiv.ofInjective residueToken residueToken_injective) r

lemma scaleToken_of_not_residue (c : (ZMod 97)ˣ) {t : Fin 239} (h : ∀ r, residueToken r ≠ t) :
    scaleToken c t = t := by
  classical
  exact Equiv.Perm.extendDomain_apply_not_subtype _ _ fun ⟨r, hr⟩ => h r hr

lemma residueToken_ne_eosToken (r : Fin 97) : residueToken r ≠ eosToken := fun h => by
  simp only [eosToken, residueToken, Fin.ext_iff] at h
  omega

/-- The quotient `c` of an equation `a / b = c`, in the field of 97 elements. -/
def quotient (x : Equation) : ZMod 97 := x.1

/-- **The numerator is zero exactly when the quotient is**: the two orbits of the scalings are
the equations with the quotient zero and the others. -/
theorem numerator_eq_zero_iff (x : Equation) : numerator x = 0 ↔ quotient x = 0 := by
  have hq : quotient x = 0 ↔ (x.1 : ℕ) = 0 := (Fin.val_eq_zero_iff (a := x.1)).symm
  rw [hq, ← Fin.val_eq_zero_iff]
  change ((x.2 : ℕ) + 1) * (x.1 : ℕ) % 97 = 0 ↔ (x.1 : ℕ) = 0
  rw [← Nat.dvd_iff_mod_eq_zero]
  refine ⟨fun h => ?_, fun h => by rw [h, mul_zero]; exact dvd_zero _⟩
  rcases (Nat.Prime.dvd_mul study_prime97).1 h with h | h
  · have := Nat.le_of_dvd (Nat.succ_pos _) h
    have := x.2.isLt
    omega
  · exact Nat.eq_zero_of_dvd_of_lt h x.1.isLt

/-- **One anchor certifies an equation of its orbit**
(`experiments/grok_division/README.md`, the section named above). If the logits of `x`, read
through the relabelling by `c` with `c · quotient a = quotient x`, are within `ε` of those of
`a` in every token after the constant `κ` is taken off, and the answer of `a` has a margin
above `2 ε`, then the answer of `x` is predicted. -/
theorem predict_eq_residueToken_of_anchor {L : Equation → Fin 239 → ℝ} {a x : Equation}
    {c : (ZMod 97)ˣ} {κ ε : ℝ} (hc : (c : ZMod 97) * quotient a = quotient x)
    (h : ∀ t, |L x (scaleToken c t) - L a t - κ| ≤ ε)
    (hm : 2 * ε < margin (L a) (residueToken a.1)) : predict (L x) = residueToken x.1 := by
  rw [predict_eq_of_abs_sub_le h hm]
  exact (scaleToken_residueToken c (quotient a)).trans (congrArg residueToken hc)

/-- **Two anchors certify every equation** (the same section): an anchor `a₁` for the
equations with a nonzero quotient, with a margin above `2 ε₁`, and an anchor `a₀` with the
quotient zero for the others, with a margin above `2 ε₀`, when every equation is within the
error of its orbit of the anchor of its orbit, after the relabelling that carries the anchor's
quotient to its own and one constant. -/
theorem predict_eq_residueToken_of_two_anchors {L : Equation → Fin 239 → ℝ} {a₁ a₀ : Equation}
    {ε₁ ε₀ : ℝ} (ha₀ : quotient a₀ = 0)
    (h₁ : ∀ x : Equation, quotient x ≠ 0 → ∃ (c : (ZMod 97)ˣ) (κ : ℝ),
      (c : ZMod 97) * quotient a₁ = quotient x ∧ ∀ t, |L x (scaleToken c t) - L a₁ t - κ| ≤ ε₁)
    (h₀ : ∀ x : Equation, quotient x = 0 → ∃ (c : (ZMod 97)ˣ) (κ : ℝ),
      ∀ t, |L x (scaleToken c t) - L a₀ t - κ| ≤ ε₀)
    (hm₁ : 2 * ε₁ < margin (L a₁) (residueToken a₁.1))
    (hm₀ : 2 * ε₀ < margin (L a₀) (residueToken a₀.1)) (x : Equation) :
    predict (L x) = residueToken x.1 := by
  by_cases hx : quotient x = 0
  · obtain ⟨c, κ, h⟩ := h₀ x hx
    exact predict_eq_residueToken_of_anchor (by rw [ha₀, hx, mul_zero]) h hm₀
  · obtain ⟨c, κ, hc, h⟩ := h₁ x hx
    exact predict_eq_residueToken_of_anchor hc h hm₁

/-! The witness: the logits that mark the answer. They move with the scalings exactly, and
every answer is certified from the anchors `3 / 5 = 20` and `0 / 5 = 0`. -/

/-- The logits that mark the answer: `1` at its token and `0` at the others. -/
def markLogits (x : Equation) (t : Fin 239) : ℝ := if t = residueToken x.1 then 1 else 0

lemma one_le_margin_markLogits (x : Equation) :
    1 ≤ margin (markLogits x) (residueToken x.1) := by
  have : Nonempty {c : Fin 239 // c ≠ residueToken x.1} :=
    ⟨⟨eosToken, (residueToken_ne_eosToken x.1).symm⟩⟩
  refine le_ciInf fun c => ?_
  simp [markLogits, c.2]

/-- The marked logits move with the quotient exactly. -/
lemma markLogits_scaleToken {a x : Equation} {c : (ZMod 97)ˣ}
    (hc : (c : ZMod 97) * quotient a = quotient x) (t : Fin 239) :
    |markLogits x (scaleToken c t) - markLogits a t - 0| ≤ 0 := by
  have hx : residueToken x.1 = scaleToken c (residueToken a.1) :=
    ((scaleToken_residueToken c (quotient a)).trans (congrArg residueToken hc)).symm
  simp [markLogits, hx]

/-- The unit that carries the quotient `20` to a nonzero quotient. -/
lemma exists_unit_mul_twenty {x : Equation} (hx : quotient x ≠ 0) :
    ∃ c : (ZMod 97)ˣ, (c : ZMod 97) * quotient (20, 4) = quotient x :=
  have h20 : (20 : ZMod 97) ≠ 0 := by decide
  ⟨Units.mk0 (quotient x / 20) (div_ne_zero hx h20), div_mul_cancel₀ _ h20⟩

example : numerator (0, 4) = 0 := (numerator_eq_zero_iff (0, 4)).2 rfl

example := scaleToken_of_not_residue 1 (t := eosToken) residueToken_ne_eosToken

example := exists_unit_mul_twenty (x := (40, 6)) (by decide)

/-- `2 · 20 = 40`: the anchor `3 / 5 = 20` certifies `86 / 7 = 40`. -/
example : predict (markLogits (40, 6)) = residueToken 40 :=
  have h2 : (2 : ZMod 97) ≠ 0 := by decide
  have hc : ((Units.mk0 2 h2 : (ZMod 97)ˣ) : ZMod 97) * quotient (20, 4) = quotient (40, 6) := by
    rw [Units.val_mk0]
    decide
  predict_eq_residueToken_of_anchor (L := markLogits) (ε := 0) hc (markLogits_scaleToken hc)
    (by linarith [one_le_margin_markLogits (20, 4)])

/-- Every answer of the marked logits is certified from the anchors `3 / 5 = 20` and
`0 / 5 = 0`. -/
example (x : Equation) : predict (markLogits x) = residueToken x.1 :=
  predict_eq_residueToken_of_two_anchors (L := markLogits) (a₁ := (20, 4)) (a₀ := (0, 4))
    (ε₁ := 0) (ε₀ := 0) rfl
    (fun _ hx => (exists_unit_mul_twenty hx).elim fun c hc => ⟨c, 0, hc, markLogits_scaleToken hc⟩)
    (fun _ hx => ⟨1, 0, markLogits_scaleToken (by rw [hx]; exact mul_zero _)⟩)
    (by linarith [one_le_margin_markLogits (20, 4)])
    (by linarith [one_le_margin_markLogits (0, 4)]) x

end Transformer.Grokking.Protocol
