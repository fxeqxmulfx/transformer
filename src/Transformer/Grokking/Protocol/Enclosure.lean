module

public import Transformer.Grokking.Protocol.HeldOutCertificate

/-!
# Enclosures of the logits decide answers and give the certificate

Source: `experiments/grok_division/src/enclosure.rs` and `src/bin/quotient.rs` (`decided`,
`sure_margin`, `sure_slack`); `experiments/grok_division/README.md`, *Enclosures bound the
rounding of the forward pass*; arXiv:2201.02177v1 (Power et al.), appendix *Additional
experimental details*, for the decoder and the validation accuracy.

The logits of the decoder are real numbers, and a program computes them with rounding. An
*enclosure* of a row `u` is two rows with `lo c ≤ u c ≤ hi c` in every class; a program that
rounds every operation outward computes one. Tests on its ends decide what the real row does:

* a class whose lower end is above every other upper end is predicted, one whose upper end is
  below another lower end is not (`predict_eq_of_enclosure`, `predict_ne_of_enclosure`), and
  labels all decided so give the accuracy `1` (`accuracy_eq_one_of_decided`);
* the margin of a class is at least its lower end less the largest other upper end
  (`le_margin_of_enclosure`);
* two rows, one read through a relabelling, differ by a constant up to half the spread of the
  differences of their ends (`abs_sub_le_of_enclosure`).

So the test of `quotient --enclose` on a held-out equation and its anchor gives the premise of
the certificate of `HeldOutCertificate` for the real logits (`enclosure_gives_certificate`),
and passed by every held-out equation it gives the held-out accuracy `1`
(`heldout_accuracy_eq_one_of_enclosed_anchors`).

That the rows a program printed enclose the logits is an argument about the program and the
processor's arithmetic, made in `enclosure.rs`. Here it is a hypothesis: Lean does not check it.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

section Rows

variable {C : Type*} [Fintype C] [LinearOrder C] [Nonempty C]

/-- **An enclosure decides a prediction** (`quotient.rs`, `decided`): a class whose lower end
is above the upper end of every other class is predicted. -/
theorem predict_eq_of_enclosure {u lo hi : C → ℝ} {y : C}
    (hu : ∀ c, lo c ≤ u c ∧ u c ≤ hi c) (h : ∀ c, c ≠ y → hi c < lo y) : predict u = y :=
  predict_eq_iff.2 ⟨fun c => by
      by_cases hc : c = y <;> [rw [hc]; linarith [h c hc, (hu c).2, (hu y).1]],
    fun c hc => by linarith [h c hc.ne, (hu c).2, (hu y).1]⟩

/-- **An enclosure refutes a prediction** (`quotient.rs`, `decided`): a class is not predicted
if the lower end of another class is above its upper end. -/
theorem predict_ne_of_enclosure {u lo hi : C → ℝ} {y c : C}
    (hu : ∀ c, lo c ≤ u c ∧ u c ≤ hi c) (h : hi y < lo c) : predict u ≠ y := fun e => by
  linarith [e ▸ le_predict u c, (hu c).1, (hu y).2]

/-- **An enclosure that decides every label of a set gives the accuracy `1`** (`quotient.rs`,
the counts `training` and `heldout` of the field `enclosed`). -/
theorem accuracy_eq_one_of_decided {X : Type*} {f lo hi : X → C → ℝ} {y : X → C}
    {H : Finset X} (hH : H.Nonempty) (hf : ∀ x c, lo x c ≤ f x c ∧ f x c ≤ hi x c)
    (h : ∀ x ∈ H, ∀ c, c ≠ y x → hi x c < lo x (y x)) : accuracy f y H = 1 :=
  (accuracy_eq_one_iff hH).2 fun x hx => predict_eq_of_enclosure (hf x) (h x hx)

omit [Fintype C] [LinearOrder C] [Nonempty C] in
/-- **A lower bound of a margin from an enclosure** (`quotient.rs`, `sure_margin`): the lower
end of the class less the upper end of any other. -/
lemma le_margin_of_enclosure {u lo hi : C → ℝ} {y : C} {m : ℝ}
    (hu : ∀ c, lo c ≤ u c ∧ u c ≤ hi c) (hm : ∀ c, c ≠ y → m ≤ lo y - hi c)
    (hne : ∃ c, c ≠ y) : m ≤ margin u y := by
  obtain ⟨c₀, hc₀⟩ := hne
  have : Nonempty {c // c ≠ y} := ⟨⟨c₀, hc₀⟩⟩
  exact le_ciInf fun c => by linarith [hm c c.2, (hu y).1, (hu c).2]

omit [Fintype C] [LinearOrder C] [Nonempty C] in
/-- **Enclosures of two rows bound the error of one against the other** (`quotient.rs`,
`sure_slack`). If `D₀` is below every lower end of `v`, read through `π`, less the upper end
of `u`, and `D₁` above every upper end less lower end, then `v` read through `π` is within
`(D₁ - D₀) / 2` of `u` after the constant `(D₀ + D₁) / 2` is taken off. -/
lemma abs_sub_le_of_enclosure {u v ul uh vl vh : C → ℝ} {π : C ≃ C} {D₀ D₁ : ℝ}
    (hu : ∀ c, ul c ≤ u c ∧ u c ≤ uh c) (hv : ∀ c, vl c ≤ v c ∧ v c ≤ vh c)
    (h₀ : ∀ c, D₀ ≤ vl (π c) - uh c) (h₁ : ∀ c, vh (π c) - ul c ≤ D₁) (c : C) :
    |v (π c) - u c - (D₀ + D₁) / 2| ≤ (D₁ - D₀) / 2 := by
  rw [abs_le]
  constructor <;> linarith [h₀ c, h₁ c, (hu c).1, (hu c).2, (hv (π c)).1, (hv (π c)).2]

end Rows

/-- **The test on the ends of two enclosures gives the premise of the certificate**
(`quotient.rs`, `sure_slack`). Let `lo` and `hi` enclose the logits `L`. For an equation `x`,
an anchor `a` and a relabelling `c`, let `D₀` be below and `D₁` above the differences of the
ends as in `abs_sub_le_of_enclosure`, and `m` below the lower end of the anchor's answer less
every other upper end. If `D₁ - D₀ < m`, the row of `x` is within some `ε` of the anchor's
after one constant, and the anchor's margin is above `2 ε`. -/
theorem enclosure_gives_certificate {L lo hi : Equation → Fin 239 → ℝ} {a x : Equation}
    {c : (ZMod 97)ˣ} {D₀ D₁ m : ℝ} (hL : ∀ z t, lo z t ≤ L z t ∧ L z t ≤ hi z t)
    (h₀ : ∀ t, D₀ ≤ lo x (scaleToken c t) - hi a t)
    (h₁ : ∀ t, hi x (scaleToken c t) - lo a t ≤ D₁)
    (hm : ∀ t, t ≠ residueToken a.1 → m ≤ lo a (residueToken a.1) - hi a t)
    (h : D₁ - D₀ < m) :
    ∃ κ ε : ℝ, (∀ t, |L x (scaleToken c t) - L a t - κ| ≤ ε) ∧
      2 * ε < margin (L a) (residueToken a.1) :=
  ⟨(D₀ + D₁) / 2, (D₁ - D₀) / 2, abs_sub_le_of_enclosure (hL a) (hL x) h₀ h₁, by
    linarith [le_margin_of_enclosure (hL a) hm
      ⟨eosToken, (residueToken_ne_eosToken a.1).symm⟩]⟩

/-- **Held-out accuracy from an enclosure of the decoder's logits.** Let `lo` and `hi` enclose
the logits of the decoder at `θ`, and let every held-out equation have a training equation
with which it passes the test of `enclosure_gives_certificate`. Then the held-out accuracy of
the decoder in exact reals is `1`. -/
theorem heldout_accuracy_eq_one_of_enclosed_anchors (θ : Param paperSizes → ℝ)
    (π : Equiv.Perm (Fin (97 * 96))) {lo hi : Equation → Fin 239 → ℝ}
    (hL : ∀ z t, lo z t ≤ answerLogits θ z t ∧ answerLogits θ z t ≤ hi z t)
    (h : ∀ x ∈ heldout equations 4656 π, ∃ (i : Fin 4656) (c : (ZMod 97)ˣ) (D₀ D₁ m : ℝ),
      (c : ZMod 97) * quotient (trainEq π i) = quotient x ∧
        (∀ t, D₀ ≤ lo x (scaleToken c t) - hi (trainEq π i) t) ∧
        (∀ t, hi x (scaleToken c t) - lo (trainEq π i) t ≤ D₁) ∧
        (∀ t, t ≠ residueToken (trainEq π i).1 →
          m ≤ lo (trainEq π i) (residueToken (trainEq π i).1) - hi (trainEq π i) t) ∧
        D₁ - D₀ < m) :
    accuracy (answerLogits θ) (fun x => residueToken x.1) (heldout equations 4656 π) = 1 :=
  heldout_accuracy_eq_one_of_training_anchors θ π fun x hx => by
    obtain ⟨i, c, D₀, D₁, m, hc, h₀, h₁, hm, hlt⟩ := h x hx
    obtain ⟨κ, ε, hε, hmar⟩ := enclosure_gives_certificate hL h₀ h₁ hm hlt
    exact ⟨i, c, κ, ε, hc, hε, hmar⟩

/-! The witnesses: the rows `(0, 2)` and `(1/2, 3/2)` of `MarginTransfer` enclosed within
`1/4`, and the logits of `solver` enclosed within `10⁻⁶`, on the split that trains on the
first 4656 equations. -/

example : predict nearRow = true :=
  predict_eq_of_enclosure (lo := fun c => nearRow c - 1 / 4) (hi := fun c => nearRow c + 1 / 4)
    (fun c => ⟨by linarith, by linarith⟩) fun c hc => by
      cases c <;> [norm_num [nearRow]; exact absurd rfl hc]

example : predict nearRow ≠ false :=
  predict_ne_of_enclosure (c := true) (lo := fun c => nearRow c - 1 / 4)
    (hi := fun c => nearRow c + 1 / 4) (fun c => ⟨by linarith, by linarith⟩)
    (by norm_num [nearRow])

example : (3 / 2 : ℝ) ≤ margin anchorRow true :=
  le_margin_of_enclosure (lo := fun c => anchorRow c - 1 / 4) (hi := fun c => anchorRow c + 1 / 4)
    (fun c => ⟨by linarith, by linarith⟩)
    (fun c hc => by cases c <;> [norm_num [anchorRow]; exact absurd rfl hc]) ⟨false, by simp⟩

example (c : Bool) : |nearRow (Equiv.refl Bool c) - anchorRow c - (-1 + 1) / 2| ≤ (1 - -1) / 2 :=
  abs_sub_le_of_enclosure (ul := fun c => anchorRow c - 1 / 4) (uh := fun c => anchorRow c + 1 / 4)
    (vl := fun c => nearRow c - 1 / 4) (vh := fun c => nearRow c + 1 / 4)
    (fun c => ⟨by linarith, by linarith⟩) (fun c => ⟨by linarith, by linarith⟩)
    (fun c => by cases c <;> norm_num [nearRow, anchorRow])
    (fun c => by cases c <;> norm_num [nearRow, anchorRow]) c

/-- The logits of the solver, enclosed within `10⁻⁶`, pass the test against any anchor of the
orbit: the differences of the ends lie within `1/52200 + 2 · 10⁻⁶` of a constant, and the
margin of the anchor has the lower bound `1/900 - 2 · 10⁻⁶`. -/
lemma solver_enclosure_test {a x : Equation} {c : (ZMod 97)ˣ}
    (hc : (c : ZMod 97) * quotient a = quotient x) :
    ∃ D₀ D₁ m : ℝ,
      (∀ t, D₀ ≤ answerLogits solver x (scaleToken c t) - 1 / 10 ^ 6 -
        (answerLogits solver a t + 1 / 10 ^ 6)) ∧
      (∀ t, answerLogits solver x (scaleToken c t) + 1 / 10 ^ 6 -
        (answerLogits solver a t - 1 / 10 ^ 6) ≤ D₁) ∧
      (∀ t, t ≠ residueToken a.1 → m ≤ answerLogits solver a (residueToken a.1) - 1 / 10 ^ 6 -
        (answerLogits solver a t + 1 / 10 ^ 6)) ∧ D₁ - D₀ < m := by
  refine ⟨(solverScale x - solverScale a) / 2 - 1 / 52200 - 2 / 10 ^ 6,
    (solverScale x - solverScale a) / 2 + 1 / 52200 + 2 / 10 ^ 6, 1 / 900 - 2 / 10 ^ 6,
    fun t => ?_, fun t => ?_, fun t ht => ?_, by linarith⟩
  · linarith [(abs_le.1 (abs_answerLogits_solver_scaleToken hc t)).1]
  · linarith [(abs_le.1 (abs_answerLogits_solver_scaleToken hc t)).2]
  · linarith [margin_le (answerLogits solver a) ht, solverScale_le_margin a,
      (solverScale_mem a).1]

example : ∃ κ ε : ℝ, (∀ t, |answerLogits solver (0, 9) (scaleToken 1 t) -
    answerLogits solver (0, 4) t - κ| ≤ ε) ∧
    2 * ε < margin (answerLogits solver (0, 4)) (residueToken 0) := by
  obtain ⟨D₀, D₁, m, h₀, h₁, hm, h⟩ :=
    solver_enclosure_test (a := (0, 4)) (x := (0, 9)) (c := 1) (by decide)
  exact enclosure_gives_certificate (lo := fun z t => answerLogits solver z t - 1 / 10 ^ 6)
    (hi := fun z t => answerLogits solver z t + 1 / 10 ^ 6)
    (fun z t => ⟨by linarith, by linarith⟩) h₀ h₁ hm h

example : accuracy (answerLogits solver) (fun x => residueToken x.1)
    (heldout equations 4656 1) = 1 :=
  accuracy_eq_one_of_decided (heldout_nonempty equations (by norm_num) 1)
    (lo := fun z t => answerLogits solver z t - 1 / 10 ^ 6)
    (hi := fun z t => answerLogits solver z t + 1 / 10 ^ 6)
    (fun z t => ⟨by linarith, by linarith⟩) fun x _ t ht => by
      linarith [margin_le (answerLogits solver x) ht, solverScale_le_margin x,
        (solverScale_mem x).1]

example : accuracy (answerLogits solver) (fun x => residueToken x.1)
    (heldout equations 4656 1) = 1 :=
  heldout_accuracy_eq_one_of_enclosed_anchors solver 1
    (lo := fun z t => answerLogits solver z t - 1 / 10 ^ 6)
    (hi := fun z t => answerLogits solver z t + 1 / 10 ^ 6)
    (fun z t => ⟨by linarith, by linarith⟩) fun x hx => by
      obtain ⟨i, c, hc⟩ := (mem_anchored_iff 1).1 one_mem_anchored x hx
      obtain ⟨D₀, D₁, m, h⟩ := solver_enclosure_test hc
      exact ⟨i, c, D₀, D₁, m, hc, h⟩

end Transformer.Grokking.Protocol
