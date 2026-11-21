module

public import Mathlib.Algebra.Field.Rat
public import Mathlib.Algebra.Group.Conj
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.NormNum
public import Transformer.GainMatroid.Section2_Gains

/-!
# Conjectures 1 and 2 and joints

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §1.2. Zaslavsky's Conjectures 1 and 2 say that the
canonical frame (resp. lift) matrices of two gain functions on a graph are projectively equivalent
if and only if the gain functions are switching (resp. switching-and-scaling) equivalent. The source
states them "leaving aside some technicalities" and then shows that joints break them: switching
cannot change the gain of a loop, while column scaling can.

As literally stated both conjectures are false: `conjecture1_fails` and `conjecture2_fails` are
counterexamples. `projEquiv_and_not_switchEquiv_of_joint` is the source's general construction.

A joint is an unbalanced loop, that is, a loop whose gain is not the identity. The source also asks
that "neither `φ(e')` nor `ψ(e')` is zero". Multiplicative gains are never zero, and what its
argument uses is that neither is the identity: a loop of gain one has a zero column, which no
scaling makes nonzero. The theorem uses this corrected condition, and `not_projEquiv_of_gain_one`
shows that it cannot be dropped.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {V E Γ F : Type*}

/-- Switching conjugates the gain on a loop (§1.2). -/
@[to_additive isAddConj_addSwitch_of_loop /-- Switching conjugates the gain on a loop (§1.2). -/]
theorem isConj_switch_of_loop [Group Γ] (G : OrientedGraph V E) (η : V → Γ) (γ : E → Γ) {e : E}
    (h : G.tail e = G.head e) : IsConj (γ e) (switch G η γ e) := by
  rw [switch_apply, h]
  exact isConj_iff.2 ⟨(η (G.head e))⁻¹, by simp⟩

/-- Over an abelian group, switching leaves the gain on a loop unchanged (§1.2). -/
@[to_additive addSwitch_eq_of_loop
  /-- Over an abelian group, switching leaves the gain on a loop unchanged (§1.2). -/]
theorem switch_eq_of_loop [CommGroup Γ] (G : OrientedGraph V E) (η : V → Γ) (γ : E → Γ) {e : E}
    (h : G.tail e = G.head e) : switch G η γ e = γ e := by
  rw [switch_apply, h, mul_comm (η (G.head e))⁻¹, inv_mul_cancel_right]

/-- The source's construction before Theorem 1 (§1.2). Let `φ` and `ψ` be `Fˣ`-gain functions that
agree on every edge that is not a joint of both, meaning a loop with non-identity gain in each. If
they differ on some edge `e'`, then their canonical frame matrices are projectively equivalent
although they are not switching equivalent. That `e'` is a joint follows from the hypotheses. The
source's wording asks that neither gain on `e'` be zero; see the module docstring for the change. -/
theorem projEquiv_and_not_switchEquiv_of_joint [DecidableEq V] [Field F]
    (G : OrientedGraph V E) {φ ψ : E → Fˣ} {e' : E}
    (hagree : ∀ e, ¬(G.tail e = G.head e ∧ φ e ≠ 1 ∧ ψ e ≠ 1) → φ e = ψ e)
    (hne : φ e' ≠ ψ e') :
    ProjEquiv (frameMatrix G φ) (frameMatrix G ψ) ∧ ¬SwitchEquiv G φ ψ := by
  classical
  have hloop : G.tail e' = G.head e' := by
    by_contra h
    exact hne (hagree e' fun hj => h hj.1)
  have hden : ∀ {g : Fˣ}, g ≠ 1 → (1 : F) - g ≠ 0 := fun hg =>
    sub_ne_zero.2 fun h => hg (Units.ext (by simpa using h.symm))
  let s : E → Fˣ := fun e =>
    if h : G.tail e = G.head e ∧ φ e ≠ 1 ∧ ψ e ≠ 1 then
      Units.mk0 ((1 - ψ e) / (1 - φ e)) (div_ne_zero (hden h.2.2) (hden h.2.1))
    else 1
  refine ⟨⟨s, congrArg rowSpace ?_⟩, ?_⟩
  · ext w e
    simp only [scaleCols, Matrix.of_apply]
    by_cases hj : G.tail e = G.head e ∧ φ e ≠ 1 ∧ ψ e ≠ 1
    · have hs : ((s e : Fˣ) : F) = (1 - ψ e) / (1 - φ e) := congrArg Units.val (dite_eq_left hj)
      have h₁ := hden hj.2.1
      rw [frameMatrix_apply, frameMatrix_apply, hs, hj.1]
      split_ifs
      · field_simp
      · simp
    · have hs : s e = 1 := dite_eq_right hj
      simp [frameMatrix_apply, hs, hagree e hj]
  · rintro ⟨η, hη⟩
    exact hne ((switch_eq_of_loop G η φ hloop).symm.trans (congrFun hη e'))

/-- One vertex with one loop. -/
def singleLoop : OrientedGraph Unit Unit := ⟨fun _ => (), fun _ => ()⟩

/-- One vertex with two loops. -/
def twoLoops : OrientedGraph Unit Bool := ⟨fun _ => (), fun _ => ()⟩

/-- The unit `2` of `ℚ`. -/
def ratTwo : ℚˣ := Units.mk0 2 (by norm_num)

/-- The unit `3` of `ℚ`. -/
def ratThree : ℚˣ := Units.mk0 3 (by norm_num)

theorem ratTwo_ne_one : ratTwo ≠ 1 := fun h => by
  have := congrArg Units.val h
  norm_num [ratTwo] at this

theorem ratThree_ne_one : ratThree ≠ 1 := fun h => by
  have := congrArg Units.val h
  norm_num [ratThree] at this

theorem ratTwo_ne_ratThree : ratTwo ≠ ratThree := fun h => by
  have := congrArg Units.val h
  norm_num [ratTwo, ratThree] at this

/-- **Conjecture 1** (Zaslavsky, §1.2) is false as stated: on one vertex with one loop the gains
`2` and `3` over `ℚ` give projectively equivalent canonical frame matrices and are not switching
equivalent. Theorem 1 is the source's corrected form. -/
theorem conjecture1_fails :
    ∃ φ ψ : Unit → ℚˣ, ProjEquiv (frameMatrix singleLoop φ) (frameMatrix singleLoop ψ) ∧
      ¬SwitchEquiv singleLoop φ ψ :=
  ⟨fun _ => ratTwo, fun _ => ratThree,
    projEquiv_and_not_switchEquiv_of_joint singleLoop (e' := ())
      (fun _ hj => (hj ⟨rfl, ratTwo_ne_one, ratThree_ne_one⟩).elim) ratTwo_ne_ratThree⟩

/-- **Conjecture 2** (Zaslavsky, §1.2) is false as stated: on one vertex with two loops the gains
`(1, 1)` and `(1, 2)` over `ℚ` give projectively equivalent canonical lift matrices and are not
switching-and-scaling equivalent. Theorem 1 is the source's corrected form. -/
theorem conjecture2_fails :
    ∃ φ ψ : Bool → ℚ, ProjEquiv (liftMatrix twoLoops φ) (liftMatrix twoLoops ψ) ∧
      ¬SwitchScaleEquiv twoLoops φ ψ := by
  refine ⟨fun _ => 1, fun b => if b then 2 else 1,
    ⟨fun b => if b then ratTwo else 1, congrArg rowSpace ?_⟩, ?_⟩
  · ext r b
    cases r <;> cases b <;> simp [scaleCols, liftMatrix, twoLoops, ratTwo]
  · rintro ⟨η, a, h⟩
    have ht := congrFun h true
    have hf := congrFun h false
    simp only [Pi.smul_apply, addSwitch_apply, smul_eq_mul, twoLoops] at ht hf
    have hc : ∀ x : ℚ, -x + 1 + x = 1 := fun x => by ring
    rw [hc] at ht hf
    norm_num at ht hf
    rw [hf] at ht
    norm_num at ht

/-- The identity must be excluded in `projEquiv_and_not_switchEquiv_of_joint`: a loop of gain one
has a zero column, so its matrix is not projectively equivalent to that of a loop of any other
gain. -/
theorem not_projEquiv_of_gain_one {g : ℚˣ} (hg : g ≠ 1) :
    ¬ProjEquiv (frameMatrix singleLoop fun _ => (1 : ℚˣ)) (frameMatrix singleLoop fun _ => g) := by
  rintro ⟨s, hs⟩
  have hA : scaleCols s (frameMatrix singleLoop fun _ => (1 : ℚˣ)) = 0 := by
    ext i j
    simp [scaleCols, frameMatrix_apply]
  have hmem := row_mem_rowSpace (frameMatrix singleLoop fun _ => g) ()
  rw [← hs, hA] at hmem
  have hzero : rowSpace (0 : Matrix Unit Unit ℚ) ≤ ⊥ :=
    Submodule.span_le.2 (by rintro _ ⟨i, rfl⟩; exact (Submodule.mem_bot ℚ).2 rfl)
  have hrow := congrFun ((Submodule.mem_bot ℚ).1 (hzero hmem)) ()
  simp only [frameMatrix_apply, singleLoop, ite_true, mul_one, Pi.zero_apply] at hrow
  exact hg (Units.ext (by simpa using (sub_eq_zero.1 hrow).symm))

example (η γ : Unit → ℚˣ) : IsConj (γ ()) (switch singleLoop η γ ()) :=
  isConj_switch_of_loop singleLoop η γ rfl

example (η γ : Unit → ℚ) : IsAddConj (γ ()) (addSwitch singleLoop η γ ()) :=
  isAddConj_addSwitch_of_loop singleLoop η γ rfl

example (η γ : Unit → ℚˣ) : switch singleLoop η γ () = γ () :=
  switch_eq_of_loop singleLoop η γ rfl

example (η γ : Unit → ℚ) : addSwitch singleLoop η γ () = γ () :=
  addSwitch_eq_of_loop singleLoop η γ rfl

example : ProjEquiv (frameMatrix singleLoop fun _ => ratTwo)
    (frameMatrix singleLoop fun _ => ratThree) ∧
      ¬SwitchEquiv singleLoop (fun _ => ratTwo) (fun _ => ratThree) :=
  projEquiv_and_not_switchEquiv_of_joint singleLoop (e' := ())
    (fun _ hj => (hj ⟨rfl, ratTwo_ne_one, ratThree_ne_one⟩).elim) ratTwo_ne_ratThree

example : ¬ProjEquiv (frameMatrix singleLoop fun _ => (1 : ℚˣ))
    (frameMatrix singleLoop fun _ => ratTwo) :=
  not_projEquiv_of_gain_one ratTwo_ne_one

end Transformer.GainMatroid
