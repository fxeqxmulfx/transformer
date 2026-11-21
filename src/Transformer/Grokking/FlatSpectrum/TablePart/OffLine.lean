module

public import Transformer.Grokking.FlatSpectrum.TablePart.FlatCycles

/-!
# PT3: a cycle that is not enough

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT3. At `p = 3` the six
cells off the line `a + b = 2` (`offLine`, `card_offLine`) form one cycle through the rows
`0, 1, 2` and the columns `0, 2, 1`, with twist zero (`exists_sixCycle_twist`). They are connected
(`connected_offLine`), all their twists vanish (`twistSpan_offLine`), and the member
`1[a + b = 2] = 2 (a + b)² + (a + b)`, with coupling `M = 1`, vanishes on them
(`not_determines_offLine`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Module SimpleGraph

/-- The six cells of the table over `ℤ/3` off the line `a + b = 2`
(`review/flat_spectrum.md`, PT3). -/
def offLine : Set ((Fin 1 → ZMod 3) × (Fin 1 → ZMod 3)) := {c | c.1 0 + c.2 0 ≠ 2}

/-- The member `2 (a + b)² + (a + b)` (`review/flat_spectrum.md`, PT3). -/
def lineMember (c : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3)) : ZMod 3 :=
  2 * (c.1 0 + c.2 0) ^ 2 + (c.1 0 + c.2 0)

/-- `2 t² + t` is the indicator of `t = 2` over `ℤ/3`. -/
lemma two_mul_sq_add (t : ZMod 3) : 2 * t ^ 2 + t = if t = 2 then 1 else 0 := by
  revert t
  decide

/-- `2 (a + b)² + (a + b)` is a member of the flat class, with coupling `M = 1`. -/
lemma lineMember_mem : lineMember ∈ flatClass (ZMod 3) (Fin 1) (Fin 1) := by
  refine mem_flatClass.2 ⟨fun u => 2 * u 0 ^ 2 + u 0, fun v => 2 * v 0 ^ 2 + v 0, 1, fun u v => ?_⟩
  have h3 : (3 : ZMod 3) = 0 := by decide
  simp only [lineMember, one_mulVec, dotProduct, Fin.sum_univ_one, Fin.isValue]
  linear_combination (u 0 * v 0) * h3

/-- The graph of the six cells is connected: they form one cycle through the rows `0, 1, 2` and the
columns `0, 2, 1`. -/
theorem connected_offLine : (patternGraph offLine).Connected := by
  have step {a b : ZMod 3} (h : a + b ≠ 2) :
      (patternGraph offLine).Adj (.inl fun _ => a) (.inr fun _ => b) :=
    patternGraph_adj_inl_inr.2 h
  have c0 : (patternGraph offLine).Reachable (.inl fun _ => 0) (.inr fun _ => 0) :=
    (step (by decide)).reachable
  have c1 : (patternGraph offLine).Reachable (.inl fun _ => 0) (.inr fun _ => 1) :=
    (step (by decide)).reachable
  have r1 : (patternGraph offLine).Reachable (.inl fun _ => 0) (.inl fun _ => 1) :=
    c0.trans (step (a := 1) (b := 0) (by decide)).symm.reachable
  have c2 : (patternGraph offLine).Reachable (.inl fun _ => 0) (.inr fun _ => 2) :=
    r1.trans (step (a := 1) (b := 2) (by decide)).reachable
  have r2 : (patternGraph offLine).Reachable (.inl fun _ => 0) (.inl fun _ => 2) :=
    c1.trans (step (a := 2) (b := 1) (by decide)).symm.reachable
  have hcases : ∀ t : ZMod 3, t = 0 ∨ t = 1 ∨ t = 2 := by decide
  refine (connected_iff_exists_forall_reachable _).2 ⟨.inl fun _ => 0, ?_⟩
  rintro (u | v)
  · obtain ⟨t, rfl⟩ : ∃ t, u = fun _ => t := ⟨u 0, funext fun i => congrArg u (Subsingleton.elim i 0)⟩
    rcases hcases t with rfl | rfl | rfl
    exacts [Reachable.refl _, r1, r2]
  · obtain ⟨t, rfl⟩ : ∃ t, v = fun _ => t := ⟨v 0, funext fun i => congrArg v (Subsingleton.elim i 0)⟩
    rcases hcases t with rfl | rfl | rfl
    exacts [c0, c1, c2]

/-- The cells off the line are six (`review/flat_spectrum.md`, PT3). -/
theorem card_offLine :
    (Finset.univ.filter fun c : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3) => c.1 0 + c.2 0 ≠ 2).card =
      6 := by
  decide +kernel

/-- The cycle through the rows `0, 1, 2` and the columns `0, 2, 1` has the twist
`(0 − 1)·0 + (1 − 2)·2 + (2 − 0)·1 = 0` (`review/flat_spectrum.md`, PT3). -/
theorem exists_sixCycle_twist :
    ∃ w : (patternGraph offLine).Walk (.inl fun _ => 0) (.inl fun _ => 0),
      w.length = 6 ∧ twist w = 0 := by
  have step {a b : ZMod 3} (h : a + b ≠ 2) :
      (patternGraph offLine).Adj (.inl fun _ => a) (.inr fun _ => b) :=
    patternGraph_adj_inl_inr.2 h
  refine ⟨.cons (step (a := 0) (b := 0) (by decide)) <|
    .cons (step (a := 1) (b := 0) (by decide)).symm <|
    .cons (step (a := 1) (b := 2) (by decide)) <|
    .cons (step (a := 2) (b := 2) (by decide)).symm <|
    .cons (step (a := 2) (b := 1) (by decide)) <|
    .cons (step (a := 0) (b := 1) (by decide)).symm .nil, rfl, ?_⟩
  simp only [twist_row_step, twist_nil, add_zero]
  ext i j
  simp [vecMulVec_apply]
  decide

/-- **A cycle is not enough** (`review/flat_spectrum.md`, PT3): the member `1[a + b = 2]` vanishes
on the six connected cells off the line. -/
theorem not_determines_offLine :
    ¬ Determines (flatClass (ZMod 3) (Fin 1) (Fin 1) :
      Set ((Fin 1 → ZMod 3) × (Fin 1 → ZMod 3) → ZMod 3)) offLine := by
  intro h
  have h0 := (determines_submodule_iff _ _).1 h lineMember lineMember_mem fun c hc => by
    have hc' : c.1 0 + c.2 0 ≠ 2 := hc
    simp [lineMember, two_mul_sq_add, hc']
  have hval : lineMember (fun _ => 2, fun _ => 0) = 0 := congrFun h0 _
  revert hval
  simp only [lineMember]
  decide

/-- The twists of the six cells all vanish: a nonzero one would span, and the part would determine
the class (`review/flat_spectrum.md`, PT3). -/
theorem twistSpan_offLine : twistSpan offLine = ⊥ := by
  by_contra hne
  obtain ⟨Z, hZ, h0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hne
  exact not_determines_offLine
    ((determines_flatClass_iff _).2 ⟨connected_offLine, twistSpan_eq_top_of_ne_zero hZ h0⟩)

end Transformer.Grokking.FlatSpectrum.TablePart
