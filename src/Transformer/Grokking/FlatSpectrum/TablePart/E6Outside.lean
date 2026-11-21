module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6Half

/-!
# PT6: the tasks of the full table outside the class

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Of the 33 tasks with
the full table at `p = 97`, twelve are in the class (`E6Tasks`). For each of the other 21
(`outside97`), two rectangles of training cells ask for two different couplings, so no function of
the class fits the task on the half (`no_function_on_half`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

attribute [local instance] fact_prime_97

/-- **There are 21 of them.** -/
theorem length_outside97 : outside97.length = 21 := by decide +kernel

/-- Each pair of rectangles refutes the class, and its columns are below 97. -/
theorem outside97_valid :
    (outside97.all fun w => refutes 97 w.2.1 w.2.2 &&
      (corners w.2.1 ++ corners w.2.2).all fun c => decide (c.2.1 < 97)) = true := by
  decide +kernel

/-- The corners of the refutations are training cells. -/
lemma line_mem_half {w : String × Rect × Rect} (hw : w ∈ outside97) {c : ℕ × ℕ × ℕ}
    (hc : c ∈ corners w.2.1 ++ corners w.2.2) : (line 97 c.1, line 97 c.2.1) ∈ half := by
  have hv : (refutes 97 w.2.1 w.2.2 &&
      (corners w.2.1 ++ corners w.2.2).all fun c => decide (c.2.1 < 97)) = true :=
    List.all_eq_true.1 outside97_valid w hw
  rw [Bool.and_eq_true] at hv
  have hb : decide (c.2.1 < 97) = true := List.all_eq_true.1 hv.2 c hc
  have hT : marksFast.2.1.testBit (97 * c.1 + c.2.1) = true :=
    List.all_eq_true.1 (List.all_eq_true.1 certificate.2.2.2.2.2.2.2.2.2 w hw) c hc
  exact mem_half (of_decide_eq_true hb) hT

example : (line 97 0, line 97 2) ∈ half :=
  line_mem_half (w := ("arithmetic-max", (0, 1, 2, 3, 2, 2, 3, 3), (0, 3, 2, 3, 2, 3, 3, 3)))
    (List.mem_of_getElem? (i := 4) rfl) (c := (0, 2, 2)) (by decide)

/-- **No function of the class fits these tasks on the half** (`review/flat_spectrum.md`, PT6): this
holds for every table with the labels of the witness on its cells. -/
theorem no_function_on_half {w : String × Rect × Rect} (hw : w ∈ outside97)
    {y : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97}
    (hy : ∀ c ∈ corners w.2.1 ++ corners w.2.2, y (line 97 c.1, line 97 c.2.1) = c.2.2) :
    ¬ ∃ f ∈ flatClass (ZMod 97) (Fin 1) (Fin 1), ∀ c ∈ half, f c = y c := by
  rintro ⟨f, hf, hfy⟩
  have hv : (refutes 97 w.2.1 w.2.2 &&
      (corners w.2.1 ++ corners w.2.2).all fun c => decide (c.2.1 < 97)) = true :=
    List.all_eq_true.1 outside97_valid w hw
  rw [Bool.and_eq_true] at hv
  have hcell {c : ℕ × ℕ × ℕ} (hc : c ∈ corners w.2.1 ++ corners w.2.2) :
      f (line 97 c.1, line 97 c.2.1) = c.2.2 := by
    rw [hfy _ (line_mem_half hw hc), hy c hc]
  exact not_fits hv.1 ⟨f, hf, fun c hc => hcell (List.mem_append_left _ hc),
    fun c hc => hcell (List.mem_append_right _ hc)⟩

/-- The table of `max(a, b)`, one of the 21: no function of the class fits it on the half. -/
example : ¬ ∃ f ∈ flatClass (ZMod 97) (Fin 1) (Fin 1), ∀ c ∈ half,
    f c = ((max (c.1 0).val (c.2 0).val : ℕ) : ZMod 97) :=
  no_function_on_half
    (w := ("arithmetic-max", (0, 1, 2, 3, 2, 2, 3, 3), (0, 3, 2, 3, 2, 3, 3, 3)))
    (List.mem_of_getElem? (i := 4) rfl) (by decide +kernel)

end Transformer.Grokking.FlatSpectrum.TablePart.E6
