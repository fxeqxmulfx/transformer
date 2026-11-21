module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6Half

/-!
# PT6: the tasks of E6 on its real half

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. A function
`g(a) + h(b) + m a b` at `p = 97` is a triple `(g, h, m)` (`form`). Twelve tasks of the arithmetic
data are of this kind (`tasks`, `tasks_formulas`). Inside the flat class the half gives each of them
on all 4705 held-out cells (`card_held_out`, `held_out_eq`). The seven with `m ≠ 0`
(`tasks_coupled`) have flat kernels, which M does not return from this half (`not_returned`). A
table that differs from a task at one held-out cell agrees with it on the half and lies outside the
class, so the class is what carries the half to the rest of the table (PT7, `update_not_mem`).

The test `lean_tasks_are_tables_of_the_data` of `experiments/table_part` checks that each formula
of `tasks` is the table of one task of the data.

That the other tasks of the data have no function of the class on their splits, and the three
tables `(a/b)⁰`, stay measurements of `experiments/table_part`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- A function of the class at `k = l = 1`, as `(g, h, m)`. -/
abbrev Form := (ZMod 97 → ZMod 97) × (ZMod 97 → ZMod 97) × ZMod 97

/-- The table of `g(a) + h(b) + m a b`. -/
def form (t : Form) : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97 :=
  fun c => t.1 (c.1 0) + t.2.1 (c.2 0) + t.2.2 * c.1 0 * c.2 0

/-- The coupling `m` as a `1 × 1` matrix. -/
def coupling (t : Form) : Matrix (Fin 1) (Fin 1) (ZMod 97) := t.2.2 • 1

/-- `form` is the table of `flatTable`. -/
lemma form_eq (t : Form) :
    form t = flatTable (fun u => t.1 (u 0)) (fun v => t.2.1 (v 0)) (coupling t) := by
  funext ⟨u, v⟩
  simp [form, flatTable, coupling, dotProduct, mulVec, Matrix.one_apply]
  ring

/-- The twelve tasks at `p = 97` with a table in the class. -/
def tasks : List Form :=
  [(fun a => a, fun b => b, 0), (fun a => a, fun b => -b, 0), (fun a => a / 2, fun b => b / 2, 0),
    (fun _ => 0, fun _ => 0, 1), (fun a => a, fun _ => 0, 1), (fun a => a ^ 2, fun _ => 0, 1),
    (fun a => a ^ 2, fun b => b ^ 2, 1), (fun a => a ^ 2 + a, fun b => b ^ 2, 1),
    (fun a => a ^ 2, fun b => b ^ 2, 0), (fun a => a ^ 2, fun b => b ^ 2, 2),
    (fun a => a ^ 3, fun b => b, 0), (fun a => a ^ 3, fun _ => 0, 1)]

/-- **The twelve tasks, by their formulas** (`review/flat_spectrum.md`, PT6). -/
theorem tasks_formulas :
    tasks.map (fun t (a b : ZMod 97) => t.1 a + t.2.1 b + t.2.2 * a * b) =
      [fun a b => a + b, fun a b => a - b, fun a b => (a + b) / 2, fun a b => a * b,
        fun a b => a + a * b, fun a b => a ^ 2 + a * b, fun a b => a ^ 2 + a * b + b ^ 2,
        fun a b => a ^ 2 + a * b + b ^ 2 + a, fun a b => a ^ 2 + b ^ 2, fun a b => (a + b) ^ 2,
        fun a b => a ^ 3 + b, fun a b => a ^ 3 + a * b] := by
  simp only [tasks, List.map_cons, List.map_nil, List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> funext a b <;> ring

/-- **Seven of the twelve have `m ≠ 0`** (`review/flat_spectrum.md`, PT6). -/
theorem tasks_coupled : (tasks.map fun t => t.2.2).countP (· ≠ 0) = 7 := by
  decide +kernel

/-- The held-out part of the table has 4705 cells. -/
theorem card_held_out :
    #(univ \ half : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))) = 4705 := by
  rw [card_sdiff_of_subset (subset_univ _), card_half, card_univ]
  simp [ZMod.card]

/-- **Inside the class the half gives every held-out label** (`review/flat_spectrum.md`, PT6). -/
theorem held_out_eq (t : Form) {f : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97}
    (hf : f ∈ flatClass (ZMod 97) (Fin 1) (Fin 1)) (hag : ∀ c ∈ half, f c = form t c)
    (c : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) : f c = form t c := by
  have hm : form t ∈ flatClass (ZMod 97) (Fin 1) (Fin 1) := by
    rw [form_eq]
    exact flatTable_mem _ _ _
  exact congrFun (determines_half f hf _ hm fun x hx => hag x (mem_coe.1 hx)) c

example (c : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) :
    form (fun a => a ^ 3, fun _ => 0, 1) c = form (fun a => a ^ 3, fun _ => 0, 1) c :=
  held_out_eq _ (by rw [form_eq]; exact flatTable_mem _ _ _) (fun _ _ => rfl) c

/-- **The class is the prior** (`review/flat_spectrum.md`, PT7): a table that differs from a task at
one held-out cell agrees with it on the whole half and lies outside the class. -/
theorem update_not_mem (t : Form) {c : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)} (hc : c ∉ half)
    {y : ZMod 97} (hy : y ≠ form t c) :
    (∀ x ∈ half, Function.update (form t) c y x = form t x) ∧
      Function.update (form t) c y ∉ flatClass (ZMod 97) (Fin 1) (Fin 1) := by
  have hag : ∀ x ∈ half, Function.update (form t) c y x = form t x := fun x hx =>
    Function.update_of_ne (fun h : x = c => hc (h ▸ hx)) _ _
  refine ⟨hag, fun hf => hy ?_⟩
  simpa using held_out_eq t hf hag c

example (t : Form) : ∃ c y, (∀ x ∈ half, Function.update (form t) c y x = form t x) ∧
    Function.update (form t) c y ∉ flatClass (ZMod 97) (Fin 1) (Fin 1) := by
  obtain ⟨c, hc⟩ : ∃ c, c ∉ half := by
    by_contra h
    exact half_ne_univ (Set.eq_univ_of_forall fun x => mem_coe.2 (not_not.1 (not_exists.1 h x)))
  exact ⟨c, form t c + 1, update_not_mem t hc (by simp)⟩

/-- A nonzero coupling is injective. -/
lemma coupling_injective {t : Form} (hm : t.2.2 ≠ 0) :
    Function.Injective (coupling t).vecMul := by
  intro a b h
  have h' : t.2.2 • a = t.2.2 • b := by simpa [coupling, vecMul_smul] using h
  exact smul_right_injective _ hm h'

example : Function.Injective (coupling (fun a => a ^ 3, fun _ => 0, 1)).vecMul :=
  coupling_injective one_ne_zero

/-- **M does not return the kernel of a task with `m ≠ 0` from this half**
(`review/flat_spectrum.md`, PT6). -/
theorem not_returned {t : Form} (hm : t.2.2 ≠ 0) :
    ¬ IsMinOn nuclearNorm
      {X | ∀ ij ∈ (half : Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))),
        X ij.1 ij.2 = bilinearKernel 97 (fun u => t.1 (u 0)) (fun v => t.2.1 (v 0)) (coupling t)
          ij.1 ij.2}
      (bilinearKernel 97 (fun u => t.1 (u 0)) (fun v => t.2.1 (v 0)) (coupling t)) :=
  (determined_and_not_returned _ _ (coupling_injective hm) half_ne_univ connected_half
    twistSpan_half).2

/-- `a³ + a b`, the task of E1: M does not return its kernel from the half of E6. -/
example := not_returned (t := (fun a => a ^ 3, fun _ => 0, 1)) one_ne_zero

end Transformer.Grokking.FlatSpectrum.TablePart.E6
