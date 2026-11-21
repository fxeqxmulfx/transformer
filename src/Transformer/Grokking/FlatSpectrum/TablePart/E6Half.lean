module

public import Transformer.Grokking.FlatSpectrum.TablePart.AgainstM
public import Transformer.Grokking.FlatSpectrum.TablePart.DegreeHalf
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Marks

/-!
# PT6: the real half of E6

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. `half` is the
training part of E6 at `p = 97`: the first 4704 entries of the shuffle of `E6Shuffle`, read as cells
`a · 97 + b`. One run of the kernel (`certificate`) computes the shuffle and shows that no word was
rejected, that the 4704 entries are different cells (`card_half`), that every row shares a column
with row 0 and no column is empty (`connected_half`), and that rows 0 and 1 meet columns 2 and 3 in
a four-cycle. So this half determines the flat class (`determines_half`).

Lean does not read the Rust code of the experiment. The same run computes the FNV-1a hash of the
order of all 9409 cells (`fingerprint`), and the test `lean_port_of_the_e6_shuffle` of
`experiments/table_part` asserts that value for the real split. The kernel run is what makes this
file slow to build: about a minute and 5.8 GB.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset

attribute [local instance] fact_prime_97

/-- The marks of the first `4704 = 2^12 + 2^9 + 2^6 + 2^5` entries, through `iterFast`. -/
def marksFast : ℕ × ℕ × ℕ :=
  iterFast mark 5 (iterFast mark 6 (iterFast mark 9 (iterFast mark 12 (0, 0, shuffleFast.2.2))))

/-- What the kernel computes: the end of the shuffle `e`, the count `m`, the mask `T` and the hash
`d` of the whole order. -/
abbrev Certificate (e m T d : ℕ) : Prop :=
  e = 0 ∧ m = 4704 ∧
    (∀ u, u < 97 → ∃ b, b < 97 ∧ T.testBit (97 * 0 + b) = true ∧ T.testBit (97 * u + b) = true) ∧
    (∀ v, v < 97 → ∃ a, a < 97 ∧ T.testBit (97 * a + v) = true) ∧
    T.testBit (97 * 0 + 2) = true ∧ T.testBit (97 * 1 + 2) = true ∧
    T.testBit (97 * 1 + 3) = true ∧ T.testBit (97 * 0 + 3) = true ∧ d = 359723728075922973

/-- **The kernel's computation on the real split of E6.** -/
theorem certificate :
    Certificate shuffleFast.1 marksFast.1 marksFast.2.1 fnvFast.2.1 := by
  decide +kernel

/-- Four runs of `iterFast` are one plain iteration. -/
lemma iterFast_four (f : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ) (a b c d : ℕ) (s : ℕ × ℕ × ℕ) :
    iterFast f a (iterFast f b (iterFast f c (iterFast f d s))) =
      f^[2 ^ a + (2 ^ b + (2 ^ c + 2 ^ d))] s := by
  simp only [iterFast_eq, Function.iterate_add_apply]

theorem shuffleFast_eq : shuffleFast = tick^[9408] (9408, 0, 0) := by
  have h := iterFast_four tick 6 7 10 13 (9408, 0, 0)
  rw [show 2 ^ 6 + (2 ^ 7 + (2 ^ 10 + 2 ^ 13)) = 9408 by norm_num] at h
  exact h

theorem marksFast_eq : marksFast = mark^[4704] (0, 0, shuffleFast.2.2) := by
  have h := iterFast_four mark 5 6 9 12 (0, 0, shuffleFast.2.2)
  rw [show 2 ^ 5 + (2 ^ 6 + (2 ^ 9 + 2 ^ 12)) = 4704 by norm_num] at h
  exact h

/-- **The shuffle ends after 9408 words**: none of them is rejected. -/
theorem shuffle_done : (tick^[9408] (9408, 0, 0)).1 = 0 :=
  shuffleFast_eq ▸ certificate.1

theorem fnvFast_eq : fnvFast = fnvStep^[9409] (0, 0xcbf29ce484222325, shuffleFast.2.2) := by
  have h := iterFast_four fnvStep 6 7 10 13 (0, 0xcbf29ce484222325, shuffleFast.2.2)
  have h2 : iterFast fnvStep 0 (iterFast fnvStep 6 (iterFast fnvStep 7 (iterFast fnvStep 10
      (iterFast fnvStep 13 (0, 0xcbf29ce484222325, shuffleFast.2.2))))) =
      fnvStep^[2 ^ 0 + (2 ^ 6 + (2 ^ 7 + (2 ^ 10 + 2 ^ 13)))]
        (0, 0xcbf29ce484222325, shuffleFast.2.2) := by
    rw [h, iterFast_eq, ← Function.iterate_add_apply]
  rw [show 2 ^ 0 + (2 ^ 6 + (2 ^ 7 + (2 ^ 10 + 2 ^ 13))) = 9409 by norm_num] at h2
  exact h2

/-- **The order of the cells is the experiment's**: its FNV-1a hash is the value that the test
`lean_port_of_the_e6_shuffle` of `experiments/table_part` gets from the Rust shuffle. -/
theorem fingerprint :
    (fnvStep^[9409] (0, 0xcbf29ce484222325, shuffleFast.2.2)).2.1 = 359723728075922973 := by
  rw [← fnvFast_eq]
  exact certificate.2.2.2.2.2.2.2.2

lemma count_marks : (mark^[4704] (0, 0, shuffleFast.2.2)).1 = 4704 :=
  marksFast_eq ▸ certificate.2.1

/-- The cell with the number `c = a · 97 + b`. -/
def cell (c : ℕ) : (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) :=
  (fun _ => ((c / 97 : ℕ) : ZMod 97), fun _ => ((c % 97 : ℕ) : ZMod 97))

/-- **The training half of E6**: the first 4704 cells of the array after the shuffle, which
`shuffleFast_eq` identifies with the plain iteration of `tick`. -/
def half : Finset ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) :=
  (range 4704).image fun i => cell (get shuffleFast.2.2 i)

lemma cell_inj {c c' : ℕ} (hc : c < 9409) (hc' : c' < 9409) (h : cell c = cell c') : c = c' := by
  have h1 := (ZMod.natCast_eq_natCast_iff' _ _ 97).1 (congrFun (Prod.ext_iff.1 h).1 0)
  have h2 := (ZMod.natCast_eq_natCast_iff' _ _ 97).1 (congrFun (Prod.ext_iff.1 h).2 0)
  omega

example : (0 : ℕ) = 0 := cell_inj (by norm_num) (by norm_num) rfl

lemma cell_add {a b : ℕ} (hb : b < 97) :
    cell (97 * a + b) = (fun _ => (a : ZMod 97), fun _ => (b : ZMod 97)) := by
  have h1 : (97 * a + b) / 97 = a := by omega
  have h2 : (97 * a + b) % 97 = b := by omega
  simp only [cell, h1, h2]

example : cell (97 * 1 + 2) = (fun _ => ((1 : ℕ) : ZMod 97), fun _ => ((2 : ℕ) : ZMod 97)) :=
  cell_add (by norm_num)

/-- **The half has 4704 different cells.** -/
theorem card_half : #half = 4704 := by
  have spec := mark_spec shuffleFast.2.2 4704 count_marks
  rw [half, card_image_of_injOn, card_range]
  intro i hi j hj hij
  have hi' : i < 4704 := mem_range.1 (mem_coe.1 hi)
  have hj' : j < 4704 := mem_range.1 (mem_coe.1 hj)
  exact spec.1 hi' hj' (cell_inj (spec.2.1 i hi') (spec.2.1 j hj') hij)

/-- A bit of the mask is a cell of the half. -/
lemma mem_half {a b : ℕ} (hb : b < 97) (h : marksFast.2.1.testBit (97 * a + b) = true) :
    ((fun _ => (a : ZMod 97), fun _ => (b : ZMod 97)) :
      (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) ∈ half := by
  rw [marksFast_eq] at h
  obtain ⟨i, hi, hc⟩ := ((mark_spec shuffleFast.2.2 4704 count_marks).2.2 _).1 h
  exact mem_image.2 ⟨i, mem_range.2 hi, by rw [hc, cell_add hb]⟩

example : ((fun _ => ((0 : ℕ) : ZMod 97), fun _ => ((2 : ℕ) : ZMod 97)) :
    (Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)) ∈ half :=
  mem_half (by norm_num) certificate.2.2.2.2.1

/-- Every line is numbered by some `a < 97`. -/
lemma exists_nat_eq (u : Fin 1 → ZMod 97) : ∃ a : ℕ, a < 97 ∧ (fun _ => (a : ZMod 97)) = u :=
  ⟨(u 0).val, ZMod.val_lt _, funext fun i => by
    rw [Subsingleton.elim i 0]
    exact ZMod.natCast_zmod_val _⟩

/-- **The graph of the half is connected**, through row 0. -/
theorem connected_half :
    (patternGraph (half : Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97)))).Connected := by
  obtain ⟨-, -, hrow, hcol, -⟩ := certificate
  refine connected_of_hub (fun _ => ((0 : ℕ) : ZMod 97)) (fun u => ?_) fun v => ?_
  · obtain ⟨a, ha, rfl⟩ := exists_nat_eq u
    obtain ⟨b, hb, h0, h1⟩ := hrow a ha
    exact ⟨fun _ => (b : ZMod 97), mem_coe.2 (mem_half hb h0), mem_coe.2 (mem_half hb h1)⟩
  · obtain ⟨b, hb, rfl⟩ := exists_nat_eq v
    obtain ⟨a, -, h⟩ := hcol b hb
    exact ⟨fun _ => (a : ZMod 97), mem_coe.2 (mem_half hb h)⟩

/-- Two different numbers below 97 are two different lines. -/
lemma line_ne {a b : ℕ} (ha : a < 97) (hb : b < 97) (hab : a ≠ b) :
    (fun _ => (a : ZMod 97) : Fin 1 → ZMod 97) ≠ fun _ => (b : ZMod 97) := fun h => hab (by
  have h' := (ZMod.natCast_eq_natCast_iff' _ _ 97).1 (congrFun h 0)
  omega)

example : (fun _ => ((0 : ℕ) : ZMod 97) : Fin 1 → ZMod 97) ≠ fun _ => ((1 : ℕ) : ZMod 97) :=
  line_ne (by norm_num) (by norm_num) (by norm_num)

/-- **The twists of the half span everything**, by its four-cycle on rows 0, 1 and columns 2, 3. -/
theorem twistSpan_half :
    twistSpan (half : Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))) = ⊤ := by
  obtain ⟨-, -, -, -, h₁, h₂, h₃, h₄, -⟩ := certificate
  exact twistSpan_eq_top_of_fourCycle
    (line_ne (a := 0) (b := 1) (by norm_num) (by norm_num) (by norm_num))
    (line_ne (a := 2) (b := 3) (by norm_num) (by norm_num) (by norm_num))
    (mem_coe.2 (mem_half (by norm_num) h₁)) (mem_coe.2 (mem_half (by norm_num) h₂))
    (mem_coe.2 (mem_half (by norm_num) h₃)) (mem_coe.2 (mem_half (by norm_num) h₄))

/-- **The real half of E6 determines the flat class** (`review/flat_spectrum.md`, PT6). -/
theorem determines_half :
    Determines (flatClass (ZMod 97) (Fin 1) (Fin 1) :
      Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97) → ZMod 97)) half :=
  (determines_flatClass_iff _).2 ⟨connected_half, twistSpan_half⟩

/-- The half is a proper part of the table. -/
theorem half_ne_univ : (half : Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))) ≠ Set.univ := by
  intro h
  have h' : half = univ := coe_eq_univ.1 h
  have hc := congrArg card h'
  rw [card_half, card_univ] at hc
  simp [ZMod.card] at hc

end Transformer.Grokking.FlatSpectrum.TablePart.E6
