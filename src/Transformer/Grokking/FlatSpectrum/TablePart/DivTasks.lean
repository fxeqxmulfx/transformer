module

public import Mathlib.Tactic.NormNum.Prime
public import Transformer.Grokking.FlatSpectrum.TablePart.DivPart

/-!
# PT6: the tables `(a / b)^k` on a part without the column `b = 0`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The table `(a / b)^0`
is the constant 1, so it lies in the class, and a part with a certificate gives it back on every
cell of the table (`constant_off_column`). For the others, two rectangles of the part whose labels
ask for two couplings leave no function of the class (`no_div_function`). The kernel checks such a
pair through `divWitnessOk`, which compares `a^k` with `y b^k` and so needs no inverse modulo `p`
(`div_pow_natCast`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

variable {p : ℕ} [Fact p.Prime]

/-- The table of `(a / b)^k`. -/
def divTable (p : ℕ) [Fact p.Prime] (k : ℕ) : (Fin 1 → ZMod p) × (Fin 1 → ZMod p) → ZMod p :=
  fun c => (c.1 0 / c.2 0) ^ k

/-- The table `(a / b)^0` is the constant 1, a member of the class. -/
lemma divTable_zero_mem : divTable p 0 ∈ flatClass (ZMod p) (Fin 1) (Fin 1) :=
  mem_flatClass.2 ⟨1, 0, 0, fun u v => by simp [divTable]⟩

/-- **Inside the class a part with a certificate gives `(a / b)^0` on the whole table**
(`review/flat_spectrum.md`, PT6). -/
theorem constant_off_column {T a a' b b' : ℕ} (hT : DivCertificate p T a a' b b')
    {f : (Fin 1 → ZMod p) × (Fin 1 → ZMod p) → ZMod p}
    (hf : f ∈ flatClass (ZMod p) (Fin 1) (Fin 1))
    (hag : ∀ c ∈ divPart p T, f c = divTable p 0 c) (u : Fin 1 → ZMod p) {v : Fin 1 → ZMod p}
    (hv : v ≠ 0) : f (u, v) = divTable p 0 (u, v) :=
  eq_off_column hT hf divTable_zero_mem hag u hv

example : divTable 3 0 (0, 1) = divTable 3 0 (0, 1) :=
  constant_off_column (T := 63) (a := 0) (a' := 1) (b := 0) (b' := 1) (by decide)
    divTable_zero_mem (fun _ _ => rfl) 0 (by decide)

/-- A label of `(a / b)^k`, checked by multiplying out. -/
lemma div_pow_natCast {a b l k : ℕ} (hb : b ≠ 0) (hbp : b < p) (h : a ^ k % p = l * b ^ k % p) :
    ((a : ZMod p) / b) ^ k = l := by
  have hb' : (b : ZMod p) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]
    exact fun hd => hb (Nat.eq_zero_of_dvd_of_lt hd hbp)
  rw [div_pow, div_eq_iff (pow_ne_zero _ hb')]
  have := (ZMod.natCast_eq_natCast_iff' (a ^ k) (l * b ^ k) p).2 h
  push_cast at this
  exact this

example : (((1 : ℕ) : ZMod 3) / ((2 : ℕ) : ZMod 3)) ^ 1 = ((2 : ℕ) : ZMod 3) :=
  div_pow_natCast (by norm_num) (by norm_num) (by norm_num)

/-- What the kernel checks of two rectangles against `(a / b)^k` on the mask `T`: they ask for two
couplings, and each corner is a cell of the part with the label of the table. -/
def divWitnessOk (p T k : ℕ) (r s : Rect) : Bool :=
  refutes p r s && (corners r ++ corners s).all fun c =>
    decide (c.1 < p) && decide (c.2.1 ≠ 0) && decide (c.2.1 < p) &&
      T.testBit ((p - 1) * c.1 + (c.2.1 - 1)) && decide (c.1 ^ k % p = c.2.2 * c.2.1 ^ k % p)

/-- **No function of the class fits `(a / b)^k` on a part with such rectangles**
(`review/flat_spectrum.md`, PT6). -/
theorem no_div_function {T k : ℕ} {r s : Rect} (h : divWitnessOk p T k r s = true) :
    ¬ ∃ f ∈ flatClass (ZMod p) (Fin 1) (Fin 1), ∀ c ∈ divPart p T, f c = divTable p k c := by
  rintro ⟨f, hf, hfy⟩
  rw [divWitnessOk, Bool.and_eq_true] at h
  have hcell {c : ℕ × ℕ × ℕ} (hc : c ∈ corners r ++ corners s) :
      f (line p c.1, line p c.2.1) = c.2.2 := by
    have hv := List.all_eq_true.1 h.2 c hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hv
    obtain ⟨⟨⟨⟨ha, hb0⟩, hbp⟩, hT⟩, hl⟩ := hv
    have hmem : (line p c.1, line p c.2.1) ∈ divPart p T := by
      have := mem_divPart (b := c.2.1 - 1) ha (by omega) hT
      rwa [Nat.sub_add_cancel (Nat.pos_of_ne_zero hb0)] at this
    rw [hfy _ hmem]
    exact div_pow_natCast hb0 hbp hl
  exact not_fits h.1 ⟨f, hf, fun c hc => hcell (List.mem_append_left _ hc),
    fun c hc => hcell (List.mem_append_right _ hc)⟩

/-- A list of such rectangles rules out each of its exponents. -/
lemma no_div_function_of_all {T : ℕ} {ws : List (String × ℕ × Rect × Rect)}
    (hv : ws.all (fun w => divWitnessOk p T w.2.1 w.2.2.1 w.2.2.2) = true) {k : ℕ}
    (hk : k ∈ ws.map (·.2.1)) :
    ¬ ∃ f ∈ flatClass (ZMod p) (Fin 1) (Fin 1), ∀ c ∈ divPart p T, f c = divTable p k c := by
  obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hk
  exact no_div_function (List.all_eq_true.1 hv w hw)

local instance fact_prime_5 : Fact (Nat.Prime 5) := ⟨by norm_num⟩

/-- `a / b` at `p = 5` on the full table: `1/2 = 3` and `1/3 = 2` ask for two couplings. -/
example : ¬ ∃ f ∈ flatClass (ZMod 5) (Fin 1) (Fin 1),
    ∀ c ∈ divPart 5 (2 ^ 20 - 1), f c = divTable 5 1 c :=
  no_div_function_of_all (ws := [("div", 1, (0, 1, 1, 2, 0, 1, 3, 0), (0, 1, 1, 3, 0, 1, 2, 0))])
    (by decide +kernel) (by simp)

example : ¬ ∃ f ∈ flatClass (ZMod 5) (Fin 1) (Fin 1),
    ∀ c ∈ divPart 5 (2 ^ 20 - 1), f c = divTable 5 1 c :=
  no_div_function (r := (0, 1, 1, 2, 0, 1, 3, 0)) (s := (0, 1, 1, 3, 0, 1, 2, 0))
    (by decide +kernel)

end Transformer.Grokking.FlatSpectrum.TablePart.E6
