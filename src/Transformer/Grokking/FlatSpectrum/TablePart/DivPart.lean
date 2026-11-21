module

public import Transformer.Grokking.FlatSpectrum.TablePart.Rectangles

/-!
# PT6: a part of a table without the column `b = 0`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The tasks with a
division have no cell with `b = 0`. Their training part has an empty column, so it does not
determine the class (PT5), but it can still fix a function of the class on the columns it has: if
one row shares a column with every row and the part holds a four-cycle, two functions of the class
that agree on the part agree on every cell of its columns (`eq_on_columns`).

Such a table has `p (p − 1)` cells, the cell `(a, b)` having the number `a (p − 1) + (b − 1)`
(`divCell`), and a part is a bit mask of these numbers (`divPart`). `DivCertificate` is what the
kernel reads from a mask, and then the part fixes the function on the whole table
(`eq_off_column`). Of the tables `(a / b)^k`, the constant `k = 0` is in the class and comes back on
the whole table (`constant_off_column`). When two rectangles of training cells ask for two
couplings, no function of the class fits the part (`no_div_function`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

variable {p : ℕ} [Fact p.Prime]

/-- **A part fixes a function of the class on the columns it has** (`review/flat_spectrum.md`, PT6):
the row `u₀` shares a column with every row, and the rows `a, a'` meet the columns `b, b'` in a
four-cycle. -/
theorem eq_on_columns {Ω : Set ((Fin 1 → ZMod p) × (Fin 1 → ZMod p))}
    {u₀ a a' b b' : Fin 1 → ZMod p} (hrow : ∀ u, ∃ v, (u₀, v) ∈ Ω ∧ (u, v) ∈ Ω) (ha : a ≠ a')
    (hb : b ≠ b') (h₁ : (a, b) ∈ Ω) (h₂ : (a', b) ∈ Ω) (h₃ : (a', b') ∈ Ω) (h₄ : (a, b') ∈ Ω)
    {f f' : (Fin 1 → ZMod p) × (Fin 1 → ZMod p) → ZMod p}
    (hf : f ∈ flatClass (ZMod p) (Fin 1) (Fin 1)) (hf' : f' ∈ flatClass (ZMod p) (Fin 1) (Fin 1))
    (hag : ∀ c ∈ Ω, f c = f' c) {u v w : Fin 1 → ZMod p} (hv : (w, v) ∈ Ω) :
    f (u, v) = f' (u, v) := by
  obtain ⟨g, h, M, hd⟩ := mem_flatClass.1 (Submodule.sub_mem _ hf hf')
  have key (x y : Fin 1 → ZMod p) : x ⬝ᵥ M *ᵥ y = x 0 * M 0 0 * y 0 := by
    simp [dotProduct, mulVec]
    ring
  have hz {x y : Fin 1 → ZMod p} (hc : (x, y) ∈ Ω) : g x + h y + x 0 * M 0 0 * y 0 = 0 := by
    have := hd x y
    rw [key, Pi.sub_apply, hag _ hc, sub_self] at this
    exact this.symm
  have ext {x y : Fin 1 → ZMod p} (hxy : x ≠ y) : x 0 - y 0 ≠ 0 :=
    sub_ne_zero.2 fun h => hxy (funext fun i => by rwa [Subsingleton.elim i 0])
  have hm : M 0 0 = 0 := by
    have h0 : (a 0 - a' 0) * (b 0 - b' 0) * M 0 0 = 0 := by
      linear_combination hz h₁ - hz h₂ + hz h₃ - hz h₄
    exact (mul_eq_zero.1 h0).resolve_left (mul_ne_zero (ext ha) (ext hb))
  have hg (x : Fin 1 → ZMod p) : g x = g u₀ := by
    obtain ⟨y, hy₀, hy⟩ := hrow x
    have h0 := hz hy₀
    have h1 := hz hy
    rw [hm] at h0 h1
    linear_combination h1 - h0
  have hfin := hd u v
  rw [key, Pi.sub_apply, hm] at hfin
  have hw := hz hv
  rw [hm] at hw
  have hgu := hg u
  have hgw := hg w
  exact sub_eq_zero.1 (by linear_combination hfin + hgu - hgw + hw)

example {f : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3) → ZMod 3}
    (hf : f ∈ flatClass (ZMod 3) (Fin 1) (Fin 1)) (u v : Fin 1 → ZMod 3) : f (u, v) = f (u, v) :=
  eq_on_columns (Ω := Set.univ) (u₀ := 0) (a := 0) (a' := 1) (b := 0) (b' := 1) (w := 0)
    (fun _ => ⟨0, trivial, trivial⟩) (by decide) (by decide) trivial trivial trivial trivial hf hf
    (fun _ _ => rfl) trivial

/-- The cell with the number `c = a (p − 1) + (b − 1)` of a table without the column `b = 0`. -/
def divCell (p c : ℕ) : (Fin 1 → ZMod p) × (Fin 1 → ZMod p) :=
  (line p (c / (p - 1)), line p (c % (p - 1) + 1))

/-- The cells whose numbers are the bits of `T`, among the `p (p − 1)` cells of the table. -/
def divPart (p T : ℕ) : Finset ((Fin 1 → ZMod p) × (Fin 1 → ZMod p)) :=
  ((range (p * (p - 1))).filter fun c => T.testBit c = true).image (divCell p)

omit [Fact p.Prime] in
/-- A bit of the mask is a cell of the part. The column is numbered from 0 here, as `b − 1`. -/
lemma mem_divPart {T a b : ℕ} (ha : a < p) (hb : b < p - 1)
    (h : T.testBit ((p - 1) * a + b) = true) : (line p a, line p (b + 1)) ∈ divPart p T := by
  have hq : 0 < p - 1 := by omega
  refine mem_image.2 ⟨(p - 1) * a + b, mem_filter.2 ⟨mem_range.2 ?_, h⟩, ?_⟩
  · calc (p - 1) * a + b < (p - 1) * a + (p - 1) := by omega
      _ = (p - 1) * (a + 1) := (Nat.mul_succ _ _).symm
      _ ≤ (p - 1) * p := Nat.mul_le_mul_left _ ha
      _ = p * (p - 1) := Nat.mul_comm _ _
  · have h1 : ((p - 1) * a + b) / (p - 1) = a := by
      rw [Nat.mul_add_div hq, Nat.div_eq_of_lt hb, add_zero]
    have h2 : ((p - 1) * a + b) % (p - 1) = b := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hb]
    simp only [divCell, h1, h2]

example : (line 3 1, line 3 (0 + 1)) ∈ divPart 3 63 :=
  mem_divPart (by norm_num) (by norm_num) (by decide)

/-- Every line has a number below `p`. -/
lemma exists_line_eq (u : Fin 1 → ZMod p) : ∃ a, a < p ∧ line p a = u :=
  ⟨(u 0).val, ZMod.val_lt _, funext fun i => by
    rw [Subsingleton.elim i 0]
    exact ZMod.natCast_zmod_val _⟩

/-- A nonzero line has a number `b + 1` with `b < p − 1`. -/
lemma exists_line_succ_eq {v : Fin 1 → ZMod p} (hv : v ≠ 0) :
    ∃ b, b < p - 1 ∧ line p (b + 1) = v := by
  obtain ⟨c, hc, rfl⟩ := exists_line_eq v
  have hc0 : c ≠ 0 := by
    rintro rfl
    exact hv (funext fun _ => by simp [line])
  exact ⟨c - 1, by omega, by rw [Nat.sub_add_cancel (Nat.pos_of_ne_zero hc0)]⟩

example : ∃ b, b < 3 - 1 ∧ line 3 (b + 1) = (1 : Fin 1 → ZMod 3) :=
  exists_line_succ_eq (by decide)

omit [Fact p.Prime] in
/-- Two different numbers below `p` are two different lines. -/
lemma line_ne_line {a b : ℕ} (ha : a < p) (hb : b < p) (hab : a ≠ b) : line p a ≠ line p b :=
  fun h => hab (by
    have h' := (ZMod.natCast_eq_natCast_iff' _ _ p).1 (congrFun h 0)
    rwa [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at h')

example : line 3 0 ≠ line 3 1 := line_ne_line (by norm_num) (by norm_num) (by norm_num)

/-- What the kernel reads from the mask `T`: the rows `a, a'` and the columns `b, b'` (numbered from
0, as `b − 1`) make a four-cycle, row 0 shares a column with every row, and no column is empty. -/
abbrev DivCertificate (p T a a' b b' : ℕ) : Prop :=
  a < p ∧ a' < p ∧ a ≠ a' ∧ b < p - 1 ∧ b' < p - 1 ∧ b ≠ b' ∧
    T.testBit ((p - 1) * a + b) = true ∧ T.testBit ((p - 1) * a' + b) = true ∧
    T.testBit ((p - 1) * a' + b') = true ∧ T.testBit ((p - 1) * a + b') = true ∧
    (∀ u, u < p → ∃ c, c < p - 1 ∧ T.testBit ((p - 1) * 0 + c) = true ∧
      T.testBit ((p - 1) * u + c) = true) ∧
    ∀ c, c < p - 1 → ∃ u, u < p ∧ T.testBit ((p - 1) * u + c) = true

/-- **A part with a certificate fixes a function of the class on the whole table**
(`review/flat_spectrum.md`, PT6). -/
theorem eq_off_column {T a a' b b' : ℕ} (hT : DivCertificate p T a a' b b')
    {f f' : (Fin 1 → ZMod p) × (Fin 1 → ZMod p) → ZMod p}
    (hf : f ∈ flatClass (ZMod p) (Fin 1) (Fin 1)) (hf' : f' ∈ flatClass (ZMod p) (Fin 1) (Fin 1))
    (hag : ∀ c ∈ divPart p T, f c = f' c) (u : Fin 1 → ZMod p) {v : Fin 1 → ZMod p} (hv : v ≠ 0) :
    f (u, v) = f' (u, v) := by
  obtain ⟨ha, ha', hne, hb, hb', hne', h₁, h₂, h₃, h₄, hrow, hcol⟩ := hT
  have hp0 : 0 < p := (Fact.out : p.Prime).pos
  have mem {x y : ℕ} (hx : x < p) (hy : y < p - 1) (h : T.testBit ((p - 1) * x + y) = true) :
      (line p x, line p (y + 1)) ∈ (divPart p T : Set ((Fin 1 → ZMod p) × (Fin 1 → ZMod p))) :=
    mem_coe.2 (mem_divPart hx hy h)
  have hrow' : ∀ x : Fin 1 → ZMod p,
      ∃ y, (line p 0, y) ∈ (divPart p T : Set ((Fin 1 → ZMod p) × (Fin 1 → ZMod p))) ∧
        (x, y) ∈ (divPart p T : Set ((Fin 1 → ZMod p) × (Fin 1 → ZMod p))) := by
    intro x
    obtain ⟨n, hn, rfl⟩ := exists_line_eq x
    obtain ⟨c, hc, h0, h1⟩ := hrow n hn
    exact ⟨line p (c + 1), mem hp0 hc h0, mem hn hc h1⟩
  obtain ⟨c, hc, rfl⟩ := exists_line_succ_eq hv
  obtain ⟨w, hw, hwc⟩ := hcol c hc
  exact eq_on_columns hrow' (line_ne_line ha ha' hne)
    (line_ne_line (by omega) (by omega) (by omega)) (mem ha hb h₁) (mem ha' hb h₂)
    (mem ha' hb' h₃) (mem ha hb' h₄) hf hf' (fun c hc => hag c (mem_coe.1 hc)) (mem hw hc hwc)

example {f : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3) → ZMod 3}
    (hf : f ∈ flatClass (ZMod 3) (Fin 1) (Fin 1)) : f (0, 1) = f (0, 1) :=
  eq_off_column (T := 63) (a := 0) (a' := 1) (b := 0) (b' := 1) (by decide) hf hf
    (fun _ _ => rfl) 0 (by decide)

end Transformer.Grokking.FlatSpectrum.TablePart.E6
