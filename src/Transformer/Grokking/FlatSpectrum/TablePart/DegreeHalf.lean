module

public import Mathlib.Algebra.MvPolynomial.NoZeroDivisors
public import Mathlib.Tactic.NormNum.Prime
public import Transformer.Grokking.FlatSpectrum.TablePart.Degree

/-!
# PT2: the count of cells, and what a half of the table holds at `q = 97`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT2. For `d < q`, every
part of `m` cells determines `H_d` exactly when `m > d q^{k−1}`
(`forall_determines_degreeClass_iff`). The table of two
variables over `ℤ/97` has 9409 cells, so a half has 4704, and `48 · 97 = 4656 < 4704 < 4753 =
49 · 97`. Every half determines every polynomial of degree at most 48
(`forall_half_determines_degree_48`). At degree 49 some half lies inside the zero set of
`∏_{c<49} (a − c)` and does not (`exists_half_zero_degree_49`). Inside `H₃`, any 292 cells
determine `a³ + a b` (`eq_cube_add_mul_of_card`). Division is the polynomial `a b^{95}`, of degree
96 (`div_eq_mul_pow`, `totalDegree_mul_pow`), and at that degree some half determines nothing
(`exists_half_not_determines_degree_96`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset MvPolynomial

section General

variable {F : Type*} [Field F]

/-- `H_d` has two different members: zero and the constant one. -/
lemma degreeClass_nontrivial (k d : ℕ) [Nonempty (Fin k → F)] :
    (degreeClass F k d : Set ((Fin k → F) → F)).Nontrivial := by
  refine ⟨0, Submodule.zero_mem _, fun _ => 1,
    mem_degreeClass.2 ⟨C 1, by simp, fun x => by simp⟩, fun h => ?_⟩
  obtain ⟨x⟩ := ‹Nonempty (Fin k → F)›
  exact zero_ne_one (congrFun h x)

example := degreeClass_nontrivial (F := ZMod 2) 1 0

/-- **PT2** (`review/flat_spectrum.md`): for `d < q`, every part of `m` cells determines `H_d`
exactly when `m > d q^{k−1}`, on `k + 1` variables here. -/
theorem forall_determines_degreeClass_iff [Fintype F] [DecidableEq F] {k d : ℕ}
    (hd : d < Fintype.card F) (m : ℕ) :
    (∀ Ω : Finset (Fin (k + 1) → F),
        #Ω = m → Determines (degreeClass F (k + 1) d : Set ((Fin (k + 1) → F) → F)) Ω) ↔
      d * Fintype.card F ^ k < m := by
  rw [forall_determines_iff (degreeClass_nontrivial _ _), minDistance_degreeClass hd]
  have hN : Fintype.card (Fin (k + 1) → F) = (Fintype.card F - d) * Fintype.card F ^ k +
      d * Fintype.card F ^ k := by
    rw [← Nat.add_mul, Nat.sub_add_cancel hd.le]
    simp [pow_succ, Nat.mul_comm]
  rw [hN, Nat.add_sub_cancel_left]

example := (forall_determines_degreeClass_iff (F := ZMod 2) (k := 0) (d := 1)
  (by simp [ZMod.card]) 2).2 (by simp)

omit [Field F] in
/-- Some `m` points with first coordinate in `T`, for `m ≤ |T| q^k`. -/
lemma exists_part_first_mem [Fintype F] [DecidableEq F] {k m : ℕ} (T : Finset F)
    (hm : m ≤ #T * Fintype.card F ^ k) :
    ∃ Ω : Finset (Fin (k + 1) → F), #Ω = m ∧ ∀ x ∈ Ω, x 0 ∈ T := by
  obtain ⟨Ω, hΩ, hcard⟩ := exists_subset_card_eq
    (s := univ.filter fun x : Fin (k + 1) → F => x 0 ∈ T) (n := m)
    (by rw [card_first_mem]; exact hm)
  exact ⟨Ω, hcard, fun x hx => (mem_filter.1 (hΩ hx)).2⟩

example := exists_part_first_mem (F := ZMod 2) (k := 0) (m := 1) {0} (by simp [ZMod.card])

end General

local instance fact_prime_97 : Fact (Nat.Prime 97) := ⟨by norm_num⟩

/-- The cells of the table of two variables over `ℤ/97`. -/
abbrev Cell97 := Fin 2 → ZMod 97

/-- **Every half determines degree 48** (`review/flat_spectrum.md`, PT2): `48 · 97 < 4704`. -/
theorem forall_half_determines_degree_48 (Ω : Finset Cell97) (hΩ : #Ω = 4704) :
    Determines (degreeClass (ZMod 97) 2 48 : Set (Cell97 → ZMod 97)) Ω :=
  (forall_determines_degreeClass_iff (F := ZMod 97) (k := 1) (d := 48)
    (by simp [ZMod.card]) 4704).2 (by simp [ZMod.card]) Ω hΩ

example : ∃ Ω : Finset Cell97, #Ω = 4704 ∧
    Determines (degreeClass (ZMod 97) 2 48 : Set (Cell97 → ZMod 97)) Ω := by
  obtain ⟨Ω, -, hΩ⟩ := exists_subset_card_eq (s := (univ : Finset Cell97)) (n := 4704)
    (by simp [card_univ, ZMod.card])
  exact ⟨Ω, hΩ, forall_half_determines_degree_48 Ω hΩ⟩

/-- `∏_{c<d} (x₀ − c)` is a member of `H_d`. -/
lemma prod_sub_mem_degreeClass (d : ℕ) :
    (fun x : Cell97 => ∏ c ∈ range d, (x 0 - (c : ZMod 97))) ∈ degreeClass (ZMod 97) 2 d := by
  refine mem_degreeClass.2 ⟨∏ c ∈ range d, (X 0 - C (c : ZMod 97)), ?_, fun x => by
    simp [map_prod]⟩
  induction d with
  | zero => simp
  | succ d ih =>
    rw [prod_range_succ]
    have h1 : (X 0 - C (d : ZMod 97) : MvPolynomial (Fin 2) (ZMod 97)).totalDegree ≤ 1 := by
      refine (totalDegree_sub _ _).trans ?_
      rw [totalDegree_X, totalDegree_C]
      norm_num
    exact (totalDegree_mul _ _).trans (by omega)

example := prod_sub_mem_degreeClass 3

/-- The residues `0, …, d − 1` modulo 97 are distinct for `d ≤ 97`. -/
lemma card_image_range_cast {d : ℕ} (hd : d ≤ 97) :
    #((range d).image fun c : ℕ => (c : ZMod 97)) = d := by
  rw [card_image_of_injOn, card_range]
  intro a ha b hb hab
  have ha' : a < 97 := (mem_range.1 ha).trans_le hd
  have hb' : b < 97 := (mem_range.1 hb).trans_le hd
  have := (ZMod.natCast_eq_natCast_iff' a b 97).1 hab
  rwa [Nat.mod_eq_of_lt ha', Nat.mod_eq_of_lt hb'] at this

example : #((range 2).image fun c : ℕ => (c : ZMod 97)) = 2 := card_image_range_cast (by norm_num)

/-- **A half inside a zero set** (`review/flat_spectrum.md`, PT2): at degree 49 some part of 4704
cells lies in the zero set of `∏_{c<49} (a − c)`, and so does not determine `H₄₉`. -/
theorem exists_half_zero_degree_49 :
    ∃ Ω : Finset Cell97, #Ω = 4704 ∧ (∀ x ∈ Ω, ∏ c ∈ range 49, (x 0 - (c : ZMod 97)) = 0) ∧
      ¬ Determines (degreeClass (ZMod 97) 2 49 : Set (Cell97 → ZMod 97)) Ω := by
  obtain ⟨T, hT⟩ : ∃ T : Finset (ZMod 97), T = (range 49).image fun c : ℕ => (c : ZMod 97) :=
    ⟨_, rfl⟩
  have hTcard : #T = 49 := hT ▸ card_image_range_cast (by norm_num)
  obtain ⟨Ω, hcard, hmem⟩ := exists_part_first_mem (F := ZMod 97) (k := 1) (m := 4704) T
    (by rw [hTcard, ZMod.card]; norm_num)
  have hzero (x : Cell97) (hx : x ∈ Ω) : ∏ c ∈ range 49, (x 0 - (c : ZMod 97)) = 0 := by
    have hxT : x 0 ∈ T := hmem x hx
    rw [hT] at hxT
    obtain ⟨c, hc, hxc⟩ := mem_image.1 hxT
    exact prod_eq_zero hc (sub_eq_zero.2 hxc.symm)
  refine ⟨Ω, hcard, hzero, fun hdet => ?_⟩
  have heq := hdet _ (prod_sub_mem_degreeClass 49) 0 (Submodule.zero_mem _) fun x hx => by
    simpa using hzero x (by exact_mod_cast hx)
  have hval := congrFun heq ![96, 0]
  simp only [Pi.zero_apply, Matrix.cons_val_zero] at hval
  obtain ⟨c, hc, hc0⟩ := prod_eq_zero_iff.1 hval
  have hc' : c < 49 := mem_range.1 hc
  have hcast : ((96 : ℕ) : ZMod 97) = (c : ZMod 97) := by
    have := sub_eq_zero.1 hc0
    exact_mod_cast this
  have := (ZMod.natCast_eq_natCast_iff' 96 c 97).1 hcast
  rw [Nat.mod_eq_of_lt (by norm_num), Nat.mod_eq_of_lt (by omega)] at this
  omega

/-- `a³ + a b` is a polynomial of degree at most 3. -/
lemma cube_add_mul_mem_degreeClass :
    (fun x : Cell97 => x 0 ^ 3 + x 0 * x 1) ∈ degreeClass (ZMod 97) 2 3 := by
  refine mem_degreeClass.2 ⟨X 0 ^ 3 + X 0 * X 1, ?_, fun x => by simp⟩
  refine (totalDegree_add _ _).trans (max_le ((totalDegree_pow _ _).trans (by simp)) ?_)
  exact (totalDegree_mul _ _).trans (by simp)

example : (fun x : Cell97 => x 0 ^ 3 + x 0 * x 1) ![1, 2] = 3 := by decide

/-- **Any 292 cells determine `a³ + a b` inside `H₃`** (`review/flat_spectrum.md`, PT2):
`3 · 97 = 291`, so a member of `H₃` that agrees with it on at least 292 cells is it. -/
theorem eq_cube_add_mul_of_card (Ω : Finset Cell97) (hΩ : 292 ≤ #Ω) {f : Cell97 → ZMod 97}
    (hf : f ∈ degreeClass (ZMod 97) 2 3) (hag : ∀ x ∈ Ω, f x = x 0 ^ 3 + x 0 * x 1) :
    f = fun x => x 0 ^ 3 + x 0 * x 1 :=
  (forall_determines_degreeClass_iff (F := ZMod 97) (k := 1) (d := 3)
    (by simp [ZMod.card]) #Ω).2 (by simpa [ZMod.card] using by omega) Ω rfl f hf _
    cube_add_mul_mem_degreeClass fun x hx => hag x (by exact_mod_cast hx)

example : (fun x : Cell97 => x 0 ^ 3 + x 0 * x 1) = fun x => x 0 ^ 3 + x 0 * x 1 :=
  eq_cube_add_mul_of_card univ (by simp [card_univ, ZMod.card]) cube_add_mul_mem_degreeClass
    fun _ _ => rfl

/-- **Division is a polynomial** (`review/flat_spectrum.md`, PT2): `a / b = a b^{95}` over `ℤ/97`,
with `a / 0 = 0` on both sides. -/
theorem div_eq_mul_pow (a b : ZMod 97) : a / b = a * b ^ 95 := by
  by_cases hb : b = 0
  · simp [hb]
  · have h96 : b ^ 95 * b = 1 := by
      have h := ZMod.pow_card_sub_one_eq_one hb
      rwa [show (97 - 1 : ℕ) = 95 + 1 from rfl, pow_succ] at h
    rw [div_eq_mul_inv, ← eq_inv_of_mul_eq_one_left h96]

example : (6 : ZMod 97) / 2 = 6 * 2 ^ 95 := div_eq_mul_pow 6 2

/-- The polynomial of division, `a b^{95}`, has degree 96 (`review/flat_spectrum.md`, PT2). -/
theorem totalDegree_mul_pow :
    (X 0 * X 1 ^ 95 : MvPolynomial (Fin 2) (ZMod 97)).totalDegree = 96 := by
  rw [totalDegree_mul_of_isDomain (X_ne_zero _) (pow_ne_zero _ (X_ne_zero _)), totalDegree_X,
    totalDegree_X_pow]

/-- **Degree guarantees nothing for division at a half** (`review/flat_spectrum.md`, PT2): some
part of 4704 cells does not determine `H₉₆`, since `96 · 97 = 9312 ≥ 4704`. -/
theorem exists_half_not_determines_degree_96 :
    ∃ Ω : Finset Cell97, #Ω = 4704 ∧
      ¬ Determines (degreeClass (ZMod 97) 2 96 : Set (Cell97 → ZMod 97)) Ω := by
  by_contra h
  push Not at h
  have hlt := (forall_determines_degreeClass_iff (F := ZMod 97) (k := 1) (d := 96)
    (by simp [ZMod.card]) 4704).1 h
  norm_num [ZMod.card] at hlt

end Transformer.Grokking.FlatSpectrum.TablePart
