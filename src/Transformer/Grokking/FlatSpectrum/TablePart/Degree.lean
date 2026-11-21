module

public import Mathlib.Algebra.MvPolynomial.CommRing
public import Mathlib.Algebra.MvPolynomial.SchwartzZippel
public import Mathlib.FieldTheory.Finite.Polynomial
public import Mathlib.RingTheory.MvPolynomial.Basic
public import Transformer.Grokking.FlatSpectrum.TablePart.Threshold

/-!
# PT2. The classical case: bounded degree

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT2. Over a finite field
`F` with `q` elements, `degreeClass F k d` is the class `H_d` of the functions on `F^k` given by a
polynomial of total degree at most `d`. A nonzero polynomial of degree at most `d` has at most
`d q^{k−1}` zeros by Schwartz–Zippel (`card_zeros_mul_le`), and `∏_{c ∈ T} (x₀ − c)` with `|T| = d`
has exactly that many (`exists_mem_degreeClass_hammingNorm_eq`). So for `d < q`

  `δ(H_d) = (q − d) q^{k−1}` (`minDistance_degreeClass`),

the minimum distance of the generalized Reed–Muller code. In Lean the number of variables is
`k + 1`, since the statement needs at least one. `DegreeHalf` turns this into a count of cells.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset MvPolynomial

variable (F : Type*) [Field F]

/-- `H_d` (`review/flat_spectrum.md`, PT2): the functions on `F^k` given by a polynomial of total
degree at most `d`. -/
noncomputable def degreeClass (k d : ℕ) : Submodule F ((Fin k → F) → F) :=
  (restrictTotalDegree (Fin k) F d).map (evalₗ F (Fin k))

variable {F}

/-- A member of `H_d` is the evaluation of a polynomial of total degree at most `d`. -/
lemma mem_degreeClass {k d : ℕ} {f : (Fin k → F) → F} :
    f ∈ degreeClass F k d ↔
      ∃ P : MvPolynomial (Fin k) F, P.totalDegree ≤ d ∧ ∀ x, f x = eval x P := by
  simp only [degreeClass, Submodule.mem_map, mem_restrictTotalDegree]
  constructor
  · rintro ⟨P, hP, rfl⟩
    exact ⟨P, hP, fun _ => rfl⟩
  · rintro ⟨P, hP, hf⟩
    exact ⟨P, hP, funext fun x => (hf x).symm⟩

example : (fun x : Fin 1 → ℚ => x 0) ∈ degreeClass ℚ 1 1 :=
  mem_degreeClass.2 ⟨X 0, by simp, fun x => by simp⟩

variable [Fintype F]

/-- **Schwartz–Zippel on the whole table** (`review/flat_spectrum.md`, PT2): a nonzero polynomial
has at most `deg P · q^{k−1}` zeros, written without division. -/
lemma card_zeros_mul_le [DecidableEq F] {k : ℕ} {P : MvPolynomial (Fin k) F} (hP : P ≠ 0) :
    #{x : Fin k → F | eval x P = 0} * Fintype.card F ≤ P.totalDegree * Fintype.card F ^ k := by
  have h := schwartz_zippel_totalDegree hP (univ : Finset F)
  simp only [Fintype.piFinset_univ, card_univ] at h
  have hq : (0 : NNRat) < Fintype.card F := by exact_mod_cast Fintype.card_pos
  rw [div_le_div_iff₀ (pow_pos hq k) hq] at h
  exact_mod_cast h

example : #{x : Fin 1 → ZMod 2 | eval x (X 0 : MvPolynomial (Fin 1) (ZMod 2)) = 0} *
    Fintype.card (ZMod 2) ≤ (X 0 : MvPolynomial (Fin 1) (ZMod 2)).totalDegree *
      Fintype.card (ZMod 2) ^ 1 :=
  card_zeros_mul_le (X_ne_zero 0)

/-- A nonzero member of `H_d` on `k + 1` variables has at least `(q − d) q^k` nonzero cells. -/
lemma mul_pow_le_hammingNorm [DecidableEq F] {k d : ℕ} {f : (Fin (k + 1) → F) → F}
    (hf : f ∈ degreeClass F (k + 1) d) (hf0 : f ≠ 0) :
    (Fintype.card F - d) * Fintype.card F ^ k ≤ hammingNorm f := by
  obtain ⟨P, hP, hfP⟩ := mem_degreeClass.1 hf
  have hP0 : P ≠ 0 := by
    rintro rfl
    exact hf0 (funext fun x => by simpa using hfP x)
  have hq : 0 < Fintype.card F := Fintype.card_pos
  have hzeros : #{x : Fin (k + 1) → F | f x = 0} ≤ d * Fintype.card F ^ k := by
    have hset : #{x : Fin (k + 1) → F | f x = 0} = #{x : Fin (k + 1) → F | eval x P = 0} :=
      congrArg card (filter_congr fun x _ => by rw [hfP x])
    rw [hset]
    refine Nat.le_of_mul_le_mul_right ?_ hq
    calc _ ≤ P.totalDegree * Fintype.card F ^ (k + 1) := card_zeros_mul_le hP0
      _ ≤ d * Fintype.card F ^ (k + 1) := Nat.mul_le_mul_right _ hP
      _ = d * Fintype.card F ^ k * Fintype.card F := by ring
  have hsplit := card_filter_add_card_filter_not (s := (univ : Finset (Fin (k + 1) → F)))
    (fun x => f x = 0)
  have hN : #(univ : Finset (Fin (k + 1) → F)) = Fintype.card F ^ k * Fintype.card F := by
    simp [card_univ, pow_succ]
  have hnorm : #{x : Fin (k + 1) → F | ¬ f x = 0} = hammingNorm f := rfl
  rw [hN, hnorm] at hsplit
  rw [Nat.sub_mul]
  have hcomm : Fintype.card F * Fintype.card F ^ k = Fintype.card F ^ k * Fintype.card F :=
    Nat.mul_comm _ _
  omega

example : (Fintype.card (ZMod 2) - 0) * Fintype.card (ZMod 2) ^ 0 ≤
    hammingNorm (fun _ => 1 : (Fin 1 → ZMod 2) → ZMod 2) :=
  mul_pow_le_hammingNorm (mem_degreeClass.2 ⟨C 1, by simp, fun x => by simp⟩)
    fun h => one_ne_zero (congrFun h 0)

omit [Fintype F] in
/-- `∏_{c ∈ T} (x₀ − c)` has total degree at most `|T|`. -/
lemma totalDegree_prod_X_sub_C_le {k : ℕ} (T : Finset F) :
    (∏ c ∈ T, (X 0 - C c : MvPolynomial (Fin (k + 1)) F)).totalDegree ≤ #T := by
  classical
  induction T using Finset.induction_on with
  | empty => simp
  | insert c T hc ih =>
    rw [prod_insert hc, card_insert_of_notMem hc]
    have h1 : (X 0 - C c : MvPolynomial (Fin (k + 1)) F).totalDegree ≤ 1 :=
      (totalDegree_sub _ _).trans (by simp)
    exact (totalDegree_mul _ _).trans (by omega)

example : (∏ c ∈ ({0} : Finset (ZMod 2)),
    (X 0 - C c : MvPolynomial (Fin 1) (ZMod 2))).totalDegree ≤ 1 :=
  totalDegree_prod_X_sub_C_le _

omit [Field F] in
/-- There are `|T| q^k` points of `F^{k+1}` whose first coordinate is in `T`. -/
lemma card_first_mem [DecidableEq F] {k : ℕ} (T : Finset F) :
    #{x : Fin (k + 1) → F | x 0 ∈ T} = #T * Fintype.card F ^ k := by
  have hset : (univ.filter fun x : Fin (k + 1) → F => x 0 ∈ T) =
      Fintype.piFinset fun i => if i = 0 then T else univ := by
    ext x
    simp only [mem_filter, mem_univ, true_and, Fintype.mem_piFinset]
    constructor
    · intro hx i
      by_cases hi : i = 0
      · subst hi
        simpa using hx
      · simp [hi]
    · intro hx
      simpa using hx 0
  rw [hset, Fintype.card_piFinset, Fin.prod_univ_succ]
  simp [Fin.succ_ne_zero]

omit [Field F] in
/-- There are `(q − |T|) q^k` points of `F^{k+1}` whose first coordinate is outside `T`. -/
lemma card_first_notMem [DecidableEq F] {k : ℕ} (T : Finset F) :
    #{x : Fin (k + 1) → F | x 0 ∉ T} = (Fintype.card F - #T) * Fintype.card F ^ k := by
  have h := card_first_mem (k := k) Tᶜ
  rw [card_compl] at h
  simpa only [mem_compl] using h

example : #{x : Fin 1 → ZMod 2 | x 0 ∉ ({0} : Finset (ZMod 2))} =
    (Fintype.card (ZMod 2) - #({0} : Finset (ZMod 2))) * Fintype.card (ZMod 2) ^ 0 :=
  card_first_notMem _

/-- For `d < q`, `H_d` has a nonzero member with exactly `(q − d) q^k` nonzero cells:
`∏_{c ∈ T} (x₀ − c)` with `|T| = d`. -/
lemma exists_mem_degreeClass_hammingNorm_eq [DecidableEq F] {k d : ℕ}
    (hd : d < Fintype.card F) :
    ∃ f ∈ degreeClass F (k + 1) d, f ≠ 0 ∧
      hammingNorm f = (Fintype.card F - d) * Fintype.card F ^ k := by
  obtain ⟨T, -, hT⟩ := exists_subset_card_eq (s := (univ : Finset F)) (n := d)
    (by simpa [card_univ] using hd.le)
  let P : MvPolynomial (Fin (k + 1)) F := ∏ c ∈ T, (X 0 - C c)
  have heval (x : Fin (k + 1) → F) : eval x P = 0 ↔ x 0 ∈ T := by
    simp [P, map_prod, prod_eq_zero_iff, sub_eq_zero]
  have hnorm : hammingNorm (fun x => eval x P) = (Fintype.card F - d) * Fintype.card F ^ k := by
    rw [← hT, ← card_first_notMem]
    exact congrArg card (filter_congr fun x _ => not_congr (heval x))
  refine ⟨fun x => eval x P, mem_degreeClass.2 ⟨P, ?_, fun _ => rfl⟩, ?_, hnorm⟩
  · simpa [hT] using totalDegree_prod_X_sub_C_le (k := k) T
  · intro h0
    have hzero : hammingNorm (fun x => eval x P) = 0 := by rw [h0, hammingNorm_zero]
    rw [hnorm] at hzero
    have hq : 0 < Fintype.card F := Fintype.card_pos
    have := Nat.pos_of_ne_zero (pow_ne_zero k hq.ne')
    rcases Nat.mul_eq_zero.1 hzero with h | h <;> omega

example := exists_mem_degreeClass_hammingNorm_eq (F := ZMod 2) (k := 0) (d := 1)
  (by simp [ZMod.card])

/-- **The minimum distance of the generalized Reed–Muller code** (`review/flat_spectrum.md`, PT2;
Kasami, Lin and Peterson, IEEE Trans. Inform. Theory 14 (1968), 189–199): for `d < q`,
`δ(H_d) = (q − d) q^{k−1}`, on `k + 1` variables here. -/
theorem minDistance_degreeClass [DecidableEq F] {k d : ℕ} (hd : d < Fintype.card F) :
    minDistance (degreeClass F (k + 1) d : Set ((Fin (k + 1) → F) → F)) =
      (Fintype.card F - d) * Fintype.card F ^ k := by
  rw [minDistance_submodule]
  refine IsLeast.csInf_eq ⟨?_, ?_⟩
  · obtain ⟨f, hf, h0, hn⟩ := exists_mem_degreeClass_hammingNorm_eq (k := k) hd
    exact ⟨f, hf, h0, hn⟩
  · rintro _ ⟨f, hf, h0, rfl⟩
    exact mul_pow_le_hammingNorm hf h0

example := minDistance_degreeClass (F := ZMod 2) (k := 1) (d := 1) (by simp [ZMod.card])

end Transformer.Grokking.FlatSpectrum.TablePart
