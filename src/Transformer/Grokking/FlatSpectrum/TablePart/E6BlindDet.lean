module

public import Transformer.Grokking.FlatSpectrum.TablePart.DivClasses
public import Transformer.Grokking.FlatSpectrum.TablePart.E6BlindInverse
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Sums
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Symmetric
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Tridiagonal

/-!
# PT6: the kernel of `⌊|a − b| / 2⌋` is invertible

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The entries of the
kernel are powers of `ζ = ψ(1)`. So it is invertible once the same powers of an element of order 97
in `ZMod 389` have an inverse (`det_aeval_ne_zero`). That inverse is the list of `E6BlindInverse`,
and the kernel of Lean multiplies it out, a quarter of the rows per theorem.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix Polynomial

attribute [local instance] fact_prime_97

/-- The table `⌊|a − b| / 2⌋` of the integers `0, …, 96`. -/
def blindTable (a b : ZMod 97) : ZMod 97 := (natDist a.val b.val / 2 : ℕ)

/-- The position in `blindInvPacked` of the entry `(i, j)`, `i ≤ j`. -/
def blindIdxOf (i j : ℕ) : ℕ :=
  bif Nat.ble (i + j) 96 then i * (98 - i) + (j - i)
  else (96 - j) * (98 - (96 - j)) + (j - i)

/-- The entry `(x, b)` of the inverse. -/
def blindInv (x b : ℕ) : ℕ :=
  (blindInvPacked >>> (9 * blindIdxOf (x - (x - b)) (x + (b - x)))) &&& 511

/-- The sum of `256 ^ ⌊|a − x| / 2⌋ · W x b` over `x < n`, added to `acc`. -/
def blindDot (a b : ℕ) : ℕ → ℕ → ℕ
  | 0, acc => acc
  | n + 1, acc => blindDot a b n (acc + 256 ^ (natDist a n / 2) * blindInv n b)

/-- Row `a` of the product of the kernel and the inverse is the row of the identity. -/
def blindRowOk (a : ℕ) : Bool :=
  (List.range 97).all fun b => blindDot a b 97 0 % 389 == if a == b then 1 else 0

theorem blind_rows_0 : (List.range' 0 25).all blindRowOk = true := by decide +kernel

theorem blind_rows_25 : (List.range' 25 25).all blindRowOk = true := by decide +kernel

theorem blind_rows_50 : (List.range' 50 25).all blindRowOk = true := by decide +kernel

theorem blind_rows_75 : (List.range' 75 22).all blindRowOk = true := by decide +kernel

lemma blindRowOk_of_lt {a : ℕ} (ha : a < 97) : blindRowOk a = true := by
  have key {s n : ℕ} (h : (List.range' s n).all blindRowOk = true) (h1 : s ≤ a)
      (h2 : a < s + n) : blindRowOk a = true :=
    List.all_eq_true.1 h a (List.mem_range'_1.2 ⟨h1, h2⟩)
  by_cases h25 : a < 25
  · exact key blind_rows_0 (by omega) (by omega)
  by_cases h50 : a < 50
  · exact key blind_rows_25 (by omega) (by omega)
  by_cases h75 : a < 75
  · exact key blind_rows_50 (by omega) (by omega)
  exact key blind_rows_75 (by omega) (by omega)

example : blindRowOk 3 = true := blindRowOk_of_lt (by norm_num)

lemma blindDot_eq (a b n acc : ℕ) :
    blindDot a b n acc = acc + ∑ x ∈ range n, 256 ^ (natDist a x / 2) * blindInv x b := by
  induction n generalizing acc with
  | zero => simp [blindDot]
  | succ n ih =>
    rw [blindDot, ih, sum_range_succ]
    ring

lemma blind_sum {a b : ℕ} (ha : a < 97) (hb : b < 97) :
    (∑ x ∈ range 97, 256 ^ (natDist a x / 2) * blindInv x b) % 389 =
      if a = b then 1 else 0 := by
  have h := List.all_eq_true.1 (blindRowOk_of_lt ha) b (List.mem_range.2 hb)
  rw [beq_iff_eq, blindDot_eq, zero_add] at h
  simpa using h

example : (∑ x ∈ range 97, 256 ^ (natDist 3 x / 2) * blindInv x 3) % 389 =
    if 3 = 3 then 1 else 0 := blind_sum (by norm_num) (by norm_num)

lemma line_eq_iff (u w : Fin 1 → ZMod 97) : u = w ↔ (u 0).val = (w 0).val := by
  constructor
  · rintro rfl
    rfl
  · intro h
    rw [← line_val u, ← line_val w, h]

/-- **The kernel of `⌊|a − b| / 2⌋` is invertible.** -/
theorem det_blind_ne_zero : (tableKernel blindTable).det ≠ 0 := by
  set A : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℤ[X] :=
    Matrix.of fun u v => X ^ (natDist (u 0).val (v 0).val / 2) with hA
  have hAζ : A.map (aeval (ZMod.stdAddChar (1 : ZMod 97) : ℂ)) = tableKernel blindTable := by
    ext u v
    simp [hA, tableKernel, blindTable, stdAddChar_natCast]
  rw [← hAζ]
  refine det_aeval_ne_zero (q := 389) (z := 256) A (by decide) (by decide +kernel)
    (Matrix.of fun v t => (blindInv (v 0).val (t 0).val : ZMod 389)) ?_
  ext u w
  rw [Matrix.mul_apply, Matrix.one_apply]
  have hterm (v : Fin 1 → ZMod 97) :
      (A.map (eval₂ (Int.castRingHom (ZMod 389)) 256)) u v *
          (Matrix.of fun v t => (blindInv (v 0).val (t 0).val : ZMod 389)) v w =
        (fun x : ZMod 97 => (256 : ZMod 389) ^ (natDist (u 0).val x.val / 2) *
          (blindInv x.val (w 0).val : ZMod 389)) ((Equiv.funUnique (Fin 1) (ZMod 97)) v) := by
    simp [hA]
  rw [Finset.sum_congr rfl fun v _ => hterm v,
    (Equiv.funUnique (Fin 1) (ZMod 97)).sum_comp fun x : ZMod 97 =>
      (256 : ZMod 389) ^ (natDist (u 0).val x.val / 2) * (blindInv x.val (w 0).val : ZMod 389),
    sum_zmod_eq_sum_range 97 fun n =>
      (256 : ZMod 389) ^ (natDist (u 0).val n / 2) * (blindInv n (w 0).val : ZMod 389)]
  have hnat := congrArg (Nat.cast : ℕ → ZMod 389)
    (blind_sum (ZMod.val_lt (u 0)) (ZMod.val_lt (w 0)))
  rw [ZMod.natCast_mod] at hnat
  push_cast at hnat
  rw [hnat]
  by_cases h : u = w
  · simp [h]
  · simp [h, (line_eq_iff u w).not.1 h]

end Transformer.Grokking.FlatSpectrum.TablePart.E6
