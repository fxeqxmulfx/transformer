module

public import Transformer.Grokking.FlatSpectrum.CertificateBalanced
public import Transformer.Grokking.FlatSpectrum.TablePart.Cyclotomic
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Lines
public import Transformer.Grokking.FlatSpectrum.TablePart.E6SumWitnesses

/-!
# PT6: tables of `a + b`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. A table `φ(a + b)`
keeps its kernel under `(a, b) ↦ (a + t, b − t)`. If the kernel is invertible, all certificates of
it agree, so `not_isMinOn_of_orbits` needs one unseen cell on every anti-diagonal `a + b = s`. The
kernel is invertible because its image in `ZMod 389`, with `ζ ↦ 256`, has a right inverse
`w(b + c)`, and the kernel of Lean checks this inverse over `ℕ` (`ConvCheck`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix Polynomial

attribute [local instance] fact_prime_97

/-- `389 = 4 · 97 + 1` is prime. -/
instance fact_prime_389 : Fact (Nat.Prime 389) := ⟨by norm_num⟩

/-- `∑_s (256 ^ e(s) + c) W(s + d) ≡ δ(d, 0)` mod 389. -/
abbrev ConvCheck (e : ℕ → ℕ) (c : ℕ) (W : List ℕ) : Prop :=
  ∀ d, d < 97 →
    (∑ s ∈ range 97, (256 ^ e s + c) * W.getD ((s + d) % 97) 0) % 389 = if d = 0 then 1 else 0

/-- The check for `(a + b)³`. -/
theorem checkPow3 : ConvCheck (fun s => s ^ 3 % 97) 0 wPow3 := by
  decide +kernel

/-- The check over `ℕ` is the identity in `ZMod 389`. -/
lemma conv_of_check {e : ℕ → ℕ} {c : ℕ} {W : List ℕ} (h : ConvCheck e c W) (d : ZMod 97) :
    ∑ s : ZMod 97, ((256 : ZMod 389) ^ e s.val + (c : ZMod 389)) *
        (W.getD (s + d).val 0 : ZMod 389) = if d = 0 then 1 else 0 := by
  have h1 := congrArg (Nat.cast : ℕ → ZMod 389) (h d.val d.val_lt)
  rw [ZMod.natCast_mod] at h1
  push_cast at h1
  have hsum : ∑ s : ZMod 97, ((256 : ZMod 389) ^ e s.val + (c : ZMod 389)) *
      (W.getD (s + d).val 0 : ZMod 389) =
      ∑ s ∈ range 97, ((256 : ZMod 389) ^ e s + (c : ZMod 389)) *
        (W.getD ((s + d.val) % 97) 0 : ZMod 389) := by
    rw [← Fin.sum_univ_eq_sum_range (fun s => ((256 : ZMod 389) ^ e s + (c : ZMod 389)) *
      (W.getD ((s + d.val) % 97) 0 : ZMod 389)) 97]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [ZMod.val_add]
    rfl
  rw [hsum]
  have h2 : ∑ s ∈ range 97, ((256 : ZMod 389) ^ e s + (c : ZMod 389)) *
      (W.getD ((s + d.val) % 97) 0 : ZMod 389) =
      ((if d.val = 0 then 1 else 0 : ℕ) : ZMod 389) := by
    simpa using h1
  rw [h2]
  by_cases hd : d = 0
  · simp [hd]
  · have hv : d.val ≠ 0 := fun h0 => hd ((ZMod.val_eq_zero d).1 h0)
    simp [hd, hv]

example (d : ZMod 97) := conv_of_check checkPow3 d

/-- A right inverse of `f(a + b)` from the vector `w`. -/
lemma conv_mul_eq_one {f w : ZMod 97 → ZMod 389}
    (h : ∀ d, ∑ s, f s * w (s + d) = if d = 0 then 1 else 0) :
    (Matrix.of fun u v : Fin 1 → ZMod 97 => f (u 0 + v 0)) *
      (Matrix.of fun v c : Fin 1 → ZMod 97 => w (v 0 + c 0)) = 1 := by
  ext u c
  rw [Matrix.mul_apply]
  simp only [of_apply]
  have hline : ∑ j : Fin 1 → ZMod 97, f (u 0 + j 0) * w (j 0 + c 0) =
      ∑ x, f (u 0 + x) * w (x + c 0) :=
    (Equiv.funUnique (Fin 1) (ZMod 97)).sum_comp fun x => f (u 0 + x) * w (x + c 0)
  rw [hline]
  have h1 := h (c 0 - u 0)
  rw [← Equiv.sum_comp (Equiv.addLeft (u 0))] at h1
  simp only [Equiv.coe_addLeft] at h1
  have h2 (x : ZMod 97) : u 0 + x + (c 0 - u 0) = x + c 0 := by ring
  simp only [h2] at h1
  rw [h1]
  by_cases huc : u = c
  · subst huc
    simp
  · rw [one_apply_ne huc, ite_eq_right]
    intro h0
    exact huc (funext fun t => by rw [Subsingleton.elim t 0]; exact (sub_eq_zero.1 h0).symm)

example : (Matrix.of fun u v : Fin 1 → ZMod 97 => (if u 0 + v 0 = 0 then 1 else 0 : ZMod 389)) *
    (Matrix.of fun v c : Fin 1 → ZMod 97 => (if v 0 + c 0 = 0 then 1 else 0 : ZMod 389)) = 1 :=
  conv_mul_eq_one (f := fun s => if s = 0 then 1 else 0) (w := fun s => if s = 0 then 1 else 0)
    fun d => by
      rw [Finset.sum_eq_single 0 (fun s _ hs => by simp [hs]) (by simp)]
      simp

/-- **The kernel `[ψ(e(a + b)) + c]` is invertible** when `ConvCheck` holds. -/
theorem det_sum_ne_zero {e : ℕ → ℕ} {c : ℕ} {W : List ℕ} (h : ConvCheck e c W) :
    (Matrix.of fun u v : Fin 1 → ZMod 97 =>
      (ZMod.stdAddChar ((e (u 0 + v 0).val : ℕ) : ZMod 97) : ℂ) + c).det ≠ 0 := by
  set A : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℤ[X] :=
    Matrix.of fun u v => X ^ e (u 0 + v 0).val + C (c : ℤ) with hA
  have hAζ : A.map (aeval (ZMod.stdAddChar (1 : ZMod 97) : ℂ)) = Matrix.of fun u v =>
      (ZMod.stdAddChar ((e (u 0 + v 0).val : ℕ) : ZMod 97) : ℂ) + c := by
    ext u v
    simp [hA, stdAddChar_natCast]
  rw [← hAζ]
  refine det_aeval_ne_zero (q := 389) (z := 256) A (by decide) (by decide +kernel)
    (Matrix.of fun v t => (W.getD (v 0 + t 0).val 0 : ZMod 389)) ?_
  have hAz : A.map (eval₂ (Int.castRingHom (ZMod 389)) 256) =
      Matrix.of fun u v => (fun s : ZMod 97 => (256 : ZMod 389) ^ e s.val + (c : ZMod 389))
        (u 0 + v 0) := by
    ext u v
    simp [hA]
  have hconv := conv_mul_eq_one
    (f := fun s : ZMod 97 => (256 : ZMod 389) ^ e s.val + (c : ZMod 389))
    (w := fun x => (W.getD x.val 0 : ZMod 389)) (conv_of_check h)
  rw [hAz]
  exact hconv

example := det_sum_ne_zero checkPow3

/-- The shift of rows by `t`. -/
def shiftRow (t : ZMod 97) : Equiv.Perm (Fin 1 → ZMod 97) := Equiv.addRight fun _ => t

/-- The shift of columns by `−t`. -/
def shiftCol (t : ZMod 97) : Equiv.Perm (Fin 1 → ZMod 97) := Equiv.subRight fun _ => t

lemma shiftRow_eq (u : Fin 1 → ZMod 97) (r : ℕ) :
    shiftRow ((r : ZMod 97) - u 0) u = fun _ => (r : ZMod 97) := by
  funext x
  rw [Subsingleton.elim x 0]
  simp [shiftRow]

lemma cast_diag {s r : ℕ} (hr : r < 97) :
    (((s + 97 - r) % 97 : ℕ) : ZMod 97) = (s : ZMod 97) - r := by
  rw [ZMod.natCast_mod, Nat.cast_sub (by omega), Nat.cast_add]
  have h : ((97 : ℕ) : ZMod 97) = 0 := ZMod.natCast_self 97
  rw [h, add_zero]

example : (((3 + 97 - 5) % 97 : ℕ) : ZMod 97) = ((3 : ℕ) : ZMod 97) - (5 : ℕ) :=
  cast_diag (by norm_num)

lemma shiftCol_eq (u v : Fin 1 → ZMod 97) {r : ℕ} (hr : r < 97) :
    shiftCol ((r : ZMod 97) - u 0) v =
      fun _ => ((((u 0 + v 0).val + 97 - r) % 97 : ℕ) : ZMod 97) := by
  funext x
  rw [Subsingleton.elim x 0, cast_diag hr, ZMod.natCast_zmod_val]
  simp only [shiftCol, Equiv.subRight_apply, Pi.sub_apply]
  ring

example (u v : Fin 1 → ZMod 97) := shiftCol_eq u v (r := 5) (by norm_num)

/-- **Every anti-diagonal has a cell outside the training half.** -/
theorem diag_cover : ∀ s, s < 97 → ∃ r, r < 97 ∧
    halfMask.testBit (97 * r + (s + 97 - r) % 97) = false := by
  decide +kernel

/-- **M does not return a table of `a + b` with an invertible kernel from the training half.** -/
theorem not_returned_sum {e : ℕ → ℕ} {W : List ℕ} (h : ConvCheck e 0 W) :
    NotReturned fun a b => (e (a + b).val : ZMod 97) := by
  have hdet : (tableKernel fun a b => (e (a + b).val : ZMod 97)).det ≠ 0 := by
    have h0 := det_sum_ne_zero h
    simpa [tableKernel] using h0
  refine not_isMinOn_of_orbits hdet shiftRow shiftCol (fun _ _ => 1) (fun _ _ => 1) (by simp)
    (by simp) (fun t u v => ?_) fun u v => ?_
  · simp [tableKernel, shiftRow, shiftCol]
  · obtain ⟨r, hr, hbit⟩ := diag_cover (u 0 + v 0).val (ZMod.val_lt _)
    refine ⟨(r : ZMod 97) - u 0, ?_⟩
    rw [shiftRow_eq, shiftCol_eq u v hr]
    intro hmem
    have hb := (mem_half_iff hr (Nat.mod_lt _ (by norm_num))).1 (mem_coe.1 hmem)
    rw [hbit] at hb
    exact Bool.false_ne_true hb

example := not_returned_sum checkPow3

end Transformer.Grokking.FlatSpectrum.TablePart.E6
