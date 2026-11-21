module

public import Transformer.Grokking.FlatSpectrum.TablePart.Cyclotomic
public import Transformer.Grokking.FlatSpectrum.TablePart.DivClasses
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Tridiagonal

/-!
# PT6: M misses `|a − b|` and `max(a, b)`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The kernel of
`|a − b|` is the matrix `[ρ^{|a − b|}]`, with `ρ = ψ(1)`, and `(1 − ρ²)` times its inverse is the
tridiagonal `tri ρ`. The kernel of `max(a, b)` is the same matrix for `σ = ψ(49)`, between two
diagonal phases, because `σ² = ρ` and `a + |a − b| + b = 2 max(a, b)`. Both keep their form under
the flip `x ↦ 96 − x`, so `not_returned_of_peel` applies to both.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- The table `|a − b|`. -/
def absDiffTable (a b : ZMod 97) : ZMod 97 := (natDist a.val b.val : ℕ)

/-- The table `max(a, b)`. -/
def maxTable (a b : ZMod 97) : ZMod 97 := (max a.val b.val : ℕ)

/-- The matrix of `tri` on the lines, between the phases `w^a` and `w^b`. -/
noncomputable def triMatrix (w ρ : ℂ) : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ :=
  Matrix.of fun u v => w ^ (u 0).val * tri ρ (u 0).val (v 0).val * w ^ (v 0).val

lemma triMatrix_line (w ρ : ℂ) {l j : ℕ} (hl : l < 97) (hj : j < 97) :
    triMatrix w ρ (line 97 l) (line 97 j) = w ^ l * tri ρ l j * w ^ j := by
  simp [triMatrix, line, ZMod.val_natCast_of_lt hl, ZMod.val_natCast_of_lt hj]

example (w ρ : ℂ) := triMatrix_line w ρ (l := 1) (j := 2) (by norm_num) (by norm_num)

lemma triMatrix_ne_zero {w ρ : ℂ} (hw : w ≠ 0) (h0 : ρ ≠ 0) (h2 : 1 + ρ ^ 2 ≠ 0) {l j : ℕ}
    (hl : l < 97) (hj : j < 97) (hn : j ∈ nb l) : triMatrix w ρ (line 97 l) (line 97 j) ≠ 0 := by
  rw [triMatrix_line w ρ hl hj]
  exact mul_ne_zero (mul_ne_zero (pow_ne_zero _ hw) (tri_ne_zero h0 h2 hl hn)) (pow_ne_zero _ hw)

lemma triMatrix_eq_zero (w ρ : ℂ) {l j : ℕ} (hl : l < 97) (hj : j < 97) (hn : j ∉ nb l) :
    triMatrix w ρ (line 97 l) (line 97 j) = 0 := by
  rw [triMatrix_line w ρ hl hj, tri_eq_zero ρ hl hj hn]
  simp

/-- One peeling step on the zero certificate: the hypotheses of `Vanishes.step` can be met. -/
example : Vanishes (0 : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ) peel1 :=
  Vanishes.step (c := 1) (T := triMatrix 1 2) (γ := fun _ => 1) (δ := fun _ => 1) one_ne_zero
    (by simp) (by simp) (by simp)
    (fun _ _ hl hj hn => triMatrix_ne_zero one_ne_zero (by norm_num) (by norm_num) hl hj hn)
    (fun _ _ hl hj hn => triMatrix_eq_zero 1 2 hl hj hn) (S := halfMask)
    (fun _ _ _ _ _ => rfl) peel_chain.1

/-- The kernel of `|a − b|` in the form that `kernel_mul_triMatrix` asks for. -/
lemma stdAddChar_absDiffTable (a b : ZMod 97) : (ZMod.stdAddChar (absDiffTable a b) : ℂ) =
    (1 : ℂ) ^ a.val * (ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ natDist a.val b.val *
      (1 : ℂ) ^ b.val := by
  simp [absDiffTable, stdAddChar_natCast]

/-- `K T = (1 − σ²) 1` for a kernel `g^a σ^{|a − b|} g^b` and `w g = 1`. -/
theorem kernel_mul_triMatrix {f : ZMod 97 → ZMod 97 → ZMod 97} {g w σ : ℂ} (hw : w * g = 1)
    (hf : ∀ a b : ZMod 97, (ZMod.stdAddChar (f a b) : ℂ) =
      g ^ a.val * σ ^ natDist a.val b.val * g ^ b.val) :
    tableKernel f * triMatrix w σ = (1 - σ ^ 2) • 1 := by
  ext u v
  rw [Matrix.mul_apply, Matrix.smul_apply]
  simp only [tableKernel, triMatrix, of_apply, hf]
  rw [sum_line_apply (fun x => g ^ (u 0).val * σ ^ natDist (u 0).val x.val * g ^ x.val *
    (w ^ x.val * tri σ x.val (v 0).val * w ^ (v 0).val))]
  have hterm (x : ZMod 97) : g ^ (u 0).val * σ ^ natDist (u 0).val x.val * g ^ x.val *
      (w ^ x.val * tri σ x.val (v 0).val * w ^ (v 0).val) =
      g ^ (u 0).val * w ^ (v 0).val * (σ ^ natDist (u 0).val x.val * tri σ x.val (v 0).val) := by
    have hx : g ^ x.val * w ^ x.val = 1 := by rw [← mul_pow, mul_comm, hw, one_pow]
    linear_combination (g ^ (u 0).val * σ ^ natDist (u 0).val x.val *
      tri σ x.val (v 0).val * w ^ (v 0).val) * hx
  simp only [hterm]
  rw [← Finset.mul_sum,
    sum_zmod_eq_sum_range 97 (fun n => σ ^ natDist (u 0).val n * tri σ n (v 0).val),
    sum_kms_tri σ (ZMod.val_lt _) (ZMod.val_lt _)]
  by_cases huv : u = v
  · subst huv
    have h1 : g ^ (u 0).val * w ^ (u 0).val = 1 := by rw [← mul_pow, mul_comm, hw, one_pow]
    rw [h1]
    simp
  · have hval : (u 0).val ≠ (v 0).val := fun h => huv (funext fun t => by
      rw [Subsingleton.elim t 0]
      exact ZMod.val_injective _ h)
    simp [one_apply_ne huv, hval]

example : tableKernel absDiffTable * triMatrix 1 (ZMod.stdAddChar (1 : ZMod 97)) =
    (1 - (ZMod.stdAddChar (1 : ZMod 97) : ℂ) ^ 2) • 1 :=
  kernel_mul_triMatrix (g := 1) (one_mul 1) stdAddChar_absDiffTable

/-- **M misses a table `g^a σ^{|a − b|} g^b` that keeps its form under the flip.** -/
theorem not_returned_of_tri {f : ZMod 97 → ZMod 97 → ZMod 97} {g w σ : ℂ} (hw : w * g = 1)
    (hσ0 : σ ≠ 0) (hσ1 : σ ^ 2 ≠ 1) (hσ2 : 1 + σ ^ 2 ≠ 0)
    (hf : ∀ a b : ZMod 97, (ZMod.stdAddChar (f a b) : ℂ) =
      g ^ a.val * σ ^ natDist a.val b.val * g ^ b.val)
    (hs : ∀ a b, f b a = f a b) {α β : ZMod 97 → ℂ} (hα : ∀ x, ‖α x‖ = 1) (hβ : ∀ x, ‖β x‖ = 1)
    (hF : ∀ a b, (ZMod.stdAddChar (f (-1 - a) (-1 - b)) : ℂ) =
      α a * ZMod.stdAddChar (f a b) * β b) : NotReturned f :=
  not_returned_of_peel (kernel_mul_triMatrix hw hf) (sub_ne_zero.2 hσ1.symm)
    (fun _ _ hl hj hn => triMatrix_ne_zero (left_ne_zero_of_mul_eq_one hw) hσ0 hσ2 hl hj hn)
    (fun _ _ hl hj hn => triMatrix_eq_zero w σ hl hj hn) hs hα hβ hF

lemma absDiffTable_comm (a b : ZMod 97) : absDiffTable b a = absDiffTable a b := by
  rw [absDiffTable, absDiffTable, natDist_comm]

lemma maxTable_comm (a b : ZMod 97) : maxTable b a = maxTable a b := by
  rw [maxTable, maxTable, max_comm]

lemma absDiffTable_flip : ∀ a b : ZMod 97, absDiffTable (-1 - a) (-1 - b) = absDiffTable a b := by
  decide +kernel

lemma maxTable_flip : ∀ a b : ZMod 97,
    maxTable (-1 - a) (-1 - b) = (-1 - a) + maxTable a b + (-b) := by
  decide +kernel

/-- `ψ(max(a, b)) = σ^a σ^{|a − b|} σ^b` for `σ = ψ(49)`, because `2 · 49 = 1`. -/
lemma stdAddChar_maxTable (a b : ZMod 97) : (ZMod.stdAddChar (maxTable a b) : ℂ) =
    (ZMod.stdAddChar (49 : ZMod 97) : ℂ) ^ a.val *
      (ZMod.stdAddChar (49 : ZMod 97) : ℂ) ^ natDist a.val b.val *
        (ZMod.stdAddChar (49 : ZMod 97) : ℂ) ^ b.val := by
  have he : a.val + natDist a.val b.val + b.val = 2 * max a.val b.val := by
    unfold natDist
    omega
  rw [← pow_add, ← pow_add, he, ← AddChar.map_nsmul_eq_pow, maxTable]
  congr 1
  generalize max a.val b.val = m
  have h : (2 : ZMod 97) * 49 = 1 := by decide +kernel
  rw [nsmul_eq_mul, Nat.cast_mul, Nat.cast_ofNat]
  linear_combination (-(m : ZMod 97)) * h

/-- **M does not return the table `|a − b|` from the training half.** -/
theorem not_returned_absDiff : NotReturned absDiffTable :=
  not_returned_of_tri (g := 1) (w := 1) (σ := ZMod.stdAddChar (1 : ZMod 97)) (one_mul 1)
    (norm_ne_zero_iff.mp (by simp))
    (stdAddChar_pow_ne_one one_ne_zero (by decide +kernel)) (one_add_sq_ne_zero one_ne_zero)
    stdAddChar_absDiffTable absDiffTable_comm
    (α := fun _ => 1) (β := fun _ => 1) (by simp) (by simp)
    (fun a b => by simp [absDiffTable_flip])

/-- **M does not return the table `max(a, b)` from the training half.** -/
theorem not_returned_max : NotReturned maxTable :=
  not_returned_of_tri (g := ZMod.stdAddChar (49 : ZMod 97)) (w := ZMod.stdAddChar (-49 : ZMod 97))
    (σ := ZMod.stdAddChar (49 : ZMod 97))
    (by rw [← AddChar.map_add_eq_mul, neg_add_cancel, AddChar.map_zero_eq_one])
    (norm_ne_zero_iff.mp (by simp))
    (stdAddChar_pow_ne_one (by decide +kernel) (by decide +kernel))
    (one_add_sq_ne_zero (by decide +kernel)) stdAddChar_maxTable maxTable_comm
    (α := fun x => ZMod.stdAddChar (-1 - x)) (β := fun x => ZMod.stdAddChar (-x))
    (by simp) (by simp)
    (fun a b => by rw [maxTable_flip, AddChar.map_add_eq_mul, AddChar.map_add_eq_mul])

/-- The hypotheses of `not_returned_of_peel` hold for the table `|a − b|`. -/
example : NotReturned absDiffTable :=
  not_returned_of_peel (T := triMatrix 1 (ZMod.stdAddChar (1 : ZMod 97)))
    (kernel_mul_triMatrix (g := 1) (one_mul 1) stdAddChar_absDiffTable)
    (sub_ne_zero.2 (stdAddChar_pow_ne_one one_ne_zero (by decide +kernel)).symm)
    (fun _ _ hl hj hn => triMatrix_ne_zero one_ne_zero (norm_ne_zero_iff.mp (by simp))
      (one_add_sq_ne_zero one_ne_zero) hl hj hn)
    (fun _ _ hl hj hn => triMatrix_eq_zero _ _ hl hj hn) absDiffTable_comm
    (α := fun _ => 1) (β := fun _ => 1) (by simp) (by simp)
    (fun a b => by simp [absDiffTable_flip])

/-- The hypotheses of `not_returned_of_tri` hold for the table `|a − b|`. -/
example : NotReturned absDiffTable :=
  not_returned_of_tri (g := 1) (w := 1) (σ := ZMod.stdAddChar (1 : ZMod 97)) (one_mul 1)
    (norm_ne_zero_iff.mp (by simp)) (stdAddChar_pow_ne_one one_ne_zero (by decide +kernel))
    (one_add_sq_ne_zero one_ne_zero) stdAddChar_absDiffTable absDiffTable_comm
    (α := fun _ => 1) (β := fun _ => 1) (by simp) (by simp)
    (fun a b => by simp [absDiffTable_flip])

end Transformer.Grokking.FlatSpectrum.TablePart.E6
