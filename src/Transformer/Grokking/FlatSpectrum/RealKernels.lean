module

public import Transformer.Grokking.FlatSpectrum.Decoding

/-!
# The real readouts of `a³ + a b`

Source: `review/flat_spectrum.md`, Torch check, "Real Fourier kernels": the readouts
`cos(2πr·f/p)` and `sin(2πr·f/p)` of `f = a³ + a b`, whose "rows a and −a are equal for cos
and opposite for sin", as `f(−a, b) = −f(a, b)`. So the cos kernel repeats a row for odd `p`
and row `a = 0` of the sin kernel is zero: both are singular, and Theorem A needs an
invertible kernel. With `ψ_r(x) = ψ(r x)`, `K = [ψ_r(a³ + a b)]` and the row flip
`P : a ↦ −a`, `conj K = P K`, so `cos = ½ (1 + P) K` and `sin = −(i/2) (1 − P) K`. For prime
`p` and `r ≠ 0`, `K K* = p I`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Real

variable {p : ℕ}

/-- The readout `cos(2π r f/p)` of `f = a³ + a b`, `review/flat_spectrum.md`, Torch check. -/
noncomputable def cosKernel (r : ZMod p) : Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (fun a b => a ^ 3 + a * b) fun c => (cos (2 * π * (r * c).val / p) : ℂ)

/-- The readout `sin(2π r f/p)` of `f = a³ + a b`, `review/flat_spectrum.md`, Torch check. -/
noncomputable def sinKernel (r : ZMod p) : Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (fun a b => a ^ 3 + a * b) fun c => (sin (2 * π * (r * c).val / p) : ℂ)

/-- The row flip `P`: `(P M)(a, b) = M(−a, b)`. -/
def rowFlip : Matrix (ZMod p) (ZMod p) ℂ := Matrix.of fun a b => if b = -a then 1 else 0

/-- `P* = P`. -/
lemma conjTranspose_rowFlip : (rowFlip : Matrix (ZMod p) (ZMod p) ℂ)ᴴ = rowFlip := by
  ext a b
  have h : a = -b ↔ b = -a := ⟨fun h => by rw [h, neg_neg], fun h => by rw [h, neg_neg]⟩
  simp only [rowFlip, conjTranspose_apply, of_apply, h]
  split_ifs <;> simp

example := conjTranspose_rowFlip (p := 5)

section NeZero

variable [NeZero p]

/-- The kernel `ψ_r(a³ + a b) = ψ(r (a³ + a b))`, `review/flat_spectrum.md`, Theorem B and
Torch check. -/
noncomputable def cubicKernelMulShift (r : ZMod p) : Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (fun a b => a ^ 3 + a * b) fun c => ZMod.stdAddChar (r * c)

/-- `cos(2π x/p) = (ψ(x) + ψ(−x))/2`. -/
lemma ofReal_cos_eq (x : ZMod p) :
    (cos (2 * π * x.val / p) : ℂ) = (ZMod.stdAddChar x + ZMod.stdAddChar (-x)) / 2 := by
  rw [AddChar.map_neg_eq_conj, Complex.add_conj, stdAddChar_eq_exp, Complex.exp_ofReal_mul_I_re,
    Complex.ofReal_mul, Complex.ofReal_ofNat]
  ring

example := ofReal_cos_eq (p := 5) 1

/-- `sin(2π x/p) = −i (ψ(x) − ψ(−x))/2`. -/
lemma ofReal_sin_eq (x : ZMod p) :
    (sin (2 * π * x.val / p) : ℂ) =
      -Complex.I * (ZMod.stdAddChar x - ZMod.stdAddChar (-x)) / 2 := by
  rw [AddChar.map_neg_eq_conj, Complex.sub_conj, stdAddChar_eq_exp, Complex.exp_ofReal_mul_I_im,
    Complex.ofReal_mul, Complex.ofReal_ofNat]
  linear_combination (sin (2 * π * x.val / p) : ℂ) * Complex.I_sq

example := ofReal_sin_eq (p := 5) 1

/-- Rows `a` and `−a` of the cos kernel are equal, `review/flat_spectrum.md`, Torch check. -/
lemma cosKernel_neg (r a : ZMod p) : cosKernel r (-a) = cosKernel r a := by
  funext b
  simp only [cosKernel, taskKernel, of_apply, ofReal_cos_eq]
  rw [show r * ((-a) ^ 3 + -a * b) = -(r * (a ^ 3 + a * b)) by ring, neg_neg, add_comm]

example := cosKernel_neg (p := 5) 1 2

/-- Rows `a` and `−a` of the sin kernel are opposite, `review/flat_spectrum.md`, Torch check. -/
lemma sinKernel_neg (r a : ZMod p) : sinKernel r (-a) = -sinKernel r a := by
  funext b
  simp only [sinKernel, taskKernel, of_apply, Pi.neg_apply, ofReal_sin_eq]
  rw [show r * ((-a) ^ 3 + -a * b) = -(r * (a ^ 3 + a * b)) by ring, neg_neg]
  ring

example := sinKernel_neg (p := 5) 1 2

/-- The sin kernel is singular: its row `a = 0` is zero. -/
theorem det_sinKernel (r : ZMod p) : (sinKernel r).det = 0 :=
  det_eq_zero_of_row_eq_zero 0 fun b => by simp [sinKernel, taskKernel]

example := det_sinKernel (p := 5) 1

/-- `P M` flips the rows of `M`. -/
lemma rowFlip_mul {n : Type*} (M : Matrix (ZMod p) n ℂ) : rowFlip * M = M.submatrix Neg.neg id := by
  ext a b
  simp [rowFlip, mul_apply]

example := rowFlip_mul (1 : Matrix (ZMod 5) (ZMod 5) ℂ)

/-- `P² = 1`. -/
lemma rowFlip_mul_rowFlip : (rowFlip : Matrix (ZMod p) (ZMod p) ℂ) * rowFlip = 1 := by
  rw [rowFlip_mul]
  ext a b
  simp [rowFlip, one_apply, eq_comm]

example := rowFlip_mul_rowFlip (p := 5)

/-- `cos = ½ (1 + P) K`. -/
lemma cosKernel_eq (r : ZMod p) :
    cosKernel r = ((1 / 2 : ℂ) • (1 + rowFlip)) * cubicKernelMulShift r := by
  rw [Matrix.smul_mul, Matrix.add_mul, Matrix.one_mul, rowFlip_mul]
  ext a b
  simp only [cosKernel, cubicKernelMulShift, taskKernel, of_apply, Matrix.smul_apply,
    Matrix.add_apply, submatrix_apply, id, smul_eq_mul, ofReal_cos_eq]
  rw [show r * ((-a) ^ 3 + -a * b) = -(r * (a ^ 3 + a * b)) by ring]
  ring

example := cosKernel_eq (p := 5) 1

/-- `sin = −(i/2) (1 − P) K`. -/
lemma sinKernel_eq (r : ZMod p) :
    sinKernel r = ((-Complex.I / 2) • (1 - rowFlip)) * cubicKernelMulShift r := by
  rw [Matrix.smul_mul, Matrix.sub_mul, Matrix.one_mul, rowFlip_mul]
  ext a b
  simp only [sinKernel, cubicKernelMulShift, taskKernel, of_apply, Matrix.smul_apply,
    Matrix.sub_apply, submatrix_apply, id, smul_eq_mul, ofReal_sin_eq]
  rw [show r * ((-a) ^ 3 + -a * b) = -(r * (a ^ 3 + a * b)) by ring]
  ring

example := sinKernel_eq (p := 5) 1

end NeZero

variable [Fact p.Prime]

/-- For odd prime `p`, `a = −a` only at `a = 0`. -/
lemma eq_neg_iff_of_ne_two (hp : p ≠ 2) {a : ZMod p} : a = -a ↔ a = 0 := by
  rw [eq_comm, ZMod.neg_eq_self_iff]
  refine ⟨fun h => h.resolve_right fun h2 => ?_, Or.inl⟩
  obtain ⟨k, hk⟩ := (Fact.out : p.Prime).odd_of_ne_two hp
  omega

example := @eq_neg_iff_of_ne_two 5 ⟨Nat.prime_five⟩ (by decide) 1

/-- The cos kernel is singular for odd prime `p`: rows `1` and `−1` are equal. -/
theorem det_cosKernel (hp : p ≠ 2) (r : ZMod p) : (cosKernel r).det = 0 :=
  det_zero_of_row_eq (fun h => one_ne_zero ((eq_neg_iff_of_ne_two hp).mp h.symm))
    (cosKernel_neg r 1)

example := @det_cosKernel 5 ⟨Nat.prime_five⟩ (by decide) 1

/-- `tr P = 1` for odd prime `p`: only `a = 0` has `−a = a`. -/
lemma trace_rowFlip (hp : p ≠ 2) : (rowFlip : Matrix (ZMod p) (ZMod p) ℂ).trace = 1 := by
  simp [trace, rowFlip, eq_neg_iff_of_ne_two hp]

example := @trace_rowFlip 5 ⟨Nat.prime_five⟩ (by decide)

/-- `K K* = p I` for `K = [ψ_r(a³ + a b)]`, prime `p` and `r ≠ 0`. -/
theorem cubicKernelMulShift_mul_conjTranspose {r : ZMod p} (hr : r ≠ 0) :
    cubicKernelMulShift r * (cubicKernelMulShift r)ᴴ = (p : ℂ) • 1 := by
  ext a c
  simp only [mul_apply, conjTranspose_apply, cubicKernelMulShift, taskKernel, of_apply,
    Matrix.smul_apply, one_apply, smul_eq_mul]
  rw [Finset.sum_congr rfl fun b _ => (by
    rw [mul_comm, star_stdAddChar_mul, show r * (a ^ 3 + a * b) - r * (c ^ 3 + c * b) =
      r * (a ^ 3 - c ^ 3) + b * (r * (a - c)) by ring, AddChar.map_add_eq_mul] :
      ZMod.stdAddChar (r * (a ^ 3 + a * b)) * star (ZMod.stdAddChar (r * (c ^ 3 + c * b))) =
        ZMod.stdAddChar (r * (a ^ 3 - c ^ 3)) * ZMod.stdAddChar (b * (r * (a - c)))),
    ← Finset.mul_sum, sum_stdAddChar_mul]
  by_cases h : a = c
  · subst h
    simp
  · simp [h, hr, sub_eq_zero]

example := @cubicKernelMulShift_mul_conjTranspose 5 ⟨Nat.prime_five⟩ 1 (by decide)

end Transformer.Grokking.FlatSpectrum
