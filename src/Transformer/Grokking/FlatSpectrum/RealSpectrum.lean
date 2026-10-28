module

public import Transformer.Grokking.FlatSpectrum.RealKernels
public import Transformer.Grokking.FlatSpectrum.PartialIsometry

/-!
# The spectra of the real readouts of `a³ + a b`

Source: `review/flat_spectrum.md`, Torch check: "cos(2πr·f/p) has rank (p+1)/2 and sin has
rank (p−1)/2, with every singular value exactly √p". The other singular values are `0`, so
"every" means every nonzero one. Proved for odd prime `p` and `r ≠ 0`; the cos kernel needs
both, as it is the all-ones matrix at `r = 0` and the invertible `[[1, 1], [−1, 1]]` at
`p = 2`. With
`K = [ψ_r(a³ + a b)]` and the row flip `P`, `cos = Q K` for `Q = ½ (1 + P)` and `sin = Q' K`
for `Q' = −(i/2) (1 − P)`. As `P² = 1`, `Q* Q = ½ (1 + P)` and `Q'* Q' = ½ (1 − P)` are
idempotent, and `K K* = p I`. So `A = X* X` has `A² = p A` for both kernels: every nonzero
singular value is `√p`, and `p rank X = tr A = p tr (Q* Q) = p (p ± 1)/2`, as `tr P = 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {p : ℕ}

/-- `Q* = Q` for `Q = ½ (1 + P)`. -/
lemma conjTranspose_cosProj :
    ((1 / 2 : ℂ) • (1 + rowFlip))ᴴ =
      (1 / 2 : ℂ) • (1 + (rowFlip : Matrix (ZMod p) (ZMod p) ℂ)) := by
  rw [conjTranspose_smul, conjTranspose_add, conjTranspose_one, conjTranspose_rowFlip, star_div₀,
    star_one, star_ofNat]

example := conjTranspose_cosProj (p := 5)

section NeZero

variable [NeZero p]

/-- `½ (1 + P)` is idempotent. -/
lemma cosProj_mul_self : ((1 / 2 : ℂ) • (1 + rowFlip)) * ((1 / 2 : ℂ) • (1 + rowFlip)) =
    (1 / 2 : ℂ) • (1 + (rowFlip : Matrix (ZMod p) (ZMod p) ℂ)) := by
  simp only [Matrix.smul_mul, Matrix.mul_smul, Matrix.add_mul, Matrix.mul_add, Matrix.one_mul,
    Matrix.mul_one, rowFlip_mul_rowFlip]
  module

example := cosProj_mul_self (p := 5)

/-- `½ (1 − P)` is idempotent. -/
lemma sinGram_mul_self : ((1 / 2 : ℂ) • (1 - rowFlip)) * ((1 / 2 : ℂ) • (1 - rowFlip)) =
    (1 / 2 : ℂ) • (1 - (rowFlip : Matrix (ZMod p) (ZMod p) ℂ)) := by
  simp only [Matrix.smul_mul, Matrix.mul_smul, Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul,
    Matrix.mul_one, rowFlip_mul_rowFlip]
  module

example := sinGram_mul_self (p := 5)

/-- `Q'* Q' = ½ (1 − P)` for `Q' = −(i/2) (1 − P)`. -/
lemma sinProj_conjTranspose_mul_self :
    ((-Complex.I / 2) • (1 - rowFlip))ᴴ * ((-Complex.I / 2) • (1 - rowFlip)) =
      (1 / 2 : ℂ) • (1 - (rowFlip : Matrix (ZMod p) (ZMod p) ℂ)) := by
  have hI : star (-Complex.I / 2) * (-Complex.I / 2) = 1 / 4 := by
    rw [star_div₀, star_neg, star_ofNat, Complex.star_def, Complex.conj_I]
    linear_combination (-1 / 4 : ℂ) * Complex.I_sq
  rw [conjTranspose_smul, conjTranspose_sub, conjTranspose_one, conjTranspose_rowFlip,
    Matrix.smul_mul, Matrix.mul_smul, smul_smul, hI]
  simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one, rowFlip_mul_rowFlip]
  module

example := sinProj_conjTranspose_mul_self (p := 5)

end NeZero

variable [Fact p.Prime]

/-- `(C* C)² = p C* C` for the cos kernel `C` and `r ≠ 0`. -/
theorem cosKernel_gram {r : ZMod p} (hr : r ≠ 0) :
    (cosKernel r)ᴴ * cosKernel r * ((cosKernel r)ᴴ * cosKernel r) =
      ((p : ℝ) : ℂ) • ((cosKernel r)ᴴ * cosKernel r) := by
  rw [Complex.ofReal_natCast, cosKernel_eq]
  exact mul_self_conjTranspose_mul_self (cubicKernelMulShift_mul_conjTranspose hr)
    (by rw [conjTranspose_cosProj, cosProj_mul_self, cosProj_mul_self])

example := @cosKernel_gram 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- `(S* S)² = p S* S` for the sin kernel `S` and `r ≠ 0`. -/
theorem sinKernel_gram {r : ZMod p} (hr : r ≠ 0) :
    (sinKernel r)ᴴ * sinKernel r * ((sinKernel r)ᴴ * sinKernel r) =
      ((p : ℝ) : ℂ) • ((sinKernel r)ᴴ * sinKernel r) := by
  rw [Complex.ofReal_natCast, sinKernel_eq]
  exact mul_self_conjTranspose_mul_self (cubicKernelMulShift_mul_conjTranspose hr)
    (by rw [sinProj_conjTranspose_mul_self, sinGram_mul_self])

example := @sinKernel_gram 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- `2 rank C = p + 1` for the cos kernel: `p rank C = tr (C* C) = p (p + 1)/2`. -/
lemma two_mul_rank_cosKernel (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    2 * (cosKernel r).rank = p + 1 := by
  have h := rank_mul_eq_trace (cosKernel_gram hr)
  rw [cosKernel_eq, trace_conjTranspose_mul_self_of_mul (cubicKernelMulShift_mul_conjTranspose hr),
    conjTranspose_cosProj, cosProj_mul_self, trace_smul, trace_add, trace_one, trace_rowFlip hp,
    ZMod.card, smul_eq_mul, ← cosKernel_eq, Complex.ofReal_natCast, mul_comm] at h
  have h2 := mul_left_cancel₀ (Nat.cast_ne_zero.mpr (Fact.out : p.Prime).ne_zero) h
  have h3 : ((2 * (cosKernel r).rank : ℕ) : ℂ) = ((p + 1 : ℕ) : ℂ) := by
    push_cast
    rw [h2]
    ring
  exact_mod_cast h3

example := @two_mul_rank_cosKernel 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide)

/-- `2 rank S + 1 = p` for the sin kernel: `p rank S = tr (S* S) = p (p − 1)/2`. -/
lemma two_mul_rank_sinKernel_add_one (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    2 * (sinKernel r).rank + 1 = p := by
  have h := rank_mul_eq_trace (sinKernel_gram hr)
  rw [sinKernel_eq, trace_conjTranspose_mul_self_of_mul (cubicKernelMulShift_mul_conjTranspose hr),
    sinProj_conjTranspose_mul_self, trace_smul, trace_sub, trace_one, trace_rowFlip hp,
    ZMod.card, smul_eq_mul, ← sinKernel_eq, Complex.ofReal_natCast, mul_comm] at h
  have h2 := mul_left_cancel₀ (Nat.cast_ne_zero.mpr (Fact.out : p.Prime).ne_zero) h
  have h3 : ((2 * (sinKernel r).rank + 1 : ℕ) : ℂ) = (p : ℂ) := by
    push_cast
    rw [h2]
    ring
  exact_mod_cast h3

example := @two_mul_rank_sinKernel_add_one 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide)

/-- The cos kernel has rank `(p + 1)/2`, `review/flat_spectrum.md`, Torch check. -/
theorem rank_cosKernel (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    (cosKernel r).rank = (p + 1) / 2 := by
  have := two_mul_rank_cosKernel hp hr
  omega

example := @rank_cosKernel 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide)

/-- The sin kernel has rank `(p − 1)/2`, `review/flat_spectrum.md`, Torch check. -/
theorem rank_sinKernel (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    (sinKernel r).rank = (p - 1) / 2 := by
  have := two_mul_rank_sinKernel_add_one hp hr
  omega

example := @rank_sinKernel 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide)

/-- Every singular value of the cos kernel is `0` or `√p`, `review/flat_spectrum.md`,
Torch check. -/
theorem singularValues_cosKernel {r : ZMod p} (hr : r ≠ 0) (i : ZMod p) :
    singularValues (cosKernel r) i = 0 ∨ singularValues (cosKernel r) i = √p :=
  singularValues_eq_zero_or (cosKernel_gram hr) i

example := @singularValues_cosKernel 5 ⟨Nat.prime_five⟩ 1 (by decide) 0

/-- Every singular value of the sin kernel is `0` or `√p`, `review/flat_spectrum.md`,
Torch check. -/
theorem singularValues_sinKernel {r : ZMod p} (hr : r ≠ 0) (i : ZMod p) :
    singularValues (sinKernel r) i = 0 ∨ singularValues (sinKernel r) i = √p :=
  singularValues_eq_zero_or (sinKernel_gram hr) i

example := @singularValues_sinKernel 5 ⟨Nat.prime_five⟩ 1 (by decide) 0

/-- `‖C‖_S1 = √p (p + 1)/2` and `r_eff C = (p + 1)/2` for the cos kernel. -/
theorem nuclearNorm_effectiveRank_cosKernel (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    nuclearNorm (cosKernel r) = (p + 1) / 2 * √p ∧ effectiveRank (cosKernel r) = (p + 1) / 2 := by
  have h : (2 * (cosKernel r).rank : ℝ) = p + 1 := by exact_mod_cast two_mul_rank_cosKernel hp hr
  have h' : ((cosKernel r).rank : ℝ) = (p + 1) / 2 := by linarith
  exact ⟨by rw [nuclearNorm_eq_rank_mul_sqrt (cosKernel_gram hr), h'], by
    rw [effectiveRank_eq_rank (by exact_mod_cast (Fact.out : p.Prime).pos) (cosKernel_gram hr), h']⟩

example := @nuclearNorm_effectiveRank_cosKernel 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide)

/-- `‖S‖_S1 = √p (p − 1)/2` and `r_eff S = (p − 1)/2` for the sin kernel. -/
theorem nuclearNorm_effectiveRank_sinKernel (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    nuclearNorm (sinKernel r) = (p - 1) / 2 * √p ∧ effectiveRank (sinKernel r) = (p - 1) / 2 := by
  have h : (2 * (sinKernel r).rank + 1 : ℝ) = p := by
    exact_mod_cast two_mul_rank_sinKernel_add_one hp hr
  have h' : ((sinKernel r).rank : ℝ) = (p - 1) / 2 := by linarith
  exact ⟨by rw [nuclearNorm_eq_rank_mul_sqrt (sinKernel_gram hr), h'], by
    rw [effectiveRank_eq_rank (by exact_mod_cast (Fact.out : p.Prime).pos) (sinKernel_gram hr), h']⟩

example := @nuclearNorm_effectiveRank_sinKernel 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide)

end Transformer.Grokking.FlatSpectrum
