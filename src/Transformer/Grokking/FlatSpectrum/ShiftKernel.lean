module

public import Transformer.Grokking.FlatSpectrum.CharacterBases

/-!
# Theorem B: the addition step read by a multiplicative character is flat

Source: `review/flat_spectrum.md`, Theorem B, "Each step is rank 1, but in different bases".
The second step adds `u = a³` and `v`. Read by `ψ_r` it is rank 1 (`rank_taskKernel_cube_add`);
read by a nontrivial multiplicative character `χ` it is flat: `C = [χ(u + v)]` has
`C* C = p I − J`, `J` the all-ones matrix, as `∑_w conj χ(w) χ(w + d) = p [d = 0] − 1`. Since
`J² = p J`, `(C* C)² = p C* C`, so `C` has rank `p − 1`, every nonzero singular value is `√p`,
and `r_eff C = p − 1`. As `|χ(x)|² = 1` off `0`, `∑ conj χ(x) χ(x) F(x) = ∑ F − F(0)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {p : ℕ} [Fact p.Prime]

/-- The second step read by a nontrivial `χ` is flat: `C* C = p I − J` for `C = [χ(u + v)]`,
`J` the all-ones matrix. -/
theorem conjTranspose_mul_taskKernel_add {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) :
    (taskKernel (· + ·) χ)ᴴ * taskKernel (· + ·) χ = (p : ℂ) • 1 - of fun _ _ => 1 := by
  ext v v'
  simp only [mul_apply, conjTranspose_apply, taskKernel, of_apply, Matrix.sub_apply,
    Matrix.smul_apply, one_apply, smul_eq_mul]
  rw [← Equiv.sum_comp (Equiv.addRight (-v)), Finset.sum_congr rfl fun w _ => by
    rw [Equiv.coe_addRight, neg_add_cancel_right, show w + -v + v' = w + (v' - v) by ring],
    sum_star_mulChar_mul_add hχ]
  by_cases h : v = v'
  · subst h
    simp
  · simp [sub_eq_zero, Ne.symm h, h]

example := @conjTranspose_mul_taskKernel_add 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `(p I − J)² = p (p I − J)`, as `J² = p J`. -/
lemma smul_one_sub_mul_self :
    ((p : ℂ) • 1 - of fun _ _ => 1 : Matrix (ZMod p) (ZMod p) ℂ) *
      ((p : ℂ) • 1 - of fun _ _ => 1) =
      (p : ℂ) • ((p : ℂ) • 1 - of fun _ _ => 1) := by
  have hJ : (of fun _ _ => 1 : Matrix (ZMod p) (ZMod p) ℂ) * (of fun _ _ => 1) =
      (p : ℂ) • of fun _ _ => 1 := by
    ext a b
    simp [mul_apply, ZMod.card]
  rw [Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_sub, Matrix.smul_mul, Matrix.smul_mul,
    Matrix.one_mul, Matrix.one_mul, Matrix.mul_smul, Matrix.mul_one, hJ]
  module

example := @smul_one_sub_mul_self 5 ⟨Nat.prime_five⟩

/-- `(C* C)² = p C* C` for `C = [χ(u + v)]` and nontrivial `χ`. -/
lemma gram_taskKernel_add {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) :
    (taskKernel (· + ·) χ)ᴴ * taskKernel (· + ·) χ *
      ((taskKernel (· + ·) χ)ᴴ * taskKernel (· + ·) χ) =
      ((p : ℝ) : ℂ) • ((taskKernel (· + ·) χ)ᴴ * taskKernel (· + ·) χ) := by
  rw [conjTranspose_mul_taskKernel_add hχ, Complex.ofReal_natCast, smul_one_sub_mul_self]

example := @gram_taskKernel_add 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `C = [χ(u + v)]` has rank `p − 1` for nontrivial `χ`. -/
theorem rank_taskKernel_add {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) :
    (taskKernel (· + ·) χ).rank = p - 1 := by
  have hp := (Fact.out : p.Prime).one_lt
  have h := rank_mul_eq_trace (gram_taskKernel_add hχ)
  have hJ : trace (of fun _ _ => (1 : ℂ) : Matrix (ZMod p) (ZMod p) ℂ) = p := by
    simp [trace, ZMod.card]
  rw [conjTranspose_mul_taskKernel_add hχ, trace_sub, trace_smul, trace_one, ZMod.card, hJ,
    Complex.ofReal_natCast, smul_eq_mul] at h
  have h' : ((taskKernel (· + ·) χ).rank : ℂ) = ((p - 1 : ℕ) : ℂ) := by
    rw [Nat.cast_sub hp.le, Nat.cast_one]
    have hp0 : (p : ℂ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
    apply mul_right_cancel₀ hp0
    rw [h]
    ring
  exact_mod_cast h'

example := @rank_taskKernel_add 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- Every singular value of `C = [χ(u + v)]` is `0` or `√p` for nontrivial `χ`. -/
theorem singularValues_taskKernel_add {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) (i : ZMod p) :
    singularValues (taskKernel (· + ·) χ) i = 0 ∨ singularValues (taskKernel (· + ·) χ) i = √p :=
  singularValues_eq_zero_or (gram_taskKernel_add hχ) i

example (i : ZMod 5) := @singularValues_taskKernel_add 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) i

/-- `r_eff [χ(u + v)] = p − 1` for nontrivial `χ`: the addition step read by a multiplicative
character is flat. -/
theorem effectiveRank_taskKernel_add {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) :
    effectiveRank (taskKernel (· + ·) χ) = p - 1 := by
  have hp := (Fact.out : p.Prime).one_lt
  rw [effectiveRank_eq_rank (by exact_mod_cast (Fact.out : p.Prime).pos)
    (gram_taskKernel_add hχ), rank_taskKernel_add hχ, Nat.cast_sub hp.le, Nat.cast_one]

example := @effectiveRank_taskKernel_add 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `|χ(x)|² = 1` off `0`: `∑ conj χ(x) χ(x) F(x) = ∑ F − F(0)`. -/
lemma sum_star_mulChar_mul_self_mul (χ : MulChar (ZMod p) ℂ) (F : ZMod p → ℂ) :
    ∑ x, star (χ x) * χ x * F x = ∑ x, F x - F 0 := by
  have hpt (x : ZMod p) : star (χ x) * χ x * F x = F x - if x = 0 then F x else 0 := by
    by_cases hx : x = 0
    · rw [hx, ite_eq_left rfl, MulChar.map_zero, star_zero, zero_mul, zero_mul, sub_self]
    · rw [ite_eq_right hx, star_mulChar_mul_self χ hx, one_mul, sub_zero]
  rw [Finset.sum_congr rfl fun x _ => hpt x, Finset.sum_sub_distrib, Finset.sum_ite_eq']
  simp

example := @sum_star_mulChar_mul_self_mul 5 ⟨Nat.prime_five⟩ 1 fun _ => 1

end Transformer.Grokking.FlatSpectrum
