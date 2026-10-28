module

public import Transformer.Grokking.FlatSpectrum.GaussSums
public import Transformer.Grokking.FlatSpectrum.PartialIsometry

/-!
# Theorem B: each step is rank 1 in its own character basis

Source: `review/flat_spectrum.md`, Theorem B: "Split the task as v = ab, then f = a³ + v. Each
step is rank 1, but in different bases: χ(ab) = χ(a)χ(b) in multiplicative characters, and
ψ(a³+v) = ψ(a³)ψ(v) in additive ones." Read by a multiplicative character `χ`, the first step
is `[χ(a b)] = χ χᵀ`, of rank 1; read by `ψ_r`, `r ≠ 0`, it is flat: `F_r = [ψ_r(a b)]` has
`F_r* F_r = p I`, so all `p` singular values are `√p`. Read by `ψ_r`, the second step is
`[ψ_r(a³ + v)] = ψ_r(a³) ψ_r(v)ᵀ`, of rank 1. For nontrivial `χ`,
`∑_w conj χ(w) χ(w + d) = −1` for `d ≠ 0`: it does not depend on `d ≠ 0`, and its sum over all
`d` is `0`. `ShiftKernel` uses it to show that the addition read by `χ` is flat.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {p : ℕ} [Fact p.Prime]

/-- The first step in multiplicative characters: `[χ(a b)] = χ χᵀ`, `review/flat_spectrum.md`,
Theorem B. -/
lemma taskKernel_mul_mulChar (χ : MulChar (ZMod p) ℂ) : taskKernel (· * ·) χ = vecMulVec χ χ := by
  ext a b
  simp [taskKernel, vecMulVec_apply]

example := @taskKernel_mul_mulChar 5 ⟨Nat.prime_five⟩ 1

/-- `[χ(a b)]` has rank 1, `review/flat_spectrum.md`, Theorem B. -/
theorem rank_taskKernel_mul_mulChar (χ : MulChar (ZMod p) ℂ) :
    (taskKernel (· * ·) χ).rank = 1 := by
  refine le_antisymm (taskKernel_mul_mulChar χ ▸ rank_vecMulVec_le _ _) ?_
  rw [Nat.one_le_iff_ne_zero, Ne, rank_eq_finrank_span_cols, Submodule.finrank_eq_zero,
    Submodule.span_eq_bot]
  intro h
  have := congrFun (h _ ⟨1, rfl⟩) 1
  simp [taskKernel, col] at this

example := @rank_taskKernel_mul_mulChar 5 ⟨Nat.prime_five⟩ 1

/-- The first step read by `ψ_r`, `r ≠ 0`, is flat: `F_r* F_r = p I` for `F_r = [ψ_r(a b)]`. -/
theorem conjTranspose_mul_taskKernel_mul {r : ZMod p} (hr : r ≠ 0) :
    (taskKernel (· * ·) fun c => ZMod.stdAddChar (r * c))ᴴ *
      taskKernel (· * ·) (fun c => ZMod.stdAddChar (r * c)) = (p : ℂ) • 1 := by
  ext b c
  simp only [mul_apply, conjTranspose_apply, taskKernel, of_apply, Matrix.smul_apply, one_apply,
    smul_eq_mul]
  rw [Finset.sum_congr rfl fun a _ => (by
    rw [star_stdAddChar_mul, show r * (a * c) - r * (a * b) = a * (r * (c - b)) by ring] :
      star (ZMod.stdAddChar (r * (a * b))) * ZMod.stdAddChar (r * (a * c)) =
        ZMod.stdAddChar (a * (r * (c - b)))), sum_stdAddChar_mul]
  by_cases h : b = c
  · subst h
    simp
  · simp [hr, sub_eq_zero, Ne.symm h, h]

example := @conjTranspose_mul_taskKernel_mul 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- Every singular value of `[ψ_r(a b)]`, `r ≠ 0`, is `√p`. -/
theorem singularValues_taskKernel_mul {r : ZMod p} (hr : r ≠ 0) (i : ZMod p) :
    singularValues (taskKernel (· * ·) fun c => ZMod.stdAddChar (r * c)) i = √p := by
  rw [singularValues, eigenvalues_eq_of_eq_smul_one _ (by
    rw [conjTranspose_mul_taskKernel_mul hr, Complex.ofReal_natCast])]

example (i : ZMod 5) := @singularValues_taskKernel_mul 5 ⟨Nat.prime_five⟩ 1 (by decide) i

/-- The second step in additive characters: `[ψ_r(a³ + v)] = ψ_r(a³) ψ_r(v)ᵀ`,
`review/flat_spectrum.md`, Theorem B. -/
lemma taskKernel_cube_add (r : ZMod p) :
    taskKernel (fun a v => a ^ 3 + v) (fun c => ZMod.stdAddChar (r * c)) =
      vecMulVec (fun a => ZMod.stdAddChar (r * a ^ 3)) fun v => ZMod.stdAddChar (r * v) := by
  ext a v
  simp [taskKernel, vecMulVec_apply, mul_add, AddChar.map_add_eq_mul]

example := @taskKernel_cube_add 5 ⟨Nat.prime_five⟩ 1

/-- `[ψ_r(a³ + v)]` has rank 1, `review/flat_spectrum.md`, Theorem B. -/
theorem rank_taskKernel_cube_add (r : ZMod p) :
    (taskKernel (fun a v => a ^ 3 + v) fun c => ZMod.stdAddChar (r * c)).rank = 1 := by
  refine le_antisymm (taskKernel_cube_add r ▸ rank_vecMulVec_le _ _) ?_
  rw [Nat.one_le_iff_ne_zero, Ne, rank_eq_finrank_span_cols, Submodule.finrank_eq_zero,
    Submodule.span_eq_bot]
  intro h
  have := congrFun (h _ ⟨0, rfl⟩) 0
  simp [taskKernel, col] at this

example := @rank_taskKernel_cube_add 5 ⟨Nat.prime_five⟩ 1

/-- Scaling `w ↦ d w` shows that `∑_w conj χ(w) χ(w + d)` is the same for every `d ≠ 0`. -/
lemma sum_star_mulChar_mul_add_eq (χ : MulChar (ZMod p) ℂ) {d : ZMod p} (hd : d ≠ 0) :
    ∑ w, star (χ w) * χ (w + d) = ∑ w, star (χ w) * χ (w + 1) := by
  rw [← Equiv.sum_comp (Equiv.mulLeft₀ d hd)]
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [Equiv.mulLeft₀_apply]
  rw [show d * t + d = d * (t + 1) by ring, map_mul, map_mul, star_mul', mul_mul_mul_comm,
    star_mulChar_mul_self χ hd, one_mul]

example := @sum_star_mulChar_mul_add_eq 5 ⟨Nat.prime_five⟩ 1 2 (by decide)

/-- `∑_w conj χ(w) χ(w + d) = p − 1` for `d = 0` and `−1` otherwise, for nontrivial `χ`. -/
lemma sum_star_mulChar_mul_add {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) (d : ZMod p) :
    ∑ w, star (χ w) * χ (w + d) = if d = 0 then (p : ℂ) - 1 else -1 := by
  have hp := (Fact.out : p.Prime).one_lt
  have h0 : ∑ w, star (χ w) * χ (w + 0) = (p : ℂ) - 1 := by
    rw [← Finset.add_sum_erase _ _ (mem_univ 0), add_zero, MulChar.map_zero, mul_zero, zero_add,
      Finset.sum_congr rfl fun w hw => by
        rw [add_zero, star_mulChar_mul_self χ (ne_of_mem_erase hw)],
      Finset.sum_const, card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul,
      mul_one, Nat.cast_sub hp.le, Nat.cast_one]
  have hsum : ∑ d : ZMod p, ∑ w, star (χ w) * χ (w + d) = 0 := by
    rw [Finset.sum_comm]
    refine Finset.sum_eq_zero fun w _ => ?_
    rw [← Finset.mul_sum, show ∑ i, χ (w + i) = ∑ i, χ i from
      Equiv.sum_comp (Equiv.addLeft w) fun t => χ t,
      MulChar.sum_eq_zero_of_ne_one hχ, mul_zero]
  rw [← Finset.add_sum_erase _ _ (mem_univ 0), h0, Finset.sum_congr rfl fun d hd =>
      sum_star_mulChar_mul_add_eq χ (ne_of_mem_erase hd), Finset.sum_const,
    card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul, Nat.cast_sub hp.le,
    Nat.cast_one] at hsum
  have hp1 : (p : ℂ) - 1 ≠ 0 := by
    rw [sub_ne_zero, Ne, Nat.cast_eq_one]
    omega
  split_ifs with hd
  · rw [hd, h0]
  · rw [sum_star_mulChar_mul_add_eq χ hd]
    have h1 : ((p : ℂ) - 1) * (1 + ∑ w, star (χ w) * χ (w + 1)) = 0 := by
      rw [mul_add, mul_one]
      exact hsum
    linear_combination (mul_eq_zero.mp h1).resolve_left hp1

example := @sum_star_mulChar_mul_add 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) 1

end Transformer.Grokking.FlatSpectrum
