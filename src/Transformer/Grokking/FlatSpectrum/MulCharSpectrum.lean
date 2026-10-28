module

public import Transformer.Grokking.FlatSpectrum.MulCharKernel
public import Mathlib.Tactic.NormNum.Prime

/-!
# Spectrum and effective rank of `[χ(a³ + a b)]`

Source: `review/flat_spectrum.md`, Theorem B: "Those equal χ₃(f), whose kernel is itself spread
out (effective rank 48 of 97)." By `A³ = (3p + 1) A² − 2p (p + 1) A`, every eigenvalue of
`A = K K*` is `0`, `2p` or `p + 1`. As `tr A = (p − 1)²` and
`tr A² = 2p (p − 1)² − (p + 1)(p − 1)`, `p + 1` occurs once and `2p` occurs `(p − 3)/2` times.
`K* K` has the same eigenvalues, so `rank K = (p − 1)/2`,
`‖K‖_S1 = √(p + 1) + (p − 3)/2 · √(2p)`, `‖K‖²_S2 = (p − 1)²` and
`r_eff K = (√(p + 1) + (p − 3)/2 · √(2p))² / (p − 1)²`. At `p = 97` the rank is `48` and
`47.9 < r_eff < 48`, so the note's 48 is `r_eff` rounded. All of it holds for every nontrivial
`χ` and odd `p`, the cubic `χ₃` of the note at `p = 97 ≡ 1 mod 3` included.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {p : ℕ} [Fact p.Prime] {χ : MulChar (ZMod p) ℂ}

lemma three_le_of_ne_two (hp : p ≠ 2) : (3 : ℝ) ≤ p := by
  have := (Fact.out : p.Prime).two_le
  exact_mod_cast (show 3 ≤ p by omega)

example := @three_le_of_ne_two 5 ⟨Nat.prime_five⟩ (by decide)

/-- Every eigenvalue of `K K*` is `0`, `2p` or `p + 1`. -/
theorem eigenvalues_cubicKernelMulChar_mem (hp : p ≠ 2) (hχ : χ ≠ 1) (i : ZMod p) :
    (isHermitian_mul_conjTranspose_self (cubicKernelMulChar χ)).eigenvalues i = 0 ∨
      (isHermitian_mul_conjTranspose_self (cubicKernelMulChar χ)).eigenvalues i = 2 * p ∨
      (isHermitian_mul_conjTranspose_self (cubicKernelMulChar χ)).eigenvalues i = p + 1 :=
  eigenvalues_mem_of_cubic _ (cubicKernelMulChar_mul_conjTranspose_cube hp hχ) i

example := @eigenvalues_cubicKernelMulChar_mem 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) 0

/-- The eigenvalue `p + 1` of `K K*` occurs once and `2p` occurs `(p − 3)/2` times. -/
theorem card_eigenvalues_cubicKernelMulChar (hp : p ≠ 2) (hχ : χ ≠ 1) :
    (#{i | (isHermitian_mul_conjTranspose_self (cubicKernelMulChar χ)).eigenvalues i =
        p + 1} : ℝ) = 1 ∧
      (#{i | (isHermitian_mul_conjTranspose_self (cubicKernelMulChar χ)).eigenvalues i =
        2 * p} : ℝ) = (p - 3) / 2 := by
  set hA := isHermitian_mul_conjTranspose_self (cubicKernelMulChar χ)
  have hp3 := three_le_of_ne_two hp
  have hαβ : (2 * p : ℝ) ≠ p + 1 := fun h => by linarith
  have hmem := eigenvalues_mem_of_cubic hA (cubicKernelMulChar_mul_conjTranspose_cube hp hχ)
  have h1 := re_trace_eq_of_mem hA hαβ hmem
  have h2 := re_trace_mul_self_eq_of_mem hA hαβ hmem
  rw [trace_cubicKernelMulChar_mul_conjTranspose hχ, show ((p : ℂ) - 1) ^ 2 =
    (((p : ℝ) - 1) ^ 2 : ℝ) by push_cast; ring, Complex.ofReal_re] at h1
  rw [trace_cubicKernelMulChar_mul_conjTranspose_sq hp hχ, show
    2 * p * ((p : ℂ) - 1) ^ 2 - ((p : ℂ) + 1) * ((p : ℂ) - 1) =
      ((2 * p * ((p : ℝ) - 1) ^ 2 - ((p : ℝ) + 1) * ((p : ℝ) - 1) : ℝ) : ℂ) by push_cast; ring,
    Complex.ofReal_re] at h2
  set m₁ : ℝ := ((#{i | hA.eigenvalues i = 2 * p} : ℕ) : ℝ)
  set m₂ : ℝ := ((#{i | hA.eigenvalues i = p + 1} : ℕ) : ℝ)
  have hm : ((p : ℝ) + 1) * ((p : ℝ) - 1) * (m₂ - 1) = 0 := by linear_combination h2 - 2 * p * h1
  have hm₂ : m₂ = 1 := by
    rcases mul_eq_zero.mp hm with h | h
    · exact absurd h (mul_ne_zero (by linarith) (by linarith))
    · linarith
  refine ⟨hm₂, ?_⟩
  rw [hm₂] at h1
  have h0 : (p : ℝ) ≠ 0 := by linarith
  have h3 : (p : ℝ) * (2 * m₁) = p * (p - 3) := by linear_combination -h1
  linarith [mul_left_cancel₀ h0 h3]

example := @card_eigenvalues_cubicKernelMulChar 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- Every singular value of `K` is `0`, `√(2p)` or `√(p + 1)`. -/
theorem singularValues_cubicKernelMulChar_mem (hp : p ≠ 2) (hχ : χ ≠ 1) (i : ZMod p) :
    singularValues (cubicKernelMulChar χ) i = 0 ∨
      singularValues (cubicKernelMulChar χ) i = √(2 * p) ∨
      singularValues (cubicKernelMulChar χ) i = √(p + 1) := by
  rw [singularValues_eq_sqrt_eigenvalues_mul_conjTranspose]
  rcases eigenvalues_cubicKernelMulChar_mem hp hχ i with h | h | h <;> rw [h]
  · exact Or.inl Real.sqrt_zero
  · exact Or.inr (Or.inl rfl)
  · exact Or.inr (Or.inr rfl)

example := @singularValues_cubicKernelMulChar_mem 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) 0

/-- `rank K = (p − 1)/2`. -/
theorem rank_cubicKernelMulChar (hp : p ≠ 2) (hχ : χ ≠ 1) :
    ((cubicKernelMulChar χ).rank : ℝ) = (p - 1) / 2 := by
  have hp3 := three_le_of_ne_two hp
  obtain ⟨h₂, h₁⟩ := card_eigenvalues_cubicKernelMulChar hp hχ
  rw [← rank_self_mul_conjTranspose, rank_eq_of_mem _ (ne_of_gt (by linarith))
    (ne_of_gt (by linarith)) (fun h => by linarith) (eigenvalues_cubicKernelMulChar_mem hp hχ),
    h₁, h₂]
  ring

example := @rank_cubicKernelMulChar 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `‖K‖_S1 = √(p + 1) + (p − 3)/2 · √(2p)`. -/
theorem nuclearNorm_cubicKernelMulChar (hp : p ≠ 2) (hχ : χ ≠ 1) :
    nuclearNorm (cubicKernelMulChar χ) = √(p + 1) + (p - 3) / 2 * √(2 * p) := by
  have hp3 := three_le_of_ne_two hp
  obtain ⟨h₂, h₁⟩ := card_eigenvalues_cubicKernelMulChar hp hχ
  simp only [nuclearNorm, singularValues_eq_sqrt_eigenvalues_mul_conjTranspose]
  rw [sum_comp_eigenvalues_of_mem _ (fun h => by linarith)
    (eigenvalues_cubicKernelMulChar_mem hp hχ) Real.sqrt Real.sqrt_zero, h₁, h₂]
  ring

example := @nuclearNorm_cubicKernelMulChar 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `‖K‖²_S2 = (p − 1)²`. -/
theorem frobeniusSq_cubicKernelMulChar (hχ : χ ≠ 1) :
    frobeniusSq (cubicKernelMulChar χ) = ((p : ℝ) - 1) ^ 2 := by
  rw [frobeniusSq_eq_re_trace, trace_mul_comm, trace_cubicKernelMulChar_mul_conjTranspose hχ,
    show ((p : ℂ) - 1) ^ 2 = (((p : ℝ) - 1) ^ 2 : ℝ) by push_cast; ring, Complex.ofReal_re]

example := @frobeniusSq_cubicKernelMulChar 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `r_eff K = (√(p + 1) + (p − 3)/2 · √(2p))² / (p − 1)²`. -/
theorem effectiveRank_cubicKernelMulChar (hp : p ≠ 2) (hχ : χ ≠ 1) :
    effectiveRank (cubicKernelMulChar χ) =
      (√(p + 1) + (p - 3) / 2 * √(2 * p)) ^ 2 / ((p : ℝ) - 1) ^ 2 := by
  have hp3 := three_le_of_ne_two hp
  rw [effectiveRank, nuclearNorm_cubicKernelMulChar hp hχ, frobeniusSq_cubicKernelMulChar hχ,
    Real.sqrt_sq (by linarith), div_pow]

example := @effectiveRank_cubicKernelMulChar 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- "Effective rank 48 of 97", `review/flat_spectrum.md`, Theorem B: at `p = 97`,
`rank K = 48` for every nontrivial `χ`. -/
theorem rank_cubicKernelMulChar_97 [Fact (Nat.Prime 97)] {χ : MulChar (ZMod 97) ℂ}
    (hχ : χ ≠ 1) : (cubicKernelMulChar χ).rank = 48 := by
  have h := rank_cubicKernelMulChar (by norm_num) hχ
  norm_num at h
  exact_mod_cast h

example := @rank_cubicKernelMulChar_97 ⟨by norm_num⟩ _
  (@quadraticChar_ringHomComp_ne_one 97 ⟨by norm_num⟩ (by decide))

/-- "Effective rank 48 of 97", `review/flat_spectrum.md`, Theorem B: at `p = 97`,
`47.9 < r_eff K < 48` for every nontrivial `χ`. -/
theorem effectiveRank_cubicKernelMulChar_97 [Fact (Nat.Prime 97)] {χ : MulChar (ZMod 97) ℂ}
    (hχ : χ ≠ 1) :
    47.9 < effectiveRank (cubicKernelMulChar χ) ∧ effectiveRank (cubicKernelMulChar χ) < 48 := by
  rw [effectiveRank_cubicKernelMulChar (by norm_num) hχ, show ((97 : ℕ) : ℝ) = 97 by norm_num,
    show (97 : ℝ) + 1 = 98 by norm_num, show ((97 : ℝ) - 3) / 2 = 47 by norm_num,
    show (2 : ℝ) * 97 = 194 by norm_num, show ((97 : ℝ) - 1) ^ 2 = 9216 by norm_num]
  have hs1 : (9.899 : ℝ) < √98 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √98 < (9.9 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ht1 : (13.928 : ℝ) < √194 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have ht2 : √194 < (13.929 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hlo := mul_self_lt_mul_self (by norm_num : (0 : ℝ) ≤ 664.515)
    (show (664.515 : ℝ) < √98 + 47 * √194 by linarith)
  have hhi := mul_self_lt_mul_self (by positivity : (0 : ℝ) ≤ √98 + 47 * √194)
    (show √98 + 47 * √194 < (664.563 : ℝ) by linarith)
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  · rw [div_lt_iff₀ (by norm_num)]
    nlinarith

example := @effectiveRank_cubicKernelMulChar_97 ⟨by norm_num⟩ _
  (@quadraticChar_ringHomComp_ne_one 97 ⟨by norm_num⟩ (by decide))

end Transformer.Grokking.FlatSpectrum
