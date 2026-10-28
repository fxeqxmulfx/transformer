module

public import Transformer.Grokking.FlatSpectrum.ZeroIndicator
public import Transformer.Grokking.FlatSpectrum.MulCharSpectrum

/-!
# The indicator of answer `0`: the multiplicities of the eigenvalues of `K K*`

Source: `review/flat_spectrum.md`, Torch check: "Uncentered, the exact value is
(√(p+1)+1+(p−3)/√2)²/(2p−1) = 31.01 and 59.53; the spectrum matches to 1e-14." Every eigenvalue
of `A = K K*`, `K = [a³ + a b = 0]`, is `0`, `1`, `2` or `p + 1` (`ZeroIndicator`). Applying `A`
twice to the unit vectors gives `(A²)_{aa} = 3` for `a ≠ 0` and `(A²)_{00} = p² + p − 1`, so
`tr A² = p² + 4p − 4`. With `tr A = 2p − 1`, the multiplicities `m₁`, `m₂`, `m_{p+1}` satisfy
`2 m₂ + p (p + 1) m_{p+1} = (p + 3)(p − 1)`. As `0 ≤ m₂ ≤ p`, this forces `m_{p+1} = 1`, then
`m₂ = (p − 3)/2` and `m₁ = 1`. As in `ThreeEigenvalues`, for eigenvalues in `{0, α, β, γ}` every
sum `∑ g(λ)` with `g 0 = 0`, so both traces and the rank, is a sum over the multiplicities.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

section FourValues

variable {n : Type*} [Fintype n] [DecidableEq n] {A : Matrix n n ℂ} (hA : A.IsHermitian)
  {α β γ : ℝ}

/-- If every eigenvalue is `0`, `α`, `β` or `γ`, pairwise distinct, then
`∑ g(λ) = m_α g(α) + m_β g(β) + m_γ g(γ)` for every `g` with `g 0 = 0`. -/
lemma sum_comp_eigenvalues_of_mem_four (hαβ : α ≠ β) (hαγ : α ≠ γ) (hβγ : β ≠ γ)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β ∨
      hA.eigenvalues i = γ) (g : ℝ → ℝ) (hg : g 0 = 0) :
    ∑ i, g (hA.eigenvalues i) = #{i | hA.eigenvalues i = α} * g α +
      #{i | hA.eigenvalues i = β} * g β + #{i | hA.eigenvalues i = γ} * g γ := by
  have hpt (i : n) : g (hA.eigenvalues i) = (if hA.eigenvalues i = α then g α else 0) +
      (if hA.eigenvalues i = β then g β else 0) + (if hA.eigenvalues i = γ then g γ else 0) := by
    rcases hmem i with h | h | h | h
    · rw [h, hg]
      split_ifs <;> subst_vars <;> simp [hg]
    · rw [h, ite_eq_left rfl, ite_eq_right hαβ, ite_eq_right hαγ, add_zero, add_zero]
    · rw [h, ite_eq_right hαβ.symm, ite_eq_left rfl, ite_eq_right hβγ, zero_add, add_zero]
    · rw [h, ite_eq_right hαγ.symm, ite_eq_right hβγ.symm, ite_eq_left rfl, zero_add, zero_add]
  rw [sum_congr rfl fun i _ => hpt i, sum_add_distrib, sum_add_distrib, ← sum_filter,
    ← sum_filter, ← sum_filter, sum_const, sum_const, sum_const, nsmul_eq_mul, nsmul_eq_mul,
    nsmul_eq_mul]

example := sum_comp_eigenvalues_of_mem_four _ (by norm_num) (by norm_num) (by norm_num)
  (@eigenvalues_zeroIndicatorKernel_mem 5 ⟨Nat.prime_five⟩ (by decide)) id rfl

/-- `tr A = m_α α + m_β β + m_γ γ`. -/
lemma re_trace_eq_of_mem_four (hαβ : α ≠ β) (hαγ : α ≠ γ) (hβγ : β ≠ γ)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β ∨
      hA.eigenvalues i = γ) :
    A.trace.re = #{i | hA.eigenvalues i = α} * α + #{i | hA.eigenvalues i = β} * β +
      #{i | hA.eigenvalues i = γ} * γ := by
  rw [hA.trace_eq_sum_eigenvalues, Complex.re_sum]
  simpa using sum_comp_eigenvalues_of_mem_four hA hαβ hαγ hβγ hmem id rfl

example := re_trace_eq_of_mem_four _ (by norm_num) (by norm_num) (by norm_num)
  (@eigenvalues_zeroIndicatorKernel_mem 5 ⟨Nat.prime_five⟩ (by decide))

/-- `tr A² = m_α α² + m_β β² + m_γ γ²`. -/
lemma re_trace_mul_self_eq_of_mem_four (hαβ : α ≠ β) (hαγ : α ≠ γ) (hβγ : β ≠ γ)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β ∨
      hA.eigenvalues i = γ) :
    (A * A).trace.re = #{i | hA.eigenvalues i = α} * α ^ 2 +
      #{i | hA.eigenvalues i = β} * β ^ 2 + #{i | hA.eigenvalues i = γ} * γ ^ 2 := by
  rw [trace_mul_self_eq_sum_sq hA,
    ← sum_comp_eigenvalues_of_mem_four hA hαβ hαγ hβγ hmem (· ^ 2) (by norm_num)]
  simp only [Complex.re_sum, Complex.ofReal_re]

example := re_trace_mul_self_eq_of_mem_four _ (by norm_num) (by norm_num) (by norm_num)
  (@eigenvalues_zeroIndicatorKernel_mem 5 ⟨Nat.prime_five⟩ (by decide))

/-- `rank A = m_α + m_β + m_γ` for distinct nonzero `α`, `β`, `γ`. -/
lemma rank_eq_of_mem_four (hα : α ≠ 0) (hβ : β ≠ 0) (hγ : γ ≠ 0) (hαβ : α ≠ β) (hαγ : α ≠ γ)
    (hβγ : β ≠ γ)
    (hmem : ∀ i, hA.eigenvalues i = 0 ∨ hA.eigenvalues i = α ∨ hA.eigenvalues i = β ∨
      hA.eigenvalues i = γ) :
    (A.rank : ℝ) = #{i | hA.eigenvalues i = α} + #{i | hA.eigenvalues i = β} +
      #{i | hA.eigenvalues i = γ} := by
  have h := sum_comp_eigenvalues_of_mem_four hA hαβ hαγ hβγ hmem
    (fun x => if x = 0 then 0 else 1) (by simp)
  simp only [hα, hβ, hγ, ite_false, mul_one] at h
  rw [hA.rank_eq_card_non_zero_eigs, Fintype.card_subtype, ← h]
  simp only [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, nsmul_eq_mul,
    mul_one, ne_eq]

example := rank_eq_of_mem_four _ one_ne_zero two_ne_zero (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (@eigenvalues_zeroIndicatorKernel_mem 5 ⟨Nat.prime_five⟩ (by decide))

end FourValues

variable {p : ℕ} [Fact p.Prime]

/-- `(A²)_{aa} = 3` for `a ≠ 0` and `(A²)_{00} = p² + p − 1`, `A = K K*`. -/
lemma zeroIndicatorKernel_mul_conjTranspose_sq_single (hp : p ≠ 2) (a : ZMod p) :
    ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ
      ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ Pi.single a 1)) a =
      3 + if a = 0 then ((p : ℂ) ^ 2 + p - 4) else 0 := by
  rcases eq_or_ne a 0 with rfl | ha
  · have hu (x : ZMod p) (hx : x ≠ 0) :
        ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ Pi.single 0 1) x = 1 := by
      rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne hp _ hx, Pi.single_eq_same,
        Pi.single_eq_of_ne hx, Pi.single_eq_of_ne (neg_ne_zero.mpr hx)]
      ring
    have hu0 : ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ Pi.single 0 1) 0 = p := by
      rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_zero, sum_pi_single',
        ite_eq_left (mem_univ _), Pi.single_eq_same]
      ring
    rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_zero, sum_eq_add_sub_one_mul hu, hu0,
      ite_eq_left rfl]
    ring
  · have hna : -a ≠ a := fun h => ha ((eq_neg_iff_of_ne_two hp).mp h.symm)
    have hu0 : ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ Pi.single a 1) 0 = 1 := by
      rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_zero, sum_pi_single',
        ite_eq_left (mem_univ _), Pi.single_eq_of_ne' ha]
      ring
    have hua : ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ Pi.single a 1) a = 1 := by
      rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne hp _ ha, Pi.single_eq_of_ne' ha,
        Pi.single_eq_same, Pi.single_eq_of_ne hna]
      ring
    have huna :
        ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ Pi.single a 1) (-a) = 1 := by
      rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne hp _ (neg_ne_zero.mpr ha),
        Pi.single_eq_of_ne' ha, Pi.single_eq_of_ne hna, neg_neg, Pi.single_eq_same]
      ring
    rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne hp _ ha, hu0, hua, huna,
      ite_eq_right ha]
    ring

example := @zeroIndicatorKernel_mul_conjTranspose_sq_single 5 ⟨Nat.prime_five⟩ (by decide) 1

/-- `tr A² = p² + 4p − 4`, `A = K K*`. -/
theorem trace_zeroIndicatorKernel_mul_conjTranspose_sq (hp : p ≠ 2) :
    ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *
      (zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ)).trace = p ^ 2 + 4 * p - 4 := by
  have h (a : ZMod p) : ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *
      (zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ)) a a =
      3 + if a = 0 then ((p : ℂ) ^ 2 + p - 4) else 0 := by
    rw [← zeroIndicatorKernel_mul_conjTranspose_sq_single hp a, mulVec_mulVec]
    simp [mulVec, dotProduct, Pi.single_apply]
  rw [trace]
  simp only [diag_apply]
  rw [sum_congr rfl fun a _ => h a, sum_eq_add_sub_one_mul fun x hx => by
    rw [ite_eq_right hx, add_zero], ite_eq_left rfl]
  ring

example := @trace_zeroIndicatorKernel_mul_conjTranspose_sq 5 ⟨Nat.prime_five⟩ (by decide)

/-- The eigenvalues `1` and `p + 1` of `K K*` occur once and `2` occurs `(p − 3)/2` times. -/
theorem card_eigenvalues_zeroIndicatorKernel (hp : p ≠ 2) :
    (#{i | (isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)).eigenvalues i = 1} :
        ℝ) = 1 ∧
      (#{i | (isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)).eigenvalues i = 2} :
        ℝ) = (p - 3) / 2 ∧
      (#{i | (isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)).eigenvalues i =
        p + 1} : ℝ) = 1 := by
  set hA := isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)
  have hp3 := three_le_of_ne_two hp
  have hmem := eigenvalues_zeroIndicatorKernel_mem hp
  have h12 : (1 : ℝ) ≠ 2 := by norm_num
  have h13 : (1 : ℝ) ≠ p + 1 := fun h => by linarith
  have h23 : (2 : ℝ) ≠ p + 1 := fun h => by linarith
  have h1 := re_trace_eq_of_mem_four hA h12 h13 h23 hmem
  have h2 := re_trace_mul_self_eq_of_mem_four hA h12 h13 h23 hmem
  rw [trace_zeroIndicatorKernel_mul_conjTranspose hp, show 2 * (p : ℂ) - 1 =
    ((2 * (p : ℝ) - 1 : ℝ) : ℂ) by push_cast; ring, Complex.ofReal_re] at h1
  rw [trace_zeroIndicatorKernel_mul_conjTranspose_sq hp, show (p : ℂ) ^ 2 + 4 * p - 4 =
    (((p : ℝ) ^ 2 + 4 * p - 4 : ℝ) : ℂ) by push_cast; ring, Complex.ofReal_re] at h2
  set m₁ : ℝ := ((#{i | hA.eigenvalues i = 1} : ℕ) : ℝ)
  set m₂ : ℝ := ((#{i | hA.eigenvalues i = 2} : ℕ) : ℝ) with hm₂
  set k : ℕ := #{i | hA.eigenvalues i = p + 1}
  have hm₂p : m₂ ≤ p := by
    have h := card_filter_le (univ : Finset (ZMod p)) fun i => hA.eigenvalues i = 2
    rw [card_univ, ZMod.card] at h
    rw [hm₂]
    exact_mod_cast h
  have hm₂0 : 0 ≤ m₂ := Nat.cast_nonneg _
  have heq : 2 * m₂ + (p : ℝ) * (p + 1) * k = (p + 3) * (p - 1) := by
    linear_combination h1 - h2
  have hP : 0 ≤ (p : ℝ) * (p + 1) := by positivity
  have hk2 : k < 2 := by
    by_contra hcon
    have h2k : (2 : ℝ) ≤ k := by exact_mod_cast not_lt.mp hcon
    nlinarith [mul_le_mul_of_nonneg_left h2k hP]
  have hk0 : k ≠ 0 := by
    intro hk
    rw [hk, Nat.cast_zero] at heq
    nlinarith
  have hk : (k : ℝ) = 1 := by exact_mod_cast (show k = 1 by omega)
  rw [hk] at heq h1
  have hm₂' : m₂ = (p - 3) / 2 := by linear_combination heq / 2
  exact ⟨by linear_combination -h1 - 2 * hm₂', hm₂', hk⟩

example := @card_eigenvalues_zeroIndicatorKernel 5 ⟨Nat.prime_five⟩ (by decide)

end Transformer.Grokking.FlatSpectrum
