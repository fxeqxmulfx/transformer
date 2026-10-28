module

public import Transformer.Grokking.FlatSpectrum.ThreeEigenvalues
public import Transformer.Grokking.FlatSpectrum.ShiftKernel
public import Transformer.Grokking.FlatSpectrum.RealKernels

/-!
# The kernel `[χ(a³ + a b)]` of a multiplicative character

Source: `review/flat_spectrum.md`, Theorem B: "Those equal χ₃(f), whose kernel is itself spread
out (effective rank 48 of 97)." For nontrivial `χ`, `K(a, b) = χ(a) χ(a² + b)`, so `A = K K*`
has `A(a, a') = χ(a) conj χ(a') (p [a'² = a²] − 1)`. For odd `p`, `a'² = a²` exactly at
`a' = ±a`, which gives `A² = 2p A − (p + 1) W` and `A W = (p + 1) W` for `W = χ χ*`. So
`A³ = (3p + 1) A² − 2p (p + 1) A`. The traces are `tr A = (p − 1)²`, `tr W = p − 1` and
`tr A² = 2p (p − 1)² − (p + 1)(p − 1)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {p : ℕ}

/-- The readout `χ(f)` of `f = a³ + a b`, `review/flat_spectrum.md`, Theorem B. -/
noncomputable def cubicKernelMulChar (χ : MulChar (ZMod p) ℂ) : Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (fun a b => a ^ 3 + a * b) χ

variable [Fact p.Prime] {χ : MulChar (ZMod p) ℂ}

/-- `a'² = a²` exactly at `a' = ±a`, two points for odd `p` and `a ≠ 0`. -/
lemma sum_ite_sq_eq (hp : p ≠ 2) {a : ZMod p} (ha : a ≠ 0) (G : ZMod p → ℂ) :
    ∑ a', (if a' ^ 2 = a ^ 2 then G a' else 0) = G a + G (-a) := by
  have hne : a ≠ -a := fun h => ha ((eq_neg_iff_of_ne_two hp).mp h)
  have hfilter : ({a' | a' ^ 2 = a ^ 2} : Finset (ZMod p)) = {a, -a} := by
    ext a'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
      Finset.mem_singleton]
    exact sq_eq_sq_iff_eq_or_eq_neg
  rw [← Finset.sum_filter, hfilter, Finset.sum_pair hne]

example := @sum_ite_sq_eq 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide) fun _ => 1

/-- `sum_ite_sq_eq` with the equation flipped. -/
lemma sum_ite_sq_eq' (hp : p ≠ 2) {a : ZMod p} (ha : a ≠ 0) (G : ZMod p → ℂ) :
    ∑ a', (if a ^ 2 = a' ^ 2 then G a' else 0) = G a + G (-a) := by
  rw [← sum_ite_sq_eq hp ha G]
  refine Finset.sum_congr rfl fun a' _ => ?_
  by_cases h : a ^ 2 = a' ^ 2
  · rw [ite_eq_left h, ite_eq_left h.symm]
  · rw [ite_eq_right h, ite_eq_right (Ne.symm h)]

example := @sum_ite_sq_eq' 5 ⟨Nat.prime_five⟩ (by decide) 1 (by decide) fun _ => 1

/-- `(K K*)(a, a') = χ(a) conj χ(a') (p [a'² = a²] − 1)` for `K = [χ(a³ + a b)]`, `χ ≠ 1`. -/
theorem cubicKernelMulChar_mul_conjTranspose_apply (hχ : χ ≠ 1) (a a' : ZMod p) :
    (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ) a a' =
      χ a * star (χ a') * ((if a' ^ 2 = a ^ 2 then (p : ℂ) else 0) - 1) := by
  simp only [mul_apply, conjTranspose_apply, cubicKernelMulChar, taskKernel, of_apply]
  have hrow (b : ZMod p) : χ (a ^ 3 + a * b) * star (χ (a' ^ 3 + a' * b)) =
      χ a * star (χ a') * (star (χ (a' ^ 2 + b)) * χ (a ^ 2 + b)) := by
    rw [show a ^ 3 + a * b = a * (a ^ 2 + b) by ring,
      show a' ^ 3 + a' * b = a' * (a' ^ 2 + b) by ring, map_mul, map_mul, star_mul']
    ring
  rw [Finset.sum_congr rfl fun b _ => hrow b, ← Finset.mul_sum,
    ← Equiv.sum_comp (Equiv.addRight (-(a' ^ 2)))]
  simp only [Equiv.coe_addRight]
  rw [Finset.sum_congr rfl fun w _ => by
    rw [show a' ^ 2 + (w + -(a' ^ 2)) = w by ring,
      show a ^ 2 + (w + -(a' ^ 2)) = w + (a ^ 2 - a' ^ 2) by ring],
    sum_star_mulChar_mul_add hχ]
  by_cases h : a ^ 2 = a' ^ 2
  · rw [ite_eq_left (sub_eq_zero.mpr h), ite_eq_left h.symm]
  · rw [ite_eq_right fun h' => h (sub_eq_zero.mp h'), ite_eq_right (Ne.symm h), zero_sub]

example := @cubicKernelMulChar_mul_conjTranspose_apply 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) 1 2

lemma zero_sq_ne_sq {a : ZMod p} (ha : a ≠ 0) : ¬ (0 : ZMod p) ^ 2 = a ^ 2 := fun h =>
  ha ((pow_eq_zero_iff two_ne_zero).mp (h.symm.trans (zero_pow two_ne_zero)))

example := @zero_sq_ne_sq 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- `A² = 2p A − (p + 1) W` for `A = K K*` and `W = χ χ*`. -/
theorem cubicKernelMulChar_mul_conjTranspose_sq (hp : p ≠ 2) (hχ : χ ≠ 1) :
    cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ *
        (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ) =
      (2 * p : ℂ) • (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ) -
        ((p : ℂ) + 1) • of fun a a' => χ a * star (χ a') := by
  ext a a''
  rw [mul_apply]
  simp only [Matrix.sub_apply, Matrix.smul_apply, of_apply, smul_eq_mul,
    cubicKernelMulChar_mul_conjTranspose_apply hχ]
  by_cases ha : a = 0
  · simp [ha, MulChar.map_zero]
  by_cases ha'' : a'' = 0
  · simp [ha'', MulChar.map_zero]
  have hfac (a' : ZMod p) :
      χ a * star (χ a') * ((if a' ^ 2 = a ^ 2 then (p : ℂ) else 0) - 1) *
        (χ a' * star (χ a'') * ((if a'' ^ 2 = a' ^ 2 then (p : ℂ) else 0) - 1)) =
      χ a * star (χ a'') * (star (χ a') * χ a' *
        (((if a' ^ 2 = a ^ 2 then (p : ℂ) else 0) - 1) *
          ((if a'' ^ 2 = a' ^ 2 then (p : ℂ) else 0) - 1))) := by ring
  have hpt (a' : ZMod p) : ((if a' ^ 2 = a ^ 2 then (p : ℂ) else 0) - 1) *
      ((if a'' ^ 2 = a' ^ 2 then (p : ℂ) else 0) - 1) =
      (if a' ^ 2 = a ^ 2 then (p : ℂ) * ((if a'' ^ 2 = a' ^ 2 then (p : ℂ) else 0) - 1)
        else 0) - ((if a'' ^ 2 = a' ^ 2 then (p : ℂ) else 0) - 1) := by
    split_ifs <;> ring
  rw [Finset.sum_congr rfl fun a' _ => hfac a', ← Finset.mul_sum, sum_star_mulChar_mul_self_mul,
    Finset.sum_congr rfl fun a' _ => hpt a', Finset.sum_sub_distrib, sum_ite_sq_eq hp ha,
    Finset.sum_sub_distrib, sum_ite_sq_eq' hp ha'', neg_sq, ite_eq_right (zero_sq_ne_sq ha),
    ite_eq_right fun h : a'' ^ 2 = (0 : ZMod p) ^ 2 => zero_sq_ne_sq ha'' h.symm]
  simp only [Finset.sum_const, card_univ, ZMod.card, nsmul_eq_mul, mul_one]
  ring

example := @cubicKernelMulChar_mul_conjTranspose_sq 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `A W = (p + 1) W`: `χ` is an eigenvector of `A = K K*` with eigenvalue `p + 1`. -/
theorem cubicKernelMulChar_mul_conjTranspose_mul_outer (hp : p ≠ 2) (hχ : χ ≠ 1) :
    cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ * (of fun a a' => χ a * star (χ a')) =
      ((p : ℂ) + 1) • of fun a a' => χ a * star (χ a') := by
  ext a a''
  rw [mul_apply]
  simp only [Matrix.smul_apply, of_apply, smul_eq_mul,
    cubicKernelMulChar_mul_conjTranspose_apply hχ]
  by_cases ha : a = 0
  · simp [ha, MulChar.map_zero]
  have hfac (a' : ZMod p) :
      χ a * star (χ a') * ((if a' ^ 2 = a ^ 2 then (p : ℂ) else 0) - 1) *
        (χ a' * star (χ a'')) = χ a * star (χ a'') * (star (χ a') * χ a' *
          ((if a' ^ 2 = a ^ 2 then (p : ℂ) else 0) - 1)) := by ring
  rw [Finset.sum_congr rfl fun a' _ => hfac a', ← Finset.mul_sum, sum_star_mulChar_mul_self_mul,
    Finset.sum_sub_distrib, sum_ite_sq_eq hp ha, ite_eq_right (zero_sq_ne_sq ha)]
  simp only [Finset.sum_const, card_univ, ZMod.card, nsmul_eq_mul, mul_one]
  ring

example := @cubicKernelMulChar_mul_conjTranspose_mul_outer 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `A³ = (3p + 1) A² − 2p (p + 1) A` for `A = K K*`. -/
theorem cubicKernelMulChar_mul_conjTranspose_cube (hp : p ≠ 2) (hχ : χ ≠ 1) :
    cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ *
        (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ) *
        (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ) =
      ((2 * p + (p + 1) : ℝ) : ℂ) • (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ *
        (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ)) -
      ((2 * p * (p + 1) : ℝ) : ℂ) • (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ) := by
  have h2 := cubicKernelMulChar_mul_conjTranspose_sq hp hχ
  have hW := cubicKernelMulChar_mul_conjTranspose_mul_outer hp hχ
  set A := cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ
  set W : Matrix (ZMod p) (ZMod p) ℂ := of fun a a' => χ a * star (χ a')
  have hW' : ((p : ℂ) + 1) • W = (2 * p : ℂ) • A - A * A := by rw [h2, sub_sub_cancel]
  have h3 : A * (A * A) = (2 * p : ℂ) • (A * A) - ((p : ℂ) + 1) • (((p : ℂ) + 1) • W) := by
    conv_lhs => rw [h2]
    rw [Matrix.mul_sub, Matrix.mul_smul, Matrix.mul_smul, hW]
  rw [Matrix.mul_assoc, h3, hW']
  push_cast
  module

example := @cubicKernelMulChar_mul_conjTranspose_cube 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `tr A = (p − 1)²` for `A = K K*`. -/
lemma trace_cubicKernelMulChar_mul_conjTranspose (hχ : χ ≠ 1) :
    (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ).trace = ((p : ℂ) - 1) ^ 2 := by
  have hpt (a : ZMod p) : (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ) a a =
      star (χ a) * χ a * ((p : ℂ) - 1) := by
    rw [cubicKernelMulChar_mul_conjTranspose_apply hχ, ite_eq_left rfl, mul_comm (χ a)]
  simp only [Matrix.trace, Matrix.diag_apply]
  rw [Finset.sum_congr rfl fun a _ => hpt a, sum_star_mulChar_mul_self_mul]
  simp only [Finset.sum_const, card_univ, ZMod.card, nsmul_eq_mul]
  ring

example := @trace_cubicKernelMulChar_mul_conjTranspose 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- `tr W = p − 1` for `W = χ χ*`. -/
lemma trace_mulChar_outer (χ : MulChar (ZMod p) ℂ) :
    (of fun a a' => χ a * star (χ a') : Matrix (ZMod p) (ZMod p) ℂ).trace = (p : ℂ) - 1 := by
  have h := sum_star_mulChar_mul_self_mul χ fun _ => (1 : ℂ)
  simp only [mul_one, Finset.sum_const, card_univ, ZMod.card, nsmul_eq_mul] at h
  simp only [Matrix.trace, Matrix.diag_apply, of_apply]
  rw [Finset.sum_congr rfl fun a _ => mul_comm (χ a) (star (χ a)), h]

example := @trace_mulChar_outer 5 ⟨Nat.prime_five⟩ 1

/-- `tr A² = 2p (p − 1)² − (p + 1)(p − 1)`, from `A² = 2p A − (p + 1) W`. -/
lemma trace_cubicKernelMulChar_mul_conjTranspose_sq (hp : p ≠ 2) (hχ : χ ≠ 1) :
    (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ *
      (cubicKernelMulChar χ * (cubicKernelMulChar χ)ᴴ)).trace =
      2 * p * ((p : ℂ) - 1) ^ 2 - ((p : ℂ) + 1) * ((p : ℂ) - 1) := by
  rw [cubicKernelMulChar_mul_conjTranspose_sq hp hχ, trace_sub, trace_smul, trace_smul,
    trace_cubicKernelMulChar_mul_conjTranspose hχ, trace_mulChar_outer, smul_eq_mul, smul_eq_mul]

example := @trace_cubicKernelMulChar_mul_conjTranspose_sq 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

end Transformer.Grokking.FlatSpectrum
