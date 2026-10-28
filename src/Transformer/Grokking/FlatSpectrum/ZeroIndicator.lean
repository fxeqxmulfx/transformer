module

public import Transformer.Grokking.FlatSpectrum.OneHotSpectrum
public import Transformer.Grokking.FlatSpectrum.MulCharKernel

/-!
# The indicator of answer `0`: the eigenvectors of `K K*`

Source: `review/flat_spectrum.md`, Torch check: "Uncentered, the exact value is
(√(p+1)+1+(p−3)/√2)²/(2p−1) = 31.01 and 59.53; the spectrum matches to 1e-14." Since
`a³ + a b = a (a² + b)`, row `0` of `K = [a³ + a b = 0]` is all ones and row `a ≠ 0` has its 1 at
`b = −a²`. For odd prime `p`, `A = K K*` acts by `(A v)_0 = (p − 1) v_0 + ∑_x v_x` and
`(A v)_a = v_0 + v_a + v_{−a}` for `a ≠ 0`, as `a' (a'² − a²) = 0` exactly at `a' ∈ {0, a, −a}`.

On an eigenvector with eigenvalue `λ ∉ {0, 2}`, `v_a = v_{−a}` and `(λ − 2) v_a = v_0` for
`a ≠ 0`, so `(λ − 1)(λ − p − 1) v_0 = 0` and `v_0 ≠ 0`: every eigenvalue of `A` is `0`, `1`, `2`
or `p + 1`. Applying `A` to the unit vectors gives `tr A = 2p − 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

/-- The indicator of answer `0` of `a³ + a b`, `review/flat_spectrum.md`, Torch check. -/
def zeroIndicatorKernel (p : ℕ) : Matrix (ZMod p) (ZMod p) ℂ :=
  classKernel (fun a b => a ^ 3 + a * b) 0

variable {p : ℕ} [Fact p.Prime]

/-- A sum over `𝔽_p` whose terms off `0` are all `c`. -/
lemma sum_eq_add_sub_one_mul {f : ZMod p → ℂ} {c : ℂ} (h : ∀ x, x ≠ 0 → f x = c) :
    ∑ x, f x = f 0 + (p - 1) * c := by
  rw [← add_sum_erase univ f (mem_univ 0), sum_congr rfl fun x hx => h x (ne_of_mem_erase hx),
    sum_const, card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul,
    Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]

example := @sum_eq_add_sub_one_mul 5 ⟨Nat.prime_five⟩ (fun _ => 1) 1 fun _ _ => rfl

/-- `a³ + a b = 0` iff `a = 0` or `b = −a²`. -/
lemma cube_add_mul_eq_zero_iff {a b : ZMod p} : a ^ 3 + a * b = 0 ↔ a = 0 ∨ b = -a ^ 2 := by
  rw [show a ^ 3 + a * b = a * (a ^ 2 + b) by ring, mul_eq_zero, add_eq_zero_iff_eq_neg']

/-- `(K w)_0 = ∑_b w_b` and `(K w)_a = w_{−a²}` for `a ≠ 0`. -/
lemma zeroIndicatorKernel_mulVec (w : ZMod p → ℂ) (a : ZMod p) :
    (zeroIndicatorKernel p *ᵥ w) a = if a = 0 then ∑ b, w b else w (-a ^ 2) := by
  simp only [mulVec, dotProduct, zeroIndicatorKernel, classKernel, taskKernel, of_apply,
    cube_add_mul_eq_zero_iff]
  rcases eq_or_ne a 0 with rfl | ha
  · simp
  · simp [ha, ite_mul]

/-- `(K* v)_b = ∑_a [a³ + a b = 0] v_a`. -/
lemma conjTranspose_zeroIndicatorKernel_mulVec (v : ZMod p → ℂ) (b : ZMod p) :
    ((zeroIndicatorKernel p)ᴴ *ᵥ v) b = ∑ a, if a ^ 3 + a * b = 0 then v a else 0 := by
  simp only [mulVec, dotProduct, conjTranspose_apply, zeroIndicatorKernel, classKernel, taskKernel,
    of_apply]
  refine sum_congr rfl fun a _ => ?_
  split_ifs <;> simp

/-- `(A v)_a = v_0 + v_a + v_{−a}` for `a ≠ 0`, `A = K K*`. -/
theorem zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne (hp : p ≠ 2) (v : ZMod p → ℂ)
    {a : ZMod p} (ha : a ≠ 0) :
    ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ v) a = v 0 + v a + v (-a) := by
  rw [← mulVec_mulVec, zeroIndicatorKernel_mulVec, ite_eq_right ha,
    conjTranspose_zeroIndicatorKernel_mulVec]
  have hpt (a' : ZMod p) : (if a' ^ 3 + a' * -a ^ 2 = 0 then v a' else 0) =
      (if a' = 0 then v a' else 0) + (if a' ^ 2 = a ^ 2 then v a' else 0) := by
    rcases eq_or_ne a' 0 with rfl | ha'
    · rw [ite_eq_left (show (0 : ZMod p) ^ 3 + 0 * -a ^ 2 = 0 by ring), ite_eq_left rfl,
        ite_eq_right (zero_sq_ne_sq ha), add_zero]
    · have h : a' ^ 3 + a' * -a ^ 2 = 0 ↔ a' ^ 2 = a ^ 2 := by
        rw [show a' ^ 3 + a' * -a ^ 2 = a' * (a' ^ 2 - a ^ 2) by ring, mul_eq_zero, sub_eq_zero,
          or_iff_right ha']
      rw [ite_eq_right ha', zero_add]
      exact if_congr h rfl rfl
  rw [sum_congr rfl fun a' _ => hpt a', sum_add_distrib, sum_ite_eq', ite_eq_left (mem_univ _),
    sum_ite_sq_eq hp ha, add_assoc]

example := @zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne 5 ⟨Nat.prime_five⟩ (by decide)
  (fun _ => 1) 1 (by decide)

/-- `(A v)_0 = (p − 1) v_0 + ∑_x v_x`, `A = K K*`. -/
theorem zeroIndicatorKernel_mul_conjTranspose_mulVec_zero (v : ZMod p → ℂ) :
    ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ v) 0 = (p - 1) * v 0 + ∑ x, v x := by
  rw [← mulVec_mulVec, zeroIndicatorKernel_mulVec, ite_eq_left rfl]
  simp only [conjTranspose_zeroIndicatorKernel_mulVec]
  rw [sum_comm]
  have hrow (a : ZMod p) : ∑ b : ZMod p, (if a ^ 3 + a * b = 0 then v a else 0) =
      v a + if a = 0 then (p - 1) * v 0 else 0 := by
    rcases eq_or_ne a 0 with rfl | ha
    · simp [ZMod.card]
      ring
    · simp [cube_add_mul_eq_zero_iff, ha]
  rw [sum_congr rfl fun a _ => hrow a, sum_add_distrib, sum_ite_eq', ite_eq_left (mem_univ _)]
  ring

/-- Every eigenvalue of `A = K K*` is `0`, `1`, `2` or `p + 1`. -/
theorem eigenvalues_zeroIndicatorKernel_mem (hp : p ≠ 2) (i : ZMod p) :
    (isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)).eigenvalues i = 0 ∨
      (isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)).eigenvalues i = 1 ∨
      (isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)).eigenvalues i = 2 ∨
      (isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)).eigenvalues i = p + 1 := by
  set hA := isHermitian_mul_conjTranspose_self (zeroIndicatorKernel p)
  set μ := hA.eigenvalues i
  have heq (a : ZMod p) : ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ
      ⇑(hA.eigenvectorBasis i)) a = (μ : ℂ) * (hA.eigenvectorBasis i) a := by
    rw [hA.mulVec_eigenvectorBasis i, Pi.smul_apply, Complex.real_smul]
  set v : ZMod p → ℂ := ⇑(hA.eigenvectorBasis i) with hv
  by_contra hcon
  simp only [not_or] at hcon
  obtain ⟨h0, h1, h2, h3⟩ := hcon
  have hμ0 : (μ : ℂ) ≠ 0 := by exact_mod_cast h0
  have hμ1 : (μ : ℂ) - 1 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast h1)
  have hμ2 : (μ : ℂ) - 2 ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast h2)
  have hμ3 : (μ : ℂ) - (p + 1) ≠ 0 := sub_ne_zero.mpr (by exact_mod_cast h3)
  have hrow (a : ZMod p) (ha : a ≠ 0) : (μ : ℂ) * v a = v 0 + v a + v (-a) :=
    (heq a).symm.trans (zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne hp v ha)
  have hsym (a : ZMod p) (ha : a ≠ 0) : v (-a) = v a := by
    have e2 := hrow (-a) (neg_ne_zero.mpr ha)
    rw [neg_neg] at e2
    have h : (μ : ℂ) * (v a - v (-a)) = 0 := by linear_combination hrow a ha - e2
    exact (sub_eq_zero.mp ((mul_eq_zero.mp h).resolve_left hμ0)).symm
  have hlin (a : ZMod p) (ha : a ≠ 0) : ((μ : ℂ) - 2) * v a = v 0 := by
    have h := hrow a ha
    rw [hsym a ha] at h
    linear_combination h
  have hS : ((μ : ℂ) - 2) * ∑ x, v x = ((μ : ℂ) - 2) * v 0 + (p - 1) * v 0 := by
    rw [mul_sum]
    exact sum_eq_add_sub_one_mul hlin
  have hz : v 0 = 0 := by
    have e0 := (heq 0).symm.trans (zeroIndicatorKernel_mul_conjTranspose_mulVec_zero v)
    have h : ((μ : ℂ) - 1) * ((μ : ℂ) - (p + 1)) * v 0 = 0 := by
      linear_combination ((μ : ℂ) - 2) * e0 + hS
    exact (mul_eq_zero.mp h).resolve_left (mul_ne_zero hμ1 hμ3)
  have hzero (a : ZMod p) : v a = 0 := by
    rcases eq_or_ne a 0 with rfl | ha
    · exact hz
    · have h := hlin a ha
      rw [hz] at h
      exact (mul_eq_zero.mp h).resolve_left hμ2
  exact hA.eigenvectorBasis.orthonormal.ne_zero i (by ext a; simpa [hv] using hzero a)

example := @eigenvalues_zeroIndicatorKernel_mem 5 ⟨Nat.prime_five⟩ (by decide) 0

/-- `A_{aa} = 1` for `a ≠ 0` and `A_{00} = p`. -/
lemma zeroIndicatorKernel_mul_conjTranspose_mulVec_single (hp : p ≠ 2) (a : ZMod p) :
    ((zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) *ᵥ Pi.single a 1) a =
      1 + if a = 0 then (p - 1 : ℂ) else 0 := by
  rcases eq_or_ne a 0 with rfl | ha
  · rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_zero, sum_pi_single', ite_eq_left (mem_univ _),
      Pi.single_eq_same, ite_eq_left rfl]
    ring
  · have hna : -a ≠ a := fun h => ha ((eq_neg_iff_of_ne_two hp).mp h.symm)
    rw [zeroIndicatorKernel_mul_conjTranspose_mulVec_of_ne hp _ ha, Pi.single_eq_of_ne' ha,
      Pi.single_eq_same, Pi.single_eq_of_ne hna, ite_eq_right ha]
    ring

example := @zeroIndicatorKernel_mul_conjTranspose_mulVec_single 5 ⟨Nat.prime_five⟩ (by decide) 1

/-- `tr A = 2p − 1`. -/
theorem trace_zeroIndicatorKernel_mul_conjTranspose (hp : p ≠ 2) :
    (zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ).trace = 2 * p - 1 := by
  have h (a : ZMod p) : (zeroIndicatorKernel p * (zeroIndicatorKernel p)ᴴ) a a =
      1 + if a = 0 then (p - 1 : ℂ) else 0 := by
    rw [← zeroIndicatorKernel_mul_conjTranspose_mulVec_single hp a]
    simp [mulVec, dotProduct, Pi.single_apply]
  rw [trace]
  simp only [diag_apply]
  rw [sum_congr rfl fun a _ => h a, sum_eq_add_sub_one_mul fun x hx => by
    rw [ite_eq_right hx, add_zero], ite_eq_left rfl]
  ring

example := @trace_zeroIndicatorKernel_mul_conjTranspose 5 ⟨Nat.prime_five⟩ (by decide)

end Transformer.Grokking.FlatSpectrum
