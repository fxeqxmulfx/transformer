module

public import Transformer.Grokking.FlatSpectrum.JacobiSums
public import Transformer.Grokking.FlatSpectrum.MulCharSpectrum
public import Transformer.Grokking.FlatSpectrum.ProductCostFlat

/-!
# What each code of a task costs one product layer

Source: `review/flat_spectrum.md`, NV10. A model need not write `ψ(f)`: it may write any code
`g(f)` that a readout can decode. For `a³ + a b` the additive code is flat and costs
`(3/2) p^{4/3}`, and by `not_isLocalMinOn_productCost` it is no local minimum among interpolants
of any proper training set. The multiplicative code costs
`(3/2) ((p + 1)^{1/3} + (p − 3)/2 · (2p)^{1/3})`, less, but still of order `p^{4/3}`. For `a b`
the multiplicative code `χ(a) χ(b)` has rank one and costs at most `(3/2) (p − 1)^{2/3}`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

section Additive

variable (p : ℕ) [NeZero p]

/-- The additive code of `a³ + a b` costs one product layer `(3/2) p^{4/3}`. -/
theorem productCost_cubicKernel :
    productCost (cubicKernel p) = 3 / 2 * (p : ℝ) ^ (4 / 3 : ℝ) := by
  have hp : (0 : ℝ) < p := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne p)
  simp only [productCost, singularValues_cubicKernel, Finset.sum_const, Finset.card_univ,
    ZMod.card, nsmul_eq_mul]
  rw [Real.sqrt_eq_rpow, ← Real.rpow_mul hp.le,
    show (4 / 3 : ℝ) = 1 + 1 / 2 * (2 / 3) by norm_num, Real.rpow_add hp, Real.rpow_one]

/-- **Theorem A for one product layer on `a³ + a b`**: the additive code is no local minimum of
the product cost among the interpolants of any proper training set. -/
theorem not_isLocalMinOn_productCost_cubicKernel {Ω : Set (ZMod p × ZMod p)} (hΩ : Ω ≠ Set.univ) :
    ¬ IsLocalMinOn productCost {X | ∀ ij ∈ Ω, X ij.1 ij.2 = cubicKernel p ij.1 ij.2}
      (cubicKernel p) :=
  not_isLocalMinOn_productCost (singularValues_cubicKernel p) (cubicKernel_apply_ne_zero p) hΩ

example := not_isLocalMinOn_productCost_cubicKernel 3 (Ω := ∅) Set.empty_ne_univ

end Additive

section RankOne

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- A rank-one table `u wᵀ` with `‖u‖² = ‖w‖² = N` costs at most `(3/2) N^{2/3}`. -/
theorem productCost_vecMulVec_le (u : m → ℂ) (w : n → ℂ) {N : ℝ} (hN : 0 < N)
    (hu : ∑ a, ‖u a‖ ^ 2 = N) (hw : ∑ b, ‖w b‖ ^ 2 = N) :
    productCost (vecMulVec u w) ≤ 3 / 2 * N ^ (2 / 3 : ℝ) := by
  obtain ⟨v, hvdef⟩ : ∃ v : ℝ, v = N ^ (1 / 3 : ℝ) := ⟨_, rfl⟩
  have hv : 0 < v := hvdef ▸ Real.rpow_pos_of_pos hN _
  have hv3 : N = v ^ 3 := by
    rw [hvdef, ← Real.rpow_natCast, ← Real.rpow_mul hN.le]
    norm_num
  have hv2 : N ^ (2 / 3 : ℝ) = v ^ 2 := by
    rw [hvdef, ← Real.rpow_natCast, ← Real.rpow_mul hN.le]
    norm_num
  obtain ⟨r, hrdef⟩ : ∃ r : ℝ, r = (√v)⁻¹ := ⟨_, rfl⟩
  have hr2 : r ^ 2 = v⁻¹ := by
    rw [hrdef, inv_pow, Real.sq_sqrt hv.le]
  have hc : (r : ℂ) ^ 2 * (v : ℂ) = 1 := by
    rw [← Complex.ofReal_pow, ← Complex.ofReal_mul, hr2, inv_mul_cancel₀ hv.ne',
      Complex.ofReal_one]
  have hfac : vecMulVec u w = (of fun a (_ : Unit) => (r : ℂ) * u a) *
      diagonal (fun _ : Unit => (v : ℂ)) * (of fun b (_ : Unit) => (r : ℂ) * star (w b))ᴴ := by
    ext a b
    simp only [vecMulVec_apply, Matrix.mul_apply, conjTranspose_apply, of_apply,
      Fintype.sum_unique, star_mul', Complex.star_def, Complex.conj_ofReal,
      starRingEnd_self_apply, diagonal_apply_eq]
    linear_combination (-(u a * w b)) * hc
  have hb := productCost_le (of fun a (_ : Unit) => (r : ℂ) * u a) (fun _ : Unit => (v : ℂ))
    (of fun b (_ : Unit) => (r : ℂ) * star (w b))
  rw [← hfac] at hb
  have hr : v⁻¹ * v ^ 3 = v ^ 2 := by field_simp
  have hG : frobeniusSq (of fun a (_ : Unit) => (r : ℂ) * u a : Matrix m Unit ℂ) = v ^ 2 := by
    simp only [frobeniusSq, of_apply, Fintype.sum_unique, norm_mul, Complex.norm_real,
      Real.norm_eq_abs, mul_pow, sq_abs]
    rw [← Finset.mul_sum, hu, hr2, hv3, hr]
  have hW : frobeniusSq (of fun b (_ : Unit) => (r : ℂ) * star (w b) : Matrix n Unit ℂ) =
      v ^ 2 := by
    simp only [frobeniusSq, of_apply, Fintype.sum_unique, norm_mul, norm_star,
      Complex.norm_real, Real.norm_eq_abs, mul_pow, sq_abs]
    rw [← Finset.mul_sum, hw, hr2, hv3, hr]
  have ho : ∑ _u : Unit, ‖(v : ℂ)‖ ^ 2 = v ^ 2 := by
    simp [Complex.norm_real, sq_abs]
  rw [hG, ho, hW] at hb
  rw [hv2]
  linarith

example := productCost_vecMulVec_le (m := Fin 1) (n := Fin 1) (fun _ => 1) (fun _ => 1) one_pos
  (by simp) (by simp)

end RankOne

section Multiplicative

variable {p : ℕ} [Fact p.Prime] {χ : MulChar (ZMod p) ℂ}

/-- `(√y)^{2/3} = y^{1/3}`. -/
lemma sqrt_rpow_two_thirds {y : ℝ} (hy : 0 ≤ y) : √y ^ (2 / 3 : ℝ) = y ^ (1 / 3 : ℝ) := by
  rw [Real.sqrt_eq_rpow, ← Real.rpow_mul hy]
  norm_num

example := sqrt_rpow_two_thirds (y := 1) zero_le_one

/-- The multiplicative code of `a³ + a b` costs one product layer
`(3/2) ((p + 1)^{1/3} + (p − 3)/2 · (2p)^{1/3})`, which at `p = 97` is `415.0`. -/
theorem productCost_cubicKernelMulChar (hp : p ≠ 2) (hχ : χ ≠ 1) :
    productCost (cubicKernelMulChar χ) =
      3 / 2 * (((p : ℝ) + 1) ^ (1 / 3 : ℝ) +
        ((p : ℝ) - 3) / 2 * (2 * (p : ℝ)) ^ (1 / 3 : ℝ)) := by
  have hp3 := three_le_of_ne_two hp
  obtain ⟨h₂, h₁⟩ := card_eigenvalues_cubicKernelMulChar hp hχ
  simp only [productCost, singularValues_eq_sqrt_eigenvalues_mul_conjTranspose]
  rw [sum_comp_eigenvalues_of_mem _ (fun h => by linarith)
    (eigenvalues_cubicKernelMulChar_mem hp hχ) (fun x => √x ^ (2 / 3 : ℝ))
    (by simp only [Real.sqrt_zero]; exact Real.zero_rpow (by norm_num)), h₁, h₂]
  simp only [sqrt_rpow_two_thirds (show (0 : ℝ) ≤ 2 * p by linarith),
    sqrt_rpow_two_thirds (show (0 : ℝ) ≤ p + 1 by linarith)]
  ring

example := @productCost_cubicKernelMulChar 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- **The code of `a b` that one layer can afford**: `χ(a b) = χ(a) χ(b)` has rank one, and costs
at most `(3/2) (p − 1)^{2/3}`, of order `p^{2/3}` against `p^{4/3}` for a flat code. -/
theorem productCost_mulCode_le (χ : MulChar (ZMod p) ℂ) :
    productCost (taskKernel (fun a b => a * b) χ) ≤ 3 / 2 * ((p : ℝ) - 1) ^ (2 / 3 : ℝ) := by
  have hp2 : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  have hsum : ∑ a : ZMod p, ‖χ a‖ ^ 2 = (p : ℝ) - 1 := by
    have h (a : ZMod p) : ‖χ a‖ ^ 2 = 1 - if a = 0 then 1 else 0 := by
      by_cases ha : a = 0
      · simp [ha, χ.map_nonunit not_isUnit_zero]
      · simp [ha, norm_mulChar_eq_one χ ha]
    simp only [h, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, ZMod.card,
      nsmul_eq_mul, mul_one, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  have hE : taskKernel (fun a b => a * b) χ = vecMulVec χ χ := by
    ext a b
    simp [taskKernel, vecMulVec_apply, map_mul]
  rw [hE]
  exact productCost_vecMulVec_le _ _ (by linarith) hsum hsum

end Multiplicative

end Transformer.Grokking.FlatSpectrum
