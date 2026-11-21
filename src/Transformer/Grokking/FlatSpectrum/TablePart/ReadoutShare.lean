module

public import Transformer.Grokking.FlatSpectrum.ReadoutFree
public import Transformer.Grokking.FlatSpectrum.SeenShare
public import Transformer.Grokking.FlatSpectrum.TablePart.RowSum

/-!
# The share of the table every code needs

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT8. Let
`f(a, b) = P(a) + a b` with `m = deg P ≥ 2` and `κ = (m − 1)(2p − 1)`. For every code `ρ` of the
answers, with centered part `ρ_c`, `(p − 1)² ‖ρ_c‖² ≤ (1 + √κ) ‖K_ρ‖²_S1`
(`centered_energy_le_nuclearNorm_sq`). The proof tests `K_ρ` against the matrix `W` that holds
`ρ_c(f(a, b))` on the rows `a ≠ 0`. Each of these rows is a permutation of `ρ_c`, so it sums to
zero and `W` kills constants. On centered vectors the readout-free bound gives
`‖W‖²_op ≤ (1 + √κ) ‖ρ_c‖²`, and `tr(Wᴴ K_ρ) = (p − 1) ‖ρ_c‖²`.

A minimum of M pays for its norm with seen cells (`SeenShare`), so the seen part of the table of
any code carries at least this much (`centered_energy_le_of_isMinOn`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset Matrix Polynomial
open scoped ComplexOrder

variable {p : ℕ} [Fact p.Prime]

/-- The code minus its mean. -/
noncomputable def centerCode (ρ : ZMod p → ℂ) (c : ZMod p) : ℂ := ρ c - (∑ x, ρ x) / p

lemma sum_centerCode (ρ : ZMod p → ℂ) : ∑ c, centerCode ρ c = 0 := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.2 (Fact.out : p.Prime).ne_zero
  simp only [centerCode, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, ZMod.card,
    nsmul_eq_mul]
  field_simp
  ring

/-- Removing the mean takes `‖Σ v‖² / p` off the squared norm. -/
lemma sum_norm_sq_centerCode (v : ZMod p → ℂ) :
    ∑ c, ‖v c‖ ^ 2 = ∑ c, ‖centerCode v c‖ ^ 2 + ‖∑ c, v c‖ ^ 2 / p := by
  have hp : (p : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Fact.out : p.Prime).ne_zero
  set μ : ℂ := (∑ x, v x) / p with hμ
  have hsplit (c : ZMod p) : ‖v c‖ ^ 2 = ‖centerCode v c‖ ^ 2 + ‖μ‖ ^ 2 +
      2 * ((centerCode v c).re * μ.re + (centerCode v c).im * μ.im) := by
    have hv : v c = centerCode v c + μ := by simp [centerCode, hμ]
    have h : ‖v c‖ ^ 2 = ‖centerCode v c + μ‖ ^ 2 := by rw [← hv]
    rw [h, Complex.sq_norm, Complex.sq_norm, Complex.sq_norm, Complex.normSq_apply,
      Complex.normSq_apply, Complex.normSq_apply, Complex.add_re, Complex.add_im]
    ring
  have hre : ∑ c, (centerCode v c).re = 0 := by
    rw [← Complex.re_sum, sum_centerCode]
    rfl
  have him : ∑ c, (centerCode v c).im = 0 := by
    rw [← Complex.im_sum, sum_centerCode]
    rfl
  have hμn : (p : ℝ) * ‖μ‖ ^ 2 = ‖∑ c, v c‖ ^ 2 / p := by
    rw [hμ, norm_div, Complex.norm_natCast]
    field_simp
  simp only [hsplit, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, ZMod.card,
    nsmul_eq_mul, ← Finset.mul_sum, ← Finset.sum_mul, hre, him]
  rw [← hμn]
  ring

/-- For `a ≠ 0`, the row `b ↦ P(a) + a b` runs once over the answers. -/
lemma sum_row_eq (P : (ZMod p)[X]) {a : ZMod p} (ha : a ≠ 0) {M : Type*} [AddCommMonoid M]
    (F : ZMod p → M) : ∑ b, F (P.eval a + a * b) = ∑ c, F c := by
  rw [← sum_mul_add_eq ha (P.eval a) F]
  exact Finset.sum_congr rfl fun b _ => by rw [add_comm]

example (F : ZMod 3 → ℂ) :
    ∑ b : ZMod 3, F ((X : (ZMod 3)[X]).eval 1 + 1 * b) = ∑ c, F c :=
  sum_row_eq X one_ne_zero F

/-- The test matrix: the centered code on the rows `a ≠ 0`. -/
noncomputable def testKernel (P : (ZMod p)[X]) (ρ : ZMod p → ℂ) : Matrix (ZMod p) (ZMod p) ℂ :=
  Matrix.of fun a b => if a = 0 then 0 else centerCode ρ (P.eval a + a * b)

/-- **The test matrix is bounded**, by the readout-free bound on centered vectors. -/
lemma sum_norm_sq_testKernel_le {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree) (ρ v : ZMod p → ℂ) :
    ∑ a, ‖(testKernel P ρ *ᵥ v) a‖ ^ 2 ≤ (∑ c, ‖centerCode ρ c‖ ^ 2) *
      (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) * ∑ b, ‖v b‖ ^ 2 := by
  have hnorm : ∑ b, ‖centerCode v b‖ ^ 2 ≤ ∑ b, ‖v b‖ ^ 2 := by
    rw [sum_norm_sq_centerCode v]
    have h : 0 ≤ ‖∑ c, v c‖ ^ 2 / p := by positivity
    linarith
  have hrow (a : ZMod p) : ‖(testKernel P ρ *ᵥ v) a‖ ^ 2 ≤
      ‖∑ b, centerCode ρ (P.eval a + a * b) * centerCode v b‖ ^ 2 := by
    by_cases ha : a = 0
    · have h : (testKernel P ρ *ᵥ v) a = 0 := by simp [mulVec, dotProduct, testKernel, ha]
      rw [h, norm_zero, zero_pow two_ne_zero]
      positivity
    · have hμ : ∑ b, centerCode ρ (P.eval a + a * b) * ((∑ x, v x) / p) = 0 := by
        rw [← Finset.sum_mul, sum_row_eq P ha (centerCode ρ), sum_centerCode, zero_mul]
      have e : (testKernel P ρ *ᵥ v) a =
          ∑ b, centerCode ρ (P.eval a + a * b) * centerCode v b := by
        calc (testKernel P ρ *ᵥ v) a = ∑ b, centerCode ρ (P.eval a + a * b) * v b := by
              simp only [mulVec, dotProduct, testKernel, of_apply, ha, ↓reduceIte]
          _ = ∑ b, centerCode ρ (P.eval a + a * b) * centerCode v b +
                ∑ b, centerCode ρ (P.eval a + a * b) * ((∑ x, v x) / p) := by
              rw [← Finset.sum_add_distrib]
              refine Finset.sum_congr rfl fun b _ => ?_
              rw [← mul_add]
              congr 1
              simp [centerCode]
          _ = _ := by rw [hμ, add_zero]
      rw [e]
  have hC := sum_norm_sq_sum_poly_add_mul_le hm (centerCode ρ) (centerCode v) (sum_centerCode v)
  have hE : 0 ≤ ∑ c, ‖centerCode ρ c‖ ^ 2 := Finset.sum_nonneg fun _ _ => by positivity
  have hA : 0 ≤ 1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1)) := by positivity
  calc ∑ a, ‖(testKernel P ρ *ᵥ v) a‖ ^ 2
      ≤ ∑ a, ‖∑ b, centerCode ρ (P.eval a + a * b) * centerCode v b‖ ^ 2 :=
        Finset.sum_le_sum fun a _ => hrow a
    _ ≤ (∑ c, ‖centerCode ρ c‖ ^ 2) * (∑ b, ‖centerCode v b‖ ^ 2) *
        (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) := hC
    _ ≤ (∑ c, ‖centerCode ρ c‖ ^ 2) * (∑ b, ‖v b‖ ^ 2) *
        (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hnorm hE) hA
    _ = _ := by ring

example (ρ v : ZMod 3 → ℂ) :=
  sum_norm_sq_testKernel_le (P := (X ^ 2 : (ZMod 3)[X])) (by simp) ρ v

/-- The pairing of the test matrix with the table is `(p − 1) ‖ρ_c‖²`. -/
lemma trace_testKernel (P : (ZMod p)[X]) (ρ : ZMod p → ℂ) :
    ((testKernel P ρ)ᴴ * taskKernel (fun a b => P.eval a + a * b) ρ).trace =
      (((p : ℝ) - 1) * ∑ c, ‖centerCode ρ c‖ ^ 2 : ℝ) := by
  rw [trace_conjTranspose_mul]
  set S : ℂ := ((∑ c, ‖centerCode ρ c‖ ^ 2 : ℝ) : ℂ) with hS
  have hrow (a : ZMod p) : ∑ b, star (testKernel P ρ a b) *
      taskKernel (fun a b => P.eval a + a * b) ρ a b = S - if a = 0 then S else 0 := by
    by_cases ha : a = 0
    · simp [testKernel, ha]
    · simp only [testKernel, taskKernel, of_apply, ha, ↓reduceIte, sub_zero]
      rw [sum_row_eq P ha fun c => star (centerCode ρ c) * ρ c]
      have e (c : ZMod p) : star (centerCode ρ c) * ρ c = star (centerCode ρ c) * centerCode ρ c +
          star (centerCode ρ c) * ((∑ x, ρ x) / p) := by
        rw [← mul_add]
        congr 1
        simp [centerCode]
      have h0 : ∑ c, star (centerCode ρ c) = 0 := by rw [← star_sum, sum_centerCode, star_zero]
      simp only [e, Finset.sum_add_distrib, ← Finset.sum_mul, h0, zero_mul, add_zero]
      exact star_dotProduct_self_eq_sum_norm_sq (centerCode ρ)
  simp only [hrow, Finset.sum_sub_distrib, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte,
    Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul]
  rw [Complex.ofReal_mul, ← hS]
  push_cast
  ring

/-- **Every code pays in nuclear norm**: `(p − 1)² ‖ρ_c‖² ≤ (1 + √κ) ‖K_ρ‖²_S1`, for every code
`ρ` of `P(a) + a b` with `deg P ≥ 2`. -/
theorem centered_energy_le_nuclearNorm_sq {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree)
    (ρ : ZMod p → ℂ) : ((p : ℝ) - 1) ^ 2 * ∑ c, ‖centerCode ρ c‖ ^ 2 ≤
      (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) *
        nuclearNorm (taskKernel (fun a b => P.eval a + a * b) ρ) ^ 2 := by
  have htr := trace_testKernel P ρ
  have hb := sum_norm_sq_testKernel_le hm ρ
  set E := ∑ c, ‖centerCode ρ c‖ ^ 2 with hE
  set A := 1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1)) with hA
  set K := taskKernel (fun a b => P.eval a + a * b) ρ with hK
  have hE0 : 0 ≤ E := Finset.sum_nonneg fun _ _ => by positivity
  have hApos : 0 < A := by positivity
  have hp1 : (0 : ℝ) ≤ (p : ℝ) - 1 := by
    have h := (Fact.out : p.Prime).one_lt
    have h2 : (1 : ℝ) < p := by exact_mod_cast h
    linarith
  rcases hE0.eq_or_lt with h0 | hEpos
  · rw [← h0, mul_zero]
    positivity
  have hEA : 0 < E * A := mul_pos hEpos hApos
  have hcpos : 0 < √(E * A) := Real.sqrt_pos.2 hEA
  have hcon : IsContraction (((√(E * A) : ℝ) : ℂ)⁻¹ • testKernel P ρ) :=
    isContraction_smul_of_sum_norm_sq_le hcpos fun v => by
      rw [Real.sq_sqrt hEA.le]
      exact hb v
  have h := norm_trace_le_nuclearNorm hcon K
  rw [conjTranspose_smul, Matrix.smul_mul, trace_smul, norm_smul, norm_star, norm_inv,
    Complex.norm_real, Real.norm_of_nonneg hcpos.le, htr, Complex.norm_real,
    Real.norm_of_nonneg (mul_nonneg hp1 hE0), inv_mul_le_iff₀ hcpos] at h
  have h2 := pow_le_pow_left₀ (mul_nonneg hp1 hE0) h 2
  rw [mul_pow, mul_pow, Real.sq_sqrt hEA.le] at h2
  have h3 : E * (((p : ℝ) - 1) ^ 2 * E) ≤ E * (A * nuclearNorm K ^ 2) := by nlinarith
  exact le_of_mul_le_mul_left h3 hEpos

example (ρ : ZMod 3 → ℂ) :=
  centered_energy_le_nuclearNorm_sq (P := (X ^ 2 : (ZMod 3)[X])) (by simp) ρ

end Transformer.Grokking.FlatSpectrum.TablePart
