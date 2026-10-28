module

public import Transformer.Grokking.FlatSpectrum.DilationFibers

/-!
# Transport along the fibers of `P(a) + a b`

Source: `review/flat_spectrum.md`, *Readout-free bound for `P(a) + a b`* (added). Row `a = 0`
of `f(a, b) = P(a) + a b` is constant, so it vanishes against a centered `h`; for `a ≠ 0` the
map `b ↦ P(a) + a b` is a bijection. Hence
`∑_{a, b} ρ(f(a, b)) g(a) h(b) = ∑_c ρ(c) G(c)` with `G(c) = ∑_{a ≠ 0} g(a) h((c − P(a)) / a)`.
Substituting `c = P(a) + a x` turns the Gram entries of `G` into affine correlations:
`‖G‖² = ∑_{a' ≠ 0} ∑_{λ ≠ 0} g(λ a') conj g(a') E_h(λ, μ_λ(a'))`. The off-diagonal energy
`∑_{a' ≠ 0} ∑_{λ ∉ {0, 1}} |E_h(λ, μ_λ(a'))|²` is at most `(m − 1)(2p − 1) ‖h‖⁴`: the
nondegenerate `λ` contribute at most `(m − 1) p ‖h‖⁴` through their slope fibers and the
affine energy, the at most `m − 1` degenerate ones at most `(m − 1)(p − 1) ‖h‖⁴`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Polynomial

variable {p : ℕ} [Fact p.Prime]

/-- **The off-diagonal affine energy** of `P(a) + a b` for `deg P = m ≥ 2` and a centered `h`:
`∑_{a ≠ 0} ∑_{λ ∉ {0, 1}} |E_h(λ, μ_λ(a))|² ≤ (m − 1)(2p − 1) ‖h‖⁴`. -/
theorem sum_sum_norm_sq_affineCorr_slope_le {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree)
    {h : ZMod p → ℂ} (hh : ∑ x, h x = 0) :
    ∑ a ∈ univ.erase (0 : ZMod p), ∑ l ∈ (univ.erase (0 : ZMod p)).erase 1,
        ‖affineCorr h l (dilationSlope P l a)‖ ^ 2 ≤
      ((P.natDegree : ℝ) - 1) * (2 * p - 1) * (∑ x, ‖h x‖ ^ 2) ^ 2 := by
  classical
  set H := ∑ x, ‖h x‖ ^ 2
  set L := (univ.erase (0 : ZMod p)).erase 1
  have hk : ((P.natDegree - 1 : ℕ) : ℝ) = (P.natDegree : ℝ) - 1 := by
    rw [Nat.cast_sub (by omega), Nat.cast_one]
  have hp : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_le
  have hS : (#(univ.erase (0 : ZMod p)) : ℝ) = (p : ℝ) - 1 := by
    rw [card_erase_of_mem (mem_univ _), card_univ, ZMod.card,
      Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]
  have hdeg : ∑ l ∈ L with IsDegenerate P l, ∑ a ∈ univ.erase (0 : ZMod p),
      ‖affineCorr h l (dilationSlope P l a)‖ ^ 2 ≤ ((P.natDegree : ℝ) - 1) * ((p - 1) * H ^ 2) := by
    have hcard : #{l ∈ L | IsDegenerate P l} ≤ P.natDegree - 1 :=
      card_le_of_forall_isDegenerate hm fun l hl => by
        obtain ⟨hlL, hld⟩ := mem_filter.mp hl
        obtain ⟨hl1, hl0⟩ := mem_erase.mp hlL
        exact ⟨ne_of_mem_erase hl0, hl1, hld⟩
    calc _ ≤ ∑ l ∈ L with IsDegenerate P l, ∑ a ∈ univ.erase (0 : ZMod p), H ^ 2 := by
          refine sum_le_sum fun l hl => sum_le_sum fun a _ => ?_
          have hl0 : l ≠ 0 := ne_of_mem_erase (mem_of_mem_erase (mem_filter.mp hl).1)
          exact pow_le_pow_left₀ (norm_nonneg _) (norm_affineCorr_le h hl0 _) 2
      _ = (#{l ∈ L | IsDegenerate P l} : ℝ) * ((p - 1) * H ^ 2) := by
          rw [sum_const, sum_const, nsmul_eq_mul, nsmul_eq_mul, hS]
      _ ≤ ((P.natDegree : ℝ) - 1) * ((p - 1) * H ^ 2) := by
          rw [← hk]
          exact mul_le_mul_of_nonneg_right (by exact_mod_cast hcard)
            (mul_nonneg (by linarith) (sq_nonneg _))
  have hnd : ∑ l ∈ L with ¬IsDegenerate P l, ∑ a ∈ univ.erase (0 : ZMod p),
      ‖affineCorr h l (dilationSlope P l a)‖ ^ 2 ≤ ((P.natDegree : ℝ) - 1) * (p * H ^ 2) := by
    calc _ ≤ ∑ l ∈ L with ¬IsDegenerate P l,
          ((P.natDegree - 1 : ℕ) : ℝ) * ∑ μ, ‖affineCorr h l μ‖ ^ 2 :=
          sum_le_sum fun l hl => sum_comp_le_of_card_fiber_le _ (dilationSlope P l)
            (F := fun μ => ‖affineCorr h l μ‖ ^ 2) (fun _ => sq_nonneg _)
            (card_slope_fiber_le (by omega) (mem_filter.mp hl).2)
      _ ≤ ((P.natDegree - 1 : ℕ) : ℝ) * ∑ l, ∑ μ, ‖affineCorr h l μ‖ ^ 2 := by
          rw [← mul_sum]
          exact mul_le_mul_of_nonneg_left (sum_le_sum_of_subset_of_nonneg (subset_univ _)
            fun l _ _ => sum_nonneg fun μ _ => sq_nonneg _) (Nat.cast_nonneg _)
      _ = ((P.natDegree : ℝ) - 1) * (p * H ^ 2) := by rw [hk, sum_sum_norm_sq_affineCorr hh]
  rw [sum_comm, ← sum_filter_add_sum_filter_not L (IsDegenerate P)]
  calc _ ≤ ((P.natDegree : ℝ) - 1) * ((p - 1) * H ^ 2) + ((P.natDegree : ℝ) - 1) * (p * H ^ 2) :=
        add_le_add hdeg hnd
    _ = _ := by ring

/-- `G(c) = ∑_{a ≠ 0} g(a) h((c − P(a)) / a)`: `h` carried along the fibers of the rows
`a ≠ 0` of `P(a) + a b`. -/
noncomputable def fiberSum (P : (ZMod p)[X]) (g h : ZMod p → ℂ) (c : ZMod p) : ℂ :=
  ∑ a ∈ univ.erase 0, g a * h ((c - P.eval a) * a⁻¹)

/-- Against a centered `h`, the trilinear sum of `P(a) + a b` is `∑_c ρ(c) G(c)`. -/
lemma sum_sum_poly_add_mul_eq (P : (ZMod p)[X]) (ρ g : ZMod p → ℂ) {h : ZMod p → ℂ}
    (hh : ∑ b, h b = 0) :
    ∑ a, ∑ b, ρ (P.eval a + a * b) * g a * h b = ∑ c, ρ c * fiberSum P g h c := by
  rw [← add_sum_erase _ _ (mem_univ (0 : ZMod p))]
  have h0 : ∑ b, ρ (P.eval 0 + 0 * b) * g 0 * h b = 0 := by
    simp_rw [zero_mul, add_zero]
    rw [← mul_sum, hh, mul_zero]
  have hrow : ∀ a ∈ univ.erase (0 : ZMod p), ∑ b, ρ (P.eval a + a * b) * g a * h b =
      ∑ c, ρ c * (g a * h ((c - P.eval a) * a⁻¹)) := by
    intro a ha
    have ha0 : a ≠ 0 := ne_of_mem_erase ha
    rw [← sum_mul_add_eq ha0 (P.eval a) fun c => ρ c * (g a * h ((c - P.eval a) * a⁻¹))]
    refine sum_congr rfl fun b _ => ?_
    rw [add_sub_cancel_right, mul_comm a b, mul_inv_cancel_right₀ ha0, add_comm (b * a),
      mul_assoc]
  rw [h0, zero_add, sum_congr rfl hrow, sum_comm]
  simp_rw [fiberSum, mul_sum]

/-- The Gram entries of `G` are affine correlations:
`∑_c h((c − P(a)) / a) conj h((c − P(a')) / a') = E_h(a / a', (P(a) − P(a')) / a')`. -/
lemma sum_fiber_mul_star (P : (ZMod p)[X]) (h : ZMod p → ℂ) {a : ZMod p} (ha : a ≠ 0)
    (a' : ZMod p) :
    ∑ c, h ((c - P.eval a) * a⁻¹) * star (h ((c - P.eval a') * a'⁻¹)) =
      affineCorr h (a * a'⁻¹) ((P.eval a - P.eval a') * a'⁻¹) := by
  rw [← sum_mul_add_eq ha (P.eval a)
    fun c => h ((c - P.eval a) * a⁻¹) * star (h ((c - P.eval a') * a'⁻¹)), affineCorr]
  refine sum_congr rfl fun x _ => ?_
  rw [add_sub_cancel_right, mul_comm a x, mul_inv_cancel_right₀ ha]
  congr 3
  ring

/-- `‖G‖² = ∑_{a' ≠ 0} ∑_{λ ≠ 0} g(λ a') conj g(a') E_h(λ, μ_λ(a'))`. -/
lemma sum_fiberSum_mul_star (P : (ZMod p)[X]) (g h : ZMod p → ℂ) :
    ∑ c, fiberSum P g h c * star (fiberSum P g h c) =
      ∑ a' ∈ univ.erase (0 : ZMod p), ∑ l ∈ univ.erase (0 : ZMod p),
        g (l * a') * star (g a') * affineCorr h l (dilationSlope P l a') := by
  have hexp : ∀ c, fiberSum P g h c * star (fiberSum P g h c) =
      ∑ a' ∈ univ.erase (0 : ZMod p), ∑ a ∈ univ.erase (0 : ZMod p),
        g a * star (g a') * (h ((c - P.eval a) * a⁻¹) * star (h ((c - P.eval a') * a'⁻¹))) := by
    intro c
    rw [fiberSum, star_sum, sum_mul_sum, sum_comm]
    refine sum_congr rfl fun a' _ => sum_congr rfl fun a _ => ?_
    rw [star_mul']
    ring
  simp_rw [hexp]
  rw [sum_comm]
  refine sum_congr rfl fun a' ha' => ?_
  rw [sum_comm]
  have ha'0 : a' ≠ 0 := ne_of_mem_erase ha'
  refine (sum_equiv (Equiv.mulRight₀ a' ha'0) (fun l => by simp [ha'0]) fun l hl => ?_).symm
  simp only [Equiv.mulRight₀_apply]
  rw [← mul_sum, sum_fiber_mul_star P h (mul_ne_zero (ne_of_mem_erase hl) ha'0), mul_inv_cancel_right₀ ha'0]
  rfl

section Examples

attribute [local instance] fact_prime_5

example := sum_sum_norm_sq_affineCorr_slope_le (p := 5) (P := X ^ 2) (by rw [natDegree_X_pow])
  (h := fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

example := sum_sum_poly_add_mul_eq (p := 5) (X ^ 2) (fun c => (c.val : ℂ)) (fun _ => 1)
  (h := fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

example (h : ZMod 5 → ℂ) := sum_fiber_mul_star (X ^ 2) h (a := 2) (by decide) 3

example (g h : ZMod 5 → ℂ) := sum_fiberSum_mul_star (X ^ 2) g h

end Examples

end Transformer.Grokking.FlatSpectrum
