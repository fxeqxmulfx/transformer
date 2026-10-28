module

public import Transformer.Grokking.FlatSpectrum.FiberTransport

/-!
# The readout-free bound for `P(a) + a b`

Source: `review/flat_spectrum.md`, *Readout-free bound for `P(a) + a b`* (added). Let
`f(a, b) = P(a) + a b` on `ℤ/p` with `m = deg P ≥ 2` and `κ = (m − 1)(2p − 1)`. Every readout
`ρ : ℤ/p → ℂ`, every `g` and every centered `h` satisfy
`|∑_{a, b} ρ(f(a, b)) g(a) h(b)|² ≤ (1 + √κ) ‖ρ‖² ‖g‖² ‖h‖²`, and in operator form
`∑_a |∑_b ρ(f(a, b)) h(b)|² ≤ (1 + √κ) ‖ρ‖² ‖h‖²`. Any centered `h` reaches `(p − 1) ‖ρ‖² ‖h‖²`
at most (`AffineReadout`), so for large `p` no readout of `f` has a centered mode with more than
a share of order `√(2m / p)`.

Proof: the sum is `∑_c ρ(c) G(c)` (`FiberTransport`), so it is at most `‖ρ‖² ‖G‖²` in square. In
`‖G‖² = ∑_{a' ≠ 0} ∑_{λ ≠ 0} g(λ a') conj g(a') E_h(λ, μ_λ(a'))` the dilation `λ = 1` gives at most
`‖g‖² ‖h‖²`, and Cauchy–Schwarz over the pairs `(a', λ)` with `λ ≠ 1` bounds the rest by
`‖g‖² · √κ ‖h‖²` through the off-diagonal energy.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Polynomial

variable {p : ℕ} [Fact p.Prime]

/-- Cauchy–Schwarz for a finite double sum. -/
lemma norm_sum_sum_mul_le {ι κ : Type*} (s : Finset ι) (t : Finset κ) (u v : ι → κ → ℂ) :
    ‖∑ i ∈ s, ∑ j ∈ t, u i j * v i j‖ ≤
      √(∑ i ∈ s, ∑ j ∈ t, ‖u i j‖ ^ 2) * √(∑ i ∈ s, ∑ j ∈ t, ‖v i j‖ ^ 2) := by
  have h := norm_sum_star_mul_le (s ×ˢ t) (fun x => star (u x.1 x.2)) fun x => v x.1 x.2
  simp only [star_star, norm_star, sum_product] at h
  exact h

example := norm_sum_sum_mul_le (univ : Finset (Fin 2)) (univ : Finset (Fin 3))
  (fun i _ => (i.val : ℂ)) fun _ j => (j.val : ℂ)

/-- **The fiber transport is bounded**: `‖G‖² ≤ (1 + √((m − 1)(2p − 1))) ‖g‖² ‖h‖²` for
`deg P = m ≥ 2` and a centered `h`. -/
theorem sum_norm_sq_fiberSum_le {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree) (g : ZMod p → ℂ)
    {h : ZMod p → ℂ} (hh : ∑ x, h x = 0) :
    ∑ c, ‖fiberSum P g h c‖ ^ 2 ≤
      (∑ a, ‖g a‖ ^ 2) * (∑ x, ‖h x‖ ^ 2) * (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) := by
  set S := univ.erase (0 : ZMod p)
  set L := S.erase 1
  set Gn := ∑ a, ‖g a‖ ^ 2
  set Hn := ∑ x, ‖h x‖ ^ 2
  set κ := ((P.natDegree : ℝ) - 1) * (2 * p - 1)
  have hGn : 0 ≤ Gn := sum_nonneg fun _ _ => sq_nonneg _
  have hHn : 0 ≤ Hn := sum_nonneg fun _ _ => sq_nonneg _
  have hκ : 0 ≤ κ := by
    have hp : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
    have hd : (2 : ℝ) ≤ P.natDegree := by exact_mod_cast hm
    exact mul_nonneg (by linarith) (by linarith)
  have hsub : ∀ F : ZMod p → ℝ, (∀ x, 0 ≤ F x) → ∑ x ∈ S, F x ≤ ∑ x, F x := fun F hF =>
    sum_le_sum_of_subset_of_nonneg (subset_univ _) fun x _ _ => hF x
  have hreal : ∑ c, ‖fiberSum P g h c‖ ^ 2 =
      ‖∑ c, fiberSum P g h c * star (fiberSum P g h c)‖ := by
    have hc : ∑ c, fiberSum P g h c * star (fiberSum P g h c) =
        ((∑ c, ‖fiberSum P g h c‖ ^ 2 : ℝ) : ℂ) := by
      push_cast
      exact sum_congr rfl fun c _ => by rw [Complex.star_def, Complex.mul_conj']
    rw [hc, Complex.norm_of_nonneg (sum_nonneg fun _ _ => sq_nonneg _)]
  have hsplit : ∑ c, fiberSum P g h c * star (fiberSum P g h c) =
      ∑ a' ∈ S, g a' * star (g a') * affineCorr h 1 (dilationSlope P 1 a') +
        ∑ a' ∈ S, ∑ l ∈ L, g (l * a') * star (g a') * affineCorr h l (dilationSlope P l a') := by
    rw [sum_fiberSum_mul_star, ← sum_add_distrib]
    refine sum_congr rfl fun a' _ => ?_
    rw [← add_sum_erase _ _ (mem_erase.mpr ⟨one_ne_zero, mem_univ (1 : ZMod p)⟩), one_mul]
  have hdiag : ‖∑ a' ∈ S, g a' * star (g a') * affineCorr h 1 (dilationSlope P 1 a')‖ ≤
      Gn * Hn := by
    refine (norm_sum_le _ _).trans ?_
    calc ∑ a' ∈ S, ‖g a' * star (g a') * affineCorr h 1 (dilationSlope P 1 a')‖
        ≤ ∑ a' ∈ S, ‖g a'‖ ^ 2 * Hn := by
          refine sum_le_sum fun a' _ => ?_
          rw [norm_mul, norm_mul, norm_star, ← sq]
          exact mul_le_mul_of_nonneg_left (norm_affineCorr_le h one_ne_zero _) (sq_nonneg _)
      _ ≤ Gn * Hn := by
          rw [← sum_mul]
          exact mul_le_mul_of_nonneg_right (hsub _ fun _ => sq_nonneg _) hHn
  have hu : √(∑ a' ∈ S, ∑ l ∈ L, ‖g (l * a') * star (g a')‖ ^ 2) ≤ Gn := by
    rw [← Real.sqrt_mul_self hGn]
    refine Real.sqrt_le_sqrt ?_
    calc ∑ a' ∈ S, ∑ l ∈ L, ‖g (l * a') * star (g a')‖ ^ 2
        = ∑ a' ∈ S, ‖g a'‖ ^ 2 * ∑ l ∈ L, ‖g (l * a')‖ ^ 2 := by
          refine sum_congr rfl fun a' _ => ?_
          rw [mul_sum]
          exact sum_congr rfl fun l _ => by rw [norm_mul, norm_star, mul_pow, mul_comm]
      _ ≤ ∑ a' ∈ S, ‖g a'‖ ^ 2 * Gn := by
          refine sum_le_sum fun a' ha' => mul_le_mul_of_nonneg_left ?_ (sq_nonneg _)
          calc ∑ l ∈ L, ‖g (l * a')‖ ^ 2 ≤ ∑ l, ‖g (l * a')‖ ^ 2 :=
                sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => sq_nonneg _
            _ = Gn := Equiv.sum_comp (Equiv.mulRight₀ a' (ne_of_mem_erase ha'))
                fun x => ‖g x‖ ^ 2
      _ ≤ Gn * Gn := by
          rw [← sum_mul]
          exact mul_le_mul_of_nonneg_right (hsub _ fun _ => sq_nonneg _) hGn
  have hv : √(∑ a' ∈ S, ∑ l ∈ L, ‖affineCorr h l (dilationSlope P l a')‖ ^ 2) ≤ √κ * Hn := by
    rw [← Real.sqrt_sq hHn, ← Real.sqrt_mul hκ]
    exact Real.sqrt_le_sqrt (sum_sum_norm_sq_affineCorr_slope_le hm hh)
  have hoff : ‖∑ a' ∈ S, ∑ l ∈ L, g (l * a') * star (g a') * affineCorr h l (dilationSlope P l a')‖
      ≤ Gn * (√κ * Hn) :=
    (norm_sum_sum_mul_le S L (fun a' l => g (l * a') * star (g a'))
      fun a' l => affineCorr h l (dilationSlope P l a')).trans
        (mul_le_mul hu hv (Real.sqrt_nonneg _) hGn)
  rw [hreal, hsplit]
  calc _ ≤ _ := norm_add_le _ _
    _ ≤ Gn * Hn + Gn * (√κ * Hn) := add_le_add hdiag hoff
    _ = Gn * Hn * (1 + √κ) := by ring

/-- **The readout-free bound for `P(a) + a b`**: for `deg P = m ≥ 2`, every readout `ρ`, every
`g` and every centered `h`,
`|∑_{a, b} ρ(P(a) + a b) g(a) h(b)|² ≤ ‖ρ‖² ‖g‖² ‖h‖² (1 + √((m − 1)(2p − 1)))`. -/
theorem norm_sq_sum_poly_add_mul_le {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree)
    (ρ g h : ZMod p → ℂ) (hh : ∑ b, h b = 0) :
    ‖∑ a, ∑ b, ρ (P.eval a + a * b) * g a * h b‖ ^ 2 ≤
      (∑ c, ‖ρ c‖ ^ 2) * (∑ a, ‖g a‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) *
        (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) := by
  have hcs := norm_sum_star_mul_le univ (fun c => star (ρ c)) (fiberSum P g h)
  simp only [star_star, norm_star] at hcs
  rw [sum_sum_poly_add_mul_eq P ρ g hh]
  calc _ ≤ (√(∑ c, ‖ρ c‖ ^ 2) * √(∑ c, ‖fiberSum P g h c‖ ^ 2)) ^ 2 :=
        pow_le_pow_left₀ (norm_nonneg _) hcs 2
    _ = (∑ c, ‖ρ c‖ ^ 2) * ∑ c, ‖fiberSum P g h c‖ ^ 2 := by
        rw [mul_pow, Real.sq_sqrt (sum_nonneg fun _ _ => sq_nonneg _),
          Real.sq_sqrt (sum_nonneg fun _ _ => sq_nonneg _)]
    _ ≤ (∑ c, ‖ρ c‖ ^ 2) * ((∑ a, ‖g a‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) *
          (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1)))) :=
        mul_le_mul_of_nonneg_left (sum_norm_sq_fiberSum_le hm g hh)
          (sum_nonneg fun _ _ => sq_nonneg _)
    _ = _ := by ring

/-- **Readout-free flatness of `P(a) + a b`**, operator form: for `deg P = m ≥ 2`, every readout
`ρ` and every centered `h`,
`∑_a |∑_b ρ(P(a) + a b) h(b)|² ≤ (1 + √((m − 1)(2p − 1))) ‖ρ‖² ‖h‖²`. -/
theorem sum_norm_sq_sum_poly_add_mul_le {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree)
    (ρ h : ZMod p → ℂ) (hh : ∑ b, h b = 0) :
    ∑ a, ‖∑ b, ρ (P.eval a + a * b) * h b‖ ^ 2 ≤
      (∑ c, ‖ρ c‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) * (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) := by
  have key := norm_sq_sum_poly_add_mul_le hm ρ (fun a => star (∑ b, ρ (P.eval a + a * b) * h b))
    h hh
  have hN : ∑ a, ∑ b, ρ (P.eval a + a * b) * star (∑ b, ρ (P.eval a + a * b) * h b) * h b =
      ((∑ a, ‖∑ b, ρ (P.eval a + a * b) * h b‖ ^ 2 : ℝ) : ℂ) := by
    push_cast
    refine sum_congr rfl fun a _ => ?_
    rw [← Complex.conj_mul', ← Complex.star_def, mul_sum]
    exact sum_congr rfl fun b _ => by ring
  simp only [norm_star] at key
  rw [hN, Complex.norm_of_nonneg (sum_nonneg fun _ _ => sq_nonneg _)] at key
  set T := ∑ a, ‖∑ b, ρ (P.eval a + a * b) * h b‖ ^ 2
  have hT : 0 ≤ T := sum_nonneg fun _ _ => sq_nonneg _
  have hR : 0 ≤ (∑ c, ‖ρ c‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) *
      (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) := by positivity
  rcases hT.eq_or_lt with h0 | hpos
  · rw [← h0]
    exact hR
  · nlinarith

section Examples

attribute [local instance] fact_prime_5

example := sum_norm_sq_fiberSum_le (p := 5) (P := X ^ 2) (by rw [natDegree_X_pow]) (fun _ => 1)
  (h := fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

example := norm_sq_sum_poly_add_mul_le (p := 5) (P := X ^ 3) (by rw [natDegree_X_pow]; omega)
  (fun c => (c.val : ℂ)) (fun _ => 1)
  (fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

example := sum_norm_sq_sum_poly_add_mul_le (p := 5) (P := X ^ 3) (by rw [natDegree_X_pow]; omega)
  (fun c => (c.val : ℂ))
  (fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

end Examples

end Transformer.Grokking.FlatSpectrum
