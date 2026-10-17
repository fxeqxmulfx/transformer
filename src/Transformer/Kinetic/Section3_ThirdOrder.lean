/-
# The third-order inequality

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: for a
product law the third cumulant `κ(F, G, H) = 𝔼[F̄ Ḡ H̄]` is bounded by first and
mixed second differences:

  `|κ| ≤ ¼ Σ_l Σ_m [s_m(F)(s²_{l,m}(G) s_l(H) + s_l(G) s²_{l,m}(H))
          + s_m(G)(s²_{l,m}(F) s_l(H) + s_l(F) s²_{l,m}(H))] + ½ Σ_l s_l(F) s_l(G) s_l(H)`
  (`abs_thirdCumulant_le`).

The source writes `κ` as a covariance and applies the Glauber calculus twice.
Here `κ = Cov(F̄Ḡ, H)`. The swap identity writes it through
`F̄Ḡ(x) - F̄Ḡ(x^l) = F̄ δ_l G + Ḡ δ_l F - δ_l F δ_l G`. The first two terms are
integrals of a centred observable against a cross term, bounded by
`abs_integral_centered_mul_le` and `abs_crossTerm_sub_le`.
-/

import Transformer.Kinetic.Section3_CrossTerm

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {N : ℕ} (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]

/-- One term of the swap identity for `F̄Ḡ` and `H`. -/
theorem abs_swap_term_le {F G H : (Idx N → ℝ) → ℝ} (hF : Measurable F) (hG : Measurable G)
    (hH : Measurable H) {CF CG CH : ℝ} (hFb : ∀ x, |F x| ≤ CF) (hGb : ∀ x, |G x| ≤ CG)
    (hHb : ∀ x, |H x| ≤ CH) {sF sG sH : Idx N → ℝ} {s2F s2G s2H : Idx N → Idx N → ℝ}
    (hsF : ∀ x l z, |F x - F (Function.update x l z)| ≤ sF l)
    (hsG : ∀ x l z, |G x - G (Function.update x l z)| ≤ sG l)
    (hsH : ∀ x l z, |H x - H (Function.update x l z)| ≤ sH l)
    (hs2F : ∀ x l m z z', m ≠ l → |F x - F (Function.update x l z) - F (Function.update x m z') +
      F (Function.update (Function.update x l z) m z')| ≤ s2F l m)
    (hs2G : ∀ x l m z z', m ≠ l → |G x - G (Function.update x l z) - G (Function.update x m z') +
      G (Function.update (Function.update x l z) m z')| ≤ s2G l m)
    (hs2H : ∀ x l m z z', m ≠ l → |H x - H (Function.update x l z) - H (Function.update x m z') +
      H (Function.update (Function.update x l z) m z')| ≤ s2H l m)
    (hdF : ∀ l, sF l ≤ s2F l l) (hdG : ∀ l, sG l ≤ s2G l l) (hdH : ∀ l, sH l ≤ s2H l l)
    (l : Idx N) :
    |∫ p, ((F p.1 - ∫ x, F x ∂(Measure.pi μs)) * (G p.1 - ∫ x, G x ∂(Measure.pi μs)) -
        (F (Function.update p.1 l (p.2 l)) - ∫ x, F x ∂(Measure.pi μs)) *
          (G (Function.update p.1 l (p.2 l)) - ∫ x, G x ∂(Measure.pi μs))) *
        (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))
        ∂((Measure.pi μs).prod (Measure.pi μs))| ≤
      1 / 2 * ∑ m, sF m * (s2G l m * sH l + sG l * s2H l m) +
      1 / 2 * ∑ m, sG m * (s2F l m * sH l + sF l * s2H l m) + sF l * sG l * sH l := by
  set μ := Measure.pi μs
  set ν := μ.prod μ
  set cF := ∫ x, F x ∂μ
  set cG := ∫ x, G x ∂μ
  have hσ : Measurable fun p : (Idx N → ℝ) × (Idx N → ℝ) => Function.update p.1 l (p.2 l) :=
    measurable_fst.comp (measurePreserving_swapAt μs l).measurable
  have hCF : 0 ≤ CF := (abs_nonneg _).trans (hFb 0)
  have hCG : 0 ≤ CG := (abs_nonneg _).trans (hGb 0)
  have hCH : 0 ≤ CH := (abs_nonneg _).trans (hHb 0)
  have hcF : |cF| ≤ CF := by
    simpa using norm_integral_le_of_norm_le_const (μ := μ) (f := F)
      (ae_of_all _ fun x => by simpa using hFb x)
  have hcG : |cG| ≤ CG := by
    simpa using norm_integral_le_of_norm_le_const (μ := μ) (f := G)
      (ae_of_all _ fun x => by simpa using hGb x)
  have hd : ∀ (X : (Idx N → ℝ) → ℝ) (C : ℝ), (∀ x, |X x| ≤ C) → ∀ x y, |X x - X y| ≤ 2 * C :=
    fun X C hX x y => (abs_sub _ _).trans (by linarith [hX x, hX y])
  have hmX : ∀ X : (Idx N → ℝ) → ℝ, Measurable X →
      Measurable fun p : (Idx N → ℝ) × (Idx N → ℝ) => X p.1 - X (Function.update p.1 l (p.2 l)) :=
    fun X hX => (hX.comp measurable_fst).sub (hX.comp hσ)
  have hmH : Measurable fun p : (Idx N → ℝ) × (Idx N → ℝ) =>
      H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2) :=
    (hH.comp (measurable_hybrid _)).sub (hH.comp (measurable_hybrid _))
  have hbH : ∀ p : (Idx N → ℝ) × (Idx N → ℝ),
      |H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)| ≤ 2 * CH := fun p => hd H CH hHb _ _
  have hct : ∀ (X : (Idx N → ℝ) → ℝ) (C : ℝ), Measurable X → (∀ x, |X x| ≤ C) →
      Measurable (crossTerm X H l) ∧ ∀ p, |crossTerm X H l p| ≤ 2 * C * (2 * CH) :=
    fun X C hX hXb => ⟨(hmX X hX).mul hmH, fun p => by
      rw [crossTerm, abs_mul]
      exact mul_le_mul (hd X C hXb _ _) (hbH p) (abs_nonneg _) (by linarith [hXb 0, abs_nonneg (X 0)])⟩
  obtain ⟨hmG', hbG'⟩ := hct G CG hG hGb
  obtain ⟨hmF', hbF'⟩ := hct F CF hF hFb
  have hI1 := abs_integral_centered_mul_le μs hF hmG' hFb hbG' hsF
    (fun x x' m z => abs_crossTerm_sub_le hsG hsH hs2G hs2H hdG hdH x x' l m z)
  have hI2 := abs_integral_centered_mul_le μs hG hmF' hGb hbF' hsG
    (fun x x' m z => abs_crossTerm_sub_le hsF hsH hs2F hs2H hdF hdH x x' l m z)
  have hsF0 : 0 ≤ sF l := by simpa using hsF 0 l ((0 : Idx N → ℝ) l)
  have hsG0 : 0 ≤ sG l := by simpa using hsG 0 l ((0 : Idx N → ℝ) l)
  have h3b : ∀ p : (Idx N → ℝ) × (Idx N → ℝ), ‖(F p.1 - F (Function.update p.1 l (p.2 l))) *
      (G p.1 - G (Function.update p.1 l (p.2 l))) *
      (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))‖ ≤ sF l * sG l * sH l := fun p => by
    rw [Real.norm_eq_abs, abs_mul, abs_mul, hybrid_succ]
    exact mul_le_mul (mul_le_mul (hsF _ _ _) (hsG _ _ _) (abs_nonneg _) hsF0) (hsH _ _ _)
      (abs_nonneg _) (mul_nonneg hsF0 hsG0)
  have hI3 : |∫ p, (F p.1 - F (Function.update p.1 l (p.2 l))) *
      (G p.1 - G (Function.update p.1 l (p.2 l))) *
      (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)) ∂ν| ≤ sF l * sG l * sH l := by
    simpa [Real.norm_eq_abs] using norm_integral_le_of_norm_le_const (μ := ν) (ae_of_all _ h3b)
  have hint1 : Integrable (fun p : (Idx N → ℝ) × (Idx N → ℝ) => (F p.1 - cF) * crossTerm G H l p) ν :=
    integrable_of_abs_le (((hF.comp measurable_fst).sub measurable_const).mul hmG') fun p => by
      rw [abs_mul]
      exact mul_le_mul ((abs_sub _ _).trans (add_le_add (hFb _) hcF)) (hbG' p) (abs_nonneg _)
        (by linarith)
  have hint2 : Integrable (fun p : (Idx N → ℝ) × (Idx N → ℝ) => (G p.1 - cG) * crossTerm F H l p) ν :=
    integrable_of_abs_le (((hG.comp measurable_fst).sub measurable_const).mul hmF') fun p => by
      rw [abs_mul]
      exact mul_le_mul ((abs_sub _ _).trans (add_le_add (hGb _) hcG)) (hbF' p) (abs_nonneg _)
        (by linarith)
  have hint3 : Integrable (fun p : (Idx N → ℝ) × (Idx N → ℝ) =>
      (F p.1 - F (Function.update p.1 l (p.2 l))) * (G p.1 - G (Function.update p.1 l (p.2 l))) *
      (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2))) ν :=
    integrable_of_abs_le (((hmX F hF).mul (hmX G hG)).mul hmH) fun p => by
      simpa [Real.norm_eq_abs] using h3b p
  have hsplit : ∀ p : (Idx N → ℝ) × (Idx N → ℝ),
      ((F p.1 - cF) * (G p.1 - cG) - (F (Function.update p.1 l (p.2 l)) - cF) *
        (G (Function.update p.1 l (p.2 l)) - cG)) *
        (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)) =
      (F p.1 - cF) * crossTerm G H l p + (G p.1 - cG) * crossTerm F H l p -
        (F p.1 - F (Function.update p.1 l (p.2 l))) * (G p.1 - G (Function.update p.1 l (p.2 l))) *
          (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)) := fun p => by
    simp only [crossTerm]
    ring
  simp_rw [hsplit]
  rw [integral_sub (f := fun p : (Idx N → ℝ) × (Idx N → ℝ) =>
      (F p.1 - cF) * crossTerm G H l p + (G p.1 - cG) * crossTerm F H l p) (hint1.add hint2) hint3,
    integral_add hint1 hint2]
  calc _ ≤ |∫ p, (F p.1 - cF) * crossTerm G H l p ∂ν| + |∫ p, (G p.1 - cG) * crossTerm F H l p ∂ν| +
        |∫ p, (F p.1 - F (Function.update p.1 l (p.2 l))) *
          (G p.1 - G (Function.update p.1 l (p.2 l))) *
          (H (hybrid l p.1 p.2) - H (hybrid ((l : ℕ) + 1) p.1 p.2)) ∂ν| :=
        (abs_sub _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
    _ ≤ _ := add_le_add (add_le_add hI1 hI2) hI3

/-- The hypotheses of `abs_swap_term_le` are satisfiable: constant observables. -/
example : ∀ (x : Idx 2 → ℝ) (l m : Idx 2) (z z' : ℝ), m ≠ l →
    |(fun _ : Idx 2 → ℝ => (0 : ℝ)) x - (fun _ : Idx 2 → ℝ => (0 : ℝ)) (Function.update x l z) -
      (fun _ : Idx 2 → ℝ => (0 : ℝ)) (Function.update x m z') +
      (fun _ : Idx 2 → ℝ => (0 : ℝ)) (Function.update (Function.update x l z) m z')| ≤
      (fun _ _ : Idx 2 => (0 : ℝ)) l m := fun _ _ _ _ _ _ => by simp

end Kinetic
end Transformer
