/-
# Bounded-difference concentration for finite product laws

The scalar concentration input for the vector Hoeffding step of Appendix B,
proof of Theorem 4.2 of arXiv:2510.22026v2. Iterating Hoeffding's lemma over
the product law avoids a separate martingale construction.
-/

import Transformer.Normalization.InitialProduct
import Mathlib.Probability.Moments.SubGaussian

open MeasureTheory Set

namespace Transformer.Normalization

variable {X : Type*} [TopologicalSpace X] [CompactSpace X] [Nonempty X]
  [MeasurableSpace X] [BorelSpace X] [FirstCountableTopology X]
  [LocallyCompactSpace X] [SecondCountableTopology X]

/-- Changing one initialization coordinate changes the scalar statistic by
at most `c`. Source: arXiv:2510.22026v2, Appendix B, vector Hoeffding step. -/
def CoordinateOscillation {n : ℕ} (f : (Fin n → X) → ℝ) (c : ℝ) : Prop :=
  ∀ x i a, |f (Function.update x i a) - f x| ≤ c

/-- Iterated scalar Hoeffding bounds the centered moment-generating
function by `exp(n c² t²/2)`. Source: arXiv:2510.22026v2, Appendix B,
the concentration input for the vector sum in the proof of Theorem 4.2. -/
theorem mgf_centered_pi_le (μ : Measure X) [IsProbabilityMeasure μ] (c : ℝ) (hc : 0 ≤ c) :
    ∀ (n : ℕ) (f : (Fin n → X) → ℝ), Continuous f → CoordinateOscillation f c →
      ∀ t : ℝ,
      (∫ x, Real.exp (t * (f x - ∫ y, f y ∂Measure.pi (fun _ : Fin n => μ)))
        ∂Measure.pi (fun _ : Fin n => μ)) ≤ Real.exp ((n : ℝ) * c ^ 2 * t ^ 2 / 2) := by
  intro n
  induction n with
  | zero =>
    intro f _ _ t
    let z : Fin 0 → X := fun i => Fin.elim0 i
    have hfx : ∀ x, f x = f z := by
      intro x
      congr 1
      exact Subsingleton.elim x z
    simp only [hfx, integral_const, probReal_univ, smul_eq_mul, one_mul,
      sub_self, mul_zero, Real.exp_zero, Nat.cast_zero, zero_mul, zero_div, le_refl]
  | succ n ih =>
    intro f hf hosc t
    let G : X → ℝ := fun a => ∫ x, f (Fin.cons a x) ∂Measure.pi (fun _ : Fin n => μ)
    have hG : Continuous G := continuous_integral_pi_cons μ n f hf
    have hmean : (∫ x, f x ∂Measure.pi (fun _ : Fin (n + 1) => μ)) = ∫ a, G a ∂μ :=
      integral_pi_cons μ n f hf
    have hslice : ∀ a, Continuous (fun x : Fin n → X => f (Fin.cons a x)) := by
      intro a
      apply hf.comp
      apply continuous_pi
      intro i
      cases i using Fin.cases <;> fun_prop
    have hsliceOsc : ∀ a, CoordinateOscillation (fun x : Fin n → X => f (Fin.cons a x)) c := by
      intro a x i b
      simpa only [Fin.cons_update] using hosc (Fin.cons a x) i.succ b
    have hGosc : ∀ a b, |G a - G b| ≤ c := by
      intro a b
      have hpoint : ∀ x : Fin n → X, |f (Fin.cons a x) - f (Fin.cons b x)| ≤ c := by
        intro x
        have heq : Function.update (Fin.cons b x : Fin (n + 1) → X) 0 a =
            (Fin.cons a x : Fin (n + 1) → X) := by
          ext i
          cases i using Fin.cases <;> simp
        simpa only [heq] using hosc (Fin.cons b x) 0 a
      dsimp [G]
      rw [← integral_sub (Perspective.integrable_of_continuous_compact (hslice a) _)
        (Perspective.integrable_of_continuous_compact (hslice b) _)]
      have h := norm_integral_le_of_norm_le_const
        (μ := Measure.pi (fun _ : Fin n => μ))
        (f := fun x => f (Fin.cons a x) - f (Fin.cons b x)) (C := c) (ae_of_all _ fun x =>
          by simpa only [Real.norm_eq_abs] using hpoint x)
      simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using h
    let a₀ : X := Classical.choice inferInstance
    have hmem : ∀ a, G a ∈ Icc (G a₀ - c) (G a₀ + c) := by
      intro a
      exact ⟨by linarith [neg_le_of_abs_le (hGosc a a₀)],
        by linarith [le_of_abs_le (hGosc a a₀)]⟩
    have hsg := ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc hG.measurable.aemeasurable
      (ae_of_all μ hmem)
    have hcoeff : (((‖(G a₀ + c) - (G a₀ - c)‖₊ / 2) ^ 2 : NNReal) : ℝ) = c ^ 2 := by
      change (‖(G a₀ + c) - (G a₀ - c)‖ / 2) ^ 2 = c ^ 2
      rw [show (G a₀ + c) - (G a₀ - c) = 2 * c by ring,
        Real.norm_eq_abs, abs_of_nonneg (by positivity)]
      ring
    have hGmgf : (∫ a, Real.exp (t * (G a - ∫ b, G b ∂μ)) ∂μ) ≤ Real.exp (c ^ 2 * t ^ 2 / 2) := by
      simpa only [ProbabilityTheory.mgf, hcoeff] using hsg.mgf_le t
    have hinner : ∀ a,
        (∫ x, Real.exp (t * (f (Fin.cons a x) - ∫ b, G b ∂μ))
          ∂Measure.pi (fun _ : Fin n => μ)) ≤
        Real.exp ((n : ℝ) * c ^ 2 * t ^ 2 / 2) * Real.exp (t * (G a - ∫ b, G b ∂μ)) := by
      intro a
      have heq : (∫ x, Real.exp (t * (f (Fin.cons a x) - ∫ b, G b ∂μ))
          ∂Measure.pi (fun _ : Fin n => μ)) =
          Real.exp (t * (G a - ∫ b, G b ∂μ)) *
            ∫ x, Real.exp (t * (f (Fin.cons a x) - G a)) ∂Measure.pi (fun _ : Fin n => μ) := by
        rw [← integral_const_mul]
        apply integral_congr_ae
        filter_upwards with x
        rw [← Real.exp_add]
        congr 1
        ring
      rw [heq, mul_comm]
      exact mul_le_mul_of_nonneg_right (ih _ (hslice a) (hsliceOsc a) t) (Real.exp_pos _).le
    rw [hmean, integral_pi_cons μ n _ (by fun_prop)]
    apply (integral_mono
      (Perspective.integrable_of_continuous_compact
        (continuous_integral_pi_cons μ n (fun x => Real.exp (t * (f x - ∫ b, G b ∂μ))) (by fun_prop)) μ)
      (Perspective.integrable_of_continuous_compact (by fun_prop) μ) hinner).trans
    rw [integral_const_mul]
    calc Real.exp ((n : ℝ) * c ^ 2 * t ^ 2 / 2) *
          (∫ a, Real.exp (t * (G a - ∫ b, G b ∂μ)) ∂μ)
        ≤ Real.exp ((n : ℝ) * c ^ 2 * t ^ 2 / 2) * Real.exp (c ^ 2 * t ^ 2 / 2) :=
          mul_le_mul_of_nonneg_left hGmgf (Real.exp_pos _).le
      _ = Real.exp (((n + 1 : ℕ) : ℝ) * c ^ 2 * t ^ 2 / 2) := by
        rw [← Real.exp_add]
        congr 1
        push_cast
        ring

/-- Constant statistics on a one-point product satisfy continuity and the
oscillation bound with `c=1`. -/
example : (0 : ℝ) ≤ 1 ∧ Continuous (fun _ : Fin 2 → Unit => (1 : ℝ)) ∧
    CoordinateOscillation (fun _ : Fin 2 → Unit => (1 : ℝ)) 1 := by
  exact ⟨by norm_num, continuous_const, fun _ _ _ => by norm_num⟩

end Transformer.Normalization
