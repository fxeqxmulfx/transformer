/-
# AdaFisher: corrected effective-step variation bounds

arXiv:2405.16397v3, Proposition 3.4, Appendix A.2, `eq:bound_1`.
Consecutive differences begin at the second positive time index. The
source's upper-bound equation repeats the same time in its C3 term;
here the two adjacent iterates are distinguished explicitly.
-/

import Transformer.AdaFisher.SectionA_Descent

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {d : ℕ}

/-- L1 variation between consecutive effective steps, Appendix A.2. -/
def stepVariation (e : ℕ → Fin d → ℝ) (T : ℕ) : ℝ :=
  ∑ t ∈ Finset.range T, ∑ i, |e t i - e (t + 1) i|

/-- Squared L2 variation in Appendix A.2, the C3 term of `eq:bound_1`. -/
def squaredStepVariation (e : ℕ → Fin d → ℝ) (T : ℕ) : ℝ :=
  ∑ t ∈ Finset.range T, ∑ i, (e t i - e (t + 1) i) ^ 2

/-- Monotone effective steps telescope exactly, Proposition 3.4,
Appendix A.2, the C2 derivation, with the undefined t=0 term removed. -/
theorem stepVariation_telescope (e : ℕ → Fin d → ℝ)
    (he : ∀ i, Antitone (fun t => e t i)) (T : ℕ) :
    stepVariation e T = ∑ i, (e 0 i - e T i) := by
  unfold stepVariation
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  calc
    _ = ∑ t ∈ Finset.range T, (e t i - e (t + 1) i) := by
      apply Finset.sum_congr rfl
      intro t ht
      exact abs_of_nonneg (sub_nonneg.mpr (he i (Nat.le_succ t)))
    _ = _ := Finset.sum_range_sub' _ T

example : ∀ i : Fin 1, Antitone (fun t => (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) t i) := by
  intro i a b hab
  norm_num

/-- The corrected C2 bound is `d*A`, where `A` bounds the first effective
step, Appendix A.2. For damping δ the general choice is A=η/δ, not η. -/
theorem stepVariation_bound (e : ℕ → Fin d → ℝ) (A : ℝ)
    (he : ∀ i, Antitone (fun t => e t i)) (hzero : ∀ t i, 0 ≤ e t i)
    (hfirst : ∀ i, e 0 i ≤ A) (T : ℕ) : stepVariation e T ≤ d * A := by
  rw [stepVariation_telescope e he T]
  calc
    _ ≤ ∑ i : Fin d, A := Finset.sum_le_sum fun i _ => by
      linarith [hfirst i, hzero T i]
    _ = _ := by simp

example :
    (∀ i : Fin 1, Antitone (fun t => (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) t i)) ∧
    (∀ (t : ℕ) (i : Fin 1), 0 ≤ (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) t i) ∧
    (∀ i : Fin 1, (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) 0 i ≤ 2) := by
  refine ⟨fun i a b hab => by norm_num, by norm_num, by norm_num⟩

/-- Corrected C3 bound `d*A²`, Proposition 3.4, Appendix A.2.
Nonnegative decreasing effective steps have differences in `[0,A]`. -/
theorem squaredStepVariation_bound (e : ℕ → Fin d → ℝ) (A : ℝ)
    (hA : 0 ≤ A) (he : ∀ i, Antitone (fun t => e t i))
    (hzero : ∀ t i, 0 ≤ e t i) (hfirst : ∀ i, e 0 i ≤ A) (T : ℕ) :
    squaredStepVariation e T ≤ d * A ^ 2 := by
  have hdiff (t : ℕ) (i : Fin d) : (e t i - e (t + 1) i) ^ 2 ≤ A * |e t i - e (t + 1) i| := by
    have h0 := sub_nonneg.mpr (he i (Nat.le_succ t))
    have h1 : e t i - e (t + 1) i ≤ A := by
      linarith [he i (Nat.zero_le t), hfirst i, hzero (t + 1) i]
    rw [abs_of_nonneg h0]
    nlinarith
  have hsum : squaredStepVariation e T ≤ A * stepVariation e T := by
    simp only [squaredStepVariation, stepVariation, Finset.mul_sum]
    exact Finset.sum_le_sum fun t _ => Finset.sum_le_sum fun i _ => hdiff t i
  have hb := mul_le_mul_of_nonneg_left (stepVariation_bound e A he hzero hfirst T) hA
  nlinarith

example : (0 : ℝ) ≤ 2 ∧
    (∀ i : Fin 1, Antitone (fun t => (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) t i)) ∧
    (∀ (t : ℕ) (i : Fin 1), 0 ≤ (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) t i) ∧
    (∀ i : Fin 1, (fun (_ : ℕ) (_ : Fin 1) => (1 : ℝ)) 0 i ≤ 2) := by
  refine ⟨by norm_num, fun i a b hab => by norm_num, by norm_num, by norm_num⟩

end Transformer.AdaFisher
