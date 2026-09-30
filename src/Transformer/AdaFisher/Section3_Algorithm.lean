/-
# AdaFisher and AdaFisherW iterations

arXiv:2405.16397v3, §3.2–3.3, `eq:expkronfactors`, Algorithm 1.
Algorithm 1 uses an undefined `h`, starts its bias denominator at zero,
and conflates raw and corrected moments. We use its computed gradient `g`,
keep raw momentum, and correct at the positive number of processed batches.
-/

import Transformer.AdaFisher.Section3_Efficient

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {d : ℕ}

/-- KF EMA, §3.2, `eq:expkronfactors`: `γ` weights the newly measured factor.
The repeated superscript on the RHS of the source is distinguished here. -/
def factorEMA (γ : ℝ) (old fresh : Fin d → ℝ) : Fin d → ℝ :=
  fun i => γ * fresh i + (1 - γ) * old i

/-- Raw gradient momentum from the intended first-moment update,
§3.3, Algorithm 1, with the source's undefined `h` corrected to `g`. -/
def momentum (β : ℝ) (g : ℕ → Fin d → ℝ) : ℕ → Fin d → ℝ
  | 0 => fun _ => 0
  | t + 1 => fun i => β * momentum β g t i + (1 - β) * g t i

/-- Corrected first moment after `t` processed batches, §3.3, Algorithm 1.
For an actual update `t` is positive; at zero Lean's total division gives zero. -/
def correctedMomentum (β : ℝ) (g : ℕ → Fin d → ℝ) (t : ℕ) : Fin d → ℝ :=
  fun i => momentum β g t i / (1 - β ^ t)

/-- AdaFisher/AdaFisherW parameter update, §3.3, Algorithm 1.
Weight decay `κ = 0` is AdaFisher; a nonzero `κ` gives AdaFisherW. -/
def parameterStep (η κ : ℝ) (f m θ : Fin d → ℝ) : Fin d → ℝ :=
  fun i => θ i - η * (m i / f i + κ * θ i)

/-- The iteration computes momentum from the current batch before updating
the parameters, §3.3, Algorithm 1, correcting its inconsistent time indices. -/
def adaFisherRun (η : ℕ → ℝ) (κ β : ℝ) (f g : ℕ → Fin d → ℝ)
    (initial : Fin d → ℝ) : ℕ → Fin d → ℝ
  | 0 => initial
  | t + 1 => parameterStep (η t) κ (f t) (correctedMomentum β g (t + 1))
      (adaFisherRun η κ β f g initial t)

/-- KF EMA preserves any common entrywise interval, §3.2, `eq:expkronfactors`. -/
theorem factorEMA_bounds (γ lo hi : ℝ) (old fresh : Fin d → ℝ)
    (hγ : 0 ≤ γ ∧ γ ≤ 1) (hold : ∀ i, lo ≤ old i ∧ old i ≤ hi)
    (hfresh : ∀ i, lo ≤ fresh i ∧ fresh i ≤ hi) (i : Fin d) :
    lo ≤ factorEMA γ old fresh i ∧ factorEMA γ old fresh i ≤ hi := by
  unfold factorEMA
  constructor <;> nlinarith [(hold i).1, (hold i).2, (hfresh i).1, (hfresh i).2,
    mul_nonneg hγ.1 (sub_nonneg.mpr (hfresh i).1),
    mul_nonneg (sub_nonneg.mpr hγ.2) (sub_nonneg.mpr (hold i).1),
    mul_nonneg hγ.1 (sub_nonneg.mpr (hfresh i).2),
    mul_nonneg (sub_nonneg.mpr hγ.2) (sub_nonneg.mpr (hold i).2)]

example : (0 : ℝ) ≤ 4 / 5 ∧ 4 / 5 ≤ 1 ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => 1) i ∧ (fun _ => 1) i ≤ 2) ∧
    (∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => 2) i ∧ (fun _ => 2) i ≤ 2) := by norm_num

/-- Every actual bias-correction denominator is positive for Algorithm 1's
β∈[0,1) and a positive batch count. This excludes the printed zero-time
denominator and the invalid β=1 endpoint of Proposition 3.4. -/
theorem correctedMomentum_denominator_pos (β : ℝ) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (t : ℕ) (ht : 0 < t) : 0 < 1 - β ^ t :=
  sub_pos.mpr (pow_lt_one₀ hβ0 hβ1 (Nat.ne_of_gt ht))

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℕ) < 1 := by norm_num

/-- The first bias-corrected moment is the first gradient, §3.3,
Algorithm 1. The source's first denominator `1 - β⁰` cannot do this. -/
theorem correctedMomentum_first (β : ℝ) (g : ℕ → Fin d → ℝ) (hβ : β ≠ 1) :
    correctedMomentum β g 1 = g 0 := by
  funext i
  simp only [correctedMomentum, momentum, mul_zero, zero_add, pow_one]
  have hden : 1 - β ≠ 0 := sub_ne_zero.mpr (Ne.symm hβ)
  field_simp

example : (9 / 10 : ℝ) ≠ 1 := by norm_num

/-- The bias denominator in the printed first step vanishes, Algorithm 1.
Replacing undefined `h` with a nonzero gradient still leaves a division by
zero, so the printed pseudocode needs the correction documented above. -/
theorem printed_initial_moment_denominator (β : ℝ) : (1 : ℝ) - β ^ (0 : ℕ) = 0 := by
  simp

/-- Raw momentum for constant gradients has the geometric bias factor,
§3.3, Table 1, the first-moment formula. -/
theorem momentum_constant (β : ℝ) (g : Fin d → ℝ) (t : ℕ) :
    momentum β (fun _ => g) t = fun i => (1 - β ^ t) * g i := by
  induction t with
  | zero => simp [momentum]
  | succ t ih =>
    funext i
    simp only [momentum, ih, pow_succ]
    ring

/-- Constant gradients are recovered exactly at every positive count
under Algorithm 1's allowed momentum range, §3.3, Table 1. -/
theorem correctedMomentum_constant (β : ℝ) (g : Fin d → ℝ)
    (hβ0 : 0 ≤ β) (hβ1 : β < 1) (t : ℕ) (ht : 0 < t) :
    correctedMomentum β (fun _ => g) t = g := by
  funext i
  rw [correctedMomentum, momentum_constant]
  have hden := ne_of_gt (correctedMomentum_denominator_pos β hβ0 hβ1 t ht)
  field_simp

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℕ) < 1 := by norm_num

/-- β=0 makes every corrected first moment the current gradient,
§3.3, Algorithm 1 and Table 1. -/
theorem correctedMomentum_zero (g : ℕ → Fin d → ℝ) (t : ℕ) :
    correctedMomentum 0 g (t + 1) = g t := by
  funext i
  simp [correctedMomentum, momentum]

/-- Closed finite-sum first moment, §3.3, Table 1. With zero-based gradients,
`t+1` processed batches correspond to the source's positive time index. -/
theorem momentum_sum (β : ℝ) (g : ℕ → Fin d → ℝ) (t : ℕ) (i : Fin d) :
    momentum β g (t + 1) i =
      (1 - β) * ∑ k ∈ Finset.range (t + 1), β ^ (t - k) * g k i := by
  induction t with
  | zero => simp [momentum]
  | succ t ih =>
    have hs : (∑ k ∈ Finset.range (t + 1), β ^ (t + 1 - k) * g k i) =
        β * ∑ k ∈ Finset.range (t + 1), β ^ (t - k) * g k i := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro k hk
      have hkt : k ≤ t := by simpa using Finset.mem_range.mp hk
      have he : t + 1 - k = (t - k) + 1 := by omega
      rw [he, pow_succ]
      ring
    change β * momentum β g (t + 1) i + (1 - β) * g (t + 1) i = _
    rw [ih, Finset.sum_range_succ (fun k => β ^ (t + 1 - k) * g k i) (t + 1), hs]
    simp only [Nat.sub_self, pow_zero, one_mul]
    ring

/-- Bias-corrected weighted gradient sum, §3.3, Table 1 and Algorithm 1. -/
theorem correctedMomentum_sum (β : ℝ) (g : ℕ → Fin d → ℝ) (t : ℕ) (i : Fin d) :
    correctedMomentum β g (t + 1) i =
      ((1 - β) * ∑ k ∈ Finset.range (t + 1), β ^ (t - k) * g k i) / (1 - β ^ (t + 1)) := by
  rw [correctedMomentum, momentum_sum]

/-- At the endpoint `β = 1`, raw momentum is forever zero,
§3.4, Proposition 3.4. This endpoint is admitted by the source's statement
but prevents convergence at any initial point with nonzero gradient. -/
theorem momentum_one (g : ℕ → Fin d → ℝ) (t : ℕ) : momentum 1 g t = 0 := by
  induction t with
  | zero => rfl
  | succ t ih =>
    funext i
    simp [momentum, ih]

/-- AdaFisherW's decay shrinks the old parameter separately from the
preconditioned gradient, §3.3, Algorithm 1. -/
theorem parameterStep_decay (η κ : ℝ) (f m θ : Fin d → ℝ) :
    parameterStep η κ f m θ = fun i => (1 - η * κ) * θ i - η * (m i / f i) := by
  funext i
  simp only [parameterStep]
  ring

/-- A zero decay and `β = 1` give a stationary run for every gradient
stream, §3.4, Proposition 3.4. The source's endpoint is a real obstruction,
independent of the particular preconditioner. -/
theorem adaFisherRun_one (η : ℕ → ℝ) (f g : ℕ → Fin d → ℝ)
    (initial : Fin d → ℝ) (t : ℕ) : adaFisherRun η 0 1 f g initial t = initial := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [adaFisherRun, ih]
    funext i
    simp [correctedMomentum, momentum_one, parameterStep]

end Transformer.AdaFisher
