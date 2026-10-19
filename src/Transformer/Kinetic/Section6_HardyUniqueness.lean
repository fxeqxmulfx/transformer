module

public import Transformer.Kinetic.Section6_HardyRegularity
public import Transformer.Kinetic.Section6_HardyContraction

/-!
# Kinetic theory for Transformers — uniqueness of the original classical profile

The source's uniqueness claim in arXiv:2605.09213v1, §6, `prop:Bessel`,
is for the classical Volterra solution. We state explicitly the two
regularity conditions its spatial integrals and finite time rectangles
use: integrability of the original Volterra integrand and a uniform
finite bound on each actual time rectangle. The earlier bare equation
predicate did not include these conditions. They are proved for the
original Bessel formula in `Section6_HardyRegularity`.

We subtract the actual source equations, using linearity of their actual
integrals, then apply the proved homogeneous short-interval argument
iterated to all nonnegative time. No norm bound or solution identity is
put into the equation predicate or assumed of the Bessel formula.
-/

@[expose] public section

open Real Set MeasureTheory

namespace Transformer
namespace Kinetic

/-- The actual homogeneous Volterra field is linear on integrable
original integrands. Source: arXiv:2605.09213v1, §6, subtracting `eq:Hardy-hom`. -/
theorem homogeneousHardyField_sub (lam a s₀ s : ℝ) (f g : ℝ → ℝ)
    (hf : IntervalIntegrable (fun r => graphon lam s r * f r) volume s₀ s)
    (hg : IntervalIntegrable (fun r => graphon lam s r * g r) volume s₀ s) :
    homogeneousHardyField lam a s₀ (fun r => f r - g r) s =
      homogeneousHardyField lam a s₀ f s - homogeneousHardyField lam a s₀ g s := by
  unfold homogeneousHardyField
  simp only [mul_sub]
  rw [intervalIntegral.integral_sub hf hg]
  ring

/-- The difference of actual source solutions satisfies the actual
homogeneous time equation, with the original Volterra field.
Source: arXiv:2605.09213v1, §6, uniqueness in `prop:Bessel`. -/
theorem hasDerivWithinAt_hardy_difference (lam a s₀ : ℝ) (f g : ℝ → ℝ → ℝ)
    (hf : IsVolterraHardySolution lam a s₀ f) (hg : IsVolterraHardySolution lam a s₀ g)
    {t s : ℝ} (ht : 0 ≤ t) (hs : s ∈ Ioc s₀ 1)
    (hIf : IntervalIntegrable (fun r => graphon lam s r * f t r) volume s₀ s)
    (hIg : IntervalIntegrable (fun r => graphon lam s r * g t r) volume s₀ s) :
    HasDerivWithinAt (fun t => f t s - g t s)
      (homogeneousHardyField lam a s₀ (fun r => f t r - g t r) s) (Ici 0) t := by
  have hd := (hf.2 s hs t ht).fun_sub (hg.2 s hs t ht)
  apply hd.congr_deriv
  rw [homogeneousHardyField_sub lam a s₀ s (f t) (g t) hIf hIg]
  unfold homogeneousHardyField
  ring

/-- The actual difference of functions bounded on finite original time
rectangles is bounded there as well.
Source: arXiv:2605.09213v1, §6, the classical difference in `prop:Bessel`. -/
theorem hardy_difference_bounded {s₀ : ℝ} (f g : ℝ → ℝ → ℝ)
    (hf : ∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc s₀ 1, |f t s| ≤ B)
    (hg : ∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc s₀ 1, |g t s| ≤ B) :
    ∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc s₀ 1, |f t s - g t s| ≤ B := by
  intro T hT
  obtain ⟨Bf, hBf⟩ := hf T hT
  obtain ⟨Bg, hBg⟩ := hg T hT
  exact ⟨Bf + Bg, fun t ht s hs => (abs_sub _ _).trans (add_le_add (hBf t ht s hs) (hBg t ht s hs))⟩

/-- Uniqueness for the actual classical Volterra-Hardy equation on the
original nonnegative-time and positive-position domain. The source's
spatial integrability and finite-rectangle bounds are explicit inputs.
Source: arXiv:2605.09213v1, §6, uniqueness in `prop:Bessel`, `eq:Hardy-hom`. -/
theorem volterraHardySolutions_eq (lam a : ℝ) {s₀ : ℝ} (hs₀ : 0 < s₀) (hs₁ : s₀ < 1)
    (f g : ℝ → ℝ → ℝ) (hf : IsVolterraHardySolution lam a s₀ f)
    (hg : IsVolterraHardySolution lam a s₀ g)
    (hIf : ∀ t : ℝ, 0 ≤ t → ∀ s ∈ Ioc s₀ 1,
      IntervalIntegrable (fun r => graphon lam s r * f t r) volume s₀ s)
    (hIg : ∀ t : ℝ, 0 ≤ t → ∀ s ∈ Ioc s₀ 1,
      IntervalIntegrable (fun r => graphon lam s r * g t r) volume s₀ s)
    (hBf : ∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc s₀ 1, |f t s| ≤ B)
    (hBg : ∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc s₀ 1, |g t s| ≤ B) :
    ∀ t : ℝ, 0 ≤ t → ∀ s ∈ Ioc s₀ 1, f t s = g t s := by
  have hzero : ∀ s ∈ Ioc s₀ 1, f 0 s - g 0 s = 0 := by
    intro s hs
    rw [hf.1 s hs, hg.1 s hs, sub_self]
  have heq := homogeneousHardy_eq_zero lam a hs₀ hs₁ (fun t s => f t s - g t s) hzero
    (hardy_difference_bounded f g hBf hBg)
    (fun t ht s hs => hasDerivWithinAt_hardy_difference lam a s₀ f g hf hg ht hs
      (hIf t ht s hs) (hIg t ht s hs))
  exact fun t ht s hs => sub_eq_zero.mp (heq t ht s hs)

/-- The actual nonzero Bessel profile at negative bias, positive mode
parameter and interior source position witnesses every regularity input
of the uniqueness and difference theorems above.
Source: arXiv:2605.09213v1, §1, `prop:Bessel`, and its §6 series solution. -/
example : IsVolterraHardySolution (-1) 1 (1 / 2)
      (fun t s => hardyProfile (-1) 1 t s (1 / 2)) ∧
    (∀ t : ℝ, 0 ≤ t → ∀ s ∈ Ioc (1 / 2 : ℝ) 1,
      IntervalIntegrable (fun r => graphon (-1) s r * hardyProfile (-1) 1 t r (1 / 2))
        volume (1 / 2) s) ∧
    (∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ s ∈ Ioc (1 / 2 : ℝ) 1,
      |hardyProfile (-1) 1 t s (1 / 2)| ≤ B) := by
  refine ⟨isVolterraHardySolution_hardyProfile (-1) zero_le_one one_half_pos, ?_, ?_⟩
  · intro t ht s hs
    exact intervalIntegrable_graphon_hardyProfile (-1) one_half_pos hs.1.le (by simpa using ht)
  · intro T _
    obtain ⟨B, -, hB⟩ := hardyProfile_bounded (-1) (a := 1) zero_le_one one_half_pos T
    exact ⟨B, hB⟩

/-- **Proposition (prop:Bessel).** For `a>0`, every real bias and
`0<σ₀<σ≤1`, the original Volterra-Hardy equation has the stated actual
`hardyProfile` as its unique classical solution, at nonnegative time.

**What the source says and what is changed here.** The source calls the
formula "the unique solution of `eq:Hardy-hom`" and its §6 proof uses
the spatial integral and a time Laplace transform. The earlier Lean
equation predicate required only pointwise time derivatives; its
totalized Bochner integral did not require integrability, and it
required no uniform bound on finite time rectangles. Uniqueness now
takes those two classical regularity conditions as explicit hypotheses.
They are proved for the original formula, so existence assumes neither
the formula's equation nor its regularity. The original eigenvalue,
bias, positions, time domain and Bessel formula are unchanged.

The proof uses the actual entire Goursat series and the actual spatial
primitive to verify `eq:Hardy-hom`, including the one-sided initial-time
derivative. Uniqueness follows by subtracting the actual integrable
equations and iterating the proved finite-time contraction argument.
The causal endpoint values stay in the original graphon and profile;
the spatial integral identities use equality on the open interval.

Source: arXiv:2605.09213v1, §1, `prop:Bessel`, `eq:Bessel-full`,
`eq:Bessel-output`, `eq:Hardy-hom`, and its proof in §6. -/
theorem hardy_bessel (lam a σ₀ : ℝ) (ha : 0 < a) (hσ₀ : 0 < σ₀) (hσ₁ : σ₀ < 1) :
    IsVolterraHardySolution lam a σ₀ (fun t σ => hardyProfile lam a t σ σ₀) ∧
      ∀ g : ℝ → ℝ → ℝ, IsVolterraHardySolution lam a σ₀ g →
        (∀ t : ℝ, 0 ≤ t → ∀ σ ∈ Ioc σ₀ 1,
          IntervalIntegrable (fun σ' => graphon lam σ σ' * g t σ') volume σ₀ σ) →
        (∀ T : ℝ, 0 ≤ T → ∃ B : ℝ, ∀ t ∈ Icc (0 : ℝ) T, ∀ σ ∈ Ioc σ₀ 1, |g t σ| ≤ B) →
        ∀ t ∈ Ici (0 : ℝ), ∀ σ ∈ Ioc σ₀ 1, g t σ = hardyProfile lam a t σ σ₀ := by
  have hsol := isVolterraHardySolution_hardyProfile lam ha.le hσ₀
  refine ⟨hsol, ?_⟩
  intro g hg hIg hBg
  apply volterraHardySolutions_eq lam a hσ₀ hσ₁ g (fun t σ => hardyProfile lam a t σ σ₀)
    hg hsol hIg
  · intro t ht σ hσ
    exact intervalIntegrable_graphon_hardyProfile lam hσ₀ hσ.1.le (mul_nonneg ha.le ht)
  · exact hBg
  · intro T _
    obtain ⟨B, -, hB⟩ := hardyProfile_bounded lam ha.le hσ₀ T
    exact ⟨B, hB⟩

/-- The original positive eigenvalue and interior source assumptions
are met at `a=1`, `σ₀=1/2`; the preceding actual Bessel-profile example
witnesses all regularity hypotheses in the uniqueness clause as well.
Source: arXiv:2605.09213v1, §1, `prop:Bessel`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧ (1 : ℝ) / 2 < 1 := by norm_num

end Kinetic
end Transformer
