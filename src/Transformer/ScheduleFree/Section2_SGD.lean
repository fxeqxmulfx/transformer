module

public import Transformer.ScheduleFree.SectionA_Conversion
public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Transformer.ScheduleFree.Section2_GradientDescent

/-!
# Stochastic online-to-batch conversion

arXiv:2405.15682v4, Theorem 2 and Appendix A. The stochastic interface
is the unbiased dual-pairing identity used in that proof. Conditional
unbiasedness given the past implies this identity for predictable `z` and
query points. We state that moment condition explicitly, together with
integrability, rather than identifying a conditional expectation with an
unconditional one as in the source's `E[g_t | z_t]` notation.

Weights here are deterministic and the initial weight is positive. The
paper allows random predictable weights, but its normalized expectation
formulas require an additional condition on the random final denominator;
`SectionC_RandomWeights` gives a counterexample to the unrestricted formula.
-/

/-!
# The Schedule-Free SGD rate

arXiv:2405.15682v4, §2, Theorem 1. The theorem is formulated for an
unbiased subgradient oracle with explicit integrability. A Lipschitz
sample loss and independent fresh samples provide the bounded oracle
used in the paper. A minimizer is a permissible comparator, but the
proof in fact works for any comparator at distance `D`.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory

noncomputable section

section

namespace Transformer.ScheduleFree

variable {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace Ω] {μ : Measure Ω}

/-- Appendix A's oracle condition, expressed as testable scalar moments.
It follows from conditional unbiasedness against a predictable displacement. -/
def UnbiasedPairings (μ : Measure Ω) (grad : E → E) (y z g : ℕ → Ω → E) : Prop :=
  ∀ t u, (∫ ω, inner ℝ (grad (y t ω)) (z t ω - u) ∂μ) =
    ∫ ω, inner ℝ (g t ω) (z t ω - u) ∂μ

/-- Theorem 2, with deterministic weights: expectation of error versus regret.
The result includes nonsmooth objectives and time-varying random interpolation
coefficients. The sole oracle assumption is `UnbiasedPairings`. -/
theorem expected_online_to_batch {F : E → ℝ} {grad : E → E} {w : ℕ → ℝ}
    (hgrad : ∀ x, IsSubgradient F x (grad x)) (hw : ∀ t, 0 ≤ w t)
    (hfirst : 0 < w 0) (α : Ω → ℕ → ℝ) (hα : ∀ ω t, 0 ≤ α ω t ∧ α ω t ≤ 1)
    (z g : ℕ → Ω → E) (u : E) (T : ℕ)
    (hunbiased : UnbiasedPairings μ grad
      (fun t ω => query w (α ω) (fun k => z k ω) t) z g)
    (hloss : Integrable (fun ω => F (average w (fun t => z t ω) T) - F u) μ)
    (htrue : ∀ t < T, Integrable
      (fun ω => inner ℝ (grad (query w (α ω) (fun k => z k ω) t)) (z t ω - u)) μ)
    (hobs : ∀ t < T, Integrable (fun ω => inner ℝ (g t ω) (z t ω - u)) μ) :
    weightSum w T * (∫ ω, F (average w (fun t => z t ω) T) - F u ∂μ) ≤
      ∫ ω, ∑ t ∈ Finset.range T, w t * inner ℝ (g t ω) (z t ω - u) ∂μ := by
  have hsum := integrable_finsetSum (Finset.range T)
    (fun t ht => (htrue t (Finset.mem_range.mp ht)).const_mul (w t))
  have h := integral_mono_ae (hloss.const_mul (weightSum w T)) hsum
    (Filter.Eventually.of_forall fun ω => online_to_batch hgrad hw hfirst (hα ω)
      (fun t => z t ω) u T)
  rw [integral_const_mul] at h
  convert h using 1
  rw [integral_finsetSum (Finset.range T)
    (fun t ht => (hobs t (Finset.mem_range.mp ht)).const_mul (w t)),
    integral_finsetSum (Finset.range T)
    (fun t ht => (htrue t (Finset.mem_range.mp ht)).const_mul (w t))]
  apply Finset.sum_congr rfl
  intro t ht
  rw [integral_const_mul, integral_const_mul, hunbiased t u]

/-- All stochastic hypotheses are simultaneously satisfiable on any probability
space: the zero loss and the stationary zero oracle. -/
example [IsProbabilityMeasure μ] (T : ℕ) :
    weightSum (fun _ => 1) T * (∫ _ω : Ω, (0 : ℝ) - 0 ∂μ) ≤
      ∫ _ω : Ω, ∑ _t ∈ Finset.range T, (1 : ℝ) * inner ℝ (0 : ℝ) (0 - 0) ∂μ := by
  apply expected_online_to_batch (F := fun _ : ℝ => 0) (grad := fun _ => 0)
    (by intro x v; simp) (fun _ => zero_le_one) zero_lt_one (fun _ _ => 0)
    (by intro ω t; exact ⟨le_rfl, zero_le_one⟩) (fun _ _ => 0) (fun _ _ => 0) 0 T
  · intro t u
    rfl
  · simp
  · intro t ht
    simp
  · intro t ht
    simp

end Transformer.ScheduleFree

end

section

namespace Transformer.ScheduleFree

variable {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Theorem 1: the sharp `DG/√T` bound, uniformly over momentum choices.
Here `α=1-β` can even vary predictably. Zero initial distance and zero-gradient
cases are included using real division's zero-step convention. -/
theorem schedule_free_sgd {F : E → ℝ} {grad : E → E} {D G : ℝ} {T : ℕ}
    (hgrad : ∀ x, IsSubgradient F x (grad x)) (hD : 0 ≤ D) (hG : 0 ≤ G) (hT : 0 < T)
    (α : Ω → ℕ → ℝ) (hα : ∀ ω t, 0 ≤ α ω t ∧ α ω t ≤ 1)
    (z g : ℕ → Ω → E) (u : E) (hinit : ∀ ω, ‖z 0 ω - u‖ = D)
    (hstep : ∀ ω t, z (t + 1) ω = z t ω - (D / (G * Real.sqrt T)) • g t ω)
    (hbound : ∀ ω t, t < T → ‖g t ω‖ ≤ G)
    (hunbiased : UnbiasedPairings μ grad
      (fun t ω => query (fun _ => 1) (α ω) (fun k => z k ω) t) z g)
    (hloss : Integrable (fun ω => F (average (fun _ => 1) (fun t => z t ω) T) - F u) μ)
    (htrue : ∀ t < T, Integrable (fun ω =>
      inner ℝ (grad (query (fun _ => 1) (α ω) (fun k => z k ω) t)) (z t ω - u)) μ)
    (hobs : ∀ t < T, Integrable (fun ω => inner ℝ (g t ω) (z t ω - u)) μ) :
    (∫ ω, F (average (fun _ => 1) (fun t => z t ω) T) - F u ∂μ) ≤
      D * G / Real.sqrt T := by
  have hb := expected_online_to_batch hgrad (fun _ => zero_le_one) zero_lt_one
    α hα z g u T hunbiased hloss htrue hobs
  simp only [weightSum_one, one_mul] at hb
  have hr (ω : Ω) : regret (fun t => z t ω) (fun t => g t ω) u T ≤
      D * G * Real.sqrt T :=
    optimal_regret_nonneg hD hG hT _ _ u (hinit ω) (hstep ω) (hbound ω)
  have hsum := integrable_finsetSum (Finset.range T)
    (fun t ht => hobs t (Finset.mem_range.mp ht))
  have hi := integral_mono_ae hsum (integrable_const (D * G * Real.sqrt T))
    (Filter.Eventually.of_forall hr)
  simp only [integral_const, probReal_univ, one_smul] at hi
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  have hs : 0 < Real.sqrt T := Real.sqrt_pos.mpr ht
  apply (le_div_iff₀ hs).mpr
  have hc := hb.trans hi
  apply (mul_le_mul_iff_left₀ hs).mp
  calc
    ((∫ ω, F (average (fun _ => 1) (fun t => z t ω) T) - F u ∂μ) * Real.sqrt T) *
        Real.sqrt T = (T : ℝ) *
          (∫ ω, F (average (fun _ => 1) (fun t => z t ω) T) - F u ∂μ) := by
            rw [mul_assoc, ← sq, Real.sq_sqrt ht.le]
            ring
    _ ≤ D * G * Real.sqrt T := hc

/-- A nonempty instance with all oracle and integrability hypotheses. -/
example : (∫ _ω : Ω, (0 : ℝ) - 0 ∂μ) ≤ 1 * 1 / Real.sqrt 1 := by
  simpa only [Nat.cast_one] using schedule_free_sgd (F := fun _ : ℝ => 0) (grad := fun _ => 0)
    (D := 1) (G := 1) (T := 1) (by intro x v; simp) zero_le_one zero_le_one
    (by decide) (fun _ _ => 0) (by intro ω t; exact ⟨le_rfl, zero_le_one⟩)
    (fun _ _ => 1) (fun _ _ => 0) 0 (by intro ω; simp)
    (by intro ω t; simp) (by intro ω t ht; simp)
    (by intro t u; rfl) (by simp) (by intro t ht; simp) (by intro t ht; simp)

end Transformer.ScheduleFree

end
