module

public import Transformer.ScheduleFree.SectionD_Expected
public import Transformer.ScheduleFree.SectionD_OracleBounds

/-!
# The accelerated rate

arXiv:2405.15682v4, Appendix D, Corollary 2. The printed condition
`α_t≤1/(5(t-1))` does not imply the preceding Corollary 1's condition
when `w_t=t`. We use the latter, `α_t≤1/(5(t+1))`, equivalently
`10w_{1:t}α_t≤w_t`. This explicit correction is retained in the statement.

The optimistic oracle assumption is stated after expectation; the
preceding `optimistic_expectation` derives it from the paper's pathwise
regret and variance assumptions, using Jensen's inequality. No particular
optimistic algorithm is assumed to satisfy that bound automatically.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory

noncomputable section

namespace Transformer.ScheduleFree

variable {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- Population squared gradient differences, with the comparator as initial query. -/
def populationEnergy (grad : E → E) (α : Ω → ℕ → ℝ) (z : ℕ → Ω → E)
    (u : E) (t : ℕ) (ω : Ω) : ℝ :=
  ‖grad (query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) t) -
    grad (pastQuery (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) u t)‖ ^ 2

/-- Appendix D, Corollary 2, with the interpolation condition corrected as
described above. The noise term is exact, without asymptotic notation. -/
theorem accelerated_rate {F : E → ℝ} {grad : E → E} {L D : ℝ}
    (hL : 0 < L) (hD : 0 ≤ D) (hgrad : ∀ x, IsSubgradient F x (grad x))
    (hsmooth : SmoothUpper F grad L) (α : Ω → ℕ → ℝ)
    (hα : ∀ ω t, 0 ≤ α ω t ∧ α ω t ≤ 1)
    (hsmall : ∀ ω t, 10 * weightSum (fun k => (k : ℝ) + 1) (t + 1) * α ω t ≤ (t : ℝ) + 1)
    (z g : ℕ → Ω → E) (u : E) (σ : ℕ → ℝ) (T : ℕ) (hT : 0 < T)
    (hinit : ∀ ω, ‖query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) 0 - u‖ ≤ D)
    (hunbiased : UnbiasedPairings P grad
      (fun t ω => query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) t) z g)
    (hoptimistic : (∫ ω, ∑ t ∈ Finset.range T,
      ((t : ℝ) + 1) * inner ℝ (g t ω) (z t ω - u) ∂P) ≤
        D * Real.sqrt ((∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 *
          ∫ ω, populationEnergy grad α z u t ω ∂P) +
            ∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * σ t ^ 2))
    (hloss : Integrable (fun ω =>
      F (average (fun k => (k : ℝ) + 1) (fun k => z k ω) T) - F u) P)
    (htrue : ∀ t < T, Integrable (fun ω => inner ℝ
      (grad (query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) t)) (z t ω - u)) P)
    (hobs : ∀ t < T, Integrable (fun ω => inner ℝ (g t ω) (z t ω - u)) P)
    (henergy : ∀ t < T, Integrable (populationEnergy grad α z u t) P) :
    (∫ ω, F (average (fun k => (k : ℝ) + 1) (fun k => z k ω) T) - F u ∂P) ≤
      14 * D ^ 2 * L / ((T : ℝ) * ((T : ℝ) + 1)) +
        2 * D * Real.sqrt (∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * σ t ^ 2) /
          ((T : ℝ) * ((T : ℝ) + 1)) := by
  let a := fun t => ∫ ω, populationEnergy grad α z u t ω ∂P
  let N := ∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * σ t ^ 2
  have ha (t : ℕ) : 0 ≤ a t := integral_nonneg (fun ω => sq_nonneg _)
  have hfirst : a 0 ≤ L ^ 2 * D ^ 2 := by
    have hp (ω : Ω) : populationEnergy grad α z u 0 ω ≤ L ^ 2 * D ^ 2 := by
      apply (smooth_gradient_sq hL hgrad hsmooth _ u).trans
      exact mul_le_mul_of_nonneg_left
        ((sq_le_sq₀ (norm_nonneg _) hD).mpr (hinit ω)) (sq_nonneg L)
    have hi := integral_mono_ae (henergy 0 hT) (integrable_const (L ^ 2 * D ^ 2))
      (Filter.Eventually.of_forall hp)
    simpa only [integral_const, probReal_univ, one_smul] using hi
  have hc := expected_negative_bregman hL hgrad hsmooth (fun k => (k : ℝ) + 1)
    (by intro t; positivity) (by norm_num) α hα hsmall z g u T hunbiased hloss htrue hobs henergy
  have hb := acceleration_algebra hL hD
    (Finset.sum_nonneg fun t _ => mul_nonneg (sq_nonneg _) (ha t))
    (show 0 ≤ N by dsimp [N]; positivity) hfirst
    (variation_absorbs_energy hL a ha T hT) hoptimistic hc
  rw [weightSum_linear] at hb
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  rw [← add_div]
  apply (le_div_iff₀ (by positivity : 0 < (T : ℝ) * ((T : ℝ) + 1))).mpr
  dsimp [N] at hb
  nlinarith

/-- Every assumption is simultaneously satisfiable, with a zero oracle and
zero loss on an arbitrary probability space. -/
example (T : ℕ) (hT : 0 < T) :
    (∫ _ω : Ω, (0 : ℝ) - 0 ∂P) ≤
      14 * 0 ^ 2 * 1 / ((T : ℝ) * ((T : ℝ) + 1)) +
        2 * 0 * Real.sqrt (∑ t ∈ Finset.range T, ((t : ℝ) + 1) ^ 2 * 0 ^ 2) /
          ((T : ℝ) * ((T : ℝ) + 1)) :=
  accelerated_rate (F := fun _ : ℝ => 0) (grad := fun _ => 0) zero_lt_one le_rfl
    (by intro x u; simp) (by intro x y; simp only [inner_zero_left, zero_add]; positivity)
    (fun _ _ => 0) (by intro ω t; exact ⟨le_rfl, zero_le_one⟩)
    (by intro ω t; simp only [mul_zero]; positivity) (fun _ _ => 0) (fun _ _ => 0) 0 (fun _ => 0) T hT
    (by intro ω; simp [query, average, mix]) (by intro t u; rfl) (by simp)
    (by simp) (by intro t ht; exact integrable_const (inner ℝ (0 : ℝ) (0 - 0)))
    (by intro t ht; exact integrable_const (inner ℝ (0 : ℝ) (0 - 0)))
    (by intro t ht; exact integrable_const (‖(0 : ℝ) - 0‖ ^ 2))

end Transformer.ScheduleFree
