module

public import Transformer.ScheduleFree.SectionD_Corollary
public import Transformer.ScheduleFree.SectionC_ExpectationTransfer

/-!
# Corollary 1 in expectation

arXiv:2405.15682v4, Appendix D. Fixed weights and unbiased scalar
pairings allow the negative population-gradient term to be retained
under expectation. The auxiliary query at time zero is the comparator;
its coefficient is zero, exactly as allowed by the paper.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory

noncomputable section

namespace Transformer.ScheduleFree

variable {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

omit [IsProbabilityMeasure P] in
/-- Appendix D, Corollary 1, with deterministic weights and explicit moments. -/
theorem expected_negative_bregman {F : E → ℝ} {grad : E → E} {L : ℝ}
    (hL : 0 < L) (hgrad : ∀ x, IsSubgradient F x (grad x))
    (hsmooth : SmoothUpper F grad L) (w : ℕ → ℝ) (hw : ∀ t, 0 ≤ w t) (hw0 : 0 < w 0)
    (α : Ω → ℕ → ℝ) (hα : ∀ ω t, 0 ≤ α ω t ∧ α ω t ≤ 1)
    (hsmall : ∀ ω t, 10 * weightSum w (t + 1) * α ω t ≤ w t)
    (z g : ℕ → Ω → E) (u : E) (T : ℕ)
    (hunbiased : UnbiasedPairings P grad
      (fun t ω => query w (α ω) (fun k => z k ω) t) z g)
    (hloss : Integrable (fun ω => F (average w (fun k => z k ω) T) - F u) P)
    (htrue : ∀ t < T, Integrable (fun ω =>
      inner ℝ (grad (query w (α ω) (fun k => z k ω) t)) (z t ω - u)) P)
    (hobs : ∀ t < T, Integrable (fun ω => inner ℝ (g t ω) (z t ω - u)) P)
    (hvar : ∀ t < T, Integrable (fun ω =>
      ‖grad (query w (α ω) (fun k => z k ω) t) -
        grad (pastQuery w (α ω) (fun k => z k ω) u t)‖ ^ 2) P) :
    weightSum w T * (∫ ω, F (average w (fun k => z k ω) T) - F u ∂P) ≤
      (∫ ω, ∑ t ∈ Finset.range T, w t * inner ℝ (g t ω) (z t ω - u) ∂P) -
        ∑ t ∈ Finset.range T, weightSum w t / (6 * L) *
          ∫ ω, ‖grad (query w (α ω) (fun k => z k ω) t) -
            grad (pastQuery w (α ω) (fun k => z k ω) u t)‖ ^ 2 ∂P := by
  let a := fun t ω => w t * inner ℝ (grad (query w (α ω) (fun k => z k ω) t)) (z t ω - u)
  let b := fun t ω => w t * inner ℝ (g t ω) (z t ω - u)
  let p := fun t ω => weightSum w t / (6 * L) *
    ‖grad (query w (α ω) (fun k => z k ω) t) -
      grad (pastQuery w (α ω) (fun k => z k ω) u t)‖ ^ 2
  have ha (t : ℕ) (ht : t < T) : Integrable (a t) P := (htrue t ht).const_mul (w t)
  have hb (t : ℕ) (ht : t < T) : Integrable (b t) P := (hobs t ht).const_mul (w t)
  have hp (t : ℕ) (ht : t < T) : Integrable (p t) P := (hvar t ht).const_mul _
  have h := expectation_penalty (fun _ => 1) (weightSum w T) T
    (fun ω => F (average w (fun k => z k ω) T) - F u) a b p
    (fun ω => by simpa only [one_mul, Finset.sum_sub_distrib] using
      negative_bregman hL hgrad hsmooth hw hw0 (hα ω) (hsmall ω) (fun t => z t ω) u T)
    (fun t ht => by dsimp [a, b]; rw [integral_const_mul, integral_const_mul, hunbiased t u])
    hloss ha hb hp
  simp only [one_mul] at h
  have hbs := integrable_finsetSum (Finset.range T) (fun t ht => hb t (Finset.mem_range.mp ht))
  have hps := integrable_finsetSum (Finset.range T) (fun t ht => hp t (Finset.mem_range.mp ht))
  have he : (∫ ω, ∑ t ∈ Finset.range T, (b t ω - p t ω) ∂P) =
      (∫ ω, ∑ t ∈ Finset.range T, b t ω ∂P) -
        ∑ t ∈ Finset.range T, ∫ ω, p t ω ∂P := by
    simp_rw [Finset.sum_sub_distrib]
    rw [integral_sub hbs hps, integral_finsetSum (Finset.range T)
      (fun t ht => hp t (Finset.mem_range.mp ht))]
  rw [he] at h
  simpa only [p, integral_const_mul] using h

example (T : ℕ) : weightSum (fun _ => 1) T * (∫ _ω : Ω, (0 : ℝ) - 0 ∂P) ≤
    (∫ _ω : Ω, ∑ _t ∈ Finset.range T, (1 : ℝ) * inner ℝ (0 : ℝ) (0 - 0) ∂P) -
      ∑ t ∈ Finset.range T, weightSum (fun _ => 1) t / (6 * 1) *
        ∫ _ω : Ω, ‖(0 : ℝ) - 0‖ ^ 2 ∂P :=
  expected_negative_bregman (F := fun _ : ℝ => 0) (grad := fun _ => 0) zero_lt_one
    (by intro a b; simp) (by intro a b; simp only [inner_zero_left, zero_add]; positivity)
    (fun _ => 1) (fun _ => zero_le_one) zero_lt_one (fun _ _ => 0)
    (by intro ω t; exact ⟨le_rfl, zero_le_one⟩) (by intro ω t; simp)
    (fun _ _ => 0) (fun _ _ => 0) 0 T (by intro t u; rfl) (by simp)
    (by intro t ht; simp) (by intro t ht; simp) (by intro t ht; simp)

end Transformer.ScheduleFree
