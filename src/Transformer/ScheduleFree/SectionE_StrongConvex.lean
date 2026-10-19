module

public import Transformer.ScheduleFree.SectionE_Conversion
public import Transformer.ScheduleFree.SectionC_ExpectationTransfer

/-!
# The stochastic strongly convex rate

arXiv:2405.15682v4, Appendix E, Theorem 6, last inequality.
We spell out `μ>0`, an initial point in the projection ball, and the
oracle moments and integrability needed to take expectations. Unlike
the general theorem's printed random-weight formulation, weights are
the deterministic `w_t=t` prescribed by this particular algorithm.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory

noncomputable section

namespace Transformer.ScheduleFree

variable {E Ω : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- Appendix E, Theorem 6: the stated `2(G+2μD)²/(μ(T+1))` rate for
the actual projected stochastic algorithm. The comparator may be a minimizer. -/
theorem strong_schedule_free {F : E → ℝ} {grad : E → E} {μ D G : ℝ}
    (hμ : 0 < μ) (hD : 0 ≤ D) (hG : 0 ≤ G) (hstrong : StrongLower F grad μ)
    (α : Ω → ℕ → ℝ) (hα : ∀ ω t, 0 ≤ α ω t ∧ α ω t ≤ 1)
    (z g : ℕ → Ω → E) (u : E) (hu : ‖u‖ ≤ D) (hinit : ∀ ω, ‖z 0 ω‖ ≤ D)
    (hstep : ∀ ω t, z (t + 1) ω = projectBall D
      (z t ω - (2 / (μ * ((t : ℝ) + 2))) • (g t ω + μ •
        (z t ω - query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) t))))
    (hbound : ∀ ω t, ‖g t ω‖ ≤ G)
    (hunbiased : UnbiasedPairings P grad
      (fun t ω => query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) t) z g)
    (T : ℕ) (hT : 0 < T)
    (hloss : Integrable (fun ω =>
      F (average (fun k => (k : ℝ) + 1) (fun k => z k ω) T) - F u) P)
    (htrue : ∀ t < T, Integrable (fun ω => inner ℝ
      (grad (query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) t)) (z t ω - u)) P)
    (hobs : ∀ t < T, Integrable (fun ω => inner ℝ (g t ω) (z t ω - u)) P)
    (hquad : ∀ t < T, Integrable (fun ω =>
      ‖u - query (fun k => (k : ℝ) + 1) (α ω) (fun k => z k ω) t‖ ^ 2) P) :
    (∫ ω, F (average (fun k => (k : ℝ) + 1) (fun k => z k ω) T) - F u ∂P) ≤
      2 * (G + 2 * μ * D) ^ 2 / (μ * ((T : ℝ) + 1)) := by
  let w := fun t : ℕ => (t : ℝ) + 1
  let y := fun ω t => query w (α ω) (fun k => z k ω) t
  let C := G + 2 * μ * D
  have hw (t : ℕ) : 0 ≤ w t := by dsimp [w]; positivity
  have hw0 : 0 < w 0 := by norm_num [w]
  have hz (ω : Ω) (t : ℕ) : ‖z t ω‖ ≤ D := by
    cases t with
    | zero => exact hinit ω
    | succ t => rw [hstep]; exact norm_projectBall hD _
  have hy (ω : Ω) (t : ℕ) : ‖y ω t‖ ≤ D :=
    mix_norm (hα ω t).1 (hα ω t).2 _ _
      (average_norm hw hw0 hD _ (hz ω) _) (hz ω t)
  have hCb (ω : Ω) (t : ℕ) : ‖g t ω + μ • (z t ω - y ω t)‖ ≤ C :=
    surrogate_gradient_bound hμ.le _ _ _ (hz ω t) (hy ω t) (hbound ω t)
  have hpath (ω : Ω) := strong_conversion hμ.le hstrong hw hw0 (hα ω) (fun t => z t ω) u T
  have hupper (ω : Ω) : (∑ t ∈ Finset.range T, w t *
      (inner ℝ (g t ω) (z t ω - u) - μ / 2 * ‖u - y ω t‖ ^ 2)) ≤ (T : ℝ) * C ^ 2 / μ := by
    apply (Finset.sum_le_sum (g := fun t => w t *
      (surrogate μ (y ω t) (g t ω) (z t ω) - surrogate μ (y ω t) (g t ω) u)) ?_).trans
      (strong_projected_regret hμ hD (by dsimp [C]; positivity) _ _ _ u hu (hstep ω) (hCb ω) T)
    intro t ht
    have hn := mul_nonneg (hw t) (show 0 ≤ μ / 2 * ‖z t ω - y ω t‖ ^ 2 by positivity)
    simp only [surrogate, inner_sub_right]
    nlinarith
  have hc := expectation_transfer w (weightSum w T) ((T : ℝ) * C ^ 2 / μ) T
    (fun ω => F (average w (fun k => z k ω) T) - F u)
    (fun t ω => inner ℝ (grad (y ω t)) (z t ω - u))
    (fun t ω => inner ℝ (g t ω) (z t ω - u))
    (fun t ω => μ / 2 * ‖u - y ω t‖ ^ 2)
    hpath hupper (fun t _ => hunbiased t u) hloss htrue hobs
    (fun t ht => (hquad t ht).const_mul (μ / 2))
  have ht : (0 : ℝ) < T := by exact_mod_cast hT
  apply (le_div_iff₀ (by positivity : 0 < μ * ((T : ℝ) + 1))).mpr
  have hh := mul_le_mul_of_nonneg_left hc (by positivity : 0 ≤ 2 * μ / (T : ℝ))
  rw [weightSum_linear] at hh
  have hleft (I : ℝ) : (2 * μ / (T : ℝ)) * (((T : ℝ) * ((T : ℝ) + 1) / 2) * I) =
      I * (μ * ((T : ℝ) + 1)) := by field_simp
  have hright : (2 * μ / (T : ℝ)) * ((T : ℝ) * C ^ 2 / μ) = 2 * C ^ 2 := by field_simp
  rw [hleft, hright] at hh
  exact hh

example (T : ℕ) (hT : 0 < T) :
    (∫ _ω : Ω, (1 : ℝ) / 2 * ‖average (fun k => (k : ℝ) + 1) (fun _ => (0 : ℝ)) T‖ ^ 2 -
      1 / 2 * ‖(0 : ℝ)‖ ^ 2 ∂P) ≤ 2 * (1 + 2 * 1 * 1) ^ 2 / (1 * ((T : ℝ) + 1)) :=
  strong_schedule_free (F := fun x : ℝ => 1 / 2 * ‖x‖ ^ 2) (grad := fun x => (1 : ℝ) • x)
    zero_lt_one zero_le_one zero_le_one (quadratic_strong 1) (fun _ _ => 0)
    (by intro ω t; exact ⟨le_rfl, zero_le_one⟩) (fun _ _ => 0) (fun _ _ => 0) 0 (by simp)
    (by intro ω; simp) (by intro ω t; simp [projectBall, query, average, mix])
    (by intro ω t; simp) (by intro t u; simp [query, average, mix]) T hT
    (by simp [average]) (by intro t ht; simp [query, average, mix]) (by intro t ht; simp)
    (by intro t ht; simp [query, average, mix])

end Transformer.ScheduleFree
