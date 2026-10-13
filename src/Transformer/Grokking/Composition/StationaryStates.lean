import Transformer.Grokking.Composition.Curvature
import Mathlib.Analysis.Real.Sqrt

/-!
# Exact stationary states and a finite penalty threshold

Source: Nanda et al., arXiv:2301.05217v1, appendix Further speculations
on grokking, subsections An intuitive explanation of grokking and
Hypothesis: Phase Transitions are inherent to composition. The source
proposes competition between weight size and useful multi-part circuits.

Explicit specialization: the actual bilinear binary CE plus the stated
coupled L2 penalty. For nonnegative lambda every nonzero stationary state
has aligned components; such states exist exactly for `0 < lambda < 1/2`.
Construct their amplitude from the ordinary log and square root, verifying
the actual gradient. Above the threshold only the origin is stationary.

These are equilibrium statements in two parameters, not global-minimum
classification, a thermodynamic limit, convergence from initialization,
native AdamW weight decay, or delayed held-out generalization.
-/

namespace Transformer.Grokking.Composition

/-- The absent-component state remains stationary with the size penalty.
Source: arXiv:2301.05217v1, appendix composition/competition hypothesis,
specialized to actual CE and a coupled quadratic penalty. -/
theorem penalized_origin_stationary (lam : ℝ) :
    penalizedGradient lam 0 0 = (0, 0) := by
  rw [penalizedGradient_eq]
  norm_num

/-- Classify all stationary states of the stated objective. Source:
arXiv:2301.05217v1, appendix competition/composition hypotheses; deviation:
nonnegative coupled L2 penalty of the bilinear binary-CE specialization. -/
theorem penalized_stationary_iff (lam x y : ℝ) (hl : 0 ≤ lam) :
    penalizedGradient lam x y = (0, 0) ↔
      x = y ∧ (x = 0 ∨ lam = 1 / (Real.exp (x ^ 2) + 1)) := by
  constructor
  · intro h
    rw [penalizedGradient_eq] at h
    let s := 1 / (Real.exp (x * y) + 1)
    have hp : 0 < lam + s := by
      have he := Real.exp_pos (x * y)
      dsimp [s]
      positivity
    have hx : lam * x - s * y = 0 := by
      have hc := congrArg Prod.fst h
      dsimp at hc
      dsimp [s]
      convert hc using 1
      ring
    have hy : lam * y - s * x = 0 := by
      have hc := congrArg Prod.snd h
      dsimp at hc
      dsimp [s]
      convert hc using 1
      ring
    have hprod : (lam + s) * (x - y) = 0 := by
      linear_combination hx - hy
    have hxy : x = y := by
      have hz := (mul_eq_zero.mp hprod).resolve_left (ne_of_gt hp)
      linarith
    refine ⟨hxy, ?_⟩
    have hroot : x * (lam - 1 / (Real.exp (x ^ 2) + 1)) = 0 := by
      dsimp [s] at hx
      rw [← hxy] at hx
      rw [show x * x = x ^ 2 by ring] at hx
      linear_combination hx
    rcases mul_eq_zero.mp hroot with hz | hs
    · exact Or.inl hz
    · exact Or.inr (by linarith)
  · rintro ⟨hxy, hz | hs⟩
    · subst y
      subst x
      exact penalized_origin_stationary lam
    · subst y
      rw [penalizedGradient_eq, show x * x = x ^ 2 by ring, hs]
      apply Prod.ext <;> dsimp <;> ring

example : 0 ≤ (1 / 10 : ℝ) := by norm_num

/-- A nonzero aligned equilibrium requires a strictly positive penalty
below the curvature threshold. Source: arXiv:2301.05217v1, appendix size
competition hypothesis, derived for the actual binary-CE specialization. -/
theorem nonzero_stationary_penalty (lam x : ℝ) (hx : x ≠ 0)
    (heq : lam = 1 / (Real.exp (x ^ 2) + 1)) : 0 < lam ∧ lam < 1 / 2 := by
  rw [heq]
  have he := Real.exp_pos (x ^ 2)
  have hb : 1 < Real.exp (x ^ 2) := by
    simpa using Real.exp_lt_exp.mpr (sq_pos_of_ne_zero hx)
  constructor
  · positivity
  · exact one_div_lt_one_div_of_lt (by norm_num) (by linarith)

example : (1 : ℝ) ≠ 0 ∧
    1 / (Real.exp ((1 : ℝ) ^ 2) + 1) = 1 / (Real.exp ((1 : ℝ) ^ 2) + 1) := by
  constructor
  · norm_num
  · rfl

/-- Closed-form amplitude proposed for the stated equilibrium equation.
Source: the explicit bilinear specialization of arXiv:2301.05217v1's
appendix competition hypothesis; stationarity is proved separately. -/
noncomputable def stationaryAmplitude (lam : ℝ) : ℝ :=
  Real.sqrt (Real.log ((1 - lam) / lam))

/-- The proposed amplitude is genuinely nonzero below the threshold.
Source: arXiv:2301.05217v1, appendix competition hypothesis, explicit
two-scalar coupled-penalty specialization. -/
theorem stationaryAmplitude_pos (lam : ℝ) (hl : 0 < lam) (hh : lam < 1 / 2) :
    0 < stationaryAmplitude lam := by
  have hr : 1 < (1 - lam) / lam := (lt_div_iff₀ hl).2 (by linarith)
  exact Real.sqrt_pos.mpr (Real.log_pos hr)

example : 0 < (1 / 10 : ℝ) ∧ (1 / 10 : ℝ) < 1 / 2 := by norm_num

/-- Substitute the closed form into the actual equilibrium coefficient.
Source: arXiv:2301.05217v1, appendix competition hypothesis, bilinear CE
specialization; the gradient equation is not assumed in the definition. -/
theorem stationaryAmplitude_coefficient (lam : ℝ) (hl : 0 < lam) (hh : lam < 1 / 2) :
    1 / (Real.exp (stationaryAmplitude lam ^ 2) + 1) = lam := by
  have hr : 1 < (1 - lam) / lam := (lt_div_iff₀ hl).2 (by linarith)
  have hp : 0 < (1 - lam) / lam := by linarith
  have hd : (1 - lam) / lam + 1 ≠ 0 := by positivity
  unfold stationaryAmplitude
  rw [Real.sq_sqrt (Real.log_pos hr).le, Real.exp_log hp]
  field_simp
  ring

example : 0 < (1 / 4 : ℝ) ∧ (1 / 4 : ℝ) < 1 / 2 := by norm_num

/-- Both sign-related nonzero equilibria satisfy the actual gradient.
Source: arXiv:2301.05217v1, appendix compositional competition hypothesis;
these are equilibrium witnesses, not assertions of dynamical attraction. -/
theorem stationaryAmplitude_states (lam : ℝ) (hl : 0 < lam) (hh : lam < 1 / 2) :
    penalizedGradient lam (stationaryAmplitude lam) (stationaryAmplitude lam) = (0, 0) ∧
      penalizedGradient lam (-stationaryAmplitude lam) (-stationaryAmplitude lam) = (0, 0) := by
  have hc := stationaryAmplitude_coefficient lam hl hh
  constructor
  · apply (penalized_stationary_iff _ _ _ hl.le).mpr
    exact ⟨rfl, Or.inr hc.symm⟩
  · apply (penalized_stationary_iff _ _ _ hl.le).mpr
    refine ⟨rfl, Or.inr ?_⟩
    rw [show (-stationaryAmplitude lam) ^ 2 = stationaryAmplitude lam ^ 2 by ring]
    exact hc.symm

example : 0 < (1 / 10 : ℝ) ∧ (1 / 10 : ℝ) < 1 / 2 := by norm_num

/-- Exact existence threshold for nonzero stationary states. Source:
arXiv:2301.05217v1, appendix composition/competition hypothesis; finite
two-parameter equilibria do not establish a thermodynamic transition. -/
theorem nonzero_stationary_iff (lam : ℝ) (hl : 0 ≤ lam) :
    (∃ x y : ℝ, (x, y) ≠ (0, 0) ∧ penalizedGradient lam x y = (0, 0)) ↔
      0 < lam ∧ lam < 1 / 2 := by
  constructor
  · rintro ⟨x, y, hne, hg⟩
    obtain ⟨hxy, hx | hc⟩ := (penalized_stationary_iff lam x y hl).mp hg
    · subst y
      subst x
      exact False.elim (hne rfl)
    · have hx : x ≠ 0 := by
        intro hz
        apply hne
        rw [hz, ← hxy, hz]
      exact nonzero_stationary_penalty lam x hx hc
  · rintro ⟨hp, hh⟩
    refine ⟨stationaryAmplitude lam, stationaryAmplitude lam, ?_,
      (stationaryAmplitude_states lam hp hh).1⟩
    intro h
    have hz := congrArg Prod.fst h
    have hs := stationaryAmplitude_pos lam hp hh
    dsimp at hz
    linarith

example : 0 ≤ (1 / 2 : ℝ) := by norm_num

/-- Only the origin is stationary at or above the threshold. Source:
arXiv:2301.05217v1, appendix competition hypothesis, specialized to
coupled L2 and bilinear CE; no optimizer convergence is asserted. -/
theorem penalized_stationary_above_threshold (lam x y : ℝ) (hl : 1 / 2 ≤ lam) :
    penalizedGradient lam x y = (0, 0) ↔ x = 0 ∧ y = 0 := by
  constructor
  · intro h
    obtain ⟨hxy, hx | hc⟩ := (penalized_stationary_iff lam x y (by linarith)).mp h
    · exact ⟨hx, hxy ▸ hx⟩
    · have hx : x = 0 := by
        by_contra hn
        have hp := nonzero_stationary_penalty lam x hn hc
        linarith
      exact ⟨hx, hxy ▸ hx⟩
  · rintro ⟨rfl, rfl⟩
    exact penalized_origin_stationary lam

example : (1 / 2 : ℝ) ≤ 1 := by norm_num

end Transformer.Grokking.Composition
