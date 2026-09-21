/-
# Measure-to-measure interpolation — Mass concentration estimates

The two technical claims of arXiv:2411.04551v3 that turn a Wasserstein bound
into a statement about mass, against the Wasserstein distance `W2` of
`Interpolation/Wasserstein.lean`:

* `Claim cl: W.to.ball` — a measure `W₂`-close to a Dirac mass puts almost
  all of its mass in a small ball.  Proved, `wToBall`, with `C = 1`: the
  mass outside `B(x₀, η₃)` has to travel at least `η₃`, so
  `μ(B(x₀,η₃)ᶜ) η₃² ≤ W₂² ≤ η₂²`, and `η₂²/η₃² ≤ η₂/η₃` or the bound is
  trivial.  The paper goes through Kantorovich–Rubinstein duality for `W_1`
  instead; the statement is the same.
* `Lemma lem: mass.concentration.Q1` — a measure putting `1 - η` of its mass
  in the positive quadrant is `Cη`-close to one carried by it.  **False as
  stated**, `not_massConcentrationQ1`: in `𝕊^0` the mass `η` sitting on the
  point `-1` has to travel distance `2 ≥ 1` into the quadrant, so
  `η ≤ W₂² ≤ C²η²`, which fails for `η < 1/C²`.  The rate the paper's own
  construction gives is `√η`, and that is proved, as a corrected statement,
  in `Interpolation/MassConcentrationSqrt.lean`.

Source: arXiv:2411.04551v3, Appendix "On condition (eq: assumption.hole)",
`lem: mass.concentration.Q1`; Appendix, `cl: W.to.ball`.
-/

import Transformer.Interpolation.Basic
import Transformer.Interpolation.Wasserstein
import Transformer.Wasserstein.LowerBound

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Interpolation

/-- **Claim (cl: W.to.ball).**

If `μ ∈ 𝒫(𝕊^{d-1})` and `x_0 ∈ 𝕊^{d-1}` satisfy `W_2(μ, δ_{x_0}) ≤ η_2`, then

  `1 - μ(B(x_0, η_3)) ≤ C η_2 / η_3`   for every `η_3 > 0`,

with `C > 0` universal.  Proved with `C = 1` from
`Wasserstein.measureReal_mul_sq_le_W2_sq`, not through the paper's `W_1`
duality.

Source: arXiv:2411.04551v3, Appendix, `cl: W.to.ball`. -/
theorem wToBall : ∃ C : ℝ, 0 < C ∧
    ∀ (d : ℕ) (μ : Perspective.ProbSphere d) (x₀ : SSphere d) (η₂ : ℝ),
      W2 d (μ : Measure (SSphere d)) (Measure.dirac x₀) ≤ η₂ →
      ∀ η₃ : ℝ, 0 < η₃ →
        1 - ((μ : Measure (SSphere d)) (Metric.ball x₀ η₃)).toReal ≤ C * η₂ / η₃ := by
  refine ⟨1, one_pos, fun d μ x₀ η₂ hW η₃ hη₃ => ?_⟩
  have : CompactSpace (SSphere d) := Metric.sphere.compactSpace (0 : EucSpace d) 1
  set m := (μ : Measure (SSphere d)).real (Metric.ball x₀ η₃)ᶜ with hm
  have hball : MeasurableSet (Metric.ball x₀ η₃) := measurableSet_ball
  have hlow := Wasserstein.measureReal_mul_sq_le_W2_sq (μ : Measure (SSphere d))
    (Measure.dirac x₀) MeasurableSet.univ Metric.isBounded_of_compactSpace (by simp) (by simp)
    hball.compl (measurableSet_singleton x₀)
    (by simp [Measure.dirac_apply' _ (measurableSet_singleton x₀).compl])
    (δ := η₃) (fun a ha b hb => by
      rw [Set.mem_singleton_iff] at hb
      subst hb
      simpa [Metric.mem_ball, not_lt] using ha) hη₃.le
  have hW0 := W2_nonneg d (μ : Measure (SSphere d)) (Measure.dirac x₀)
  have hWsq : W2 d (μ : Measure (SSphere d)) (Measure.dirac x₀) ^ 2 ≤ η₂ ^ 2 :=
    pow_le_pow_left₀ hW0 hW 2
  have hη₂ : 0 ≤ η₂ := hW0.trans hW
  have hcompl : 1 - ((μ : Measure (SSphere d)) (Metric.ball x₀ η₃)).toReal = m := by
    rw [hm, measureReal_compl hball, probReal_univ, measureReal_def]
  rw [hcompl, one_mul]
  have hm0 : 0 ≤ m := measureReal_nonneg
  have hm1 : m ≤ 1 := by rw [hm]; exact measureReal_le_one
  have hmsq : m * η₃ ^ 2 ≤ η₂ ^ 2 := hlow.trans hWsq
  rw [le_div_iff₀ hη₃]
  rcases le_or_gt η₃ η₂ with h | h
  · nlinarith
  · nlinarith

/-- The positive quadrant is measurable: an intersection of `d` open
half-spaces. -/
theorem measurableSet_positiveQuadrant (d : ℕ) : MeasurableSet (positiveQuadrant d) := by
  have : positiveQuadrant d = ⋂ i : Fin d,
      {x : SSphere d | 0 < (EuclideanSpace.equiv _ ℝ ((x : EucSpace d))) i} := by
    ext x; simp [positiveQuadrant]
  rw [this]
  exact MeasurableSet.iInter fun i => measurableSet_lt measurable_const
    ((continuous_apply i).comp ((EuclideanSpace.equiv _ ℝ).continuous.comp
      continuous_subtype_val)).measurable

/-- The point `-1` of `𝕊^0` is not in the positive quadrant. -/
theorem antipode_notMem_positiveQuadrant :
    antipode 1 (basePoint 0) ∉ positiveQuadrant 1 := by
  intro h
  have := h 0
  simp [antipode, basePoint] at this
  linarith

/-- Every point of the positive quadrant of `𝕊^0` is at distance at least `1`
from `-1` — in fact it is `+1`, at distance `2`, but `1` is what is used. -/
theorem one_le_dist_antipode_of_mem {b : SSphere 1} (hb : b ∈ positiveQuadrant 1) :
    1 ≤ dist (antipode 1 (basePoint 0)) b := by
  have h0 := hb 0
  simp only [EuclideanSpace.equiv] at h0
  rw [Subtype.dist_eq, dist_eq_norm, EuclideanSpace.norm_eq]
  simp [antipode, basePoint]
  have h0' : 0 < (b : EucSpace 1).ofLp 0 := h0
  rw [abs_of_neg (by linarith)]
  linarith

/-- **Refutation of Lemma (lem: mass.concentration.Q1).**

The lemma: let `ν_0^i ∈ 𝒫(𝕊^{d-1})` and `η > 0`, and let
`φ_η : 𝕊^{d-1} → 𝕊^{d-1}` be Lipschitz and invertible with
`(φ_η)_# ν_0^i (ℚ_1^{d-1}) = 1 - η` for every `i`; then for every `i` there is
`μ_0^i ∈ 𝒫(ℚ_1^{d-1})` with `W_2((φ_η)_# ν_0^i, μ_0^i) ≤ C η`, `C > 0`
universal.  "`μ_0^i ∈ 𝒫(ℚ_1^{d-1})`" is read as a probability measure on the
sphere giving no mass to the complement of the quadrant.

It is false.  Take `d = 1`, `N = 1`, `φ_η = id`, and
`ν = (1 - η) δ_{+1} + η δ_{-1}`.  Every point of `ℚ_1^0 = {+1}` is at distance
`2` from `-1`, so any `μ` carried by the quadrant has `η ≤ W_2(ν, μ)² ≤ C²η²`;
`η = 1/(C² + 2)` contradicts it.  The universal `C` is quantified first, as
"universal" demands; the same example refutes any `C` independent of `η`.

What the paper needs of the lemma is only `W_2 → 0` as `η → 0`, for its
continuity argument, and its own construction delivers `2√η`: that is
`massConcentrationQ1_sqrt`.

Source: arXiv:2411.04551v3, `lem: mass.concentration.Q1`. -/
theorem not_massConcentrationQ1 : ¬ ∃ C : ℝ, 0 < C ∧
    ∀ (d N : ℕ) (ν : Idx N → Perspective.ProbSphere d) (η : ℝ), 0 < η →
      ∀ (φ : SSphere d → SSphere d) (L : NNReal), LipschitzWith L φ →
        Function.Bijective φ →
        (∀ i : Idx N,
          Measure.map φ (ν i : Measure (SSphere d)) (positiveQuadrant d)
            = ENNReal.ofReal (1 - η)) →
        ∀ i : Idx N, ∃ μ : Perspective.ProbSphere d,
          (μ : Measure (SSphere d)) (positiveQuadrant d)ᶜ = 0 ∧
          W2 d (Measure.map φ (ν i : Measure (SSphere d)))
              (μ : Measure (SSphere d)) ≤ C * η := by
  rintro ⟨C, hC, hQ1⟩
  have : CompactSpace (SSphere 1) := Metric.sphere.compactSpace (0 : EucSpace 1) 1
  set η : ℝ := 1 / (C ^ 2 + 2) with hηdef
  have hη : 0 < η := by positivity
  have hη1 : η < 1 := by rw [hηdef, div_lt_one (by positivity)]; nlinarith
  set e : SSphere 1 := basePoint 0
  set e' : SSphere 1 := antipode 1 (basePoint 0)
  let ν0 : Measure (SSphere 1) :=
    ENNReal.ofReal (1 - η) • Measure.dirac e + ENNReal.ofReal η • Measure.dirac e'
  have hQ := measurableSet_positiveQuadrant 1
  have heQ : e ∈ positiveQuadrant 1 := basePoint_mem_positiveQuadrant
  have he'Q : e' ∉ positiveQuadrant 1 := antipode_notMem_positiveQuadrant
  have hprob : IsProbabilityMeasure ν0 := by
    constructor
    simp only [ν0, Measure.add_apply, Measure.smul_apply, measure_univ, smul_eq_mul, mul_one]
    rw [← ENNReal.ofReal_add (by linarith) hη.le]
    simp
  let ν : Idx 1 → Perspective.ProbSphere 1 := fun _ => ⟨ν0, hprob⟩
  have hmass : ∀ i : Idx 1, Measure.map id (ν i : Measure (SSphere 1)) (positiveQuadrant 1)
      = ENNReal.ofReal (1 - η) := by
    intro i
    simp only [Measure.map_id, ν, ProbabilityMeasure.coe_mk, ν0, Measure.add_apply,
      Measure.smul_apply, smul_eq_mul, Measure.dirac_apply' _ hQ,
      Set.indicator_of_mem heQ, Set.indicator_of_notMem he'Q]
    simp
  obtain ⟨μ, hμQ, hW⟩ := hQ1 1 1 ν η hη id 1 LipschitzWith.id Function.bijective_id hmass 0
  simp only [Measure.map_id] at hW
  have hlow := Wasserstein.measureReal_mul_sq_le_W2_sq (ν 0 : Measure (SSphere 1))
    (μ : Measure (SSphere 1)) MeasurableSet.univ Metric.isBounded_of_compactSpace
    (by simp) (by simp) (measurableSet_singleton e') hQ hμQ (δ := 1)
    (fun a ha b hb => by
      rw [Set.mem_singleton_iff] at ha
      subst ha
      exact one_le_dist_antipode_of_mem hb) zero_le_one
  have hne : e' ≠ e := fun h => he'Q (h ▸ heQ)
  have hA : (ν 0 : Measure (SSphere 1)).real {e'} = η := by
    simp only [measureReal_def, ν, ProbabilityMeasure.coe_mk, ν0, Measure.add_apply,
      Measure.smul_apply, smul_eq_mul, Measure.dirac_apply' _ (measurableSet_singleton e'),
      Set.indicator_of_mem (Set.mem_singleton e'),
      Set.indicator_of_notMem (fun h => hne (Set.mem_singleton_iff.1 h).symm)]
    simp [hη.le]
  rw [hA, one_pow, mul_one] at hlow
  have hW0 := W2_nonneg 1 (ν 0 : Measure (SSphere 1)) (μ : Measure (SSphere 1))
  have hsq : W2 1 (ν 0 : Measure (SSphere 1)) (μ : Measure (SSphere 1)) ^ 2 ≤ (C * η) ^ 2 :=
    pow_le_pow_left₀ hW0 hW 2
  have h1 : η ≤ C ^ 2 * η ^ 2 := by nlinarith
  have h2 : 1 ≤ C ^ 2 * η := by nlinarith
  have h3 : C ^ 2 * η < 1 := by
    rw [hηdef, mul_one_div, div_lt_one (by positivity)]; linarith
  linarith

end Interpolation
end Transformer
