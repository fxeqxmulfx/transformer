module

public import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
public import Transformer.Landscape.SaddleAvoidance.SublevelConvergence
public import Transformer.Grokking.FlatSpectrum.Training.Convergent

/-!
# Small steps on a sublevel set of the weight-decayed loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD9.
Let the loss `L` be real-analytic in the entries and bounded below, let `G` be a continuous
field, and let `λ > 0`. No bound on the gradient and no convexity is assumed here. On the
factor space the sublevel set `{F ≤ c}` of the weight-decayed loss is compact
(`isCompact_setOf_flatLoss_le`), as `F ≤ c` bounds `λ ‖W‖² / 2`. The theorems of
`Transformer.Landscape.SaddleAvoidance` for an analytic function with a compact sublevel set
then give a step `η₀ > 0` such that for every `0 < η < η₀` (`exists_pos_forall_sublevel`)

* gradient descent maps `{F ≤ c}` into itself;
* every run from `{F ≤ c}` converges to a critical point in it;
* the starts in `{F ≤ c}` whose run converges to a strict saddle form a null set;
* every step matrix `1 − η M` in `{F ≤ c}` is positive definite, as `G` is bounded there.

The bounds behind `η₀` come from compactness and are not explicit.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology MeasureTheory Set
open scoped ComplexOrder
open Transformer.Landscape.SaddleAvoidance (gradientStep IsStrictSaddle)

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] {L : Matrix m n ℂ → ℝ}
  {G : Matrix m n ℂ → Matrix m n ℂ} {lam l₀ : ℝ}

/-- The sublevel sets of the weight-decayed loss on the factor space are compact when the loss
is bounded below and `λ > 0`: `F x ≤ c` gives `‖x‖² ≤ 2 (c − l₀) / λ`. -/
lemma isCompact_setOf_flatLoss_le
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) (c : ℝ) :
    IsCompact {x : EuclideanSpace ℂ ((m ⊕ n) × k) | flatLoss L lam x ≤ c} := by
  have hcont : Continuous (flatLoss (k := k) L lam) :=
    continuous_iff_continuousAt.mpr fun x => (analyticAt_flatLoss hL lam x).continuousAt
  refine (isCompact_closedBall (0 : EuclideanSpace ℂ ((m ⊕ n) × k))
    √(2 * (c - l₀) / lam)).of_isClosed_subset (isClosed_le hcont continuous_const) fun x hx => ?_
  have hx' : flatLoss L lam x ≤ c := hx
  rw [mem_closedBall_zero_iff]
  refine Real.le_sqrt_of_sq_le ?_
  rw [le_div_iff₀ hlam]
  have h := hlow ((factors x).1 * (factors x).2ᴴ)
  rw [flatLoss, decayedLoss, ← norm_sq_eq_frobeniusSq_factors] at hx'
  nlinarith

variable [DecidableEq m] [DecidableEq n]

/-- **Small steps on a sublevel set** (`review/flat_spectrum.md`, item WD9): for a loss that
is real-analytic in the entries and bounded below, a continuous field `G`, a weight decay
`λ > 0` and a level `c`, some `η₀ > 0` has, for every `0 < η < η₀`: gradient descent on the
factor space maps `{F ≤ c}` into itself, the starts in `{F ≤ c}` whose run converges to a
strict saddle form a null set, every run from `{F ≤ c}` converges to a critical point in
`{F ≤ c}`, and every step matrix `1 − η M` in `{F ≤ c}` is positive definite. -/
theorem exists_pos_forall_sublevel (hG : Continuous G)
    (hL : ∀ y : EuclideanSpace ℂ (m × n),
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => L (of fun i j => y (i, j))) y)
    (hlam : 0 < lam) (hlow : ∀ X, l₀ ≤ L X) (c : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ →
      MapsTo (gradientStep (flatLoss L lam) η)
        {x : EuclideanSpace ℂ ((m ⊕ n) × k) | flatLoss L lam x ≤ c}
        {x | flatLoss L lam x ≤ c} ∧
      volume ({x : EuclideanSpace ℂ ((m ⊕ n) × k) | flatLoss L lam x ≤ c} ∩
        {x | ∃ z, IsStrictSaddle (flatLoss L lam) z ∧
          Tendsto (fun t => (gradientStep (flatLoss L lam) η)^[t] x) atTop (𝓝 z)}) = 0 ∧
      (∀ x : EuclideanSpace ℂ ((m ⊕ n) × k), flatLoss L lam x ≤ c →
        ∃ w, flatLoss L lam w ≤ c ∧ gradient (flatLoss L lam) w = 0 ∧
          Tendsto (fun t => (gradientStep (flatLoss L lam) η)^[t] x) atTop (𝓝 w)) ∧
      ∀ x : EuclideanSpace ℂ ((m ⊕ n) × k), flatLoss L lam x ≤ c →
        (1 - η • lift lam (G ((factors x).1 * (factors x).2ᴴ))).PosDef := by
  have hf : AnalyticOnNhd ℝ (flatLoss (k := k) L lam) univ :=
    fun x _ => analyticAt_flatLoss hL lam x
  have hK := isCompact_setOf_flatLoss_le (k := k) hL hlam hlow c
  obtain ⟨η₁, hη₁, h1⟩ :=
    Landscape.SaddleAvoidance.exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero volume
      hf.contDiff hK
  obtain ⟨η₂, hη₂, h2⟩ := Landscape.SaddleAvoidance.exists_pos_tendsto_of_mapsTo hf.contDiff hK
    fun z _ _ => Landscape.SaddleAvoidance.exists_isLojasiewiczAt_of_analyticOnNhd hf z
  have hcG : Continuous fun x : EuclideanSpace ℂ ((m ⊕ n) × k) =>
      frobeniusSq (G ((factors x).1 * (factors x).2ᴴ)) :=
    continuous_frobeniusSq.comp (hG.comp (continuous_mul_conjTranspose.comp continuous_factors))
  obtain ⟨C, hC⟩ := hK.exists_bound_of_continuousOn hcG.continuousOn
  have hC0 := abs_nonneg C
  refine ⟨min (min η₁ η₂) (1 / (lam + (|C| + 1) + 1)),
    lt_min (lt_min hη₁ hη₂) (by positivity), fun η hη hlt => ?_⟩
  have hlt1 : η < η₁ := hlt.trans_le ((min_le_left _ _).trans (min_le_left _ _))
  have hlt2 : η < η₂ := hlt.trans_le ((min_le_left _ _).trans (min_le_right _ _))
  have hlt3 := hlt.trans_le (min_le_right _ _)
  rw [lt_div_iff₀ (by positivity)] at hlt3
  obtain ⟨hmap, hnull⟩ := h1 η hη hlt1
  refine ⟨hmap, hnull, fun x hx => h2 η hη hlt2 hmap x hx, fun x hx => ?_⟩
  have hCx : frobeniusSq (G ((factors x).1 * (factors x).2ᴴ)) ≤ (|C| + 1) ^ 2 := by
    have h := hC x hx
    rw [Real.norm_eq_abs] at h
    nlinarith [le_abs_self (frobeniusSq (G ((factors x).1 * (factors x).2ᴴ))), le_abs_self C]
  refine one_sub_smul_lift_posDef hη.le (by positivity) hCx ?_
  rw [abs_of_pos hlam]
  nlinarith

/-- The squared loss of the `1 × 1` kernel `1` at width one, `λ = 1`, below the level `1`. -/
example := exists_pos_forall_sublevel (k := Fin 1)
  (continuous_mask_sub (Set.univ : Set (Fin 1 × Fin 1)) 1) (analyticAt_fitLoss Set.univ 1)
  one_pos (fitLoss_nonneg Set.univ 1) 1

example := isCompact_setOf_flatLoss_le (k := Fin 1)
  (analyticAt_fitLoss (Set.univ : Set (Fin 1 × Fin 1)) 1) one_pos (fitLoss_nonneg Set.univ 1) 1

end Transformer.Grokking.FlatSpectrum.Training
