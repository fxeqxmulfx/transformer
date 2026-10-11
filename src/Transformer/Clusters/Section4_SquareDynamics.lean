import Transformer.Clusters.Section4_SquareAmplitude
import Transformer.Clusters.Section4_SquareGeometry

/-!
# The square is an actual attention trajectory

Source: arXiv:2305.05465v6, §4 `c:codim`, `eq:trans_dyn`, and `e:Rres`.
Set `Q = K = V = I₂` and use the four tokens `(±u,±u)`.
The attention numerator factors over the two coordinate signs;
the partition function is `(exp(u²) + exp(-u²))²` for every row.

Direct substitution in the weighted vector sum gives the same
scalar drift in both coordinates, with the appropriate corner sign.
Thus every global solution of the genuine scalar equation gives
a global solution of the original four-token attention system.
The previously proved change of variables transfers that solution
to the exact rescaled dynamics.

Starting with amplitude two, all four rescaled trajectories have
limits `(±q,±q)` for one `q ≥ 1`. These statements use the actual
exponential attention weights, rather than an assumed scalar model.
This supplies the dynamical part of the counterexample to the
codimension conjecture's proposed bound of three subspaces.
-/

open scoped BigOperators
open Real Filter Topology
namespace Transformer.Clusters

/-- The actual square partition function is the product of the two sign partition functions.
Source: arXiv:2305.05465v6, `eq:P`, evaluated at the four planar corners. -/
theorem squareAttention_denominator (u : ℝ) (i : Idx 4) :
    (∑ j : Idx 4, Real.exp (inner (𝕜 := ℝ) (squareCorner u i) (squareCorner u j))) =
      (Real.exp (u ^ 2) + Real.exp (-u ^ 2)) ^ 2 := by
  simp only [Fin.sum_univ_four, exp_inner_squareCorner]
  fin_cases i <;> norm_num [squareSignX, squareSignY] <;> ring

/-- All rows of the square have a strictly positive literal partition function.
Source: arXiv:2305.05465v6, `eq:P`, its positive exponential denominator. -/
theorem squareAttention_denominator_pos (u : ℝ) (i : Idx 4) :
    0 < ∑ j : Idx 4, Real.exp (inner (𝕜 := ℝ) (squareCorner u i) (squareCorner u j)) := by
  rw [squareAttention_denominator]
  positivity

/-- The actual four-corner attention weights factor over the two signs.
Source: arXiv:2305.05465v6, `eq:P`, with `Q = K = I₂`. -/
theorem attentionMatrix_square (u : ℝ) (i j : Idx 4) :
    attentionMatrix (1 : ParamMatrix 2) 1 (squareCorner u) i j =
      (Real.exp (u ^ 2 * squareSignX i * squareSignX j) *
        Real.exp (u ^ 2 * squareSignY i * squareSignY j)) /
        (Real.exp (u ^ 2) + Real.exp (-u ^ 2)) ^ 2 := by
  unfold attentionMatrix Perspective.softmaxWeight
  simp only [one_apply_eq_self]
  rw [squareAttention_denominator, exp_inner_squareCorner]

/-- The genuine attention drift of every corner is the scalar square drift in its own direction.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, with `Q = K = V = I₂`.
This verifies the scalar reduction directly from the vector attention average.
The partition denominator stays positive even at zero amplitude. -/
theorem attentionDrift_square (u : ℝ) (i : Idx 4) :
    (∑ j : Idx 4, attentionMatrix (1 : ParamMatrix 2) 1 (squareCorner u) i j • squareCorner u j) =
      squareCorner (squareScalarDrift u) i := by
  rw [Fin.sum_univ_four]
  ext k
  fin_cases k <;> fin_cases i <;>
    simp [squareCorner, squareSignX, squareSignY, squareScalarDrift, attentionMatrix_square]
  all_goals
    field_simp [ne_of_gt (squareScalar_denominator_pos u)]
    ring

/-- A genuine scalar solution produces a solution of the original vector attention system.
Source: arXiv:2305.05465v6, `eq:trans_dyn`, the verified square specialization. -/
theorem transformerDynamics_square (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) :
    TransformerDynamics (1 : ParamMatrix 2) 1 1 (fun t => squareCorner (u t)) := by
  intro t i
  simp only [one_apply_eq_self, attentionDrift_square]
  exact hasDerivAt_squareCorner (hu t) i

example : ∃ u : ℝ → ℝ, ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := by
  obtain ⟨u, _, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu⟩

/-- The derivative of each actual rescaled square token follows its normalized scalar amplitude.
Source: arXiv:2305.05465v6, `e:Rres`, the literal square solution. -/
theorem hasDerivAt_rescaled_square (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (t : ℝ) (i : Idx 4) :
    HasDerivAt (fun s => squareCorner (squareAmplitude u s) i)
      (squareCorner (Real.exp (-t) * (squareScalarDrift (u t) - u t)) i) t := by
  exact hasDerivAt_squareCorner (hasDerivAt_squareAmplitude (hu t)) i

example : ∃ u : ℝ → ℝ, ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := by
  obtain ⟨u, _, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu⟩

/-- The actual square has the same initial configuration before and after rescaling.
Source: arXiv:2305.05465v6, §3, the initial-condition statement after `e:Rres`. -/
theorem square_rescaled_initial (u : ℝ → ℝ) (h0 : u 0 = 2) :
    (fun t => squareCorner (squareAmplitude u t)) 0 = squareCorner 2 := by
  simp only [squareAmplitude, neg_zero, Real.exp_zero, one_mul, h0]

example : ∃ u : ℝ → ℝ, u 0 = 2 := by
  obtain ⟨u, h0, _⟩ := exists_squareScalar_solution 2
  exact ⟨u, h0⟩

/-- All four actual square tokens share one positive limiting amplitude.
Source: arXiv:2305.05465v6, §4 `c:codim`, the limiting configuration of its square test. -/
theorem exists_rescaled_square_limits (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2) :
    ∃ q : ℝ, 1 ≤ q ∧ ∀ i : Idx 4,
      Tendsto (fun t => squareCorner (squareAmplitude u t) i) atTop (nhds (squareCorner q i)) := by
  obtain ⟨q, hq, hlim⟩ := exists_squareAmplitude_limit u hu h0
  exact ⟨q, hq, fun i => tendsto_squareCorner hlim i⟩

example : ∃ u : ℝ → ℝ, u 0 = 2 ∧
    ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := exists_squareScalar_solution 2

/-- The literal original square and its normalized amplitude obey the paper's change of variables.
Source: arXiv:2305.05465v6, §3, the rescaling preceding `e:Rres`. -/
theorem square_rescaling (u : ℝ → ℝ) (t : ℝ) (i : Idx 4) :
    squareCorner (u t) i = expTime (1 : ParamMatrix 2) t (squareCorner (squareAmplitude u t) i) := by
  rw [expTime_one_apply, smul_squareCorner, squareAmplitude, ← mul_assoc,
    ← Real.exp_add, add_neg_cancel, Real.exp_zero, one_mul]

/-- A genuine scalar solution gives the exact rescaled vector attention dynamics.
Source: arXiv:2305.05465v6, `e:Rres`, by the proved change of variables. -/
theorem rescaledDynamics_square (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) :
    RescaledDynamics (1 : ParamMatrix 2) 1 1 (fun t => squareCorner (squareAmplitude u t)) := by
  exact (transformerDynamics_iff_rescaled 1 1 1
    (fun t => squareCorner (u t)) (fun t => squareCorner (squareAmplitude u t))
    (square_rescaling u)).mp (transformerDynamics_square u hu)

example : ∃ u : ℝ → ℝ, ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t := by
  obtain ⟨u, _, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu⟩

/-- There is an actual global rescaled four-token trajectory with four positive-amplitude limits.
Source: arXiv:2305.05465v6, §4 `c:codim`, the square test of its universal conclusion.
The trajectory starts at `(±2,±2)` and solves `e:Rres` at every real time. -/
theorem exists_rescaled_square :
    ∃ (Z : ℝ → Idx 4 → EucSpace 2) (q : ℝ),
      RescaledDynamics (1 : ParamMatrix 2) 1 1 Z ∧ Z 0 = squareCorner 2 ∧
        1 ≤ q ∧ ∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (squareCorner q i)) := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  obtain ⟨q, hq, hlim⟩ := exists_rescaled_square_limits u hu h0
  exact ⟨fun t => squareCorner (squareAmplitude u t), q, rescaledDynamics_square u hu,
    square_rescaled_initial u h0, hq, hlim⟩

end Transformer.Clusters
