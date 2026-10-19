module

public import Transformer.Clusters.Section7_UnboundedParticles
public import Transformer.Landscape.GlobalFlow.Existence

/-!
# Four scalar tokens for the ReLU feed-forward dynamics

Section 12 of arXiv:2305.05465v6, equation `eq: mlp.transfo`, applies
componentwise ReLU to the rescaled attention drift. In dimension one,
with `Q = K = V = W = I₁`, the scores are
`exp(2t) z_i z_j`. Consider the four coordinates `(y, -w, 0, w)`.
The scalar drift below is their literal attention average of differences,
before applying ReLU. No equation is imposed by its definition.

The extra coordinate `w` permits an autonomous ODE with linear growth:
it will be fixed at one, while the clock has derivative `w`. The state
space is `ℝ × (ℝ × ℝ)`, with coordinates `(clock, y, w)`.
The exponential scores are smooth even though their derivatives are
not globally bounded. Their normalized weights still give a linear
bound for the attention drift.

These facts support a genuine global trajectory, which will test the
source's proposed extension of its three-hyperplane conclusion.
-/

@[expose] public section

open scoped BigOperators
open Real
namespace Transformer.Clusters

/-- Four scalar coordinates used to test §12, `eq: mlp.transfo`:
one moving coordinate and three reference coordinates. -/
def reluFourCoords (y w : ℝ) : Idx 4 → ℝ := ![y, -w, 0, w]

/-- The unactivated scalar attention drift of §12, `eq: mlp.transfo`,
at scores `b z_i z_j`, for the four coordinates `(y, -w, 0, w)`. -/
noncomputable def reluFourDrift (b y w : ℝ) (i : Idx 4) : ℝ :=
  ∑ j : Idx 4,
    Perspective.softmaxWeight
      (fun k => b * reluFourCoords y w i * reluFourCoords y w k) j *
        (reluFourCoords y w j - reluFourCoords y w i)

/-- The autonomous clock, moving token, and fixed scale for the
scalar specialization of §12, `eq: mlp.transfo`. -/
abbrev ReluClockState := ℝ × (ℝ × ℝ)

/-- The moving token's unactivated drift with the actual clock-dependent
scores of §12, `eq: mlp.transfo`, at `V = I₁`. -/
noncomputable def reluClockDrift (s : ReluClockState) : ℝ :=
  reluFourDrift (Real.exp (2 * s.1)) s.2.1 s.2.2 0

/-- The moving coordinate is exactly the zeroth coordinate.
Source: arXiv:2305.05465v6, §12, the configuration used to test
`eq: mlp.transfo`. -/
theorem reluFourCoords_zero (y w : ℝ) : reluFourCoords y w 0 = y := rfl

/-- Each coordinate is bounded by the larger coordinate scale.
Source: arXiv:2305.05465v6, §12, the four-token test configuration. -/
theorem norm_reluFourCoords_le (y w : ℝ) (i : Idx 4) :
    ‖reluFourCoords y w i‖ ≤ max ‖y‖ ‖w‖ := by
  fin_cases i
  · exact le_max_left _ _
  · simp [reluFourCoords]
  · simp [reluFourCoords]
  · exact le_max_right _ _

/-- Attention differences have at most twice the coordinate scale.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo` and the
triangle inequality for the four-token configuration. -/
theorem norm_reluFourCoords_sub_le (y w : ℝ) (i j : Idx 4) :
    ‖reluFourCoords y w j - reluFourCoords y w i‖ ≤ 2 * max ‖y‖ ‖w‖ := by
  calc
    _ ≤ ‖reluFourCoords y w j‖ + ‖reluFourCoords y w i‖ := norm_sub_le _ _
    _ ≤ max ‖y‖ ‖w‖ + max ‖y‖ ‖w‖ :=
      add_le_add (norm_reluFourCoords_le y w j) (norm_reluFourCoords_le y w i)
    _ = _ := by ring

/-- Normalization bounds the drift independently of the score magnitude.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`;
its positive normalized attention weights form a convex average. -/
theorem norm_reluFourDrift_le (b y w : ℝ) (i : Idx 4) :
    ‖reluFourDrift b y w i‖ ≤ 2 * max ‖y‖ ‖w‖ := by
  let a := fun j => Perspective.softmaxWeight
    (fun k => b * reluFourCoords y w i * reluFourCoords y w k) j
  have ha (j : Idx 4) : 0 ≤ a j := Perspective.softmaxWeight_nonneg _ j
  have hsum : ∑ j : Idx 4, a j = 1 := Perspective.sum_softmaxWeight (by norm_num) _
  calc
    _ ≤ ∑ j : Idx 4, ‖a j * (reluFourCoords y w j - reluFourCoords y w i)‖ :=
      norm_sum_le _ _
    _ = ∑ j : Idx 4, a j * ‖reluFourCoords y w j - reluFourCoords y w i‖ := by
      apply Finset.sum_congr rfl
      intro j hj
      rw [norm_mul, Real.norm_of_nonneg (ha j)]
    _ ≤ ∑ j : Idx 4, a j * (2 * max ‖y‖ ‖w‖) := by
      apply Finset.sum_le_sum
      intro j hj
      exact mul_le_mul_of_nonneg_left (norm_reluFourCoords_sub_le y w i j) (ha j)
    _ = _ := by rw [← Finset.sum_mul, hsum, one_mul]

/-- Each reference coordinate varies smoothly with the autonomous state.
Source: arXiv:2305.05465v6, §12, its four-token scalar specialization. -/
theorem contDiff_reluFourCoords (i : Idx 4) :
    ContDiff ℝ ⊤ (fun s : ReluClockState => reluFourCoords s.2.1 s.2.2 i) := by
  fin_cases i
  · change ContDiff ℝ ⊤ (fun s : ReluClockState => s.2.1)
    fun_prop
  · change ContDiff ℝ ⊤ (fun s : ReluClockState => -s.2.2)
    fun_prop
  · change ContDiff ℝ ⊤ (fun _ : ReluClockState => (0 : ℝ))
    fun_prop
  · change ContDiff ℝ ⊤ (fun s : ReluClockState => s.2.2)
    fun_prop

/-- The scores of the moving token are smooth in the autonomous state.
Source: arXiv:2305.05465v6, §12, the scores in `eq: mlp.transfo`
at identity matrices in dimension one. -/
theorem contDiff_reluFourScore (i : Idx 4) :
    ContDiff ℝ ⊤ (fun s : ReluClockState =>
      Real.exp (2 * s.1) * s.2.1 * reluFourCoords s.2.1 s.2.2 i) := by
  exact (by fun_prop : ContDiff ℝ ⊤
    (fun s : ReluClockState => Real.exp (2 * s.1) * s.2.1)).mul
      (contDiff_reluFourCoords i)

/-- The genuine unactivated moving drift is smooth.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`;
the finite exponential partition is strictly positive. -/
theorem contDiff_reluClockDrift : ContDiff ℝ ⊤ reluClockDrift := by
  have hden : ContDiff ℝ ⊤ (fun s : ReluClockState =>
      ∑ k : Idx 4, Real.exp (Real.exp (2 * s.1) * s.2.1 * reluFourCoords s.2.1 s.2.2 k)) :=
    ContDiff.sum (fun k hk => (contDiff_reluFourScore k).exp)
  have hpos (s : ReluClockState) : 0 <
      ∑ k : Idx 4, Real.exp (Real.exp (2 * s.1) * s.2.1 * reluFourCoords s.2.1 s.2.2 k) :=
    Finset.sum_pos (fun k hk => Real.exp_pos _) Finset.univ_nonempty
  unfold reluClockDrift reluFourDrift
  simp only [reluFourCoords_zero, Perspective.softmaxWeight]
  apply ContDiff.sum
  intro j hj
  exact ((contDiff_reluFourScore j).exp.fun_div hden (fun s => (hpos s).ne')).mul
    ((contDiff_reluFourCoords j).sub (by fun_prop))

/-- The actual clock-dependent moving drift has linear growth in its state.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`;
the score-independent convex-average bound controls the autonomous ODE. -/
theorem norm_reluClockDrift_le (s : ReluClockState) :
    ‖reluClockDrift s‖ ≤ 2 * ‖s‖ := by
  have hy : ‖s.2.1‖ ≤ ‖s‖ := (norm_fst_le s.2).trans (norm_snd_le s)
  have hw : ‖s.2.2‖ ≤ ‖s‖ := (norm_snd_le s.2).trans (norm_snd_le s)
  exact (norm_reluFourDrift_le _ _ _ _).trans
    (mul_le_mul_of_nonneg_left (max_le hy hw) (by norm_num))

end Transformer.Clusters
