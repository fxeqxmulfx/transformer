import Transformer.Clusters.Section12_Feedforward
import Transformer.Clusters.Section12_ReluLimits
import Transformer.Clusters.Section7_Symmetric

/-!
# Four actual limiting tokens in the ReLU feed-forward equation

We verify the complete vector equation `eq: mlp.transfo` of §12,
arXiv:2305.05465v6, for the scalar trajectory constructed above.
Set `d = 1`, `n = 4`, and `Q = K = V = W = I₁`. The vectors
are the four scalar coordinates `(y(t), -1, 0, 1)` times the
unit vector of `ℝ¹`. Identity-matrix exponentiation gives exactly
the scalar scores `exp(2t) z_i z_j` used in the construction.

The first token's unactivated drift is nonnegative, so ReLU
leaves it unchanged. Each of the other three unactivated drifts
is nonpositive, so ReLU makes it zero. This verifies every token
equation at every real time, including negative times. The source
only asks for nonnegative times.

The proved trajectory starts at four distinct tokens and converges
to four distinct vectors `(q, -1, 0, 1)`, with `-8 ≤ q ≤ -7`.
In dimension one, codimension-one affine hyperplanes are points.
The next module uses these actual limits to refute the universal
three-hyperplane extension proposed in the source's last figure.
-/

open scoped BigOperators
open Real Filter Topology
namespace Transformer.Clusters

/-- The four vectors testing §12, `eq: mlp.transfo`,
at identity matrices in dimension one. -/
noncomputable def reluFourVectors (y : ℝ) (i : Idx 4) : EucSpace 1 :=
  reluFourCoords y 1 i • unit1

/-- The actual exponential attention scores equal the scalar scores.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`,
at `Q = K = V = I₁` and the four-token test configuration. -/
theorem attentionMatrix_reluFourVectors (t y : ℝ) (i j : Idx 4) :
    attentionMatrix (1 : ParamMatrix 1) 1
      (fun k => expTime (1 : ParamMatrix 1) t (reluFourVectors y k)) i j =
        Perspective.softmaxWeight
          (fun k => Real.exp (2 * t) * reluFourCoords y 1 i * reluFourCoords y 1 k) j := by
  unfold attentionMatrix
  simp only [one_apply_eq_self, expTime_one_apply, reluFourVectors,
    smul_smul, inner_smul_unit1]
  congr 1
  funext k
  rw [show (Real.exp t * reluFourCoords y 1 i) *
      (Real.exp t * reluFourCoords y 1 k) =
      (Real.exp t * Real.exp t) * reluFourCoords y 1 i * reluFourCoords y 1 k by ring,
    ← Real.exp_add, show t + t = 2 * t by ring]

/-- The literal vector attention difference average is the scalar drift.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`;
this checks the reduction before applying the activation. -/
theorem attentionDrift_reluFourVectors (t y : ℝ) (i : Idx 4) :
    (∑ j : Idx 4, attentionMatrix (1 : ParamMatrix 1) 1
      (fun k => expTime (1 : ParamMatrix 1) t (reluFourVectors y k)) i j •
        (reluFourVectors y j - reluFourVectors y i)) =
          reluFourDrift (Real.exp (2 * t)) y 1 i • unit1 := by
  simp_rw [attentionMatrix_reluFourVectors]
  simp only [reluFourVectors, ← sub_smul,
    smul_smul, ← Finset.sum_smul, reluFourDrift]

/-- In dimension one, componentwise ReLU is scalar ReLU times the unit vector.
Source: arXiv:2305.05465v6, §12, the activation in `eq: mlp.transfo`. -/
theorem actPointwise_relu_smul_unit1 (a : ℝ) :
    actPointwise relu (a • unit1) = max a 0 • unit1 := by
  ext k
  fin_cases k
  simp [actPointwise_apply, unit1, relu]

/-- A proved scalar trajectory solves every equation of the actual
four-token ReLU feed-forward system. Source: arXiv:2305.05465v6,
§12, `eq: mlp.transfo`, at the identity good triple in dimension one. -/
theorem mlpRescaledDynamics_reluFourVectors (y : ℝ → ℝ)
    (hy : ∀ t : ℝ, y t ≤ -7)
    (hD : ∀ t : ℝ, HasDerivAt y (reluFourDrift (Real.exp (2 * t)) (y t) 1 0) t) :
    MlpRescaledDynamics 1 relu (1 : ParamMatrix 1) 1 1
      (fun t => reluFourVectors (y t)) := by
  intro t i
  simp only [one_apply_eq_self, attentionDrift_reluFourVectors, actPointwise_relu_smul_unit1]
  have hlow : y t ≤ -6 := by linarith [hy t]
  fin_cases i
  · change HasDerivAt (fun s => y s • unit1)
      (max (reluFourDrift (Real.exp (2 * t)) (y t) 1 0) 0 • unit1) t
    rw [max_eq_left (reluFourDrift_zero_nonneg (Real.exp (2 * t)) hlow)]
    exact (hD t).smul_const unit1
  · change HasDerivAt (fun _ : ℝ => (-1 : ℝ) • unit1)
      (max (reluFourDrift (Real.exp (2 * t)) (y t) 1 1) 0 • unit1) t
    rw [max_eq_right (reluFourDrift_one_nonpos (Real.exp_pos (2 * t)) hlow), zero_smul]
    exact hasDerivAt_const t ((-1 : ℝ) • unit1)
  · change HasDerivAt (fun _ : ℝ => (0 : ℝ) • unit1)
      (max (reluFourDrift (Real.exp (2 * t)) (y t) 1 2) 0 • unit1) t
    rw [max_eq_right (reluFourDrift_two_nonpos (Real.exp (2 * t)) hlow), zero_smul]
    exact hasDerivAt_const t (0 : EucSpace 1)
  · change HasDerivAt (fun _ : ℝ => (1 : ℝ) • unit1)
      (max (reluFourDrift (Real.exp (2 * t)) (y t) 1 3) 0 • unit1) t
    rw [max_eq_right (reluFourDrift_three_nonpos (Real.exp (2 * t)) hlow), zero_smul]
    exact hasDerivAt_const t ((1 : ℝ) • unit1)

/-- The scalar solution hypotheses have a proved global witness. -/
example : ∃ y : ℝ → ℝ, (∀ t : ℝ, y t ≤ -7) ∧
    ∀ t : ℝ, HasDerivAt y (reluFourDrift (Real.exp (2 * t)) (y t) 1 0) t := by
  obtain ⟨y, h0, hy, hD⟩ := exists_reluFour_scalar
  exact ⟨y, hy, hD⟩

/-- Scalar convergence gives the four genuine vector limits.
Source: arXiv:2305.05465v6, §12, the four-token test of its
proposed extension, with the three reference coordinates fixed. -/
theorem tendsto_reluFourVectors (y : ℝ → ℝ) (q : ℝ)
    (hq : Tendsto y atTop (nhds q)) (i : Idx 4) :
    Tendsto (fun t => reluFourVectors (y t) i) atTop (nhds (reluFourVectors q i)) := by
  fin_cases i
  · exact hq.smul_const unit1
  · exact tendsto_const_nhds
  · exact tendsto_const_nhds
  · exact tendsto_const_nhds

example : Tendsto (fun _ : ℝ => (-8 : ℝ)) atTop (nhds (-8)) := tendsto_const_nhds

/-- The four vector limits are pairwise distinct.
Source: arXiv:2305.05465v6, §12, the obstruction in the
four-token test of its proposed three-hyperplane extension. -/
theorem reluFourVectors_injective {q : ℝ} (hq : q ≤ -7) :
    Function.Injective (reluFourVectors q) := by
  intro i j hij
  apply reluFourCoords_limit_injective hq
  have h := congrArg (fun x : EucSpace 1 => x 0) hij
  simpa only [reluFourVectors, coord_smul_unit1] using h

example : (-8 : ℝ) ≤ -7 := by norm_num

/-- There are exactly four limiting vectors in the test configuration.
Source: arXiv:2305.05465v6, §12, the cardinality relevant to
the proposed three-hyperplane bound. -/
theorem card_reluFourVectors_image {q : ℝ} (hq : q ≤ -7) :
    (Finset.univ.image (reluFourVectors q)).card = 4 := by
  classical
  rw [Finset.card_image_of_injective _ (reluFourVectors_injective hq)]
  simp

example : (-8 : ℝ) ≤ -7 := by norm_num

/-- The full ReLU feed-forward equation has an actual global trajectory
with four distinct limiting tokens. Source: arXiv:2305.05465v6,
§12, `eq: mlp.transfo` and the proposed three-hyperplane extension. -/
theorem exists_mlp_relu_four_limits :
    ∃ (Z : ℝ → Idx 4 → EucSpace 1) (q : ℝ),
      MlpRescaledDynamics 1 relu (1 : ParamMatrix 1) 1 1 Z ∧
        Z 0 = reluFourVectors (-8) ∧ -8 ≤ q ∧ q ≤ -7 ∧
          ∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (reluFourVectors q i)) := by
  obtain ⟨y, q, h0, hy, hD, hql, hqu, hlim⟩ := exists_reluFour_scalar_limit
  refine ⟨fun t => reluFourVectors (y t), q,
    mlpRescaledDynamics_reluFourVectors y hy hD, ?_, hql, hqu, ?_⟩
  · change reluFourVectors (y 0) = reluFourVectors (-8)
    rw [h0]
  · exact tendsto_reluFourVectors y q hlim

end Transformer.Clusters
