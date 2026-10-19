module

public import Transformer.Interpolation.Section1_FullFieldLipschitz

/-!
# Uniform spatial bounds for the complete one-atom network

ArXiv:2411.04551v3, §1, `eq: ips` and `eq: vf`,
and the norm estimate in `thm: targets.atoms`.
At a Dirac law, the normalized original attention is exactly its atom.
The score matrix therefore cancels from the complete field even at
an ambient query distinct from the atom. Its removal here is an
identity of the actual original fields, with no architecture change.

Changing the atom contributes the value norm times its displacement.
Changing the unit query contributes the projection and perceptron
spatial constants. Together they give 3K+5K² for the velocities of
two distinct actual unit Dirac laws when the parameter blocks are
bounded by K. The same bound follows from the original sum norm.
-/

@[expose] public section

open Real MeasureTheory
open scoped NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Zeroing the score matrix leaves the complete original Dirac field
unchanged at every ambient query. Source: arXiv:2411.04551v3,
§1, the exact normalization in `eq: vf`. -/
theorem fullVF_dirac_zero_score (θ : TimeParams d) (t : ℝ)
    (p : SSphere d) (x : EucSpace d) :
    fullVF d (fun s => { θ s with B := 0 }) (Perspective.diracProb d p) t x =
      fullVF d θ (Perspective.diracProb d p) t x := by
  rw [fullVF_dirac_eq, fullVF_dirac_eq]

/-- At a fixed unit query, changing the original atom costs at most
its displacement times the actual value norm. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem norm_fullVF_dirac_law_sub_le (θ : TimeParams d) (t : ℝ)
    (p q : SSphere d) (x : EucSpace d) (hx : ‖x‖ = 1) :
    ‖fullVF d θ (Perspective.diracProb d p) t x -
      fullVF d θ (Perspective.diracProb d q) t x‖ ≤
      ‖(θ t).V‖ * ‖(p : EucSpace d) - (q : EucSpace d)‖ := by
  have he : fullVF d θ (Perspective.diracProb d p) t x -
      fullVF d θ (Perspective.diracProb d q) t x =
      proj d x ((θ t).V ((p : EucSpace d) - (q : EucSpace d))) := by
    rw [fullVF_dirac_eq, fullVF_dirac_eq]
    simp only [map_sub, proj, inner_sub_right, sub_smul]
    module
  rw [he]
  exact (norm_proj_le hx _).trans ((θ t).V.le_opNorm _)

/-- The original field of a fixed atom has a spatial constant
independent of its score norm. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: vf` and the characteristic flow. -/
theorem norm_fullVF_dirac_query_sub_le (θ : TimeParams d) (t : ℝ)
    (p : SSphere d) (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    ‖fullVF d θ (Perspective.diracProb d p) t x -
      fullVF d θ (Perspective.diracProb d p) t y‖ ≤
      (‖(θ t).W‖ * ‖(θ t).U‖ +
        2 * (‖(θ t).V‖ + ‖(θ t).W‖ * (‖(θ t).U‖ + ‖(θ t).b‖))) * ‖x - y‖ := by
  have h := norm_fullVF_sub_le (fun s => { θ s with B := 0 })
    (Perspective.diracProb d p) t x y hx hy
  simpa only [fullVF_dirac_zero_score, norm_zero, mul_zero, zero_mul, zero_add] using h

/-- Comparing the complete velocities at two different original atoms
retains both law and query changes. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: ips` and `eq: vf`. -/
theorem norm_fullVF_dirac_pair_sub_le (θ : TimeParams d) (t : ℝ) (p q : SSphere d) :
    ‖fullVF d θ (Perspective.diracProb d p) t p -
      fullVF d θ (Perspective.diracProb d q) t q‖ ≤
      (3 * ‖(θ t).V‖ + 3 * ‖(θ t).W‖ * ‖(θ t).U‖ +
        2 * ‖(θ t).W‖ * ‖(θ t).b‖) * ‖(p : EucSpace d) - (q : EucSpace d)‖ := by
  have hp : ‖(p : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hq : ‖(q : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp q.2
  have h := norm_sub_le_norm_sub_add_norm_sub
    (fullVF d θ (Perspective.diracProb d p) t p)
    (fullVF d θ (Perspective.diracProb d q) t p)
    (fullVF d θ (Perspective.diracProb d q) t q)
  exact h.trans ((add_le_add (norm_fullVF_dirac_law_sub_le θ t p q p hp)
    (norm_fullVF_dirac_query_sub_le θ t q p q hp hq)).trans (by ring_nf; rfl))

/-- The actual one-atom spherical velocity is Lipschitz, with its
explicit original parameter constant. Source auxiliary:
arXiv:2411.04551v3, §1, the one-token case of `eq: ips`. -/
theorem lipschitzWith_fullVF_dirac_diagonal_sphere (θ : TimeParams d) (t : ℝ) :
    LipschitzWith ⟨3 * ‖(θ t).V‖ + 3 * ‖(θ t).W‖ * ‖(θ t).U‖ +
      2 * ‖(θ t).W‖ * ‖(θ t).b‖, by positivity⟩
      (fun p : SSphere d => fullVF d θ (Perspective.diracProb d p) t p) := by
  apply LipschitzWith.of_dist_le_mul
  intro p q
  change ‖fullVF d θ (Perspective.diracProb d p) t p -
    fullVF d θ (Perspective.diracProb d q) t q‖ ≤
      _ * ‖(p : EucSpace d) - (q : EucSpace d)‖
  exact norm_fullVF_dirac_pair_sub_le θ t p q

/-- The original one-atom velocity is continuous on the sphere for
every complete tuple, regardless of its score matrix. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: ips` and `eq: vf`. -/
theorem continuous_fullVF_dirac_diagonal_sphere (θ : TimeParams d) (t : ℝ) :
    Continuous (fun p : SSphere d => fullVF d θ (Perspective.diracProb d p) t p) :=
  (lipschitzWith_fullVF_dirac_diagonal_sphere θ t).continuous

/-- A common component norm bound gives the original one-atom
velocity constant 3K+5K². Source auxiliary: arXiv:2411.04551v3,
§1.1, the norm estimate in `thm: targets.atoms`. -/
theorem norm_fullVF_dirac_pair_sub_le_of_component_bounds (θ : TimeParams d)
    (t K : ℝ) (p q : SSphere d) (hV : ‖(θ t).V‖ ≤ K) (hW : ‖(θ t).W‖ ≤ K)
    (hU : ‖(θ t).U‖ ≤ K) (hb : ‖(θ t).b‖ ≤ K) :
    ‖fullVF d θ (Perspective.diracProb d p) t p -
      fullVF d θ (Perspective.diracProb d q) t q‖ ≤
      (3 * K + 5 * K ^ 2) * ‖(p : EucSpace d) - (q : EucSpace d)‖ := by
  have hK : 0 ≤ K := (norm_nonneg _).trans hV
  have hwu := mul_le_mul hW hU (norm_nonneg _) hK
  have hwb := mul_le_mul hW hb (norm_nonneg _) hK
  apply (norm_fullVF_dirac_pair_sub_le θ t p q).trans
  apply mul_le_mul_of_nonneg_right _ (norm_nonneg _)
  nlinarith

/-- The same constant follows from the actual sum norm of the five
original blocks, as used by `thm: targets.atoms`. Source auxiliary:
arXiv:2411.04551v3, §1.1, that theorem's parameter norm estimate. -/
theorem norm_fullVF_dirac_pair_sub_le_of_sum_bound (θ : TimeParams d)
    (t K : ℝ) (p q : SSphere d)
    (hθ : ‖(θ t).V‖ + ‖(θ t).B‖ + ‖(θ t).W‖ + ‖(θ t).U‖ + ‖(θ t).b‖ ≤ K) :
    ‖fullVF d θ (Perspective.diracProb d p) t p -
      fullVF d θ (Perspective.diracProb d q) t q‖ ≤
      (3 * K + 5 * K ^ 2) * ‖(p : EucSpace d) - (q : EucSpace d)‖ := by
  have hV := norm_nonneg (θ t).V
  have hB := norm_nonneg (θ t).B
  have hW := norm_nonneg (θ t).W
  have hU := norm_nonneg (θ t).U
  have hb := norm_nonneg (θ t).b
  exact norm_fullVF_dirac_pair_sub_le_of_component_bounds θ t K p q
    (by linarith) (by linarith) (by linarith) (by linarith)

/-- Distinct genuine unit atoms, a unit query and nonzero original
matrices and bias jointly realize all hypotheses, including the source's
sum norm and all four component bounds. -/
example : ∃ (θ : TimeParams 2) (p q : SSphere 2) (x : EucSpace 2) (K : ℝ),
    p ≠ q ∧ ‖x‖ = 1 ∧
    ‖(θ 0).V‖ ≤ K ∧ ‖(θ 0).W‖ ≤ K ∧ ‖(θ 0).U‖ ≤ K ∧ ‖(θ 0).b‖ ≤ K ∧
    ‖(θ 0).V‖ + ‖(θ 0).B‖ + ‖(θ 0).W‖ + ‖(θ 0).U‖ + ‖(θ 0).b‖ ≤ K := by
  let p : SSphere 2 := basePoint 1
  let q := antipode 2 p
  let I := ContinuousLinearMap.id ℝ (EucSpace 2)
  let θ : TimeParams 2 := fun _ => ⟨I, I, I, I, (p : EucSpace 2)⟩
  have hp : ‖(p : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  have hn : ‖I‖ = 1 := ContinuousLinearMap.norm_id
  refine ⟨θ, p, q, p, 5, (antipode_ne 2 p).symm, hp, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp only [θ, hn, hp] <;> norm_num

end Transformer.Interpolation
