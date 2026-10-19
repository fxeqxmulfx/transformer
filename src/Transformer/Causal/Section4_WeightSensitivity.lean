module

public import Transformer.Causal.Section4_SecondOverlap
public import Transformer.Causal.Section4_FlowContinuity

/-!
# Original weight sensitivity and antipodal follower geometry

ArXiv:2411.04990v2, §4, proof of `thm1`.
Every original masked row weight is smooth in the original ambient
tokens and therefore Lipschitz on the maximum-norm unit ball.
For two original tuples with identical predecessors, their ambient
distance is exactly the distance of their last tokens. The weight
estimate thus controls actual follower-weight changes by that last
distance, without perturbing the original shared predecessor path.

The remaining identities concern the actual spherical projection.
Its contribution to the separation of two unit followers is an
explicit multiple of their squared distance. Near the antipode of
a unit anchor that multiple is positive, while the projected
predecessor itself is small. These estimates give the main positive
term and the small weight-variation error in the original antipodal
instability argument.
-/

@[expose] public section

open scoped BigOperators NNReal
open Real

namespace Transformer.Causal

variable {d n : ℕ}

/-- Each actual original causal weight is Lipschitz on the entire compact ambient unit ball.
Source: arXiv:2411.04990v2, §4, smoothness of the original normalized field in `thm1`. -/
theorem exists_lipschitzOnWith_csaTokenWeight (β : ℝ) (Q K : ParamMatrix d) (i j : Idx n) :
    ∃ L : ℝ≥0, LipschitzOnWith L (fun Y : Idx n → EucSpace d => csaTokenWeight β Q K Y i j)
      (Metric.closedBall 0 1) := by
  exact (contDiff_csaTokenWeight β Q K i j).contDiffOn.exists_lipschitzOnWith
    (by simp) (convex_closedBall (0 : Idx n → EucSpace d) 1) (isCompact_closedBall 0 1)

/-- A bound chosen only from the proved Lipschitz estimate for the actual original row entry.
Source: arXiv:2411.04990v2, §4, the original weights in `thm1`. -/
noncomputable def csaWeightLipschitz (β : ℝ) (Q K : ParamMatrix d) (i j : Idx n) : ℝ≥0 :=
  Classical.choose (exists_lipschitzOnWith_csaTokenWeight β Q K i j)

/-- The chosen constant controls the original weight on the original ambient unit ball.
Source: arXiv:2411.04990v2, §4, the normalized original field in `thm1`. -/
theorem lipschitzOnWith_csaTokenWeight_unit (β : ℝ) (Q K : ParamMatrix d) (i j : Idx n) :
    LipschitzOnWith (csaWeightLipschitz β Q K i j)
      (fun Y : Idx n → EucSpace d => csaTokenWeight β Q K Y i j) (Metric.closedBall 0 1) :=
  Classical.choose_spec (exists_lipschitzOnWith_csaTokenWeight β Q K i j)

/-- When original predecessors agree, their entire ambient tuple difference is their last-token difference.
Source: arXiv:2411.04990v2, §4, conditioning on the original preceding tokens in `thm1`. -/
theorem norm_csaTuple_sub_of_prefix_eq (Y Z : SphereTuple d (n + 1))
    (hp : ∀ j : Idx n, Y j.castSucc = Z j.castSucc) :
    ‖(fun j => (Y j : EucSpace d)) - (fun j => (Z j : EucSpace d))‖ ≤
      ‖(Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d)‖ := by
  apply (pi_norm_le_iff_of_nonneg (norm_nonneg _)).mpr
  intro i
  refine Fin.lastCases ?_ (fun j => ?_) i
  · exact le_rfl
  · simp only [Pi.sub_apply, hp j, sub_self, norm_zero]
    exact norm_nonneg _

/-- Actual original weights of followers with the same actual prefix have a last-token Lipschitz estimate.
Source: arXiv:2411.04990v2, §4, the original conditional follower field in `thm1`. -/
theorem abs_csaWeight_sub_le_of_prefix_eq (β : ℝ) (Q K : ParamMatrix d)
    (Y Z : SphereTuple d (n + 1)) (hp : ∀ j : Idx n, Y j.castSucc = Z j.castSucc)
    (i j : Idx (n + 1)) :
    |csaTokenWeight β Q K (fun l => (Y l : EucSpace d)) i j -
      csaTokenWeight β Q K (fun l => (Z l : EucSpace d)) i j| ≤
        (csaWeightLipschitz β Q K i j : ℝ) *
          ‖(Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d)‖ := by
  have h := (lipschitzOnWith_csaTokenWeight_unit β Q K i j).norm_sub_le
    (coe_sphereTuple_mem_closedBall Y) (coe_sphereTuple_mem_closedBall Z)
  rw [Real.norm_eq_abs] at h
  exact h.trans (mul_le_mul_of_nonneg_left (norm_csaTuple_sub_of_prefix_eq Y Z hp)
    (NNReal.coe_nonneg _))

/-- The sum of the proved original predecessor-weight sensitivity bounds for the actual last row.
Source: arXiv:2411.04990v2, §4, the conditional original follower field of `thm1`. -/
noncomputable def csaPredecessorLipschitz (β : ℝ) (Q K : ParamMatrix d) (n : ℕ) : ℝ :=
  ∑ j : Idx n, (csaWeightLipschitz β Q K (Fin.last n) j.castSucc : ℝ)

/-- The actual total predecessor-weight sensitivity bound is nonnegative.
Source: arXiv:2411.04990v2, §4, the original antipodal instability in `thm1`. -/
theorem csaPredecessorLipschitz_nonneg (β : ℝ) (Q K : ParamMatrix d) :
    0 ≤ csaPredecessorLipschitz β Q K n :=
  Finset.sum_nonneg (fun j _ => NNReal.coe_nonneg (csaWeightLipschitz β Q K (Fin.last n) j.castSucc))

/-- The actual total predecessor mass has the corresponding original last-token sensitivity bound.
Source: arXiv:2411.04990v2, §4, the original conditional predecessor force in `thm1`. -/
theorem abs_csaPredecessorMass_sub_le (β : ℝ) (Q K : ParamMatrix d)
    (Y Z : SphereTuple d (n + 1)) (hp : ∀ j : Idx n, Y j.castSucc = Z j.castSucc) :
    |(∑ j : Idx n, csaTokenWeight β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc) -
      ∑ j : Idx n, csaTokenWeight β Q K (fun l => (Z l : EucSpace d)) (Fin.last n) j.castSucc| ≤
        csaPredecessorLipschitz β Q K n *
          ‖(Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d)‖ := by
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  calc
    _ ≤ ∑ j : Idx n, (csaWeightLipschitz β Q K (Fin.last n) j.castSucc : ℝ) *
        ‖(Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d)‖ := by
      apply Finset.sum_le_sum
      intro j hj
      exact abs_csaWeight_sub_le_of_prefix_eq β Q K Y Z hp _ _
    _ = _ := by rw [csaPredecessorLipschitz, Finset.sum_mul]

/-- The original projected common-predecessor force has an exact squared-separation identity.
Source: arXiv:2411.04990v2, §4, the positive spherical linearization at the antipodal equilibrium in `thm1`. -/
theorem csaProjected_separation (x y : SSphere d) (p : EucSpace d) :
    inner ℝ ((x : EucSpace d) - (y : EucSpace d))
      (proj d (x : EucSpace d) p - proj d (y : EucSpace d) p) =
        -((inner ℝ (x : EucSpace d) p + inner ℝ (y : EucSpace d) p) / 2) *
          ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 := by
  rw [norm_sub_sq_real]
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_right,
    real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp x.2,
    mem_sphere_zero_iff_norm.mp y.2, one_pow, real_inner_comm (x : EucSpace d) (y : EucSpace d)]
  ring

/-- The actual projected predecessor force is small near the original antipode of its anchor.
Source: arXiv:2411.04990v2, §4, the vanishing projected force at the antipodal configuration in `thm1`. -/
theorem norm_csaProj_near_antipode (y : SSphere d) (p a : EucSpace d) :
    ‖proj d (y : EucSpace d) p‖ ≤ ‖p - a‖ + ‖(y : EucSpace d) + a‖ := by
  have he : proj d (y : EucSpace d) p = proj d (y : EucSpace d) (p + (y : EucSpace d)) := by
    simp only [proj, inner_add_right, real_inner_self_eq_norm_sq,
      mem_sphere_zero_iff_norm.mp y.2, one_pow, add_smul, one_smul]
    module
  calc
    _ = ‖proj d (y : EucSpace d) (p + (y : EucSpace d))‖ := by rw [he]
    _ ≤ ‖p + (y : EucSpace d)‖ := norm_proj_le (mem_sphere_zero_iff_norm.mp y.2) _
    _ = ‖(p - a) + ((y : EucSpace d) + a)‖ := by congr 1; module
    _ ≤ _ := norm_add_le _ _

/-- The actual original predecessor has uniformly negative overlap with a follower near its original antipode.
Source: arXiv:2411.04990v2, §4, the positive spherical Jacobian at the antipodal configuration in `thm1`. -/
theorem inner_csa_near_antipode (x p a : SSphere d) :
    inner ℝ (x : EucSpace d) (p : EucSpace d) ≤ -1 +
      ‖(x : EucSpace d) + (a : EucSpace d)‖ + ‖(p : EucSpace d) - (a : EucSpace d)‖ := by
  have h1 := (abs_le.mp (abs_real_inner_le_norm
    ((x : EucSpace d) + (a : EucSpace d)) (p : EucSpace d))).2
  have h2 := (abs_le.mp (abs_real_inner_le_norm
    (a : EucSpace d) ((p : EucSpace d) - (a : EucSpace d)))).1
  simp only [mem_sphere_zero_iff_norm.mp p.2, mem_sphere_zero_iff_norm.mp a.2,
    one_mul, mul_one, inner_add_left, inner_sub_right, real_inner_self_eq_norm_sq, one_pow] at h1 h2
  linarith

/-- A pair of actual nonzero original tuples jointly realizes the shared-prefix hypotheses.
Source: arXiv:2411.04990v2, §4, conditioning on an original consensus prefix in `thm1`. -/
example : ∀ j : Idx 1, (fun _ : Idx 2 => basePoint 1) j.castSucc =
    (fun _ : Idx 2 => (basePoint 1 : SSphere 2)) j.castSucc := fun _ => rfl

end Transformer.Causal
