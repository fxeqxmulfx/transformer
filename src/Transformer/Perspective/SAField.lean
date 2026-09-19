/-
# The right-hand side of `SA` as a vector field

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The right-hand side of `eq: SA` is a time-independent vector field on
`(ℝ^d)^n`: the softmax weights of `Perspective.softmaxWeight` at the scores
`β⟨y_i, y_j⟩`, averaged and projected.  This file names it (`saField`), proves
that solving `eq: SA` is being one of its integral curves (`SA_iff`), and
records the two facts a Grönwall comparison needs about it: the attention
average of unit vectors is again short, and at `n = 1` the field vanishes.

That it is Lipschitz — Step 1 of Appendix D — is
`Perspective.lipschitzOnWith_saField`, in `Perspective.SALipschitz`.
-/

import Transformer.Perspective.Beta0Field
import Transformer.Perspective.Softmax

open scoped BigOperators NNReal
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The attention scores seen by the `i`-th token: `j ↦ β ⟨y_i, y_j⟩`.

Source: arXiv:2312.10794v5, §2, `eq: SA`. -/
noncomputable def saScore (β : ℝ) (Y : Idx n → EucSpace d) (i : Idx n) : Idx n → ℝ :=
  fun j => β * inner (𝕜 := ℝ) (Y i) (Y j)

/-- **The vector field of `eq: SA`:**
`(SA_β Y)_i = Proj_{y_i}( Σ_j σ(β⟨y_i, y_·⟩)_j y_j )`.

Source: arXiv:2312.10794v5, §2, `eq: SA`. -/
noncomputable def saField (β : ℝ) (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  fun i => proj d (Y i) (∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j)

/-- A path solves `eq: SA` exactly when it is an integral curve of `saField`. -/
theorem SA_iff (β : ℝ) (X : ℝ → SphereTuple d n) :
    SA d n β X ↔
      ∀ t : ℝ, ∀ i : Idx n,
        HasDerivAt (fun s => (X s i : EucSpace d))
          (saField d n β (fun j => ((X t j : EucSpace d))) i) t := by
  have hdrift : ∀ (t : ℝ) (i : Idx n),
      (partitionSA d n β X t i)⁻¹ •
          ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) ((X t i : EucSpace d))
                ((X t j : EucSpace d))) • ((X t j : EucSpace d))
        = ∑ j : Idx n,
            softmaxWeight (saScore d n β (fun k => ((X t k : EucSpace d))) i) j
              • ((X t j : EucSpace d)) := by
    intro t i
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hZ : partitionSA d n β X t i
        = ∑ k : Idx n, Real.exp (saScore d n β (fun k => ((X t k : EucSpace d))) i k) := by
      rw [partitionSA]
      rfl
    rw [smul_smul, softmaxWeight, saScore, inv_mul_eq_div, hZ]
  constructor
  · intro h t i
    exact (h t i).congr_deriv (by rw [saField, hdrift t i])
  · intro h t i
    exact (h t i).congr_deriv (by rw [saField, hdrift t i])

/-- The attention average of a tuple of unit vectors is a convex combination of
unit vectors, hence sits in the unit ball. -/
theorem norm_saAvg_le (β : ℝ) (hn : 0 < n) {Y : Idx n → EucSpace d}
    (hY : ∀ j : Idx n, ‖Y j‖ = 1) (i : Idx n) :
    ‖∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j‖ ≤ 1 := by
  refine (norm_sum_le _ _).trans ?_
  have hterm : ∀ j : Idx n, ‖softmaxWeight (saScore d n β Y i) j • Y j‖
      = softmaxWeight (saScore d n β Y i) j := by
    intro j
    rw [norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (softmaxWeight_nonneg (saScore d n β Y i) j), hY j, mul_one]
  simp only [hterm]
  exact le_of_eq (sum_softmaxWeight hn _)

/-- The hypotheses of `norm_saAvg_le` are satisfiable: one token at the base
point. -/
example : (0 : ℕ) < 1 ∧ ∀ _j : Idx 1, ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
  ⟨one_pos, fun _ => mem_sphere_zero_iff_norm.mp (basePoint 0).2⟩

/-- **A lone token does not move:** at `n = 1` the only attention weight is `1`
and `Proj_x x = 0`, so `saField` vanishes on unit tuples. -/
theorem saField_of_one (β : ℝ) {Y : Idx 1 → EucSpace d}
    (hY : ∀ j : Idx 1, ‖Y j‖ = 1) (i : Idx 1) : saField d 1 β Y i = 0 := by
  obtain rfl : i = 0 := Subsingleton.elim i 0
  have hw : softmaxWeight (saScore d 1 β Y 0) 0 = 1 := by
    rw [softmaxWeight]
    simp
  rw [saField, Fin.sum_univ_one, hw, one_smul, proj, real_inner_self_eq_norm_mul_norm,
    hY 0, mul_one, one_smul, sub_self]

/-- The hypothesis of `saField_of_one` is satisfiable: the base point. -/
example : ∀ _j : Idx 1, ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
  fun _ => mem_sphere_zero_iff_norm.mp (basePoint 0).2

end Perspective
end Transformer
