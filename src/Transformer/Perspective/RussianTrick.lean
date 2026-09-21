/-
# The "Russian trick"

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix A, `e:russiantrick`.

The identity the Hessian formula `e:helpcl` is summed against: there are `d`
skew-symmetric endomorphisms `B_1, …, B_d` of `ℝ^d` with

  `-I_d = (1/(d-1)) Σ_k B_k²`.

The paper builds each `B_k` from the canonical `(d-1)`-dimensional symplectic
form by zeroing out the `k`-th block, which is where its hypothesis that `d` is
odd comes from: an orthogonal complex structure on `e_k^⊥` needs `d-1` even.
That construction is not the only one, and not the cheapest.  A *rank-two*
block suffices: for a unit pair `u ⊥ v`,

  `B x = c (⟨u,x⟩ v - ⟨v,x⟩ u)`    has   `B² x = -c² (⟨u,x⟩ u + ⟨v,x⟩ v)`,

so `B²` is `-c²` times the projection onto `span{u,v}`.  Taking the `d` pairs
`(e_k, e_{k+1})` around the cycle `ℤ/d`, every basis vector is hit exactly
twice — once as the first member of a pair and once as the second — so

  `Σ_k B_k² = -2c² I`,   and `c = √((d-1)/2)` gives `-(d-1) I`.

Nothing here needs `d` odd; it needs `d ≥ 2`, which is also where the
statement makes sense: at the odd dimension `d = 1` the factor `1/(d-1)` of the
paper is undefined, and the only skew matrix of `ℝ^1` is `0` anyway.  So the hypothesis `Odd d` is replaced by `2 ≤ d`, which
covers every odd `d` at which the identity is meaningful.
-/

import Transformer.Basic

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- A skew-symmetric endomorphism of `ℝ^d`: `⟨Bx, y⟩ = -⟨x, By⟩`.

These are exactly the generators of the rotations, `e^{tB} ∈ O(d)`, which is
why the perturbation of `Perspective.AppendixA_Saddle` stays on the sphere. -/
def IsSkew (d : ℕ) (B : ParamMatrix d) : Prop :=
  ∀ x y : EucSpace d, inner (𝕜 := ℝ) (B x) y = -inner (𝕜 := ℝ) x (B y)

/-- The rank-two map `x ↦ c (⟨u,x⟩ v - ⟨v,x⟩ u)`, the elementary rotation
generator in the plane of `u` and `v`. -/
noncomputable def skewPair (d : ℕ) (c : ℝ) (u v : EucSpace d) : ParamMatrix d :=
  c • ((innerSL ℝ u).smulRight v - (innerSL ℝ v).smulRight u)

theorem skewPair_apply (c : ℝ) (u v x : EucSpace d) :
    skewPair d c u v x
      = c • ((inner (𝕜 := ℝ) u x : ℝ) • v - (inner (𝕜 := ℝ) v x : ℝ) • u) := rfl

/-- `skewPair` is skew for *any* pair `u`, `v`: the antisymmetry is in the
shape of the formula, not in the geometry. -/
theorem skewPair_isSkew (c : ℝ) (u v : EucSpace d) : IsSkew d (skewPair d c u v) := by
  intro x y
  simp only [skewPair_apply, inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right, real_inner_comm x u, real_inner_comm x v]
  ring

/-- **The square of a rank-two generator.**  On an orthonormal pair,
`B²` is `-c²` times the orthogonal projection onto `span{u, v}`. -/
theorem skewPair_sq (c : ℝ) {u v : EucSpace d}
    (hu : inner (𝕜 := ℝ) u u = (1 : ℝ)) (hv : inner (𝕜 := ℝ) v v = (1 : ℝ))
    (huv : inner (𝕜 := ℝ) u v = (0 : ℝ)) (x : EucSpace d) :
    skewPair d c u v (skewPair d c u v x)
      = (-c ^ 2) • ((inner (𝕜 := ℝ) u x : ℝ) • u + (inner (𝕜 := ℝ) v x : ℝ) • v) := by
  have hvu : inner (𝕜 := ℝ) v u = (0 : ℝ) := by rw [real_inner_comm]; exact huv
  simp only [skewPair_apply, inner_sub_right, real_inner_smul_right, hu, hv, huv, hvu,
    smul_sub, smul_smul, smul_add]
  module

/-- **Every basis vector is covered exactly twice.**  If `σ` has no fixed
point, the pairs `(e_k, e_{σ k})` are honest orthonormal pairs, and summing
their projections over `k` counts each `e_j` once as a first member and once as
a second: `Σ_k B_k² = -2c² I`. -/
theorem sum_skewPair_sq (c : ℝ) (e : OrthonormalBasis (Idx d) ℝ (EucSpace d))
    (σ : Equiv.Perm (Idx d)) (hσ : ∀ k, σ k ≠ k) (x : EucSpace d) :
    ∑ k : Idx d, skewPair d c (e k) (e (σ k)) (skewPair d c (e k) (e (σ k)) x)
      = (-(2 * c ^ 2)) • x := by
  have hon : ∀ i j : Idx d, inner (𝕜 := ℝ) (e i) (e j) = if i = j then (1 : ℝ) else 0 :=
    orthonormal_iff_ite.1 e.orthonormal
  have hstep : ∀ k : Idx d,
      skewPair d c (e k) (e (σ k)) (skewPair d c (e k) (e (σ k)) x)
        = (-c ^ 2) • ((inner (𝕜 := ℝ) (e k) x : ℝ) • e k
            + (inner (𝕜 := ℝ) (e (σ k)) x : ℝ) • e (σ k)) := fun k =>
    skewPair_sq c (by simp) (by simp) (by rw [hon]; simp [Ne.symm (hσ k)]) x
  rw [Finset.sum_congr rfl fun k _ => hstep k, ← Finset.smul_sum, Finset.sum_add_distrib,
    e.sum_repr' x,
    Fintype.sum_equiv σ (fun k => (inner (𝕜 := ℝ) (e (σ k)) x : ℝ) • e (σ k))
      (fun j => (inner (𝕜 := ℝ) (e j) x : ℝ) • e j) (fun _ => rfl), e.sum_repr' x]
  module

/-- **Equation (e:russiantrick).** *"Russian trick".*  In dimension `d ≥ 2`
there are `d` skew-symmetric matrices `B_1, …, B_d` with

  `-I_d = (1/(d-1)) Σ_k B_k²`.

The identity is what turns the Hessian formula `e:helpcl` into a *sum* of
directions along which `𝖤_0` can be increased.

Deviation from the source: the paper states this in odd dimension, because its
own construction — the canonical symplectic form with the `k`-th `2×2` block
removed — needs `d-1` even.  The construction used here, the cycle of rank-two
generators described at the head of this file, needs only `d ≥ 2`.  The one
odd dimension this leaves out is `d = 1`, where `1/(d-1)` is undefined on
paper; in Lean it would read `0⁻¹ = 0`, and the statement would fail for that
reason alone, which says nothing about the survey.

Source: arXiv:2312.10794v5, Appendix A, `e:russiantrick`. -/
theorem russian_trick (hd : 2 ≤ d) :
    ∃ B : Idx d → ParamMatrix d,
      (∀ k : Idx d, IsSkew d (B k)) ∧
      ∀ x : EucSpace d, (((d : ℝ) - 1)⁻¹) • ∑ k : Idx d, B k (B k x) = -x := by
  have : NeZero d := ⟨by omega⟩
  have hd' : (2 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hne : ((d : ℝ) - 1) ≠ 0 := by linarith
  set c : ℝ := Real.sqrt (((d : ℝ) - 1) / 2) with hc
  have hc2 : c ^ 2 = ((d : ℝ) - 1) / 2 := Real.sq_sqrt (by linarith)
  set σ : Equiv.Perm (Idx d) := Equiv.addRight (1 : Idx d) with hσdef
  have hσ : ∀ k : Idx d, σ k ≠ k := by
    intro k hk
    simp only [hσdef, Equiv.coe_addRight, add_eq_left] at hk
    have h1 := congrArg Fin.val hk
    rw [Fin.val_one', Fin.val_zero, Nat.mod_eq_of_lt (by omega)] at h1
    exact one_ne_zero h1
  refine ⟨fun k => skewPair d c (EuclideanSpace.basisFun (Idx d) ℝ k)
      (EuclideanSpace.basisFun (Idx d) ℝ (σ k)), fun k => skewPair_isSkew _ _ _, ?_⟩
  intro x
  rw [sum_skewPair_sq c (EuclideanSpace.basisFun (Idx d) ℝ) σ hσ x, smul_smul, hc2]
  have hscal : ((d : ℝ) - 1)⁻¹ * -(2 * (((d : ℝ) - 1) / 2)) = -1 := by
    field_simp
  rw [hscal, neg_one_smul]

/-- The hypothesis of `russian_trick` is satisfiable: `d = 2`. -/
example : 2 ≤ 2 := le_refl 2

end Perspective
end Transformer
