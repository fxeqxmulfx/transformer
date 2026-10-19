module

public import Transformer.Perspective.CircleSA
public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Mathlib.Analysis.Calculus.ContDiff.WithLp
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# SA on the circle as a gradient flow in a metric

The angles are the points of the Euclidean space `EucSpace n`, on which
`𝖤_β` is `eucTorusEnergy`: smooth (`contDiff_eucTorusEnergy`), with the
partial derivatives of `hasDerivAt_torusEnergy` as the coordinates of its
gradient (`gradient_eucTorusEnergy_apply`). So along a solution of
`angularSA`, for every curve `φ` of angles through `θ(t)` with velocity `b`,

  `d/ds (n² 𝖤_β)(φ(s))|_{s=0} = Σ_i Z_i(θ(t)) θ̇_i(t) b_i`

(`hasDerivAt_torusEnergy_of_angularSA`): SA in angles is the gradient flow of
the energy of §3.4 for the metric `diag(Z_i)`, `e:scalarproduct` in angles, as
`sa_is_gradient_flow` states it on spheres. With the gain
`G = diag(n²/Z_i)`, SA in angles is `θ̇ = -G(θ) ∇f(θ)` for `f = -𝖤_β`
(`saGain_gradient_apply`), the form of `Landscape.SaddleAvoidance.metricGradientFlow`;
`𝖤_β` and `G` are invariant under the periods `2π e_k`
(`cos_sub_add_period`), which span the space (`span_periods`).

Source: arXiv:2312.10794v5, §3.4 (`e:scalarproduct`, `eq:r3`) and §7.1
(`eq:energyF`).
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- `𝖤_β` on the Euclidean space of angles. -/
noncomputable def eucTorusEnergy (β : ℝ) (ψ : EucSpace n) : ℝ :=
  torusEnergy n β ψ.ofLp

/-- `𝖤_β` is smooth. -/
theorem contDiff_eucTorusEnergy (β : ℝ) {k : WithTop ℕ∞} :
    ContDiff ℝ k (eucTorusEnergy n β) := by
  unfold eucTorusEnergy torusEnergy
  fun_prop

/-- **The gradient of `𝖤_β` has the partial derivatives as coordinates**:
`(∇𝖤_β)_i = (1/n²) Σ_j e^{β cos(θ_j - θ_i)} sin(θ_j - θ_i)`. The derivative
along `e_i` is that of `x ↦ 𝖤_β(θ + (x - θ_i) e_i)`, the partial map of
`hasDerivAt_torusEnergy`. -/
theorem gradient_eucTorusEnergy_apply (β : ℝ) (hβ : β ≠ 0) (ψ : EucSpace n) (i : Idx n) :
    (gradient (eucTorusEnergy n β) ψ).ofLp i = (1 / (n : ℝ) ^ 2) * ∑ j : Idx n,
      Real.exp (β * Real.cos (ψ.ofLp j - ψ.ofLp i)) * Real.sin (ψ.ofLp j - ψ.ofLp i) := by
  set e : EucSpace n := EuclideanSpace.single i 1
  have hline : HasDerivAt (fun x : ℝ => ψ + (x - ψ.ofLp i) • e) e (ψ.ofLp i) := by
    simpa using (((hasDerivAt_id (ψ.ofLp i)).sub_const (ψ.ofLp i)).smul_const e).const_add ψ
  have hcomp := ((contDiff_eucTorusEnergy n β (k := 1)).differentiable one_ne_zero
    ψ).hasFDerivAt.comp_hasDerivAt_of_eq (ψ.ofLp i) hline (by simp)
  have hupd : (eucTorusEnergy n β ∘ fun x : ℝ => ψ + (x - ψ.ofLp i) • e) =
      fun x => torusEnergy n β (Function.update ψ.ofLp i x) := by
    funext x
    refine congrArg (torusEnergy n β) (funext fun j => ?_)
    by_cases hj : j = i
    · subst hj
      simp [e]
    · simp [e, hj]
  rw [hupd] at hcomp
  rw [← hcomp.unique (hasDerivAt_torusEnergy n β hβ ψ.ofLp i),
    ← InnerProductSpace.toDual_symm_apply (𝕜 := ℝ), real_inner_comm,
    EuclideanSpace.inner_single_left]
  simp [gradient]

/-- The gradient of `f = -𝖤_β` is `-∇𝖤_β`. -/
theorem gradient_neg_eucTorusEnergy (β : ℝ) (ψ : EucSpace n) :
    gradient (fun φ => -eucTorusEnergy n β φ) ψ = -gradient (eucTorusEnergy n β) ψ := by
  simp only [gradient, fderiv_fun_neg, map_neg]

/-- Adding `2π` to one angle changes no cosine of a difference of angles. -/
theorem cos_sub_add_period (ψ : EucSpace n) (k i j : Idx n) :
    Real.cos ((ψ + (2 * π) • EuclideanSpace.single k (1 : ℝ)).ofLp i -
      (ψ + (2 * π) • EuclideanSpace.single k (1 : ℝ)).ofLp j) =
      Real.cos (ψ.ofLp i - ψ.ofLp j) := by
  have h : (ψ + (2 * π) • EuclideanSpace.single k (1 : ℝ)).ofLp i -
      (ψ + (2 * π) • EuclideanSpace.single k (1 : ℝ)).ofLp j = ψ.ofLp i - ψ.ofLp j +
        (((if i = k then 1 else 0) - (if j = k then 1 else 0) : ℤ) : ℝ) * (2 * π) := by
    simp only [WithLp.ofLp_add, WithLp.ofLp_smul, Pi.add_apply, Pi.smul_apply, smul_eq_mul,
      PiLp.single_apply]
    split_ifs <;> push_cast <;> ring
  rw [h, Real.cos_add_int_mul_two_pi]

/-- The periods `2π e_k` span the space of angles. -/
theorem span_periods :
    Submodule.span ℝ (Set.range fun k : Idx n => (2 * π) • EuclideanSpace.single k (1 : ℝ)) =
      ⊤ := by
  rw [Set.range_smul, Submodule.span_smul_eq_of_isUnit _ _ (isUnit_iff_ne_zero.2 two_pi_pos.ne')]
  have h := (EuclideanSpace.basisFun (Idx n) ℝ).toBasis.span_eq
  rwa [OrthonormalBasis.coe_toBasis, show ⇑(EuclideanSpace.basisFun (Idx n) ℝ) =
    fun k => EuclideanSpace.single k (1 : ℝ) from funext (EuclideanSpace.basisFun_apply _ _)] at h

theorem eucTorusEnergy_add_period (β : ℝ) (ψ : EucSpace n) (k : Idx n) :
    eucTorusEnergy n β (ψ + (2 * π) • EuclideanSpace.single k (1 : ℝ)) = eucTorusEnergy n β ψ := by
  simp only [eucTorusEnergy, torusEnergy, cos_sub_add_period]

/-- **SA in angles is `θ̇ = -G(θ) ∇f(θ)` with `f = -𝖤_β`**: the `i`-th
coordinate of `G ∇f` is `(1/Z_i) Σ_j e^{β cos(θ_i - θ_j)} sin(θ_i - θ_j)`. -/
theorem saGain_gradient_apply (β : ℝ) (hβ : β ≠ 0) (ψ : EucSpace n) (i : Idx n) :
    (saGain n β ψ (gradient (fun φ => -eucTorusEnergy n β φ) ψ)).ofLp i =
      (circlePartition n β ψ.ofLp i)⁻¹ * ∑ j : Idx n,
        Real.exp (β * Real.cos (ψ.ofLp i - ψ.ofLp j)) * Real.sin (ψ.ofLp i - ψ.ofLp j) := by
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Fin.pos i).ne'
  have hZ := (circlePartition_pos n β ψ.ofLp i).ne'
  have hodd : ∀ j : Idx n,
      Real.exp (β * Real.cos (ψ.ofLp j - ψ.ofLp i)) * Real.sin (ψ.ofLp j - ψ.ofLp i)
        = -(Real.exp (β * Real.cos (ψ.ofLp i - ψ.ofLp j)) * Real.sin (ψ.ofLp i - ψ.ofLp j)) := by
    intro j
    rw [show ψ.ofLp j - ψ.ofLp i = -(ψ.ofLp i - ψ.ofLp j) by ring, Real.cos_neg, Real.sin_neg]
    ring
  rw [saGain_apply, gradient_neg_eucTorusEnergy, WithLp.ofLp_neg, Pi.neg_apply,
    gradient_eucTorusEnergy_apply n β hβ, Finset.sum_congr rfl fun j _ => hodd j,
    Finset.sum_neg_distrib]
  field_simp

/-- A curve of angles has the velocity `b` in `EucSpace n` exactly when each
angle has the velocity `b_i`. -/
theorem hasDerivAt_toLp_iff {θ : ℝ → Angles n} {b : EucSpace n} {t : ℝ} :
    HasDerivAt (fun s => WithLp.toLp 2 (θ s)) b t ↔
      ∀ i, HasDerivAt (fun s => θ s i) (b.ofLp i) t := by
  rw [← hasDerivAt_pi]
  let e := PiLp.continuousLinearEquiv 2 ℝ fun _ : Idx n => ℝ
  exact ⟨fun h => e.hasFDerivAt.comp_hasDerivAt t h,
    fun h => e.symm.hasFDerivAt.comp_hasDerivAt t h⟩

/-- **SA in angles is the gradient flow of `n² 𝖤_β` for the metric
`diag(Z_i)`**: along a solution `θ` of `angularSA`, for every curve `φ` of
angles through `θ(t)` with velocity `b`,

  `d/ds (n² 𝖤_β)(φ(s))|_{s=0} = Σ_i Z_i(θ(t)) θ̇_i(t) b_i`,

`eq:r3` in angles. Here `n² 𝖤_β = (1/2β) Σ_i Σ_j e^{β cos(θ_i - θ_j)}` is the
energy of §3.4, and `Σ_i Z_i a_i b_i` is `e:scalarproduct`: at
`(cos θ_i, sin θ_i)` the tangent vectors `a_i τ_i` and `b_i τ_i`, `τ_i` the
unit tangent, have the inner product `a_i b_i`. -/
theorem hasDerivAt_torusEnergy_of_angularSA (β : ℝ) (hβ : β ≠ 0) {θ : ℝ → Angles n}
    (hθ : angularSA n β θ) (t : ℝ) {φ : ℝ → Angles n} {b : Angles n} (hφ0 : φ 0 = θ t)
    (hφ : ∀ i, HasDerivAt (fun s => φ s i) (b i) 0) :
    HasDerivAt (fun s => (n : ℝ) ^ 2 * torusEnergy n β (φ s))
      (∑ i : Idx n, circlePartition n β (θ t) i * deriv (fun s => θ s i) t * b i) 0 := by
  have hγ : HasDerivAt (fun s => WithLp.toLp 2 (φ s)) (WithLp.toLp 2 b) 0 :=
    (hasDerivAt_toLp_iff n).2 hφ
  have hF := (((contDiff_eucTorusEnergy n β (k := 1)).differentiable one_ne_zero
    _).hasFDerivAt.comp_hasDerivAt 0 hγ).const_mul ((n : ℝ) ^ 2)
  refine hF.congr_deriv ?_
  have hgrad : ∀ ψ v : EucSpace n, fderiv ℝ (eucTorusEnergy n β) ψ v =
      inner ℝ (gradient (eucTorusEnergy n β) ψ) v := fun ψ v => by
    rw [gradient, InnerProductSpace.toDual_symm_apply]
  simp only [hφ0, hgrad, PiLp.inner_apply, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (Fin.pos i).ne'
  have hZ := (circlePartition_pos n β (θ t) i).ne'
  have hodd : ∀ j : Idx n,
      Real.exp (β * Real.cos (θ t j - θ t i)) * Real.sin (θ t j - θ t i)
        = -(Real.exp (β * Real.cos (θ t i - θ t j)) * Real.sin (θ t i - θ t j)) := by
    intro j
    rw [show θ t j - θ t i = -(θ t i - θ t j) by ring, Real.cos_neg, Real.sin_neg]
    ring
  rw [(hθ t i).deriv, gradient_eucTorusEnergy_apply n β hβ]
  simp only [RCLike.inner_apply, conj_trivial]
  rw [Finset.sum_congr rfl fun j _ => hodd j, Finset.sum_neg_distrib]
  field_simp

/-- The hypotheses of the theorems above are satisfiable: `β = 1 ≠ 0`; the
consensus `θ_i = c` solves `angularSA`, and the rotation `φ(s)_i = c + s`
passes through it at `s = 0` with velocity `b_i = 1`. -/
example (c : ℝ) : (1 : ℝ) ≠ 0 ∧ angularSA n 1 (fun _ _ => c) ∧
    (fun s : ℝ => fun _ : Idx n => c + s) 0 = (fun _ _ => c : ℝ → Angles n) 0 ∧
    ∀ _ : Idx n, HasDerivAt (fun s : ℝ => c + s) 1 0 :=
  ⟨one_ne_zero, fun t i => by simpa using hasDerivAt_const t c, by simp,
    fun _ => by simpa using (hasDerivAt_id (0 : ℝ)).const_add c⟩

end Perspective
end Transformer
