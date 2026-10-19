module

public import Transformer.Landscape.SaddleAvoidance.GradientDescent
public import Transformer.Landscape.SaddleAvoidance.Descent
public import Mathlib.Algebra.Module.ZLattice.Basic
public import Mathlib.LinearAlgebra.Basis.VectorSpace
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Saddle avoidance — periodic functions

A `C²` function on a finite-dimensional real inner product space `E` that is
invariant under the translations by a set `Λ` spanning `E`, such as a lattice
spanning `E` or the periods `2πℤⁿ` of a function of `n` angles, has a bounded
gradient, a bounded Hessian and a globally Lipschitz gradient
(`exists_lipschitzWith_gradient_of_periodic`). These are the hypotheses on
`f` of the gradient-descent theorems of this directory, and, through the
bound and the Lipschitz constant of `-∇f`, of `Landscape.GlobalFlow.IsFlowField`.

The integer combinations of `Λ` leave the function invariant
(`periodic_of_mem_span`). Reducing modulo the lattice spanned by a basis of
`E` taken from `Λ` moves every point into one closed ball
(`exists_add_mem_closedBall_of_span_eq_top`). The gradient and the Hessian
are invariant too (`gradient_add_eq`, `fderiv_gradient_add_eq`), so their
bounds on that compact ball hold everywhere (`exists_bound_of_periodic`), and
the mean value inequality turns the bound on the Hessian into a Lipschitz
constant of the gradient (`exists_lipschitzWith_gradient_of_isCompact`). The
same argument bounds every invariant `C¹` map and its Lipschitz constant
(`exists_lipschitzWith_of_periodic`), such as the field `G ∇f` of a flow
`ẋ = -G(x) ∇f(x)` with `G` and `f` invariant.
-/

@[expose] public section

open Metric
open scoped NNReal

namespace Transformer.Landscape.SaddleAvoidance

section Translate

variable {E : Type*} [AddCommGroup E]

/-- **Integer combinations of periods are periods**: if every translation by
an element of `Λ` leaves `f` invariant, so does every translation by an
element of the subgroup that `Λ` generates. -/
theorem periodic_of_mem_span {α : Type*} {f : E → α} {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {v : E} (hv : v ∈ Submodule.span ℤ Λ) (y : E) :
    f (y + v) = f y := by
  induction hv using Submodule.span_induction generalizing y with
  | mem v hv => exact hΛ v hv y
  | zero => rw [add_zero]
  | add u w _ _ hu hw => rw [← add_assoc, hw, hu]
  | smul c u _ hu => exact Function.Periodic.zsmul hu c y

/-- **A continuous periodic function is bounded**: if the translations by `Λ`
leave the continuous `g` invariant and every point has a translate by an
element of `Λ` in the compact set `K`, the bound of `‖g‖` on `K` holds
everywhere. -/
theorem exists_bound_of_periodic [TopologicalSpace E] {F : Type*} [SeminormedAddCommGroup F]
    {g : E → F} (hg : Continuous g) {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, g (y + v) = g y)
    {K : Set E} (hK : IsCompact K) (hKΛ : ∀ y, ∃ v ∈ Λ, y + v ∈ K) :
    ∃ C, ∀ y, ‖g y‖ ≤ C := by
  obtain ⟨C, hC⟩ := hK.exists_bound_of_continuousOn hg.continuousOn
  refine ⟨C, fun y => ?_⟩
  obtain ⟨v, hv, hyv⟩ := hKΛ y
  rw [← hΛ v hv y]
  exact hC _ hyv

end Translate

section Ball

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]

/-- **A spanning set of translations moves every point into one ball.** If
`Λ` spans `E`, there is a radius `R` such that every `y` has a translate
`y + v` with `‖y + v‖ ≤ R` by an integer combination `v` of `Λ`: subtract from
`y` the integer parts of its coordinates in a basis of `E` taken from `Λ`
(`ZSpan.fract`). -/
theorem exists_add_mem_closedBall_of_span_eq_top {Λ : Set E}
    (hspan : Submodule.span ℝ Λ = ⊤) :
    ∃ R, ∀ y : E, ∃ v ∈ Submodule.span ℤ Λ, y + v ∈ closedBall (0 : E) R := by
  obtain ⟨ι, b, hb⟩ : ∃ (ι : Type _) (b : Module.Basis ι ℝ E), Set.range b ⊆ Λ :=
    ⟨_, Module.Basis.ofSpan hspan.ge, Module.Basis.ofSpan_subset hspan.ge⟩
  have := Module.Finite.finite_basis b
  let _ : Fintype ι := Fintype.ofFinite ι
  refine ⟨∑ i, ‖b i‖, fun y => ⟨-(ZSpan.floor b y : E), ?_, ?_⟩⟩
  · exact Submodule.neg_mem _ (Submodule.span_mono hb (ZSpan.floor b y).2)
  · rw [mem_closedBall_zero_iff, ← sub_eq_add_neg, ← ZSpan.fract_apply]
    exact ZSpan.norm_fract_le b y

/-- **A `C¹` map invariant under a set of translations spanning `E` is
bounded and globally Lipschitz**, with one constant `L`: its derivative is
invariant too, and both are bounded on the ball of
`exists_add_mem_closedBall_of_span_eq_top`. -/
theorem exists_lipschitzWith_of_periodic {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    {g : E → F} (hg : ContDiff ℝ 1 g) {Λ : Set E} (hspan : Submodule.span ℝ Λ = ⊤)
    (hΛ : ∀ v ∈ Λ, ∀ y, g (y + v) = g y) :
    ∃ L : ℝ≥0, (∀ x, ‖g x‖ ≤ L) ∧ LipschitzWith L g := by
  obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
  have hper : ∀ v ∈ Submodule.span ℤ Λ, ∀ y, g (y + v) = g y :=
    fun _ hv => periodic_of_mem_span hΛ hv
  obtain ⟨M, hM⟩ := exists_bound_of_periodic hg.continuous hper (isCompact_closedBall 0 R) hR
  obtain ⟨H, hH⟩ := exists_bound_of_periodic (hg.continuous_fderiv one_ne_zero)
    (fun v hv y => by
      rw [← fderiv_comp_add_right]
      exact congrArg (fun G => fderiv ℝ G y) (funext fun x => hper v hv x))
    (isCompact_closedBall 0 R) hR
  have hHL : ∀ x, ‖fderiv ℝ g x‖ ≤ (max M H).toNNReal := fun x =>
    (hH x).trans ((le_max_right M H).trans (Real.le_coe_toNNReal _))
  exact ⟨(max M H).toNNReal, fun x => (hM x).trans ((le_max_left M H).trans
    (Real.le_coe_toNNReal _)),
    lipschitzWith_of_nnnorm_fderiv_le (hg.differentiable one_ne_zero) fun x => hHL x⟩

/-- The hypotheses of `exists_lipschitzWith_of_periodic` hold for `sin` on the
line and the period `2π`, which spans `ℝ`. -/
example : ContDiff ℝ 1 Real.sin ∧ Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
    ∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, Real.sin (y + v) = Real.sin y := by
  refine ⟨Real.contDiff_sin, Ideal.span_singleton_eq_top.2
    (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne'), fun v hv y => ?_⟩
  rw [Set.mem_singleton_iff.1 hv, Real.sin_add_two_pi]

end Ball

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- A function invariant under translation by `v` has a Hessian invariant
under it. -/
theorem fderiv_gradient_add_eq {f : E → ℝ} {v : E} (hv : ∀ y, f (y + v) = f y) (y : E) :
    fderiv ℝ (gradient f) (y + v) = fderiv ℝ (gradient f) y := by
  rw [← fderiv_comp_add_right]
  exact congrArg (fun F => fderiv ℝ F y) (funext (gradient_add_eq hv))

variable [FiniteDimensional ℝ E]

/-- **A periodic `C²` function has a bounded gradient, a bounded Hessian and
a globally Lipschitz gradient**, with one constant `L`, when the translations
by `Λ` leave it invariant and every point has a translate by an element of
`Λ` in a compact set `K`. -/
theorem exists_lipschitzWith_gradient_of_isCompact {f : E → ℝ} (hf : ContDiff ℝ 2 f)
    {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {K : Set E} (hK : IsCompact K)
    (hKΛ : ∀ y, ∃ v ∈ Λ, y + v ∈ K) :
    ∃ L : ℝ≥0, (∀ x, ‖gradient f x‖ ≤ L) ∧ (∀ x, ‖fderiv ℝ (gradient f) x‖ ≤ L) ∧
      LipschitzWith L (gradient f) := by
  have hg := contDiff_gradient hf
  obtain ⟨M, hM⟩ := exists_bound_of_periodic hg.continuous
    (fun v hv => gradient_add_eq (hΛ v hv)) hK hKΛ
  obtain ⟨H, hH⟩ := exists_bound_of_periodic (hg.continuous_fderiv one_ne_zero)
    (fun v hv => fderiv_gradient_add_eq (hΛ v hv)) hK hKΛ
  have hHL : ∀ x, ‖fderiv ℝ (gradient f) x‖ ≤ (max M H).toNNReal := fun x =>
    (hH x).trans ((le_max_right M H).trans (Real.le_coe_toNNReal _))
  refine ⟨(max M H).toNNReal, fun x => (hM x).trans ((le_max_left M H).trans
    (Real.le_coe_toNNReal _)), hHL, ?_⟩
  exact lipschitzWith_of_nnnorm_fderiv_le (hg.differentiable one_ne_zero) fun x => hHL x

/-- **A `C²` function invariant under a set of translations spanning `E` has
a bounded gradient, a bounded Hessian and a globally Lipschitz gradient**,
with one constant `L`. A lattice spanning `E` (`IsZLattice.span_top`) and
the periods `2πℤⁿ` of a function of `n` angles are such sets; neither
discreteness nor a group structure is needed. -/
theorem exists_lipschitzWith_gradient_of_periodic {f : E → ℝ} (hf : ContDiff ℝ 2 f)
    {Λ : Set E} (hspan : Submodule.span ℝ Λ = ⊤) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) :
    ∃ L : ℝ≥0, (∀ x, ‖gradient f x‖ ≤ L) ∧ (∀ x, ‖fderiv ℝ (gradient f) x‖ ≤ L) ∧
      LipschitzWith L (gradient f) := by
  obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
  exact exists_lipschitzWith_gradient_of_isCompact hf
    (fun _ hv => periodic_of_mem_span hΛ hv) (isCompact_closedBall 0 R) hR

/-- The hypotheses of the theorems above hold for `f(x) = 1 - cos x` on the
line, the period `2π`, which spans `ℝ`, its multiples, and the compact
interval `[0, 2π]`, which holds a translate of every point by a multiple of
`2π`. -/
example : let f := fun x : ℝ => 1 - Real.cos x
    let Λ := Set.range fun k : ℤ => k * (2 * Real.pi)
    ContDiff ℝ 2 f ∧ Continuous f ∧ Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      2 * (2 * Real.pi) ∈ Submodule.span ℤ {2 * Real.pi} ∧ (∀ v ∈ Λ, ∀ y, f (y + v) = f y) ∧
      IsCompact (Set.Icc 0 (2 * Real.pi)) ∧ ∀ y, ∃ v ∈ Λ, y + v ∈ Set.Icc 0 (2 * Real.pi) := by
  intro f Λ
  refine ⟨contDiff_const.sub Real.contDiff_cos, continuous_const.sub Real.continuous_cos,
    Ideal.span_singleton_eq_top.2 (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne'),
    fun v hv y => ?_, ?_, ?_, isCompact_Icc, fun y => ?_⟩
  · rw [Set.mem_singleton_iff.1 hv]
    simp [f]
  · rw [show 2 * (2 * Real.pi) = (2 : ℤ) • (2 * Real.pi) by norm_num]
    exact Submodule.smul_mem _ _ (Submodule.subset_span rfl)
  · rintro _ ⟨k, rfl⟩ y
    simp only [f, Real.cos_add_int_mul_two_pi]
  · refine ⟨_, ⟨-toIcoDiv Real.two_pi_pos 0 y, rfl⟩, ?_⟩
    have h := toIcoMod_mem_Ico Real.two_pi_pos 0 y
    rw [zero_add, ← self_sub_toIcoDiv_zsmul, zsmul_eq_mul] at h
    show y + ((-toIcoDiv Real.two_pi_pos 0 y : ℤ) : ℝ) * (2 * Real.pi) ∈ _
    rw [Int.cast_neg, neg_mul, ← sub_eq_add_neg]
    exact Set.Ico_subset_Icc_self h

end Transformer.Landscape.SaddleAvoidance
