/-
# Attention's forward pass and Frank-Wolfe — an open set of token mergers

The null-set argument in the proof of lem:singleLeader must allow singular
updates of the configuration. At V = I₁, P = I₁/2 and B = -I₁, every pair
of tokens a > 0 > b merges at its midpoint in one step. Thus a nonempty
open set of initial configurations has an image of Lebesgue measure zero.

Source: arXiv:2508.09628v1, §2.1, the proof of lem:singleLeader, and
§1.2, eq:hardmax.dynamics.V.
-/

import Transformer.FrankWolfe.Section2_LeaderNull

open MeasureTheory

namespace Transformer.FrankWolfe

/-- With V = I₁, P = I₁/2 and B = -I₁, two tokens of opposite signs each
select the other token and both land at (a + b)/2. This is the singular
configuration update omitted by the source's null-set argument.
Source: arXiv:2508.09628v1, §1.2, eq:hardmax.dynamics.V, and §2.1,
the proof of lem:singleLeader. -/
theorem hardmaxAverageStep_opposite_signs_eq_midpoint (a b : ℝ) (ha : 0 < a) (hb : b < 0) :
    let X : Idx 2 → EucSpace 1 :=
      ![EuclideanSpace.single 0 a, EuclideanSpace.single 0 b]
    ∀ i : Idx 2,
      hardmaxAverageStep ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace 1))
        (-ContinuousLinearMap.id ℝ (EucSpace 1)) X i =
      EuclideanSpace.single 0 ((a + b) / 2) := by
  intro X
  have hab : a * b < a * a := by
    nlinarith [mul_pos ha ha, mul_neg_of_pos_of_neg ha hb]
  have hba : b * a < b * b := by
    nlinarith [mul_pos_of_neg_of_neg hb hb, mul_neg_of_neg_of_pos hb ha]
  have h0 : leaderSet (-ContinuousLinearMap.id ℝ (EucSpace 1)) X 0 =
      {EuclideanSpace.single 0 b} := by
    simp [leaderSet, X, Finset.univ_fin2, Fin.forall_fin_two, Finset.filter_insert,
      Finset.filter_singleton, inner_neg_left, EuclideanSpace.inner_single_left,
      hab.le, not_le.mpr hab]
  have h1 : leaderSet (-ContinuousLinearMap.id ℝ (EucSpace 1)) X 1 =
      {EuclideanSpace.single 0 a} := by
    simp [leaderSet, X, Finset.univ_fin2, Fin.forall_fin_two, Finset.filter_insert,
      Finset.filter_singleton, inner_neg_left, EuclideanSpace.inner_single_left,
      hba.le, not_le.mpr hba]
  intro i
  fin_cases i
  · change hardmaxAverageStep _ _ X 0 = _
    rw [hardmaxAverageStep, h0]
    ext j
    fin_cases j
    simp [X]
    ring
  · change hardmaxAverageStep _ _ X 1 = _
    rw [hardmaxAverageStep, h1]
    ext j
    fin_cases j
    simp [X]
    ring

/-- The hypotheses of the midpoint formula hold at a = 1 and b = -1.
The preconditioner is that of the source's value matrix V = I₁, and the
key-query matrix B = -I₁ is invertible.
Source: arXiv:2508.09628v1, §1.2, eq:hardmax.dynamics.V, and §2.1. -/
example : (0 : ℝ) < 1 ∧ (-1 : ℝ) < 0 ∧
    IsPreconditioner (ContinuousLinearMap.id ℝ (EucSpace 1))
      ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    Function.Bijective (-ContinuousLinearMap.id ℝ (EucSpace 1)) := by
  refine ⟨by norm_num, by norm_num, ?_, neg_bijective⟩
  intro x
  ext i
  simp
  ring

/-- A counterexample to the measure-preservation argument used to iterate
lem:singleLeader: at V = I₁ and B = -I₁, the joint hardmax update maps an
open set of positive measure into the null diagonal of two-token tuples.

The source asserts that piecewise affine updates cannot send sets of
positive measure to null sets. The coupling through the evolving leader
sets allows a singular affine branch, as the midpoint formula shows.

Source: arXiv:2508.09628v1, §2.1, the proof of lem:singleLeader. -/
theorem hardmaxAverageStep_maps_open_set_to_null :
    ∃ U : Set (Idx 2 → EucSpace 1), IsOpen U ∧ 0 < volume U ∧
      volume ((fun X : Idx 2 → EucSpace 1 => fun i =>
        hardmaxAverageStep ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace 1))
          (-ContinuousLinearMap.id ℝ (EucSpace 1)) X i) '' U) = 0 := by
  let U : Set (Idx 2 → EucSpace 1) := {X | 0 < X 0 0 ∧ X 1 0 < 0}
  have hU : IsOpen U :=
    (isOpen_lt continuous_const (by fun_prop : Continuous fun X : Idx 2 → EucSpace 1 =>
      X 0 0)).inter
    (isOpen_lt (by fun_prop : Continuous fun X : Idx 2 → EucSpace 1 => X 1 0)
      continuous_const)
  have hUn : U.Nonempty :=
    ⟨![EuclideanSpace.single 0 1, EuclideanSpace.single 0 (-1)], by simp [U]⟩
  let f : (Idx 2 → EucSpace 1) →ₗ[ℝ] EucSpace 1 :=
    (LinearMap.proj 0 : (Idx 2 → EucSpace 1) →ₗ[ℝ] EucSpace 1) - LinearMap.proj 1
  have hproper : f.ker ≠ ⊤ := by
    intro htop
    have hmem : (![EuclideanSpace.single 0 1, (0 : EucSpace 1)] : Idx 2 → EucSpace 1)
        ∈ f.ker := by rw [htop]; trivial
    have hzero := LinearMap.mem_ker.mp hmem
    have hcoord := congrArg (fun z : EucSpace 1 => z 0) hzero
    simp [f, LinearMap.proj] at hcoord
  refine ⟨U, hU, hU.measure_pos volume hUn, ?_⟩
  apply measure_mono_null ?_ (Measure.addHaar_submodule volume f.ker hproper)
  rintro Y ⟨X, hX, rfl⟩
  have hrepr : X = ![EuclideanSpace.single 0 (X 0 0), EuclideanSpace.single 0 (X 1 0)] := by
    funext i
    fin_cases i <;> ext j <;> fin_cases j <;> simp
  have hmerge (i : Idx 2) :
      hardmaxAverageStep ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace 1))
        (-ContinuousLinearMap.id ℝ (EucSpace 1)) X i =
      EuclideanSpace.single 0 ((X 0 0 + X 1 0) / 2) := by
    have h := hardmaxAverageStep_opposite_signs_eq_midpoint (X 0 0) (X 1 0) hX.1 hX.2 i
    simpa only [← hrepr] using h
  change
    hardmaxAverageStep ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace 1))
      (-ContinuousLinearMap.id ℝ (EucSpace 1)) X 0 -
    hardmaxAverageStep ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace 1))
      (-ContinuousLinearMap.id ℝ (EucSpace 1)) X 1 = 0
  rw [hmerge, hmerge, sub_self]

end Transformer.FrankWolfe
