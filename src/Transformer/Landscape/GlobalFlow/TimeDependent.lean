/-
# Global trajectories for bounded time-dependent Lipschitz fields

A field continuous in time and uniformly Lipschitz in space, with a
uniform global bound, has one complete trajectory through each initial
point in a complete real normed space.

Picard–Lindelöf supplies trajectories on each compact time interval.
Uniform spatial uniqueness makes the trajectories agree on overlaps.
Choosing an interval around each observation time then glues them into
one differentiable curve on the entire real line.

This is a standard ODE prerequisite for the characteristics of the
attention equation in arXiv:2411.04551v3, §1, `eq: ips` and
`eq: cauchy.pb`. Time dependence is explicit in the actual field.
-/

import Transformer.Landscape.GlobalFlow.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

open Real Set Metric Filter Topology
open scoped NNReal

namespace Transformer.Landscape.GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Uniform spatial Lipschitz regularity gives uniqueness on an open
time interval. Standard Picard–Lindelöf uniqueness, as used for the
characteristics of arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem timeDependent_eqOn_Ioo (F : ℝ → E → E) (K : ℝ≥0)
    (hK : ∀ t, LipschitzWith K (F t)) (f g : ℝ → E)
    (a b t₀ : ℝ) (ht₀ : t₀ ∈ Ioo a b)
    (hf : ∀ t ∈ Ioo a b, HasDerivAt f (F t (f t)) t)
    (hg : ∀ t ∈ Ioo a b, HasDerivAt g (F t (g t)) t) (h0 : f t₀ = g t₀) :
    EqOn f g (Ioo a b) :=
  ODE_solution_unique_of_mem_Ioo (v := F) (s := fun _ => univ)
    (fun t _ => (hK t).lipschitzOnWith) ht₀
    (fun t ht => ⟨hf t ht, mem_univ _⟩) (fun t ht => ⟨hg t ht, mem_univ _⟩) h0

/-- Complete trajectories of the same time-dependent field are fixed
by their initial value. Standard ODE uniqueness, used for
arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem timeDependent_unique (F : ℝ → E → E) (K : ℝ≥0)
    (hK : ∀ t, LipschitzWith K (F t)) (f g : ℝ → E)
    (hf : ∀ t, HasDerivAt f (F t (f t)) t)
    (hg : ∀ t, HasDerivAt g (F t (g t)) t) (h0 : f 0 = g 0) : f = g := by
  funext t
  exact timeDependent_eqOn_Ioo F K hK f g (-(|t| + 1)) (|t| + 1) 0
    ⟨by linarith [abs_nonneg t], by linarith [abs_nonneg t]⟩
    (fun s _ => hf s) (fun s _ => hg s) h0
    ⟨by linarith [neg_abs_le t], by linarith [le_abs_self t]⟩

variable [CompleteSpace E]

/-- A bounded time-dependent field admits a trajectory on every finite
open interval around zero. Standard Picard–Lindelöf existence, used for
arXiv:2411.04551v3, §1, the characteristic construction. -/
theorem exists_timeDependent_local (F : ℝ → E → E) (K : ℝ≥0) (C : ℝ) (hC : 0 ≤ C)
    (hK : ∀ t, LipschitzWith K (F t)) (hF : ∀ x, Continuous (fun t => F t x))
    (hbound : ∀ t x, ‖F t x‖ ≤ C) (x₀ : E) (T : ℝ) (hT : 0 < T) :
    ∃ f : ℝ → E, f 0 = x₀ ∧ ∀ t ∈ Ioo (-T) T, HasDerivAt f (F t (f t)) t := by
  have h0 : (0 : ℝ) ∈ Icc (-T) T := ⟨by linarith, hT.le⟩
  have hPL : IsPicardLindelof F ⟨0, h0⟩ x₀ (C * T).toNNReal 0 C.toNNReal K := by
    refine ⟨fun t _ => (hK t).lipschitzOnWith,
      fun x _ => (hF x).continuousOn, ?_, ?_⟩
    · intro t _ x _
      rw [Real.coe_toNNReal C hC]
      exact hbound t x
    · rw [Real.coe_toNNReal C hC, Real.coe_toNNReal _ (mul_nonneg hC hT.le)]
      simp only [NNReal.coe_zero, sub_zero, zero_sub, neg_neg, max_self]
      exact le_rfl
  obtain ⟨f, hinit, hf⟩ := hPL.exists_eq_forall_mem_Icc_hasDerivWithinAt₀
  exact ⟨f, hinit, fun t ht => (hf t (Ioo_subset_Icc_self ht)).hasDerivAt (Icc_mem_nhds ht.1 ht.2)⟩

/-- Uniqueness glues the finite-interval trajectories into a complete
actual trajectory. Standard global ODE argument, used for
arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem exists_timeDependent_global (F : ℝ → E → E) (K : ℝ≥0) (C : ℝ) (hC : 0 ≤ C)
    (hK : ∀ t, LipschitzWith K (F t)) (hF : ∀ x, Continuous (fun t => F t x))
    (hbound : ∀ t x, ‖F t x‖ ≤ C) (x₀ : E) :
    ∃ f : ℝ → E, f 0 = x₀ ∧ ∀ t, HasDerivAt f (F t (f t)) t := by
  choose α hα0 hα using fun N : ℕ =>
    exists_timeDependent_local F K C hC hK hF hbound x₀ (N + 1) (by positivity)
  have hagree (N N' : ℕ) (s : ℝ) (hs : |s| < min (N : ℝ) N' + 1) : α N s = α N' s := by
    let M := min (N : ℝ) N' + 1
    have hM : 0 < M := by dsimp [M]; positivity
    have hsub (A : ℕ) (hA : M ≤ A + 1) : Ioo (-M) M ⊆ Ioo (-((A : ℝ) + 1)) (A + 1) :=
      Ioo_subset_Ioo (by linarith) hA
    exact timeDependent_eqOn_Ioo F K hK (α N) (α N') (-M) M 0 ⟨by linarith, hM⟩
      (fun t ht => hα N t (hsub N (by dsimp [M]; simp) ht))
      (fun t ht => hα N' t (hsub N' (by dsimp [M]; simp) ht))
      ((hα0 N).trans (hα0 N').symm) (abs_lt.mp hs)
  refine ⟨fun t => α ⌈|t|⌉₊ t, by simpa using hα0 0, fun t => ?_⟩
  let N := ⌈|t|⌉₊
  have htN : |t| < N + 1 := (Nat.le_ceil _).trans_lt (by linarith)
  have hI : t ∈ Ioo (-((N : ℝ) + 1)) (N + 1) :=
    ⟨by linarith [neg_abs_le t], by linarith [le_abs_self t]⟩
  have heq : (fun s => α ⌈|s|⌉₊ s) =ᶠ[nhds t] α N := by
    filter_upwards [Ioo_mem_nhds hI.1 hI.2] with s hs
    apply hagree
    rw [← min_add_add_right]
    exact lt_min_iff.mpr ⟨(Nat.le_ceil _).trans_lt (by linarith),
      abs_lt.mpr ⟨by linarith [hs.1], hs.2⟩⟩
  exact (hα N t hI).congr_of_eventuallyEq heq

/-- Translating the clock constructs an actual complete trajectory
through any prescribed time and point. Standard time-dependent ODE
existence, used for the invertible characteristics in arXiv:2411.04551v3, §1. -/
theorem exists_timeDependent_through (F : ℝ → E → E) (K : ℝ≥0) (C : ℝ) (hC : 0 ≤ C)
    (hK : ∀ t, LipschitzWith K (F t)) (hF : ∀ x, Continuous (fun t => F t x))
    (hbound : ∀ t x, ‖F t x‖ ≤ C) (t₀ : ℝ) (x₀ : E) :
    ∃ f : ℝ → E, f t₀ = x₀ ∧ ∀ t, HasDerivAt f (F t (f t)) t := by
  obtain ⟨f, h0, hf⟩ := exists_timeDependent_global (fun t x => F (t₀ + t) x) K C hC
    (fun t => hK (t₀ + t)) (fun x => (hF x).comp (continuous_const.add continuous_id))
    (fun t x => hbound (t₀ + t) x) x₀
  refine ⟨fun t => f (t - t₀), by simpa only [sub_self] using h0, fun t => ?_⟩
  have hd := (hf (t - t₀)).scomp t ((hasDerivAt_id t).sub_const t₀)
  have he : t₀ + (t - t₀) = t := by ring
  simpa only [Function.comp_def, id_eq, one_smul, he] using hd

/-- Every initial point has exactly one actual complete trajectory for
a bounded field continuous in time and uniformly Lipschitz in space.
Standard ODE well-posedness, used for arXiv:2411.04551v3, §1. -/
theorem exists_unique_timeDependent_global (F : ℝ → E → E) (K : ℝ≥0) (C : ℝ) (hC : 0 ≤ C)
    (hK : ∀ t, LipschitzWith K (F t)) (hF : ∀ x, Continuous (fun t => F t x))
    (hbound : ∀ t x, ‖F t x‖ ≤ C) (x₀ : E) :
    ∃! f : ℝ → E, f 0 = x₀ ∧ ∀ t, HasDerivAt f (F t (f t)) t := by
  obtain ⟨f, h0, hf⟩ := exists_timeDependent_global F K C hC hK hF hbound x₀
  refine ⟨f, ⟨h0, hf⟩, fun g hg => ?_⟩
  exact timeDependent_unique F K hK g f hg.2 hf (hg.1.trans h0.symm)

/-- An actual field varying in both arguments is bounded and uniformly
Lipschitz, and has the nonconstant complete trajectory `sin(t)`. It
realizes every existence, uniqueness and interval hypothesis jointly. -/
example : ∃ F : ℝ → ℝ → ℝ,
    (∀ t, LipschitzWith 1 (F t)) ∧ (∀ x, Continuous (fun t => F t x)) ∧
    (∀ t x, ‖F t x‖ ≤ 2) ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ∈ Ioo (-1 : ℝ) 1 ∧
    (∀ t, HasDerivAt sin (F t (sin t)) t) ∧ sin (0 : ℝ) = sin 0 := by
  let F (t x : ℝ) := cos t + sin (x - sin t)
  refine ⟨F, ?_, ?_, ?_, by norm_num, zero_lt_one, by norm_num, ?_, rfl⟩
  · intro t
    apply LipschitzWith.of_dist_le_mul
    intro x y
    have hh := Real.lipschitzWith_sin.dist_le_mul (x - sin t) (y - sin t)
    have hd : (x - sin t) - (y - sin t) = x - y := by ring
    have he : F t x - F t y = sin (x - sin t) - sin (y - sin t) := by dsimp [F]; ring
    simpa only [NNReal.coe_one, one_mul, Real.dist_eq, he, hd] using hh
  · intro x
    exact continuous_cos.add (continuous_sin.comp (continuous_const.sub continuous_sin))
  · intro t x
    calc
      _ ≤ |cos t| + |sin (x - sin t)| := abs_add_le _ _
      _ ≤ 1 + 1 := add_le_add (abs_cos_le_one _) (abs_sin_le_one _)
      _ = 2 := by norm_num
  · intro t
    simpa only [F, sub_self, sin_zero, add_zero] using hasDerivAt_sin t

end Transformer.Landscape.GlobalFlow
