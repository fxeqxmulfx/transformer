module

public import Transformer.Clusters.Section8_Polytope
public import Mathlib.Analysis.Calculus.MeanValue
public import Mathlib.Analysis.Normed.Group.Uniform

/-!
# Uniform continuity of rescaled token paths

ArXiv:2305.05465v6, §8.1, Step 3 of `t:Idcase11`. The source observes
that bounded token positions imply bounded velocities in `e:dynIdzi`.
We make the bound explicit: inside a ball of radius `R`, the drift has
norm at most `2R`, since it is a convex combination of displacements.
The mean value estimate then gives a Lipschitz bound for each token and
for the entire finite tuple on the future half-line.

The tuple uses the product supremum norm, so its bound has no extra token
count. Empty token tuples are also covered. Every dynamical input is an
actual solution of `e:dynIdzi`; no regularity assumption is added to it.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- Attention coefficients in `e:dynIdzi` are nonnegative at every
temperature. Source: arXiv:2305.05465v6, §8.1, `p:noninc`. -/
theorem scaledAttention_nonneg (A : ParamMatrix d) (c : ℝ)
    (Y : Idx n → EucSpace d) (i j : Idx n) : 0 ≤ scaledAttention A c Y i j :=
  Perspective.softmaxWeight_nonneg _ _

/-- Each indexed attention row sums to one; the row index witnesses
nonemptiness. Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
theorem sum_scaledAttention (A : ParamMatrix d) (c : ℝ)
    (Y : Idx n → EucSpace d) (i : Idx n) : ∑ j : Idx n, scaledAttention A c Y i j = 1 :=
  Perspective.sum_softmaxWeight (Fin.pos_iff_nonempty.mpr ⟨i⟩) _

/-- The drift of a configuration in a radius-`R` ball has norm at most
`2R`. Source: arXiv:2305.05465v6, §8.1, the bounded-speed observation
at the start of Step 3 of `t:Idcase11`. -/
theorem norm_scaledAttentionDrift_le (A : ParamMatrix d) (c : ℝ)
    (Y : Idx n → EucSpace d) (R : ℝ) (hY : ∀ j, ‖Y j‖ ≤ R) (i : Idx n) :
    ‖∑ j : Idx n, scaledAttention A c Y i j • (Y j - Y i)‖ ≤ 2 * R := by
  calc
    _ ≤ ∑ j : Idx n, scaledAttention A c Y i j * (2 * R) := by
      apply (norm_sum_le _ _).trans
      apply Finset.sum_le_sum
      intro j _
      rw [norm_smul, Real.norm_of_nonneg (scaledAttention_nonneg A c Y i j)]
      apply mul_le_mul_of_nonneg_left _ (scaledAttention_nonneg A c Y i j)
      exact (norm_sub_le _ _).trans (by linarith [hY i, hY j])
    _ = 2 * R := by rw [← Finset.sum_mul, sum_scaledAttention, one_mul]

/-- The position-bound hypothesis holds for an origin configuration
with two tokens and radius one. Source: arXiv:2305.05465v6, §8.1. -/
example : ∀ j : Idx 2, ‖(fun _ : Idx 2 => (0 : EucSpace 1)) j‖ ≤ (1 : ℝ) := by simp

/-- Each actual token path is continuous, as a consequence of the ODE.
Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
theorem IdRescaledDynamics.continuous_token (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (i : Idx n) :
    Continuous (fun t => Z t i) :=
  continuous_iff_continuousAt.mpr fun t => (hZ t i).continuousAt

/-- The continuity input is realized by stationary unit tokens;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := idRescaledDynamics_const 1 _

/-- The entire finite token tuple is continuous.
Source: arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
theorem IdRescaledDynamics.continuous_tuple (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) : Continuous Z :=
  continuous_pi fun i => hZ.continuous_token A Z i

/-- A stationary two-token solution satisfies the tuple-continuity input;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example (z : EucSpace 1) : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => z) := idRescaledDynamics_const 1 z

/-- The bounded velocity yields a displacement bound over any future
time interval. Source: arXiv:2305.05465v6, §8.1, Step 3 of `t:Idcase11`. -/
theorem IdRescaledDynamics.norm_sub_le (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (R : ℝ)
    (hR : ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖Z t i‖ ≤ R) (i : Idx n)
    {t s : ℝ} (ht : 0 ≤ t) (hts : t ≤ s) : ‖Z s i - Z t i‖ ≤ (2 * R) * (s - t) :=
  norm_image_sub_le_of_norm_deriv_le_segment' (f := fun u => Z u i)
    (fun u _ => (hZ u i).hasDerivWithinAt)
    (fun u hu => norm_scaledAttentionDrift_le A (Real.exp (2 * u)) (Z u) R
      (fun j => hR j u (ht.trans hu.1)) i) s ⟨hts, le_rfl⟩

/-- All dynamical, radius and interval inputs hold for the stationary
origin path, with the interval `[0,1]`. Source: arXiv:2305.05465v6, §8.1. -/
example : IdRescaledDynamics (1 : ParamMatrix 1) (fun _ (_ : Idx 2) => 0) ∧
    (∀ (i : Idx 2) (t : ℝ), 0 ≤ t → ‖(fun _ (_ : Idx 2) => (0 : EucSpace 1)) t i‖ ≤ 1) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by
  exact ⟨idRescaledDynamics_const 1 _, fun _ _ _ => by simp, le_rfl, zero_le_one⟩

/-- The finite tuple is `2R`-Lipschitz on future times, in the product
supremum norm. Source: arXiv:2305.05465v6, §8.1, Step 3 of `t:Idcase11`. -/
theorem IdRescaledDynamics.lipschitzOn_tuple (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) (R : ℝ) (hR0 : 0 ≤ R)
    (hR : ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖Z t i‖ ≤ R) :
    LipschitzOnWith ⟨2 * R, by positivity⟩ Z (Set.Ici 0) := by
  apply lipschitzOnWith_iff_norm_sub_le.mpr
  intro t ht s hs
  change ‖Z t - Z s‖ ≤ (2 * R) * ‖t - s‖
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg (by positivity) (norm_nonneg _))).mpr
  intro i
  change ‖Z t i - Z s i‖ ≤ (2 * R) * ‖t - s‖
  rcases le_or_gt s t with hst | hst
  · rw [Real.norm_of_nonneg (sub_nonneg.mpr hst)]
    exact hZ.norm_sub_le A Z R hR i hs hst
  · rw [norm_sub_rev, Real.norm_eq_abs, abs_of_nonpos (sub_nonpos.mpr hst.le)]
    have hbound := hZ.norm_sub_le A Z R hR i ht hst.le
    linarith

/-- The radius and ODE inputs are jointly satisfied by two stationary
origin tokens in a unit ball. Source: arXiv:2305.05465v6, §8.1. -/
example : IdRescaledDynamics (1 : ParamMatrix 1) (fun _ (_ : Idx 2) => 0) ∧
    (0 : ℝ) ≤ 1 ∧ ∀ (i : Idx 2) (t : ℝ), 0 ≤ t →
      ‖(fun _ (_ : Idx 2) => (0 : EucSpace 1)) t i‖ ≤ 1 :=
  ⟨idRescaledDynamics_const 1 _, zero_le_one, fun _ _ _ => by simp⟩

/-- A common future position bound can be chosen nonnegative, even for
the empty tuple. Source: arXiv:2305.05465v6, §8.1, `c:theziunifbdd`. -/
theorem IdRescaledDynamics.exists_nonneg_bound (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) :
    ∃ R : ℝ, 0 ≤ R ∧ ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖Z t i‖ ≤ R := by
  obtain ⟨R, hR⟩ := exists_bound_idRescaled A Z hZ
  exact ⟨max R 0, le_max_right _ _, fun i t ht => (hR i t ht).trans (le_max_left _ _)⟩

/-- The boundedness input holds for stationary unit tokens;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := idRescaledDynamics_const 1 _

/-- Uniform continuity on the future half-line follows from the actual
bounded dynamics. Source: arXiv:2305.05465v6, §8.1, `c:theziunifbdd` and
Step 3 of `t:Idcase11`; no separate boundedness hypothesis is assumed. -/
theorem IdRescaledDynamics.uniformContinuousOn_tuple (A : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : IdRescaledDynamics A Z) :
    UniformContinuousOn Z (Set.Ici 0) := by
  obtain ⟨R, hR0, hR⟩ := hZ.exists_nonneg_bound A Z
  exact (hZ.lipschitzOn_tuple A Z R hR0 hR).uniformContinuousOn

/-- The uniform-continuity input holds for genuine stationary unit tokens;
arXiv:2305.05465v6, §8.1, `e:dynIdzi`. -/
example : IdRescaledDynamics (1 : ParamMatrix 1)
    (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) := idRescaledDynamics_const 1 _

end Transformer.Clusters
