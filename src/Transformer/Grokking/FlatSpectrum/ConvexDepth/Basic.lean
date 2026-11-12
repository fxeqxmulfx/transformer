module

public import Transformer.Grokking.FlatSpectrum.ConvexRecall.Loss
public import Mathlib.Analysis.Normed.Operator.Basic

/-!
# Additive baseline: convex training of injections at arbitrary depth

Source: `review/convex_depth.md`, CD1. This baseline uses fixed linear transports
and fixed input features, with a separate learned linear injection at each depth. It
differs from arXiv:2512.24880v2, Eq. (3), and DiffGPT's phase-mHC: neither routing nor
injection features depend on the evolving learned state. The affine dependence below
is proved from the recurrence; it is also the architecture's expressivity limitation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {X E F C S : Type*}
variable [NormedAddCommGroup E] [NormedSpace ℝ E]
variable [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Fixed transport, fixed input features, and independently trained depth injections
(CD1). The initial state and residual scales are fixed during optimization. -/
noncomputable def run (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (W : ℕ → F →L[ℝ] E) (x : X) : ℕ → E
  | 0 => initial x
  | d + 1 => U d (run U φ α initial W x d) + α d • W d (φ d x)

/-- All depth parameters enter affinely even though the hidden state is propagated
through every intervening block (CD1). The fixed initial state requires `a+b=1`. -/
theorem run_affine (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (W V : ℕ → F →L[ℝ] E) (x : X) (d : ℕ)
    (a b : ℝ) (hab : a + b = 1) :
    run U φ α initial (a • W + b • V) x d =
      a • run U φ α initial W x d + b • run U φ α initial V x d := by
  induction d with
  | zero => simp only [run, ← add_smul, hab, one_smul]
  | succ d ih =>
    simp only [run, ih, map_add, map_smul, Pi.add_apply, Pi.smul_apply,
      add_apply, smul_apply, smul_add]
    simp only [smul_comm (α d) a, smul_comm (α d) b]
    abel

/-- Fixed linear readout and ordinary observed CE; no softmax-probability loss or
auxiliary layerwise targets are substituted (CD1). -/
noncomputable def loss [Fintype C] [Fintype S]
    (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ) (initial : X → E)
    (readout : E →ₗ[ℝ] (C → ℝ)) (inputs : S → X) (labels : S → C) (d : ℕ)
    (W : ℕ → F →L[ℝ] E) : ℝ :=
  ∑ s, crossEntropy (readout (run U φ α initial W (inputs s) d)) (labels s)

/-- End-to-end ordinary CE is jointly convex in every depth injection (CD1).
This does not permit the fixed readout, transports or feature maps to train jointly. -/
theorem loss_convex [Fintype C] [Fintype S]
    (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ) (initial : X → E)
    (readout : E →ₗ[ℝ] (C → ℝ)) (inputs : S → X) (labels : S → C) (d : ℕ) :
    ConvexOn ℝ Set.univ (loss U φ α initial readout inputs labels d) := by
  refine ⟨convex_univ, ?_⟩
  intro W _ V _ a b ha hb hab
  simp only [loss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  rw [run_affine U φ α initial W V _ d a b hab, map_add, map_smul, map_smul]
  exact (ConvexRecall.convex_crossEntropy (labels s)).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- Separate operator-norm upper bounds are convex spectral budgets, not equal-
singular-value or orthogonality equalities on the trained injections (CD1). -/
def budget (d : ℕ) (ρ : ℕ → ℝ) : Set (ℕ → F →L[ℝ] E) :=
  {W | ∀ k < d, ‖W k‖ ≤ ρ k}

/-- The complete depth budget is a convex feasible set (CD1). -/
theorem budget_convex (d : ℕ) (ρ : ℕ → ℝ) :
    Convex ℝ (budget (E := E) (F := F) d ρ) := by
  intro W hW V hV a b ha hb hab k hk
  change ‖a • W k + b • V k‖ ≤ ρ k
  calc
    _ ≤ ‖a • W k‖ + ‖b • V k‖ := norm_add_le _ _
    _ = a * ‖W k‖ + b * ‖V k‖ := by
      rw [norm_smul, norm_smul, Real.norm_of_nonneg ha, Real.norm_of_nonneg hb]
    _ ≤ a * ρ k + b * ρ k := add_le_add
      (mul_le_mul_of_nonneg_left (hW k hk) ha)
      (mul_le_mul_of_nonneg_left (hV k hk) hb)
    _ = ρ k := by rw [← add_mul, hab, one_mul]

/-- Nonnegative radii give a concrete feasible starting point at every depth (CD1). -/
theorem zero_mem_budget (d : ℕ) (ρ : ℕ → ℝ) (hρ : ∀ k < d, 0 ≤ ρ k) :
    (0 : ℕ → F →L[ℝ] E) ∈ budget d ρ := by
  intro k hk
  simpa using hρ k hk

/-- Spectral budgets preserve joint convexity of the same ordinary CE (CD1). -/
theorem budget_loss_convex [Fintype C] [Fintype S]
    (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ) (initial : X → E)
    (readout : E →ₗ[ℝ] (C → ℝ)) (inputs : S → X) (labels : S → C)
    (d : ℕ) (ρ : ℕ → ℝ) :
    ConvexOn ℝ (budget d ρ) (loss U φ α initial readout inputs labels d) := by
  refine ⟨budget_convex d ρ, ?_⟩
  intro W _ V _ a b ha hb hab
  exact (loss_convex U φ α initial readout inputs labels d).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

example := run_affine (fun _ => ContinuousLinearMap.id ℝ ℝ) (fun _ (x : ℝ) => x)
  (fun _ => 1) id (fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ => 2 • ContinuousLinearMap.id ℝ ℝ) 1 3 (1 / 3) (2 / 3) (by norm_num)

example := zero_mem_budget (E := ℝ) (F := ℝ) 3 (fun _ => 1)
  (by intro k hk; norm_num)

/-- Two nonzero, separately learned injections both affect the actual final state
(CD1). This is not a zero-branch embedding of a shallow model. -/
theorem two_active_injections (a b x : ℝ) :
    run (fun _ => ContinuousLinearMap.id ℝ ℝ) (fun _ (y : ℝ) => y) (fun _ => 1)
      id (fun k => (if k = 0 then a else b) • ContinuousLinearMap.id ℝ ℝ) x 2 =
      (1 + a + b) * x := by
  simp [run]
  ring

/-- The final effect of a parameter change is linear and independent of the old
parameters (CD1). This exact identity identifies the limitation: the recurrence
does not learn interactions between features produced by different trained layers. -/
theorem run_difference_linear (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (W V : ℕ → F →L[ℝ] E) (x : X) (d : ℕ) :
    run U φ α initial W x d - run U φ α initial V x d =
      run U φ α (fun _ => 0) (W - V) x d := by
  induction d with
  | zero => simp [run]
  | succ d ih =>
    simp only [run, ← ih, map_sub, Pi.sub_apply, sub_apply, smul_sub]
    abel

/-- The affine restriction is checked on two nonzero parameter families, not
only at a zero initialization. Their change propagates through three blocks. -/
example :
    run (fun _ => ContinuousLinearMap.id ℝ ℝ) (fun _ (x : ℝ) => x)
        (fun _ => 1) id (fun _ => 2 • ContinuousLinearMap.id ℝ ℝ) 1 3 -
      run (fun _ => ContinuousLinearMap.id ℝ ℝ) (fun _ (x : ℝ) => x)
        (fun _ => 1) id (fun _ => ContinuousLinearMap.id ℝ ℝ) 1 3 = 3 := by
  norm_num [run]

end Transformer.Grokking.FlatSpectrum.ConvexDepth
