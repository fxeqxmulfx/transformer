module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Basic

/-!
# Depth bounds without products of learned operator norms

Source: `review/convex_depth.md`, CD2. Fixed nonexpansive transports propagate the state;
learned injections obey convex spectral upper bounds. The accumulated norm and parameter
perturbation bounds are additive in depth. No claim is made about a dynamic mHC router,
nonlinear trained branch, floating-point execution, or convergence of an optimizer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators

variable {X E F : Type*}
variable [NormedAddCommGroup E] [NormedSpace ℝ E]
variable [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- A norm-preserving or contracting skip path gives an additive depth budget (CD2).
The feature norm is evaluated on the fixed supplied input, never on a hidden state. -/
theorem run_norm_le (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (W : ℕ → F →L[ℝ] E) (x : X) (ρ : ℕ → ℝ)
    (hU : ∀ k y, ‖U k y‖ ≤ ‖y‖) (d : ℕ) (hW : W ∈ budget d ρ) :
    ‖run U φ α initial W x d‖ ≤
      ‖initial x‖ + ∑ k ∈ Finset.range d, ‖α k‖ * ρ k * ‖φ k x‖ := by
  induction d with
  | zero => simp [run]
  | succ d ih =>
    have hd := ih (fun k hk => hW k (Nat.lt_trans hk (Nat.lt_succ_self d)))
    have hinj : ‖W d (φ d x)‖ ≤ ρ d * ‖φ d x‖ :=
      (W d).le_opNorm _ |>.trans
        (mul_le_mul_of_nonneg_right (hW d (Nat.lt_succ_self d)) (norm_nonneg _))
    calc
      _ ≤ ‖U d (run U φ α initial W x d)‖ + ‖α d • W d (φ d x)‖ :=
        norm_add_le _ _
      _ ≤ ‖run U φ α initial W x d‖ + ‖α d‖ * (ρ d * ‖φ d x‖) := by
        rw [norm_smul]
        exact add_le_add (hU d _) (mul_le_mul_of_nonneg_left hinj (norm_nonneg _))
      _ ≤ (‖initial x‖ + ∑ k ∈ Finset.range d, ‖α k‖ * ρ k * ‖φ k x‖) +
          ‖α d‖ * (ρ d * ‖φ d x‖) := add_le_add hd le_rfl
      _ = _ := by rw [Finset.sum_range_succ]; ring

/-- Parameter changes add across depth rather than multiplying changing block
Jacobians (CD2). Both runs share the same fixed source features and transports. -/
theorem run_difference_le (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (W V : ℕ → F →L[ℝ] E) (x : X)
    (hU : ∀ k y, ‖U k y‖ ≤ ‖y‖) (d : ℕ) :
    ‖run U φ α initial W x d - run U φ α initial V x d‖ ≤
      ∑ k ∈ Finset.range d, ‖α k‖ * ‖W k - V k‖ * ‖φ k x‖ := by
  induction d with
  | zero => simp [run]
  | succ d ih =>
    have he : run U φ α initial W x (d + 1) - run U φ α initial V x (d + 1) =
        U d (run U φ α initial W x d - run U φ α initial V x d) +
          α d • (W d - V d) (φ d x) := by
      simp only [run, map_sub, sub_apply, smul_sub]
      abel
    rw [he, Finset.sum_range_succ]
    calc
      _ ≤ ‖U d (run U φ α initial W x d - run U φ α initial V x d)‖ +
          ‖α d • (W d - V d) (φ d x)‖ := norm_add_le _ _
      _ ≤ ‖run U φ α initial W x d - run U φ α initial V x d‖ +
          ‖α d‖ * (‖W d - V d‖ * ‖φ d x‖) := by
        rw [norm_smul]
        exact add_le_add (hU d _)
          (mul_le_mul_of_nonneg_left ((W d - V d).le_opNorm _) (norm_nonneg _))
      _ ≤ _ := by
        simpa only [mul_assoc] using add_le_add ih
          (le_rfl : ‖α d‖ * (‖W d - V d‖ * ‖φ d x‖) ≤
            ‖α d‖ * (‖W d - V d‖ * ‖φ d x‖))

/-- With zero learned injection, isometric transports preserve the input norm at
every depth (CD2). Flatness concerns the transport, not its trained perturbations. -/
theorem zero_injection_norm (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (x : X) (hU : ∀ k y, ‖U k y‖ = ‖y‖) (d : ℕ) :
    ‖run U φ α initial 0 x d‖ = ‖initial x‖ := by
  induction d with
  | zero => rfl
  | succ d ih => simpa only [run, Pi.zero_apply, zero_apply, smul_zero, add_zero, hU] using ih

/-- A fixed linear classifier transfers the parameter-motion bound to its actual
logits (CD2). A state-dependent normalization is not covered by this statement. -/
theorem readout_difference_le {G : Type*} [NormedAddCommGroup G] [NormedSpace ℝ G]
    (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (W V : ℕ → F →L[ℝ] E) (x : X)
    (hU : ∀ k y, ‖U k y‖ ≤ ‖y‖) (d : ℕ) (R : E →L[ℝ] G) :
    ‖R (run U φ α initial W x d) - R (run U φ α initial V x d)‖ ≤
      ‖R‖ * ∑ k ∈ Finset.range d, ‖α k‖ * ‖W k - V k‖ * ‖φ k x‖ := by
  rw [← map_sub]
  exact (R.le_opNorm _).trans
    (mul_le_mul_of_nonneg_left (run_difference_le U φ α initial W V x hU d) (norm_nonneg R))

/-- A bound on the total injection budget gives a bound independent of the
number of blocks (CD2). Such a budget must be imposed; flat transport alone
does not stop learned injections from accumulating or cancelling the state. -/
theorem run_norm_le_total (U : ℕ → E →L[ℝ] E) (φ : ℕ → X → F) (α : ℕ → ℝ)
    (initial : X → E) (W : ℕ → F →L[ℝ] E) (x : X) (ρ : ℕ → ℝ)
    (hU : ∀ k y, ‖U k y‖ ≤ ‖y‖) (d : ℕ) (hW : W ∈ budget d ρ)
    (R : ℝ) (hR : (∑ k ∈ Finset.range d, ‖α k‖ * ρ k * ‖φ k x‖) ≤ R) :
    ‖run U φ α initial W x d‖ ≤ ‖initial x‖ + R := by
  exact (run_norm_le U φ α initial W x ρ hU d hW).trans (add_le_add le_rfl hR)

example := run_norm_le (fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ (x : ℝ) => x) (fun _ => 1 / 2) id
  (fun _ => ContinuousLinearMap.id ℝ ℝ) 2 (fun _ => 1)
  (by intro k y; exact le_rfl) 3 (by intro k hk; simp)

example := run_difference_le (fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ (x : ℝ) => x) (fun _ => 1 / 2) id
  (fun _ => ContinuousLinearMap.id ℝ ℝ) (fun _ => 2 • ContinuousLinearMap.id ℝ ℝ)
  2 (by intro k y; exact le_rfl) 3

example := zero_injection_norm (fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ (x : ℝ) => x) (fun _ => 1) id 2 (by intro k y; rfl) 3

example := readout_difference_le (fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ (x : ℝ) => x) (fun _ => 1 / 2) id
  (fun _ => ContinuousLinearMap.id ℝ ℝ) (fun _ => 2 • ContinuousLinearMap.id ℝ ℝ)
  2 (by intro k y; exact le_rfl) 3 (ContinuousLinearMap.id ℝ ℝ)

/-- Nonzero injections with a total budget of one at four blocks. The residual
scales are fixed before training and the example verifies their actual sum. -/
example := run_norm_le_total (fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ (x : ℝ) => x) (fun _ => 1 / 4) id
  (fun _ => ContinuousLinearMap.id ℝ ℝ) 1 (fun _ => 1)
  (by intro k y; exact le_rfl) 4 (by intro k hk; simp) 1
  (by norm_num [Finset.sum_range_succ])

end Transformer.Grokking.FlatSpectrum.ConvexDepth
