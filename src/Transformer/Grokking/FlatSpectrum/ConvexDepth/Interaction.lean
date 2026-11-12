module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PathLoss

/-!
# A learned joint tensor preserves a nonlinear residual interaction

Source: `review/convex_depth.md`, CD7. The first residual block creates a signed
feature. The second multiplies that actual hidden feature by another signed input.
A feasible correlated routing tensor preserves their product while an independent
tensor with identical single-layer marginals erases it. This is a capacity and CE
geometry witness, not a stress experiment or a convergence theorem.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators

/-- Two possible learned block choices in the finite witness (CD7). -/
def bitSign (b : Bool) : ℝ := if b then 1 else -1

/-- The residual branches actually read the evolving state. At depth one, the
product uses the feature created by depth zero, not the original first input (CD7). -/
def interactionBranch (k : ℕ) (j : Bool) (h : ℝ × ℝ) : ℝ × ℝ :=
  if k = 0 then ((bitSign j - 1) * h.1, 0)
  else (bitSign j * h.1 * h.2 - h.1, 0)

/-- Direct evaluation exhibits the product of both block choices and both inputs
inside an ordinary two-block residual stack (CD7). -/
theorem interaction_path (a b : Bool) (x y : ℝ) :
    pathRun (residualBlocks interactionBranch) 2 (b, (a, ())) (x, y) =
      (bitSign b * bitSign a * x * y, y) := by
  dsimp [pathRun, residualBlocks, interactionBranch]
  ext <;> ring

/-- Equal signs at the two depths receive probability one half each (CD7).
This is a trainable joint-tensor point, not a constraint imposed on all fits. -/
noncomputable def correlatedWeights (p : Path Bool 2) : ℝ :=
  if p.1 = p.2.1 then 1 / 2 else 0

/-- The interacting tensor obeys every convex domain constraint (CD7). -/
theorem correlatedWeights_mem : correlatedWeights ∈ pathDomain 2 := by
  constructor
  · intro p
    unfold correlatedWeights
    split_ifs <;> norm_num
  · change (∑ p : Bool × (Bool × Unit), correlatedWeights p) = 1
    norm_num [Fintype.sum_prod_type, correlatedWeights]

/-- Both marginal distributions remain uniform; the learned information is the
cross-depth correlation, which individual layer averages cannot retain (CD7). -/
theorem correlatedWeights_marginals (j : Bool) :
    (∑ i : Bool, correlatedWeights (i, (j, ()))) = 1 / 2 ∧
      (∑ i : Bool, correlatedWeights (j, (i, ()))) = 1 / 2 := by
  cases j <;> norm_num [correlatedWeights]

/-- The joint tensor is genuinely outside the independently factorized family
(CD7). This records precisely which equality was relaxed to obtain convexity. -/
theorem correlatedWeights_not_product :
    ¬ ∃ θ : ℕ → Bool → ℝ, correlatedWeights = productWeights θ 2 := by
  rintro ⟨θ, he⟩
  have hff := congrFun he (false, (false, ()))
  have hft := congrFun he (false, (true, ()))
  have htf := congrFun he (true, (false, ()))
  have htt := congrFun he (true, (true, ()))
  change (1 / 2 : ℝ) = θ 1 false * (θ 0 false * 1) at hff
  change (0 : ℝ) = θ 1 false * (θ 0 true * 1) at hft
  change (0 : ℝ) = θ 1 true * (θ 0 false * 1) at htf
  change (1 / 2 : ℝ) = θ 1 true * (θ 0 true * 1) at htt
  simp only [mul_one] at hff hft htf htt
  have hd : (θ 1 false * θ 0 false) * (θ 1 true * θ 0 true) =
      (θ 1 false * θ 0 true) * (θ 1 true * θ 0 false) := by ring
  rw [← hff, ← htt, ← hft, ← htf] at hd
  norm_num at hd

/-- Mixing complete residual paths retains their nonlinear interaction (CD7). -/
theorem correlated_prediction (x y : ℝ) :
    pathPredict (residualBlocks interactionBranch) 2 correlatedWeights (x, y) = (x * y, y) := by
  unfold pathPredict
  change (∑ p : Bool × (Bool × Unit), correlatedWeights p •
    pathRun (residualBlocks interactionBranch) 2 p (x, y)) = _
  simp only [Fintype.sum_prod_type]
  ext <;> simp [correlatedWeights, pathRun, residualBlocks, interactionBranch, bitSign] <;> ring

/-- Independent layer choices with the same marginal probabilities lose the
interaction, despite the identical nonlinear block library (CD7). -/
theorem independent_prediction (x y : ℝ) :
    pathPredict (residualBlocks interactionBranch) 2
      (productWeights (fun _ (_ : Bool) => 1 / 2) 2) (x, y) = (0, y) := by
  unfold pathPredict
  change (∑ p : Bool × (Bool × Unit), productWeights (fun _ (_ : Bool) => 1 / 2) 2 p •
    pathRun (residualBlocks interactionBranch) 2 p (x, y)) = _
  simp only [Fintype.sum_prod_type]
  ext <;> simp [productWeights, pathRun, residualBlocks, interactionBranch, bitSign] <;> ring

/-- A fixed binary readout of each path supplies ordinary class logits (CD7). -/
def interactionDecode (h : ℝ × ℝ) (c : Bool) : ℝ := if c then h.1 else -h.1

/-- The actual final-logit mixture carries the input product, with both choices
evaluated before the fixed decoder (CD7). -/
theorem correlated_logits (x y : ℝ) :
    pathLogits (residualBlocks interactionBranch) 2 interactionDecode correlatedWeights (x, y) =
      (fun c => if c then x * y else -(x * y)) := by
  unfold pathLogits
  change (∑ p : Bool × (Bool × Unit), correlatedWeights p •
    interactionDecode (pathRun (residualBlocks interactionBranch) 2 p (x, y))) = _
  simp only [Fintype.sum_prod_type]
  funext c
  cases c <;>
    simp [correlatedWeights, pathRun, residualBlocks, interactionBranch, bitSign, interactionDecode] <;>
    ring

/-- Signed parity has strict margin two on all four inputs in the lifted
ordinary-residual model (CD7). This does not assert that an optimizer found it. -/
theorem residual_interaction_margin (x y : Bool) :
    let z := pathLogits (residualBlocks interactionBranch) 2 interactionDecode
      correlatedWeights (bitSign x, bitSign y)
    z (x == y) - z (!(x == y)) = 2 := by
  dsimp only
  rw [correlated_logits]
  cases x <;> cases y <;> norm_num [bitSign]

/-- An affine score in the original two input features cannot strictly separate
the same parity labels (CD7). Adding their product to a frozen feature bank would
remove this separation; the comparison fixes the original input features. -/
theorem no_affine_interaction :
    ¬ ∃ a b c : ℝ, 0 < a + b + c ∧ 0 < -a - b + c ∧
      a - b + c < 0 ∧ -a + b + c < 0 := by
  rintro ⟨a, b, c, hpp, hnn, hpn, hnp⟩
  linarith

end Transformer.Grokking.FlatSpectrum.ConvexDepth
