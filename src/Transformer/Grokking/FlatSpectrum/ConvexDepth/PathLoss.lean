module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Paths

/-!
# Convex answer loss for joint nonlinear paths

Source: `review/convex_depth.md`, CD5. Even the terminal decoder may be nonlinear,
provided it is evaluated on each complete path before mixing logits. Independent
per-layer distributions embed as product tensors, but the learned domain permits
cross-layer correlations. This changes the coordinates and increases their count;
it is not a convexity theorem for trainable factorized transformer parameters.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {J : Type} [Fintype J] {E C S : Type*}

/-- Each full path retains its terminal normalization and decoder; only its final
logits are mixed (CD5). The decoder is fixed while the joint tensor learns. -/
noncomputable def pathLogits (B : ℕ → J → E → E) (d : ℕ) (decode : E → C → ℝ)
    (w : Path J d → ℝ) (x : E) : C → ℝ :=
  ∑ p, w p • decode (pathRun B d p x)

/-- Composed nonlinear path features enter linearly in the trained joint tensor
(CD5), without averaging hidden states before applying nonlinearities. -/
theorem pathLogits_linear (B : ℕ → J → E → E) (d : ℕ) (decode : E → C → ℝ)
    (w v : Path J d → ℝ) (x : E) (a b : ℝ) :
    pathLogits B d decode (a • w + b • v) x =
      a • pathLogits B d decode w x + b • pathLogits B d decode v x := by
  simp only [pathLogits, Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_smul,
    mul_smul, Finset.sum_add_distrib, Finset.smul_sum]

/-- A one-path tensor preserves the entire decoder, including nonlinear operations
such as normalization (CD5). -/
theorem pathLogits_vertex (B : ℕ → J → E → E) (d : ℕ) (decode : E → C → ℝ)
    (p : Path J d) (x : E) :
    pathLogits B d decode (pathVertex p) x = decode (pathRun B d p x) := by
  classical
  simp [pathLogits, pathVertex]

/-- Ordinary CE on the mixed logits. It is not an average of the separate path
losses and is not the negative logarithm of a latent probability coordinate (CD5). -/
noncomputable def pathLoss [Fintype C] [Fintype S]
    (B : ℕ → J → E → E) (d : ℕ) (decode : E → C → ℝ)
    (inputs : S → E) (labels : S → C) (w : Path J d → ℝ) : ℝ :=
  ∑ s, crossEntropy (pathLogits B d decode w (inputs s)) (labels s)

/-- The interacting deep-path model is jointly convex in its complete tensor
of routing weights, on a convex normalized feasible domain (CD5). -/
theorem pathLoss_convex [Fintype C] [Fintype S]
    (B : ℕ → J → E → E) (d : ℕ) (decode : E → C → ℝ)
    (inputs : S → E) (labels : S → C) :
    ConvexOn ℝ (pathDomain d) (pathLoss B d decode inputs labels) := by
  refine ⟨pathDomain_convex d, ?_⟩
  intro w _ v _ a b ha hb hab
  simp only [pathLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s _
  rw [pathLogits_linear]
  exact (ConvexRecall.convex_crossEntropy (labels s)).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- Independent per-depth routing probabilities map to their joint product
distribution. Optimization below need not retain this factorization (CD5). -/
def productWeights (θ : ℕ → J → ℝ) : (d : ℕ) → Path J d → ℝ
  | 0, _ => 1
  | d + 1, p => θ d p.1 * productWeights θ d p.2

/-- Every independent collection of layer distributions remains feasible in the
larger joint model (CD5). Correlated tensors supply additional routing choices. -/
theorem productWeights_mem (θ : ℕ → J → ℝ) (d : ℕ)
    (hθ : ∀ k < d, (∀ j, 0 ≤ θ k j) ∧ ∑ j, θ k j = 1) :
    productWeights θ d ∈ pathDomain d := by
  induction d with
  | zero =>
    constructor
    · intro p
      norm_num [productWeights]
    · change (∑ p : Unit, (1 : ℝ)) = 1
      simp
  | succ d ih =>
    have hp := ih (fun k hk => hθ k (Nat.lt_trans hk (Nat.lt_succ_self d)))
    have hd := hθ d (Nat.lt_succ_self d)
    constructor
    · intro p
      exact mul_nonneg (hd.1 p.1) (hp.1 p.2)
    · change (∑ p : J × Path J d, θ d p.1 * productWeights θ d p.2) = 1
      rw [Fintype.sum_prod_type]
      simp only [← Finset.mul_sum, hp.2, mul_one]
      exact hd.2

/-- The independent-routing feasibility premises are inhabited at arbitrary depth
by a nondegenerate binary distribution (CD5). -/
example (d : ℕ) : productWeights (fun _ (_ : Bool) => (1 / 2 : ℝ)) d ∈ pathDomain d := by
  apply productWeights_mem
  intro k hk
  norm_num

/-- Plain residual stacking is retained even with nonlinear branches (CD5).
This example squares its own previous hidden state at each of two depths. -/
example :
    pathRun (residualBlocks (fun _ (_ : Unit) (x : ℝ) => x ^ 2)) 2
      (choicePath (fun _ => ()) 2) 1 = 6 := by
  norm_num [pathRun, residualBlocks, choicePath]

end Transformer.Grokking.FlatSpectrum.ConvexDepth
