import Transformer.GPTMini.Sparsemax.Failure
import Mathlib.Analysis.Calculus.FDeriv.Linear
import Mathlib.Analysis.Calculus.FDeriv.Add

/-!
# A corrective score loss on a saturated sparsemax route

arXiv:1602.02068v2, §3.2, equation `sparsemax_loss`, and §3.3,
equation `sparsemax_loss_specialcases1`. The paper's convex score loss
uses the projection's quadratic potential and a target-score term. It
is not an outer loss depending only on the projected probabilities.

We use the existing causal `sparseWeights`, prove the loss nonnegative
for visible targets, and prove its full score derivative on strict
singleton regions. The previous wrong-route counterexample receives a
nonzero target derivative and one score-gradient step repairs its route.
This assumes a supplied routing target; it is not a guarantee for latent
attention or a claim about the benchmark's end-to-end cross-entropy.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open Filter
open scoped BigOperators Topology

/-- The paper's sparsemax loss, using the actual causal projection's
quadratic potential. Source: arXiv:1602.02068v2, §3.2, `sparsemax_loss`.
Our `routingObjective` is one half of `sum p²/2 - sum p*score`, hence
the factor two. Masking to positions at most `i` is the only deviation. -/
def rowSparsemaxLoss {T : ℕ} (target i : Fin T) (scores : Fin T → ℝ) : ℝ :=
  -scores target - 2 * routingObjective (fun j => -scores j / 2)
    (sparseWeights scores i) + 1 / 2

/-- The potential form retains both score and probability dependence.
Source: arXiv:1602.02068v2, §3.2, `sparsemax_loss`; on positive
coordinates `p = score - threshold`, so each summand equals
`(score² - threshold²)/2`, and inactive summands are zero. -/
theorem rowSparsemaxLoss_eq_probability_potential {T : ℕ} (target i : Fin T)
    (scores : Fin T → ℝ) :
    rowSparsemaxLoss target i scores = -scores target +
      (∑ j, (sparseWeights scores i j * scores j -
        (sparseWeights scores i j) ^ 2 / 2)) + 1 / 2 := by
  have hcost : (∑ j, sparseWeights scores i j * (-scores j / 2)) =
      -(∑ j, sparseWeights scores i j * scores j) / 2 := by
    calc
      _ = ∑ j, -(sparseWeights scores i j * scores j) / 2 := by
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = _ := by rw [← Finset.sum_div, Finset.sum_neg_distrib]
  unfold rowSparsemaxLoss routingObjective
  rw [hcost, Finset.sum_sub_distrib, ← Finset.sum_div]
  ring

/-- Any feasible route bounds the projection's score loss from below.
Source: arXiv:1602.02068v2, §2.1 and §3.2; the quadratic projection
minimizes its objective over the causal simplex. -/
theorem rowSparsemaxLoss_lower_bound {T : ℕ} (target i : Fin T)
    (scores route : Fin T → ℝ) (hroute : route ∈ simplexOn {j : Fin T | j ≤ i}) :
    -scores target - 2 * routingObjective (fun j => -scores j / 2) route + 1 / 2 ≤
      rowSparsemaxLoss target i scores := by
  have hmin := (sparseWeights_spec scores i).2 route hroute
  unfold rowSparsemaxLoss
  linarith

/-- Feasible comparison routes exist with a visible target.
Source context: arXiv:1602.02068v2, §2.1, masked simplex. -/
example : basis (1 : Fin 2) ∈ simplexOn {j : Fin 2 | j ≤ 1} :=
  basis_mem_simplex _ _ (by decide)

/-- The scaled row objective assigns the one-hot penalty `1/4`.
Source: arXiv:1602.02068v2, §3.2, `sparsemax_loss`, with factor-two scaling. -/
private theorem routingObjective_basis_score {T : ℕ} (scores : Fin T → ℝ)
    (winner : Fin T) :
    routingObjective (fun j => -scores j / 2) (basis winner) =
      1 / 4 - scores winner / 2 := by
  classical
  simp [routingObjective, basis]
  ring

/-- The score loss is nonnegative when the target is a visible slot.
Source: arXiv:1602.02068v2, §3.2, Proposition 3, nonnegativity claim.
The visible-target premise is required by the causal masking extension. -/
theorem rowSparsemaxLoss_nonneg {T : ℕ} (target i : Fin T)
    (scores : Fin T → ℝ) (htarget : target ≤ i) :
    0 ≤ rowSparsemaxLoss target i scores := by
  have h := rowSparsemaxLoss_lower_bound target i scores (basis target)
    (basis_mem_simplex _ target htarget)
  rw [routingObjective_basis_score] at h
  linarith

/-- Nonnegativity's visible-target hypotheses are inhabited.
Source context: arXiv:1602.02068v2, §3.2, causal extension. -/
example : (1 : Fin 2) ≤ 1 := by decide

/-- An actual singleton projection has loss equal to the winner-target
score gap, including wrong routes. Source: arXiv:1602.02068v2, §3.3,
`sparsemax_loss_specialcases1`, specialized to a causal row. -/
theorem rowSparsemaxLoss_eq_score_gap {T : ℕ} (target i winner : Fin T)
    (scores : Fin T → ℝ) (hroute : sparseWeights scores i = basis winner) :
    rowSparsemaxLoss target i scores = scores winner - scores target := by
  unfold rowSparsemaxLoss
  rw [hroute, routingObjective_basis_score]
  ring

/-- The singleton-loss premise uses the same wrong route as the plateau.
Source context: arXiv:1602.02068v2, §3.3. -/
example : sparseWeights (separatedScores 0) 1 = basis (0 : Fin 2) :=
  sparseWeights_eq_basis_of_gap _ _ _ (by decide)
    (fun j _ hn => le_of_lt (separatedScores_gap 0 j hn))

/-- The full score derivative remains corrective on a strict singleton
region: it is evaluation at the winner minus evaluation at the target.
Source: arXiv:1602.02068v2, §3.2, `sparsemax_loss_gradient`, specialized
to strict singleton support; boundaries and other supports are not asserted
by this theorem. No derivative is stipulated in the loss definition. -/
theorem rowSparsemaxLoss_hasFDerivAt_of_gap {T : ℕ} (target i winner : Fin T)
    (scores : Fin T → ℝ) (hw : winner ≤ i)
    (hgap : ∀ j, j ≤ i → j ≠ winner → scores j + 1 < scores winner) :
    HasFDerivAt (𝕜 := ℝ) (rowSparsemaxLoss target i)
      ((ContinuousLinearMap.proj winner : (Fin T → ℝ) →L[ℝ] ℝ) -
        ContinuousLinearMap.proj target) scores := by
  have hd := ((ContinuousLinearMap.proj winner : (Fin T → ℝ) →L[ℝ] ℝ).hasFDerivAt
    (x := scores)).sub ((ContinuousLinearMap.proj target).hasFDerivAt (x := scores))
  apply hd.congr_of_eventuallyEq
  exact (sparseWeights_eventually_eq_basis scores i winner hw hgap).mono
    fun z hz => rowSparsemaxLoss_eq_score_gap target i winner z hz

/-- Strict singleton derivative hypotheses hold on the concrete plateau.
Source context: arXiv:1602.02068v2, §3.2, singleton specialization. -/
example : (0 : Fin 2) ≤ 1 ∧ ∀ j : Fin 2,
    j ≤ 1 → j ≠ 0 → separatedScores 0 j + 1 < separatedScores 0 0 :=
  ⟨by decide, fun j _ hn => separatedScores_gap 0 j hn⟩

/-- Adding the score loss supplies its derivative alongside any locally
flat outer probability loss. Source: arXiv:1602.02068v2, §3.2,
`sparsemax_loss_gradient`; this auxiliary objective is a derived extension
requiring a supplied target, not the paper's attention training recipe. -/
theorem rowLoss_add_sparsemaxLoss_hasFDerivAt_of_gap {T : ℕ}
    (loss : (Fin T → ℝ) → ℝ) (weight : ℝ) (target i winner : Fin T)
    (scores : Fin T → ℝ) (hw : winner ≤ i)
    (hgap : ∀ j, j ≤ i → j ≠ winner → scores j + 1 < scores winner) :
    HasFDerivAt (𝕜 := ℝ) (fun z => loss (sparseWeights z i) +
      weight * rowSparsemaxLoss target i z)
      (weight • ((ContinuousLinearMap.proj winner : (Fin T → ℝ) →L[ℝ] ℝ) -
        ContinuousLinearMap.proj target)) scores := by
  have ho := rowLoss_hasFDerivAt_zero loss scores i winner hw hgap
  have ha := (rowSparsemaxLoss_hasFDerivAt_of_gap target i winner scores hw hgap).const_smul weight
  simpa only [Pi.add_def, Pi.smul_def, zero_add, smul_eq_mul] using ho.add ha

/-- Inhabited auxiliary-loss hypotheses; arXiv:1602.02068v2, §3.2. -/
example : (0 : Fin 2) ≤ 1 ∧ ∀ j : Fin 2,
    j ≤ 1 → j ≠ 0 → separatedScores 0 j + 1 < separatedScores 0 0 :=
  ⟨by decide, fun j _ hn => separatedScores_gap 0 j hn⟩

/-- The positive-loss, zero-derivative outer-loss counterexample has a
negative target-coordinate derivative under the paper's score loss.
Source: arXiv:1602.02068v2, §3.2, `sparsemax_loss_gradient`, applied to
`wrong_route_positive_stationary_point`; this is a row-level repair. -/
theorem wrong_route_sparsemax_loss_has_corrective_derivative :
    rowSparsemaxLoss 1 1 (separatedScores 0) = 2 ∧
    HasFDerivAt (𝕜 := ℝ) (rowSparsemaxLoss 1 1)
      ((ContinuousLinearMap.proj 0 : (Fin 2 → ℝ) →L[ℝ] ℝ) -
        ContinuousLinearMap.proj 1) (separatedScores 0) ∧
    ((ContinuousLinearMap.proj (0 : Fin 2) : (Fin 2 → ℝ) →L[ℝ] ℝ) -
      (ContinuousLinearMap.proj (1 : Fin 2) : (Fin 2 → ℝ) →L[ℝ] ℝ))
        (basis (1 : Fin 2)) = -1 := by
  have hroute := sparseWeights_eq_basis_of_gap (separatedScores 0) 1 0
    (by decide) (fun j _ hn => le_of_lt (separatedScores_gap 0 j hn))
  refine ⟨?_, rowSparsemaxLoss_hasFDerivAt_of_gap 1 1 0 _ (by decide)
    (fun j _ hn => separatedScores_gap 0 j hn), ?_⟩
  · rw [rowSparsemaxLoss_eq_score_gap 1 1 0 _ hroute]
    norm_num [separatedScores]
  · norm_num [basis]

/-- A score-gradient step of size 3/2 changes the same wrong singleton
into the correct exact route with zero score loss. Source: the gradient
in arXiv:1602.02068v2, §3.2; this concrete step is a derived example,
not a convergence claim for arbitrary steps or network parameters. -/
theorem wrong_route_one_gradient_step_repaired :
    let next : Fin 2 → ℝ := fun j => separatedScores 0 j -
      (3 / 2) * (sparseWeights (separatedScores 0) 1 j - basis 1 j)
    sparseWeights next 1 = basis 1 ∧ rowSparsemaxLoss 1 1 next = 0 := by
  dsimp only
  have hroute := sparseWeights_eq_basis_of_gap (separatedScores 0) 1 0
    (by decide) (fun j _ hn => le_of_lt (separatedScores_gap 0 j hn))
  let next : Fin 2 → ℝ := fun j => separatedScores 0 j -
    (3 / 2) * (sparseWeights (separatedScores 0) 1 j - basis 1 j)
  have hnext : sparseWeights next 1 = basis 1 := by
    apply sparseWeights_eq_basis_of_gap next 1 1 (by decide)
    intro j _ hn
    fin_cases j <;> norm_num [next, hroute, separatedScores, basis] at *
  refine ⟨hnext, ?_⟩
  rw [rowSparsemaxLoss_eq_score_gap 1 1 1 next hnext]
  ring

end Transformer.GPTMini.Sparsemax
