module

public import Transformer.Grokking.Convex.ConvexDepth.ClassGeometry.Basic
public import Transformer.Grokking.Convex.ConvexDepth.SignedContext

/-!
# Complete bounded complex realization of the identifying decoder

Source: `review/convex/moments.md`, CM56. A fixed four-point unitary code carries
the scalar feature and a constant through signed normalized contexts. Every
real and imaginary coefficient, for every output class, is charged. The three
shared coordinates enter affinely; ordinary full and minibatch CE are convex.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ClassGeometry

open scoped BigOperators

variable {C X : Type*}

/-- Fixed signed context for a latent scalar and a constant, CM56. -/
def row (x : ℝ) : Fin 4 → ℝ := ![x, 1, 0, 0]

/-- Actual realification of the normalized complex feature product, CM56. -/
noncomputable def features : Fin 4 × Bool → ℝ → ℝ :=
  ContextInterpolation.features (fun x => SignedContext.normalized (row x))
    (complexFeatures fourier4)

/-- Complete shared expansion, including the imaginary coordinates, CM56. -/
noncomputable def expansion (e : C → ℝ) (p : Fin 3) (k : Fin 4 × Bool) (c : C) : ℝ :=
  ![2 * e c * complexFeatures fourier4 k 0,
    2 * e c * complexFeatures fourier4 k 1,
    -(e c) ^ 2 * complexFeatures fourier4 k 1] p

/-- The complete head is affine in the scaled slope, intercept and margin,
CM56. There are no separately learned factors multiplied together. -/
noncomputable def head (e : C → ℝ) := ParameterSharing.parameterMap (expansion e) 0

/-- Flatness bounds these actual signed features; it does not identify the
latent feature or output coordinates, CM56. -/
theorem features_bound (x : ℝ) (k : Fin 4 × Bool) : |features k x| ≤ 1 :=
  SignedContext.complex_features_bound fourier4 fourier4_unitary row x k

/-- Every occurrence of each parameter is included in its cost, CM56. -/
theorem expansion_bound (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (p : Fin 3) (c : C) :
    ∑ k, |expansion e p k c| ≤ costs p := by
  have h := he c
  have hs : (e c) ^ 2 ≤ 1 := by
    nlinarith [sq_abs (e c), mul_nonneg (sub_nonneg.mpr h) (abs_nonneg (e c))]
  fin_cases p <;>
    norm_num [expansion, costs, Fintype.sum_prod_type, Fintype.sum_bool,
      Fin.sum_univ_succ, complexFeatures, fourier4, abs_mul, abs_div, abs_neg,
      abs_of_nonneg (sq_nonneg (e c))] <;> nlinarith

/-- The expanded source logits are exactly the two required table rows,
CM56. This identity checks the actual eight realified coordinates. -/
theorem source_logits (e : C → ℝ) (θ : Fin 3 → ℝ) (s : Fin 4) (c : C) :
    headLogits (complexFeatures fourier4) (head e θ) s c =
      ![2 * e c * θ 0, 2 * e c * θ 1 - θ 2 * (e c) ^ 2, 0, 0] s := by
  fin_cases s <;>
    norm_num [headLogits, head, ParameterSharing.parameterMap,
      ParameterSharing.coefficientMap, expansion, Fintype.sum_prod_type,
      Fintype.sum_bool, Fin.sum_univ_succ, complexFeatures, fourier4] <;> ring

/-- The actual complex product equals the identifying score divided by a
positive input-only normalizer, CM56. -/
theorem head_logits (e : C → ℝ) (θ : Fin 3 → ℝ) (x : ℝ) (c : C) :
    headLogits features (head e θ) x c = score e θ x c / SignedContext.normalizer (row x) := by
  rw [features, SignedContext.normalized_logits]
  simp_rw [source_logits]
  simp only [Fin.sum_univ_succ, row, Matrix.cons_val_zero, Matrix.cons_val_succ,
    Matrix.cons_val_fin_one, mul_zero, zero_add, one_mul]
  congr 1
  unfold score
  ring

/-- The complete coefficient budget follows from the sufficient compact
budget, including all raw classes and imaginary entries, CM56. -/
theorem head_budget (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) {R : ℝ} {θ : Fin 3 → ℝ}
    (hθ : θ ∈ ParameterSharing.weightedDomain costs R) : head e θ ∈ headBudget R := by
  simpa [head] using ParameterSharing.parameterMap_budget (expansion e) 0 costs
    (expansion_bound e he) (zero_mem_headBudget (le_refl (0 : ℝ))) hθ

/-- Ordinary full-vocabulary CE is convex in these actual shared head
parameters, for any fixed scalar input feature, CM56. -/
theorem loss_convex [Fintype C] {S : Type*} [Fintype S]
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (inputs : S → ℝ) (labels : S → C) (R : ℝ) :
    ConvexOn ℝ (ParameterSharing.weightedDomain costs R)
      (fun θ => headLoss features inputs labels (head e θ)) :=
  ParameterSharing.loss_convex (expansion e) 0 costs costs_nonneg features inputs labels
    R 0 (expansion_bound e he) (zero_mem_headBudget (le_refl (0 : ℝ)))

/-- Repeated record IDs and the true short-batch denominator are retained,
CM56. Convexity asserts neither identification of the feature nor convergence. -/
theorem batchLoss_convex [Fintype C] {S : Type*}
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (inputs : S → ℝ) (labels : S → C)
    (batch : Finset S) (R : ℝ) :
    ConvexOn ℝ (ParameterSharing.weightedDomain costs R)
      (fun θ => Fallback.batchLoss features inputs labels batch (head e θ)) :=
  ParameterSharing.batchLoss_convex (expansion e) 0 costs costs_nonneg features inputs labels
    batch R 0 (expansion_bound e he) (zero_mem_headBudget (le_refl (0 : ℝ)))

/-- Every feasible parameter predicts the target at every input under the
explicit affine realizability prior, CM56. Only two observed labels constrain
the head; the theorem derives, rather than assumes, its query logits. -/
theorem identified_prediction (e : C → ℝ) (he : Function.Injective e)
    (f : X → ℝ) (target : X → C) (a b : ℝ)
    (realizable : ∀ x, e (target x) = a * f x + b)
    (x₀ x₁ : X) (hx : f x₀ ≠ f x₁) {R δ : ℝ} (hδ : 0 < δ)
    {θ : Fin 3 → ℝ} (hθ : θ ∈ domain e (f x₀) (f x₁) (target x₀) (target x₁) R δ)
    (x : X) (c : C) (hc : c ≠ target x) :
    headLogits features (head e θ) (f x) c <
      headLogits features (head e θ) (f x) (target x) := by
  obtain ⟨ha, hb⟩ := anchors_identify e θ (f x₀) (f x₁) a b (target x₀) (target x₁)
    hx (realizable x₀) (realizable x₁) hθ.2.2.1 hθ.2.2.2
  have hcenter : θ 0 * f x + θ 1 = θ 2 * e (target x) := by
    rw [ha, hb, realizable x]
    ring
  rw [head_logits, head_logits]
  exact div_lt_div_of_pos_right (strict_score e he θ (f x) (target x) c
    (lt_of_lt_of_le hδ hθ.2.1) hc hcenter) (SignedContext.normalizer_pos _)

/-- The full ordinary CE objective remains convex after imposing the observed
anchors and positive-margin constraint, CM56. -/
theorem domain_loss_convex [Fintype C] {S : Type*} [Fintype S]
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (inputs : S → ℝ) (labels : S → C)
    (x₀ x₁ : ℝ) (c₀ c₁ : C) (R δ : ℝ) :
    ConvexOn ℝ (domain e x₀ x₁ c₀ c₁ R δ)
      (fun θ => headLoss features inputs labels (head e θ)) :=
  (loss_convex e he inputs labels R).subset (fun _ h => h.1)
    (domain_convex e x₀ x₁ c₀ c₁ R δ)

/-- The same convex feasible domain supports true normalized minibatches,
including repeated input values and a short tail, CM56. -/
theorem domain_batchLoss_convex [Fintype C] {S : Type*}
    (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (inputs : S → ℝ) (labels : S → C)
    (batch : Finset S) (x₀ x₁ : ℝ) (c₀ c₁ : C) (R δ : ℝ) :
    ConvexOn ℝ (domain e x₀ x₁ c₀ c₁ R δ)
      (fun θ => Fallback.batchLoss features inputs labels batch (head e θ)) :=
  (batchLoss_convex e he inputs labels batch R).subset (fun _ h => h.1)
    (domain_convex e x₀ x₁ c₀ c₁ R δ)

end Transformer.Grokking.Convex.ConvexDepth.ClassGeometry
