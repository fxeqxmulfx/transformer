module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PathLoss

/-!
# Train one output tensor over complete nonlinear path features

Source: `review/convex_depth.md`, CD8. A fixed terminal decoder can be too
restrictive for a new label vocabulary. Train the combined path/feature/class
coefficients directly, with convex per-class absolute-sum budgets. Do not
factor them into separately trained routing weights and decoder weights.
Every feature still comes from a complete path's own nonlinear hidden state.
The complex transport and branch library remain fixed during this optimization.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {K X C S : Type*} [Fintype K]

/-- Ordinary real class logits from one learned combined coefficient tensor
and fixed features, which may contain complete nonlinear paths (CD8). -/
noncomputable def headLogits (φ : K → X → ℝ) (W : K → C → ℝ) (x : X) : C → ℝ :=
  fun c => ∑ k, W k c * φ k x

/-- The whole coefficient tensor enters linearly, without a hidden product
between learned routing and learned decoder parameters (CD8). -/
theorem headLogits_linear (φ : K → X → ℝ) (W V : K → C → ℝ) (x : X) (a b : ℝ) :
    headLogits φ (a • W + b • V) x = a • headLogits φ W x + b • headLogits φ V x := by
  funext c
  simp [headLogits, add_mul, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]

/-- A per-class absolute-sum upper bound on the combined coefficients (CD8).
This is a convex sufficient logit bound, not a flat-spectrum equality. -/
def headBudget (R : ℝ) : Set (K → C → ℝ) := {W | ∀ c, ∑ k, |W k c| ≤ R}

/-- The entire output budget is jointly convex in all coordinates (CD8). -/
theorem headBudget_convex (R : ℝ) : Convex ℝ (headBudget (K := K) (C := C) R) := by
  intro W hW V hV a b ha hb hab c
  change (∑ k, |a * W k c + b * V k c|) ≤ R
  calc
    _ ≤ ∑ k, (a * |W k c| + b * |V k c|) := by
      apply Finset.sum_le_sum
      intro k hk
      simpa [abs_mul, abs_of_nonneg ha, abs_of_nonneg hb] using
        abs_add_le (a * W k c) (b * V k c)
    _ = a * (∑ k, |W k c|) + b * (∑ k, |V k c|) := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    _ ≤ a * R + b * R := add_le_add
      (mul_le_mul_of_nonneg_left (hW c) ha) (mul_le_mul_of_nonneg_left (hV c) hb)
    _ = R := by rw [← add_mul, hab, one_mul]

/-- Nonnegative budgets have an explicit feasible starting point (CD8). -/
theorem zero_mem_headBudget {R : ℝ} (hR : 0 ≤ R) :
    (0 : K → C → ℝ) ∈ headBudget R := by
  intro c
  simpa using hR

/-- Usual observed cross-entropy after the complete learned readout (CD8). -/
noncomputable def headLoss [Fintype C] [Fintype S] (φ : K → X → ℝ)
    (inputs : S → X) (labels : S → C) (W : K → C → ℝ) : ℝ :=
  ∑ s, crossEntropy (headLogits φ W (inputs s)) (labels s)

/-- Actual ordinary CE remains jointly convex while every output coefficient
learns. The nonlinear feature library is fixed, and the feasible set is convex
(CD8). This does not prove joint convexity in the library's raw block weights. -/
theorem headLoss_convex [Fintype C] [Fintype S] (φ : K → X → ℝ)
    (inputs : S → X) (labels : S → C) (R : ℝ) :
    ConvexOn ℝ (headBudget R) (headLoss φ inputs labels) := by
  refine ⟨headBudget_convex R, ?_⟩
  intro W _ V _ a b ha hb hab
  simp only [headLoss, smul_eq_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro s hs
  rw [headLogits_linear]
  exact (ConvexRecall.convex_crossEntropy (labels s)).2
    (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- Feature bounds, for example from a flat skip's path-state bound, control
every learned logit through the explicit coefficient budget (CD8). -/
theorem headLogits_bound (φ : K → X → ℝ) {W : K → C → ℝ} {R M : ℝ}
    (hW : W ∈ headBudget R) (hM : 0 ≤ M) (x : X) (hφ : ∀ k, |φ k x| ≤ M) (c : C) :
    |headLogits φ W x c| ≤ R * M := by
  calc
    _ ≤ ∑ k, |W k c * φ k x| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k, |W k c| * M := by
      apply Finset.sum_le_sum
      intro k hk
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hφ k) (abs_nonneg _)
    _ = (∑ k, |W k c|) * M := by rw [Finset.sum_mul]
    _ ≤ R * M := mul_le_mul_of_nonneg_right (hW c) hM

variable {J H E : Type} [Fintype J] [Fintype H]

/-- Each native head feature is extracted after its whole nonlinear path,
before learned coefficients mix outputs. The terminal feature map is fixed (CD8). -/
def headFeatures (B : ℕ → J → E → E) (d : ℕ) (terminal : E → H → ℝ) :
    Path J d × H → E → ℝ := fun k x => terminal (pathRun B d k.1 x) k.2

/-- Lift a path router and a linear terminal decoder into the combined coordinates
(CD8). Independent factorization is allowed as a point, not imposed as a domain. -/
def liftHead {d : ℕ} (w : Path J d → ℝ) (D : H → C → ℝ) : Path J d × H → C → ℝ :=
  fun k c => w k.1 * D k.2 c

/-- The combined head exactly embeds CD5 with a linear decoder of fixed terminal
features, retaining every intermediate nonlinear branch evaluation (CD8). -/
theorem liftHead_logits (B : ℕ → J → E → E) (d : ℕ) (terminal : E → H → ℝ)
    (w : Path J d → ℝ) (D : H → C → ℝ) (x : E) :
    headLogits (headFeatures B d terminal) (liftHead w D) x =
      pathLogits B d (fun h c => ∑ i, D i c * terminal h i) w x := by
  funext c
  simp [headLogits, headFeatures, liftHead, pathLogits, Fintype.sum_prod_type,
    Finset.mul_sum, mul_assoc]

/-- A normalized nonnegative path router and a bounded shared decoder embed
within the very same output budget (CD8). This includes independent routers
through `productWeights_mem`; it does not assert convexity of their factors. -/
theorem liftHead_budget {d : ℕ} {w : Path J d → ℝ} {D : H → C → ℝ} {R : ℝ}
    (hw : w ∈ pathDomain d) (hD : D ∈ headBudget R) :
    liftHead w D ∈ headBudget R := by
  intro c
  calc
    (∑ k, |liftHead w D k c|) = ∑ p, w p * ∑ h, |D h c| := by
      simp only [liftHead, Fintype.sum_prod_type, abs_mul, abs_of_nonneg (hw.1 _),
        Finset.mul_sum]
    _ ≤ ∑ p, w p * R := by
      apply Finset.sum_le_sum
      intro p hp
      exact mul_le_mul_of_nonneg_left (hD c) (hw.1 p)
    _ = R := by rw [← Finset.sum_mul, hw.2, one_mul]

/-- Exact feature count makes the exponential depth cost explicit (CD8).
The additional output coefficients multiply this by the class count. -/
theorem head_feature_card (d : ℕ) :
    Fintype.card (Path J d × H) = Fintype.card J ^ d * Fintype.card H := by
  rw [Fintype.card_prod, path_card]

example := zero_mem_headBudget (K := Fin 2) (C := Bool) (by norm_num : (0 : ℝ) ≤ 1)

example : (fun (_ : Fin 2) (_ : Bool) => (1 / 2 : ℝ)) ∈ headBudget 1 := by
  intro c
  norm_num [Fin.sum_univ_two]

example := headLogits_bound (K := Fin 2) (C := Bool) (fun k (x : ℝ) => if k = 0 then x else x^2)
  (W := fun _ _ => 1 / 2) (R := 1) (M := 4)
  (by intro c; norm_num [Fin.sum_univ_two]) (by norm_num) 2
  (by intro k; fin_cases k <;> norm_num) true

example : liftHead (pathVertex (choicePath (fun _ => true) 2))
    (fun (_ : Fin 2) (_ : Bool) => (1 / 2 : ℝ)) ∈ headBudget 1 := by
  apply liftHead_budget (pathVertex_mem _)
  intro c
  norm_num [Fin.sum_univ_two]

example : headLogits (headFeatures (fun _ (_ : Unit) (x : ℝ) => x + x^2) 2
    (fun x (_ : Unit) => x)) (fun _ (_ : Bool) => 1) 1 true = 6 := by
  norm_num [headLogits, headFeatures, pathRun, path_card]

end Transformer.Grokking.FlatSpectrum.ConvexDepth
