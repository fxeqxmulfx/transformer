module

public import Transformer.Grokking.Convex.ConvexDepth.CoordinatePrograms.Basic

/-!
# Complete complex products for a finite program bank

Source: `review/convex/moments.md`, CM57. Each fixed input-only program has two
complete Fourier blocks. One signed normalization covers the entire bank;
every real and imaginary coefficient enters the exact product and budget.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.CoordinatePrograms

open scoped BigOperators

variable {C : Type*}

/-- Source positions in two exact four-point Fourier blocks, CM57. -/
abbrev Source := Fin 2 × Fin 4

/-- Complete real and imaginary coefficient positions, CM57. -/
abbrev Coordinate := Fin 2 × (Fin 4 × Bool)

/-- Blockwise realification of the actual complex Fourier code, CM57. -/
noncomputable def source (k : Coordinate) (s : Source) : ℝ :=
  if k.1 = s.1 then complexFeatures fourier4 k.2 s.2 else 0

/-- Input-only signed row for five features and the class-offset constant,
CM57. No parameter or target label is an argument. -/
def row (h : Fin 5 → ℝ) (s : Source) : ℝ :=
  if s.1 = 0 then ![h 0, h 1, h 2, h 3] s.2 else ![h 4, 1, 0, 0] s.2

/-- Blockwise unitary bounds control the actual source coordinates, CM57. -/
theorem source_bound (k : Coordinate) (s : Source) : |source k s| ≤ 1 := by
  unfold source
  split
  · exact ConvexDepth.complexFeatures_bound fourier4 fourier4_unitary k.2 s.2
  · norm_num

/-- The complete coefficient expansion is fixed in every learned parameter,
CM57. The final term is the learned mass times a fixed quadratic class feature. -/
noncomputable def localExpansion (e : C → ℝ) (j : Fin 6) (k : Coordinate) (c : C) : ℝ :=
  ![2 * e c * source k (0, 0), 2 * e c * source k (0, 1),
    2 * e c * source k (0, 2), 2 * e c * source k (0, 3),
    2 * e c * source k (1, 0), -(e c) ^ 2 * source k (1, 1)] j

/-- Actual complete affine head for a single program, CM57. -/
noncomputable def localHead (e : C → ℝ) := ParameterSharing.parameterMap (localExpansion e) 0

/-- Source logits whose actual complex realization is checked below, CM57. -/
def sourceScore (e : C → ℝ) (θ : Fin 6 → ℝ) (s : Source) (c : C) : ℝ :=
  if s.1 = 0 then ![2 * e c * θ 0, 2 * e c * θ 1,
    2 * e c * θ 2, 2 * e c * θ 3] s.2
  else ![2 * e c * θ 4, -θ 5 * (e c) ^ 2, 0, 0] s.2

set_option maxHeartbeats 800000 in
/-- Exact source products over all sixteen realified coordinates, CM57. -/
theorem local_source_logits (e : C → ℝ) (θ : Fin 6 → ℝ) (s : Source) (c : C) :
    headLogits source (localHead e θ) s c = sourceScore e θ s c := by
  rcases s with ⟨b, s⟩
  fin_cases b <;> fin_cases s <;>
    norm_num [headLogits, localHead, ParameterSharing.parameterMap,
      ParameterSharing.coefficientMap, localExpansion, source, sourceScore,
      Fintype.sum_prod_type, Fintype.sum_bool, Fin.sum_univ_succ,
      complexFeatures, fourier4] <;> ring

/-- The input-only row contracts those actual source logits to the intended
linear coordinate score; both Fourier blocks contribute, CM57. -/
theorem row_score (e : C → ℝ) (θ : Fin 6 → ℝ) (h : Fin 5 → ℝ) (c : C) :
    (∑ s, row h s * sourceScore e θ s c) = score e θ h c := by
  norm_num [row, sourceScore, score, center, Fintype.sum_prod_type, Fin.sum_univ_succ]
  ring

set_option maxHeartbeats 800000 in
/-- Complete column costs, including every imaginary coefficient and every
output class, CM57. No dense coefficient is dropped from this bound. -/
theorem local_expansion_bound (e : C → ℝ) (he : ∀ c, |e c| ≤ 1)
    (j : Fin 6) (c : C) : (∑ k, |localExpansion e j k c|) ≤ costs j := by
  have h := he c
  have hs : (e c) ^ 2 ≤ 1 := by
    nlinarith [sq_abs (e c), mul_nonneg (sub_nonneg.mpr h) (abs_nonneg (e c))]
  fin_cases j <;>
    norm_num [localExpansion, source, costs, Fintype.sum_prod_type,
      Fintype.sum_bool, Fin.sum_univ_succ, complexFeatures, fourier4,
      abs_mul, abs_div, abs_neg, abs_of_nonneg (sq_nonneg (e c))] <;> nlinarith

variable {P X : Type*} [Fintype P] [DecidableEq P]

/-- All source blocks of the fixed program bank, CM57. -/
noncomputable def bankSource (k : P × Coordinate) (s : P × Source) : ℝ :=
  if k.1 = s.1 then source k.2 s.2 else 0

/-- A single fixed input row over the entire bank, CM57. -/
def bankRow (h : P → X → Fin 5 → ℝ) (x : X) (s : P × Source) : ℝ := row (h s.1 x) s.2

/-- Actual normalized complex features; inputs alone determine them, CM57. -/
noncomputable def features (h : P → X → Fin 5 → ℝ) : (P × Coordinate) → X → ℝ :=
  ContextInterpolation.features (fun x => SignedContext.normalized (bankRow h x)) bankSource

omit [Fintype P] in
/-- Every bank source coordinate inherits its individual Fourier bound,
CM57. No flat empirical Gram matrix is assumed. -/
theorem bank_source_bound (k : P × Coordinate) (s : P × Source) : |bankSource k s| ≤ 1 := by
  unfold bankSource
  split
  · exact source_bound k.2 s.2
  · norm_num

/-- Normalizing the full signed input row bounds every actual feature, CM57. -/
theorem features_bound (h : P → X → Fin 5 → ℝ) (x : X) (k : P × Coordinate) :
    |features h k x| ≤ 1 :=
  SignedContext.normalized_features_bound (bankRow h) bankSource x
    (by norm_num) bank_source_bound k

/-- One fixed affine map for every program mass and decoder coordinate,
CM57. There is no learned factor inside this map. -/
noncomputable def expansion (e : C → ℝ) (p : P × Fin 6) (k : P × Coordinate) (c : C) : ℝ :=
  if p.1 = k.1 then localExpansion e p.2 k.2 c else 0

/-- The actual complete head over all programs and both Fourier blocks, CM57. -/
noncomputable def head (e : C → ℝ) := ParameterSharing.parameterMap (expansion (P := P) e) 0

/-- A program's coefficients equal its local affine expansion, CM57. -/
theorem head_block (e : C → ℝ) (θ : P × Fin 6 → ℝ) (p : P) (k : Coordinate) (c : C) :
    head e θ (p, k) c = localHead e (fun j => θ (p, j)) k c := by
  simp [head, localHead, ParameterSharing.parameterMap, ParameterSharing.coefficientMap,
    expansion, Fintype.sum_prod_type, ite_mul, Finset.sum_ite_irrel]

/-- Complete global source products reduce to the checked local products,
CM57. Off-program blocks contribute exactly zero. -/
theorem bank_source_logits (e : C → ℝ) (θ : P × Fin 6 → ℝ) (p : P) (s : Source) (c : C) :
    headLogits bankSource (head e θ) (p, s) c = sourceScore e (fun j => θ (p, j)) s c := by
  calc
    _ = headLogits source (localHead e (fun j => θ (p, j))) s c := by
      simp [headLogits, bankSource, Fintype.sum_prod_type, mul_ite,
        Finset.sum_ite_irrel, head_block]
    _ = _ := local_source_logits e (fun j => θ (p, j)) s c

/-- The actual normalized complete complex head is exactly the sum of the
joint coordinate scores divided by one positive input-only denominator, CM57. -/
theorem head_logits (e : C → ℝ) (θ : P × Fin 6 → ℝ)
    (h : P → X → Fin 5 → ℝ) (x : X) (c : C) :
    headLogits (features h) (head e θ) x c =
      (∑ p, score e (fun j => θ (p, j)) (h p x) c) /
        SignedContext.normalizer (bankRow h x) := by
  rw [features, SignedContext.normalized_logits, Fintype.sum_prod_type]
  congr 1
  apply Finset.sum_congr rfl
  intro p hp
  simp only [bankRow, bank_source_logits]
  exact row_score e (fun j => θ (p, j)) (h p x) c

/-- The complete expansion cost sums over all programs and all real/imaginary
coordinates, not only the compact parameter count, CM57. -/
theorem expansion_bound (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) (p : P × Fin 6) (c : C) :
    (∑ k, |expansion e p k c|) ≤ costs p.2 := by
  simpa [expansion, Fintype.sum_prod_type, apply_ite abs, Finset.sum_ite_irrel] using
    local_expansion_bound e he p.2 c

/-- The global sufficient budget controls every complete raw-class column,
CM57. The perspective constraints are additional convex restrictions. -/
theorem head_budget (e : C → ℝ) (he : ∀ c, |e c| ≤ 1) {R : ℝ} {θ : P × Fin 6 → ℝ}
    (hθ : θ ∈ ParameterSharing.weightedDomain (fun p => costs p.2) R) :
    head e θ ∈ headBudget R := by
  simpa [head] using ParameterSharing.parameterMap_budget (expansion e) 0
    (fun p => costs p.2) (expansion_bound e he)
    (zero_mem_headBudget (le_refl (0 : ℝ))) hθ

end Transformer.Grokking.Convex.ConvexDepth.CoordinatePrograms
