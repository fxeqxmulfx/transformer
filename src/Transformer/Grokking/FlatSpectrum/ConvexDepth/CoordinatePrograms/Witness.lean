module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms.Identification

/-!
# Multiple active decoders and an unseen raw answer

Source: `review/flat_spectrum.md`, CM57. Two distinct quantity coordinates have
different unknown affine decoders. A third constant program cannot satisfy the
two distinct observed answers at positive mass. Training has repeated records;
class 3 and the copy query are absent. All claims use the actual complete head.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms

open scoped BigOperators

noncomputable def exampleCoordinates := ClassGeometry.exampleCoordinates

/-- Fixed input-only quantities: two affine representations and one constant,
CM57. -/
noncomputable def exampleQuantity (p : Fin 3) (x : Fin 6) : ℝ :=
  if p = 0 then (x.val : ℝ) / 2 else if p = 1 then (x.val : ℝ) - 1 else 0

/-- Fixed quantity, structural-role and copy features, CM57. Their construction
does not consult the target function or training labels. -/
noncomputable def exampleFeatures (p : Fin 3) (x : Fin 6) : Fin 5 → ℝ :=
  if x.val < 3 then ![exampleQuantity p x, 1, 0, 0, 0]
  else if x = 3 then ![0, 0, 1, 0, 0]
  else if x = 4 then ![0, 0, 0, 1, 0] else ![0, 0, 0, 0, 1]

def exampleTarget : Fin 6 → Fin 5 := ![1, 2, 3, 0, 4, 4]
def exampleInputs : Fin 5 → Fin 6 := ![0, 0, 1, 3, 4]
def exampleLabels (a : Fin 5) : Fin 5 := exampleTarget (exampleInputs a)
noncomputable def exampleRows (p : Fin 3) (a : Fin 5) := exampleFeatures p (exampleInputs a)
noncomputable def exampleValues (a : Fin 5) : ℝ := exampleCoordinates (exampleLabels a)

/-- Two distinct scaled decoders have masses four and two; the third is zero,
CM57. The copy coefficients are tied to their respective masses. -/
noncomputable def exampleParameters (p : Fin 3 × Fin 6) : ℝ :=
  ![![2, 1, 0, 4, 4, 4], ![1 / 2, 1, 0, 2, 2, 2], ![0, 0, 0, 0, 0, 0]] p.1 p.2

/-- Witnesses to semantic realizability, not independently learned model
factors; active programs have different slopes and intercepts, CM57. -/
noncomputable def exampleDecoder (p : Fin 3) : Fin 5 → ℝ :=
  if p = 0 then ![1 / 2, 1 / 4, 0, 1, 1] else ![1 / 4, 1 / 2, 0, 1, 1]

/-- Signed extrapolation from the two distinct quantity anchors, CM57. -/
noncomputable def exampleWeights : Option (Fin 5) → ℝ
  | none => 0
  | some a => ![-1, 0, 2, 0, 0] a

/-- The nonzero joint point has complete charge78, within radius128; the
perspective budget is16 per unit mass. All training constraints hold, CM57. -/
theorem example_feasible : exampleParameters ∈ domain exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1 := by
  norm_num [domain, ParameterSharing.weightedDomain, Fintype.sum_prod_type,
    Fin.sum_univ_succ, costs, charge, exampleParameters, center,
    exampleRows, exampleValues, exampleFeatures, exampleQuantity,
    exampleCoordinates, ClassGeometry.exampleCoordinates, exampleLabels,
    exampleTarget, exampleInputs, Fin.forall_fin_succ]

/-- Both active feature/decoder pairs realize the nonconstant target, including
the unseen class and the explicit copy direction, CM57. -/
theorem example_realizable (p : Fin 3) (hp : p ≠ 2) (x : Fin 6) :
    exampleCoordinates (exampleTarget x) = dot (exampleDecoder p) (exampleFeatures p x) := by
  fin_cases p <;> fin_cases x <;>
    norm_num [exampleCoordinates, ClassGeometry.exampleCoordinates, exampleTarget,
      dot, exampleDecoder, exampleFeatures, exampleQuantity, Fin.sum_univ_succ] at *

/-- The quantity query has an observable signed span certificate for both
active programs. No unseen answer occurs in this certificate, CM57. -/
theorem example_span (p : Fin 3) (j : Fin 5) :
    exampleFeatures p 2 j = ∑ a, exampleWeights a * augmentedRows (exampleRows p) a j := by
  fin_cases p <;> fin_cases j <;>
    norm_num [exampleFeatures, exampleQuantity, exampleWeights, augmentedRows,
      exampleRows, exampleInputs, copyRow, Fintype.sum_option, Fin.sum_univ_succ]

/-- A constant quantity program is forced inactive by distinct observed class
coordinates, rather than removed by an assumed correct-program index, CM57. -/
theorem rejected_program {θ : Fin 3 × Fin 6 → ℝ}
    (hθ : θ ∈ domain exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1) :
    ∀ j, θ (2, j) = 0 := by
  have h0 := hθ.2.2.2.2.2 2 0
  have h1 := hθ.2.2.2.2.2 2 2
  norm_num [center, exampleRows, exampleFeatures, exampleQuantity, exampleInputs,
    exampleValues, exampleCoordinates, ClassGeometry.exampleCoordinates,
    exampleLabels, exampleTarget, Fin.sum_univ_succ] at h0 h1
  have hg : θ (2, 5) = 0 := by linarith
  exact zero_mass (fun j => θ (2, j)) 16 (hθ.2.2.2.1 2) hg

/-- Coverage and semantic realizability hold for every active program of
every feasible witness head, CM57. The rejected program cannot be active. -/
theorem example_query_coverage {θ : Fin 3 × Fin 6 → ℝ}
    (hθ : θ ∈ domain exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1) :
    ∀ p, 0 < θ (p, 5) → ∃ (β : Fin 5 → ℝ) (weights : Option (Fin 5) → ℝ),
      β 4 = 1 ∧ (∀ a, exampleValues a = dot β (exampleRows p a)) ∧
      exampleCoordinates 3 = dot β (exampleFeatures p 2) ∧
      ∀ j, exampleFeatures p 2 j = ∑ a, weights a * augmentedRows (exampleRows p) a j := by
  intro p hp
  have hn : p ≠ 2 := by
    intro he
    subst p
    rw [rejected_program hθ 5] at hp
    linarith
  refine ⟨exampleDecoder p, exampleWeights, ?_, ?_, ?_, example_span p⟩
  · fin_cases p <;> norm_num [exampleDecoder]
  · intro a
    exact example_realizable p hn (exampleInputs a)
  · exact example_realizable p hn 2

/-- Every feasible joint head identifies the unseen quantity answer, not just
the explicitly displayed parameter point. The active decoders and span are
verified from the witness definitions, CM57. -/
theorem example_unseen_prediction {θ : Fin 3 × Fin 6 → ℝ}
    (hθ : θ ∈ domain exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1)
    (c : Fin 5) (hc : c ≠ 3) :
    headLogits (features exampleFeatures) (head exampleCoordinates θ) 2 c <
      headLogits (features exampleFeatures) (head exampleCoordinates θ) 2 3 :=
  identified_prediction exampleCoordinates ClassGeometry.exampleCoordinates_injective
    exampleFeatures exampleRows (fun _ => exampleValues) (fun _ => 16) (by norm_num : (0 : ℝ) < 1)
    hθ 2 3 c hc (example_query_coverage hθ)

example := example_realizable 1 (by decide) 2
example := rejected_program example_feasible
example := example_query_coverage example_feasible
example := example_unseen_prediction example_feasible 4 (by decide)

example : exampleLabels 0 = exampleLabels 1 ∧
    (∀ a, exampleLabels a ≠ 3) ∧ (∀ a, exampleInputs a ≠ 5) := by
  decide

example : exampleParameters (0, 5) > 0 ∧ exampleParameters (1, 5) > 0 ∧
    exampleDecoder 0 0 ≠ exampleDecoder 1 0 := by
  norm_num [exampleParameters, exampleDecoder]

/-- A pure copy query is covered by the declared copy-prior row alone, CM57. -/
noncomputable def copyWeights : Option (Fin 5) → ℝ
  | none => 1
  | some _ => 0

/-- Every feasible head identifies the unseen copy query using the explicitly
supplied copy geometry. It is not inferred from flatness or a seen query, CM57. -/
theorem example_copy_prediction {θ : Fin 3 × Fin 6 → ℝ}
    (hθ : θ ∈ domain exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1)
    (c : Fin 5) (hc : c ≠ 4) :
    headLogits (features exampleFeatures) (head exampleCoordinates θ) 5 c <
      headLogits (features exampleFeatures) (head exampleCoordinates θ) 5 4 := by
  apply identified_prediction exampleCoordinates ClassGeometry.exampleCoordinates_injective
    exampleFeatures exampleRows (fun _ => exampleValues) (fun _ => 16) (by norm_num : (0 : ℝ) < 1)
    hθ 5 4 c hc
  intro p hp
  have hn : p ≠ 2 := by
    intro he
    subst p
    rw [rejected_program hθ 5] at hp
    linarith
  refine ⟨exampleDecoder p, copyWeights, ?_, ?_, ?_, ?_⟩
  · fin_cases p <;> norm_num [exampleDecoder]
  · intro a
    exact example_realizable p hn (exampleInputs a)
  · exact example_realizable p hn 5
  · intro j
    fin_cases j <;> norm_num [exampleFeatures, copyWeights, augmentedRows, copyRow,
      Fintype.sum_option, Fin.sum_univ_succ]

example := example_copy_prediction example_feasible 0 (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms
