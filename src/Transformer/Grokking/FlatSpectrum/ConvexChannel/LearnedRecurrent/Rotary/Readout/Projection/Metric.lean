module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection.Magnitudes
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Domain

/-!
# The concrete complete-row formula is a Euclidean closest point

Source: `review/convex/moments.md`, CM102. The scalar optimization and complex
reverse-triangle bound establish the actual formula's closest-point property.
All classes/components remain admissible in the original learned model family.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection

open Pair
open scoped BigOperators

/-- CM102: concrete complete complex projection, including total zero cases. -/
noncomputable def project (z : State) : State :=
  (resize z.1 (magnitudes (‖z.1‖, ‖z.2‖)).1,
    resize z.2 (magnitudes (‖z.1‖, ‖z.2‖)).2)

/-- CM102: the actual complex formula realizes both projected magnitudes. -/
theorem project_norms (z : State) :
    (‖(project z).1‖, ‖(project z).2‖) = magnitudes (‖z.1‖, ‖z.2‖) := by
  have h := magnitude_bounds (‖z.1‖, ‖z.2‖) (norm_nonneg _) (norm_nonneg _)
  apply Prod.ext
  · exact resize_norm _ _ h.1 h.2.1
  · exact resize_norm _ _ h.2.2.1 h.2.2.2.1

/-- CM102: the concrete full complex projection satisfies the original row budget. -/
theorem project_bound (z : State) : rowSize (project z) ≤ 1 := by
  have h := magnitude_bounds (‖z.1‖, ‖z.2‖) (norm_nonneg _) (norm_nonneg _)
  have he := project_norms z
  have h₁ := congrArg Prod.fst he
  have h₂ := congrArg Prod.snd he
  dsimp only at h₁ h₂
  change ‖(project z).1‖ + ‖(project z).2‖ ≤ 1
  rw [h₁, h₂]
  exact h.2.2.2.2

/-- CM102: every already feasible complete complex row is unchanged. -/
theorem project_eq (z : State) (hz : rowSize z ≤ 1) : project z = z := by
  change ‖z.1‖ + ‖z.2‖ ≤ 1 at hz
  have hm : magnitudes (‖z.1‖, ‖z.2‖) = (‖z.1‖, ‖z.2‖) := by
    simp [magnitudes, hz]
  simp only [project, hm, resize_self]

/-- CM102: actual phase-preserving Euclidean distance equals the scalar optimum. -/
theorem project_distance (z : State) :
    squaredDistance z (project z) =
      (‖z.1‖ - (magnitudes (‖z.1‖, ‖z.2‖)).1) ^ 2 +
      (‖z.2‖ - (magnitudes (‖z.1‖, ‖z.2‖)).2) ^ 2 := by
  have h := magnitude_bounds (‖z.1‖, ‖z.2‖) (norm_nonneg _) (norm_nonneg _)
  change ‖z.1 - resize z.1 _‖ ^ 2 + ‖z.2 - resize z.2 _‖ ^ 2 = _
  rw [resize_distance _ _ h.1 h.2.1,
    resize_distance _ _ h.2.2.1 h.2.2.2.1]

/-- CM102: the explicit formula is a closest point over every feasible complex row. -/
theorem project_closest (z v : State) (hv : rowSize v ≤ 1) :
    squaredDistance z (project z) ≤ squaredDistance z v := by
  rw [project_distance]
  calc
    _ ≤ (‖z.1‖ - ‖v.1‖) ^ 2 + (‖z.2‖ - ‖v.2‖) ^ 2 :=
      magnitude_optimal (‖z.1‖, ‖z.2‖) (norm_nonneg _) (norm_nonneg _) _ _
        (norm_nonneg _) (norm_nonneg _) hv
    _ ≤ squaredDistance z v :=
      add_le_add (magnitude_distance_le _ _) (magnitude_distance_le _ _)

/-- CM102: every original class/component keeps a complete feasible readout. -/
theorem table_mem (J O : Type*) (u : J → O → State) :
    (fun j c => project (u j c)) ∈ rowDomain J O := fun j c => project_bound (u j c)

/-- CM102: full learned recurrent content survives the new constrained step. -/
theorem replacement_unit {O : Type*} (f : Parameters O) (hf : Unit f)
    (u : O → State) : Unit (replace f (fun c => project (u c))) :=
  replace_unit f hf _ (fun c => project_bound (u c))

/-- CM102: the entire projected class/component table stays in the model capacity. -/
theorem model_feasible {J O : Type*} [Fintype J] (B : ℝ) (w : J → ℝ)
    (f : J → Parameters O) (hw : ∀ j, 0 ≤ w j) (hm : ∑ j, w j = 1)
    (hf : ∀ j, Unit (f j)) (u : J → O → State) :
    scoreMap B w f (fun j c => project (u j c)) ∈ domain O B :=
  score_feasible B w f hw hm hf _ (table_mem _ _ u)

example : project (Complex.I, 0) = (Complex.I, 0) :=
  project_eq _ (by simp [rowSize])

example : squaredDistance (3 * Complex.I, 2) (project (3 * Complex.I, 2)) ≤
    squaredDistance (3 * Complex.I, 2) ((1 / 2 : ℝ) • Complex.I, (1 / 2 : ℂ)) :=
  project_closest _ _ (by norm_num [rowSize, norm_smul, norm_div])

example : Unit (replace positionWitness (fun c => project
    (if c then (2 * Complex.I, 1) else (1, 2 * Complex.I)))) :=
  replacement_unit _ position_unit _

example : scoreMap 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    (fun _ c => project (if c then (2 * Complex.I, 1) else (1, 2 * Complex.I))) ∈
      domain Bool 16 :=
  model_feasible _ _ _ (by norm_num) (by simp) (fun _ => position_unit) _

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Projection
