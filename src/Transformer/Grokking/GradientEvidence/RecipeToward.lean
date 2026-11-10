module

public import Transformer.Grokking.GradientEvidence.RecipeRescale

/-!
# A decay toward a fixed value

Source: `review/flat_spectrum.md`, *The numbers of the tokens* (added), item NT3.
The AdamW of the main model shrinks a coordinate toward zero: `p ← (1 − η ρ) p − η u`. With a
fixed value `c` for the coordinate, add `η ρ c` after the update (`mainAdamPathToward`):

`p ← (1 − η ρ) p − η u + η ρ c = p − η ρ (p − c) − η u`

(`mainAdamPathToward_succ`). The moments are those of the same gradients. Then `p − c`
follows the path of RR1 from `p₀ − c` (`mainAdamPathToward_eq`), and a coordinate without
gradient returns to `c` by the product of the decay factors (`mainAdamPathToward_zero`): what
the gradients do not feed is left at the fixed value, not at zero.

In closed loop, with a gradient field `G` on all coordinates and a table `c` of fixed values
(`mainAdamRunToward`): the run is `c` plus the run, with the decay toward zero, of the field
`q ↦ G (c + q)` from `p₀ − c`, with the same moments (`mainAdamRunToward_eq`). So it is the run
of the model whose weights are `c + q`, with `q` decayed toward zero. A coordinate whose
gradient is zero at every point returns to its fixed value (`mainAdamRunToward_of_zero`).
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

open Finset

/-- The path of a coordinate under the AdamW of the main model with `η ρ c` added after every
update: a decay toward `c`. -/
def mainAdamPathToward (c p0 ρ eps : ℝ) (η g : ℕ → ℝ) : ℕ → ℝ
  | 0 => p0
  | t + 1 => (1 - η t * ρ) * mainAdamPathToward c p0 ρ eps η g t -
      η t * mainDirection g eps t + η t * ρ * c

/-- **NT3**, `review/flat_spectrum.md`: the added term turns the decay toward `c`. -/
theorem mainAdamPathToward_succ (c p0 ρ eps : ℝ) (η g : ℕ → ℝ) (t : ℕ) :
    mainAdamPathToward c p0 ρ eps η g (t + 1) = mainAdamPathToward c p0 ρ eps η g t -
      η t * ρ * (mainAdamPathToward c p0 ρ eps η g t - c) - η t * mainDirection g eps t := by
  rw [mainAdamPathToward]
  ring

/-- **NT3**, `review/flat_spectrum.md`: the distance to `c` follows the path of RR1. -/
theorem mainAdamPathToward_eq (c p0 ρ eps : ℝ) (η g : ℕ → ℝ) (t : ℕ) :
    mainAdamPathToward c p0 ρ eps η g t = c + mainAdamPath (p0 - c) ρ eps η g t := by
  induction t with
  | zero => simp [mainAdamPathToward, mainAdamPath]
  | succ t ih =>
    rw [mainAdamPathToward, ih, mainAdamPath]
    ring

/-- With `c = 0` it is the path of the main model. -/
theorem mainAdamPathToward_zero_value (p0 ρ eps : ℝ) (η g : ℕ → ℝ) (t : ℕ) :
    mainAdamPathToward 0 p0 ρ eps η g t = mainAdamPath p0 ρ eps η g t := by
  rw [mainAdamPathToward_eq, sub_zero, zero_add]

/-- **NT3**, `review/flat_spectrum.md`: a coordinate without gradient returns to `c` by the
product of the decay factors of the schedule. -/
theorem mainAdamPathToward_zero (c p0 ρ eps : ℝ) (η : ℕ → ℝ) (t : ℕ) :
    mainAdamPathToward c p0 ρ eps η (fun _ => 0) t =
      c + (∏ s ∈ range t, (1 - η s * ρ)) * (p0 - c) := by
  rw [mainAdamPathToward_eq, mainAdamPath_zero]

/-- A constant rate `0.001` with the decay `10`: after 5000 updates a coordinate without
gradient is within `exp (-50)` of its distance from `c` at the start. -/
theorem mainAdamPathToward_remainder (c p0 eps : ℝ) :
    |mainAdamPathToward c p0 10 eps (fun _ => 1 / 1000) (fun _ => 0) 5000 - c| ≤
      Real.exp (-50) * |p0 - c| := by
  rw [mainAdamPathToward_zero, add_sub_cancel_left, prod_const, card_range, abs_mul]
  refine mul_le_mul_of_nonneg_right ?_ (abs_nonneg _)
  have h := one_sub_pow_le_exp (x := 1 / 1000 * 10) (by norm_num) 5000
  rw [abs_of_nonneg (pow_nonneg (by norm_num) _)]
  refine h.trans (le_of_eq ?_)
  norm_num

variable {ι : Type*}

/-- One update of every coordinate with the decay toward the table `c`: the update of the main
model, then `η ρ c` added to the weights. -/
def mainAdamStepToward (c : ι → ℝ) (G : (ι → ℝ) → ι → ℝ) (ρ eps η : ℝ) (t : ℕ)
    (s : (ι → ℝ) × (ι → ℝ) × (ι → ℝ)) : (ι → ℝ) × (ι → ℝ) × (ι → ℝ) :=
  (fun i => (mainAdamStep G ρ eps η t s).1 i + η * ρ * c i, (mainAdamStep G ρ eps η t s).2)

/-- The main model's AdamW with the decay toward the table `c`, from `p0` with zero moments,
the gradient taken at the weights. -/
def mainAdamRunToward (c : ι → ℝ) (G : (ι → ℝ) → ι → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ)
    (p0 : ι → ℝ) : ℕ → (ι → ℝ) × (ι → ℝ) × (ι → ℝ)
  | 0 => (p0, fun _ => 0, fun _ => 0)
  | t + 1 => mainAdamStepToward c G ρ eps (η t) t (mainAdamRunToward c G ρ eps η p0 t)

/-- **NT3 in closed loop**, `review/flat_spectrum.md`: the run with the decay toward the table
`c` is `c` plus the run of the field `q ↦ G (c + q)` from `p0 − c` with the decay toward zero,
and its moments are the moments of that run. -/
theorem mainAdamRunToward_eq (c : ι → ℝ) (G : (ι → ℝ) → ι → ℝ) (ρ eps : ℝ) (η : ℕ → ℝ)
    (p0 : ι → ℝ) (t : ℕ) :
    mainAdamRunToward c G ρ eps η p0 t =
      (fun i => c i + (mainAdamRun (fun q => G fun j => c j + q j) ρ eps η
          (fun i => p0 i - c i) t).1 i,
        (mainAdamRun (fun q => G fun j => c j + q j) ρ eps η (fun i => p0 i - c i) t).2) := by
  induction t with
  | zero => simp [mainAdamRunToward, mainAdamRun]
  | succ t ih =>
    rw [mainAdamRunToward, ih, mainAdamRun]
    simp only [mainAdamStepToward, mainAdamStep, bufferedAdamUpdate]
    refine Prod.ext (funext fun i => ?_) rfl
    ring

/-- **NT3 in closed loop**, `review/flat_spectrum.md`: a coordinate whose gradient is zero at
every point returns to its fixed value by the product of the decay factors. -/
theorem mainAdamRunToward_of_zero (c : ι → ℝ) (G : (ι → ℝ) → ι → ℝ) (i : ι)
    (hi : ∀ w, G w i = 0) (ρ eps : ℝ) (η : ℕ → ℝ) (p0 : ι → ℝ) (t : ℕ) :
    (mainAdamRunToward c G ρ eps η p0 t).1 i =
      c i + (∏ s ∈ range t, (1 - η s * ρ)) * (p0 i - c i) := by
  rw [mainAdamRunToward_eq]
  exact congrArg (c i + ·) (mainAdamRun_of_zero _ i (fun _ => hi _) ρ eps η _ t)

/-- The second of two coordinates has no gradient and the fixed value `1`; the first has the
gradient `1 / w`. -/
example (t : ℕ) :=
  mainAdamRunToward_of_zero (fun _ => 1) (fun w (i : Fin 2) => if i = 0 then 1 / w 0 else 0) 1
    (fun _ => by simp) (1 / 10) (1 / 100000000) (fun _ => 1 / 1000) (fun _ => 2) t

end Transformer.Grokking.GradientEvidence
