module

public import Transformer.Phasor.GeometricMemory

/-!
# Causal memory updated by learned affine projections

Source: the affine-projection trial in `experiments/phasor_value_transport/README.md`.
Each token and coordinate pair specifies a line. Reading the token projects
the pair's preceding state onto that line. Repeating a token has no further
effect, while projections for different tokens can depend on their order.
The adjoint differentiates both line parameters and the preceding state.
-/

@[expose] public section

namespace Transformer.Phasor

/-- Distance coefficient along the normal `(1, s)` to `x + s y = b`. -/
noncomputable def projectionShift (s b x y : ℝ) : ℝ :=
  (b - x - s * y) / (1 + s^2)

/-- One coordinate pair after the token writes its affine constraint. -/
noncomputable def affineMemoryWrite (s b : ℝ) (z : ℝ × ℝ) : ℝ × ℝ :=
  let a := projectionShift s b z.1 z.2
  (z.1 + a, z.2 + s * a)

private theorem projectionDenominator (s : ℝ) : 1 + s^2 ≠ 0 := by positivity

/-- Every write lands on the token's line for arbitrary learned parameters. -/
theorem affineMemoryWrite_on_line (s b : ℝ) (z : ℝ × ℝ) :
    (affineMemoryWrite s b z).1 + s * (affineMemoryWrite s b z).2 = b := by
  simp only [affineMemoryWrite, projectionShift]
  field_simp [projectionDenominator]
  ring

/-- Once the constraint holds, another occurrence leaves the state fixed. -/
theorem affineMemoryWrite_idempotent (s b : ℝ) (z : ℝ × ℝ) :
    affineMemoryWrite s b (affineMemoryWrite s b z) = affineMemoryWrite s b z := by
  let u := affineMemoryWrite s b z
  have hu : u.1 + s * u.2 = b := affineMemoryWrite_on_line _ _ _
  have shift : projectionShift s b u.1 u.2 = 0 := by
    unfold projectionShift
    rw [show b - u.1 - s * u.2 = 0 by linarith, zero_div]
  change (u.1 + projectionShift s b u.1 u.2, u.2 + s * projectionShift s b u.1 u.2) = u
  simp [shift]

private theorem fold_positive_writes {α : Type*} (s b : α → ℝ) (a : α) (n : ℕ) (z : ℝ × ℝ) :
    (List.replicate (n+1) a).foldl (fun u t => affineMemoryWrite (s t) (b t) u) z =
      affineMemoryWrite (s a) (b a) z := by
  induction n generalizing z with
  | zero => rfl
  | succ n ih =>
    rw [List.replicate_succ, List.foldl_cons, ih, affineMemoryWrite_idempotent]

/-- For arbitrary tokens and any initial state, independently lengthening
every run preserves the state, even when different tokens share parameters. -/
theorem affineMemory_stutter {α : Type*} (s b : α → ℝ) (runs : List (α × ℕ)) (z : ℝ × ℝ) :
    let write := fun u t => affineMemoryWrite (s t) (b t) u
    (runs.flatMap (fun p => List.replicate (p.2+1) p.1)).foldl write z =
      (runs.map Prod.fst).foldl write z := by
  induction runs generalizing z with
  | nil => rfl
  | cons run rest ih =>
    simp only [List.flatMap_cons, List.foldl_append, List.map_cons, List.foldl_cons,
      fold_positive_writes, ih]

/-- With zero intercepts the zero state remains zero on any word for any
slopes. The entire previous residual is thereby a submodel. -/
theorem affineMemory_zero_bias {α : Type*} (s : α → ℝ) (word : List α) :
    word.foldl (fun u t => affineMemoryWrite (s t) 0 u) (0, 0) = (0, 0) := by
  induction word with
  | nil => rfl
  | cons a rest ih => simpa [affineMemoryWrite, projectionShift] using ih

/-- First-order coefficient when the slope, intercept and both input
coordinates all vary. -/
noncomputable def projectionShiftVariation (s b x y ds db dx dy : ℝ) : ℝ :=
  (db - dx - s * dy - (y + 2 * s * projectionShift s b x y) * ds) / (1 + s^2)

/-- Exact second- and third-order terms after scaling by the positive
perturbed denominator. In particular the displayed velocity is the linear
part of the shift for a simultaneous perturbation of all four inputs. -/
theorem projectionShift_remainder (s b x y ds db dx dy t : ℝ) :
    let a := projectionShift s b x y
    let v := projectionShiftVariation s b x y ds db dx dy
    (1 + (s+t*ds)^2) * (projectionShift (s+t*ds) (b+t*db) (x+t*dx) (y+t*dy) - a - t*v) =
      -t^2 * (ds*dy + ds^2*a + 2*s*ds*v) - t^3 * ds^2*v := by
  simp only [projectionShiftVariation, projectionShift]
  field_simp [projectionDenominator]
  ring

/-- Pairing the output velocity with an arbitrary cotangent gives the
native reverse sweep's four contributions. The same two scalars suffice
to propagate the state gradient and accumulate both learned parameters. -/
theorem affineMemoryWrite_adjoint (s b x y ds db dx dy qx qy : ℝ) :
    let a := projectionShift s b x y
    let v := projectionShiftVariation s b x y ds db dx dy
    let k := (qx + s*qy) / (1 + s^2)
    qx * (dx+v) + qy * (dy+s*v+a*ds) =
      (qx-k)*dx + (qy-s*k)*dy + k*db + (a*qy-k*(y+2*s*a))*ds := by
  simp only [projectionShiftVariation]
  generalize projectionShift s b x y = a
  field_simp [projectionDenominator]
  ring

/-- Distinct tokens can write in different orders with different results. -/
example : affineMemoryWrite 1 0 (affineMemoryWrite 0 1 (0, 0)) ≠
    affineMemoryWrite 0 1 (affineMemoryWrite 1 0 (0, 0)) := by
  norm_num [affineMemoryWrite, projectionShift]

end Transformer.Phasor
