module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Domain

/-!
# The fully trainable scalar complex recurrence

Source: `review/flat_spectrum.md`, CM95. This is a concrete one-dimensional
member of the general operator family: every transition, initial state and
readout is a learned complex scalar. It is commutative; higher-dimensional
operators are needed for general order-dependent behavior.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Scalar

variable {O : Type*}

/-- Executable complex scalar multiplication as a continuous linear map,
CM95. Real and imaginary entries are both numerical parameters. -/
noncomputable def multiply (z : ℂ) : ℂ →L[ℂ] ℂ := z • ContinuousLinearMap.id ℂ ℂ

/-- The actual scalar transition is ordinary complex multiplication,
CM95. -/
theorem multiply_apply (z v : ℂ) : multiply z v = z * v := by
  simp [multiply]

/-- Its operator capacity is exactly the scalar magnitude, CM95. -/
theorem multiply_norm (z : ℂ) : ‖multiply z‖ = ‖z‖ := by
  simp [multiply, norm_smul]

/-- All scalar numerical paths remain trainable, CM95. This declaration
contains no target rule, prescribed phase or object interpretation. -/
noncomputable def parameters (initial : ℂ) (transition output : O → ℂ) : Parameters O ℂ :=
  ⟨initial, fun o => multiply (transition o), fun c => multiply (output c)⟩

/-- Complete scalar capacities imply the general recurrent bound, CM95. -/
theorem unit (a : ℂ) (t u : O → ℂ) (ha : ‖a‖ ≤ 1)
    (ht : ∀ o, ‖t o‖ ≤ 1) (hu : ∀ c, ‖u c‖ ≤ 1) : Unit (parameters a t u) := by
  exact ⟨ha, fun o => by simpa [parameters, multiply_norm] using ht o,
    fun c => by simpa [parameters, multiply_norm] using hu c⟩

/-- Every repeated object executes its learned transition again, CM95.
The product identity is specific to the scalar subfamily. -/
theorem run_product (a : ℂ) (t u : O → ℂ) (v : ℂ) (xs : List O) :
    runFrom (parameters a t u) v xs = v * (xs.map t).prod := by
  induction xs generalizing v with
  | nil => simp [runFrom]
  | cons o xs ih =>
    change runFrom (parameters a t u) (multiply (t o) v) xs = _
    rw [ih, multiply_apply]
    simp only [List.map_cons, List.prod_cons]
    ring

/-- Complete executed complex output of the learned scalar family,
CM95, before taking the real class score. -/
theorem complex_product (a : ℂ) (t u : O → ℂ) (xs : List O) (c : O) :
    complex (parameters a t u) xs c = u c * (a * (xs.map t).prod) := by
  change multiply (u c) (runFrom (parameters a t u) a xs) = _
  rw [multiply_apply, run_product]

example : Unit (parameters Complex.I (fun _ : Bool => Complex.I)
    (fun c : Bool => if c then Complex.I else -Complex.I)) :=
  unit _ _ _ (by simp) (by intro o; simp) (by intro c; cases c <;> simp)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Scalar
