module

public import Transformer.SequenceModel.Attention.Execution.Basic
public import Transformer.Precision.ArithmeticError

/-!
# Scalar execution certificates imply tensor operation bounds

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C1--C3.
These adapters discharge vector error contracts from represented scalar
brackets or explicit sequential product/add rounding. They do not assign
this reduction order to an unspecified CUDA matrix-multiplication kernel.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Stability Precision

/-- Coordinatewise rounding of a real operation result. -/
def roundedVector {d : ℕ} (Q : ℝ → ℝ) (x : EucSpace d) : EucSpace d :=
  WithLp.toLp 2 (fun c => Q (x c))

/-- C3: represented brackets at every coordinate imply a Euclidean operation
error. This applies when the selected primitive rounds its exact result. -/
theorem roundedVector_error {d : ℕ} {G : Set ℝ} {Q : ℝ → ℝ} (hQ : IsNearest G Q)
    (x : EucSpace d) (lo hi : Fin d → ℝ) (η : ℝ) (hη : 0 ≤ η)
    (hb : ∀ c, lo c ∈ G ∧ hi c ∈ G ∧ lo c ≤ x c ∧ x c ≤ hi c)
    (hw : ∀ c, (hi c - lo c) / 2 ≤ η) :
    ‖roundedVector Q x - x‖ ≤ Real.sqrt (d : ℝ) * η := by
  apply norm_le_coordinates _ η hη
  intro c
  obtain ⟨hl, hu, hxl, hxu⟩ := hb c
  exact (hQ.error_of_bracket hl hu hxl hxu).trans (hw c)

/-- A matrix product with a specified sequential reduction in each row. -/
def roundedMatrix {m : ℕ} (mulRound addRound : ℝ → ℝ) (n : ℕ)
    (A : Fin m → ℕ → ℝ) (x : ℕ → ℝ) : EucSpace m :=
  WithLp.toLp 2 (fun r => roundedDot mulRound addRound (A r) x n)

/-- C3: rowwise scalar product/add certificates give a matrix operation's
Euclidean error; both input and output widths are explicit. -/
theorem roundedMatrix_error {m : ℕ} (mulRound addRound : ℝ → ℝ) (n : ℕ)
    (A : Fin m → ℕ → ℝ) (x : ℕ → ℝ) (P R : ℝ) (hP : 0 ≤ P) (hR : 0 ≤ R)
    (hp : ∀ r k, k < n → |mulRound (A r k * x k) - A r k * x k| ≤ P)
    (ha : ∀ r k, k < n →
      |addRound (roundedDot mulRound addRound (A r) x k + mulRound (A r k * x k)) -
        (roundedDot mulRound addRound (A r) x k + mulRound (A r k * x k))| ≤ R) :
    ‖roundedMatrix mulRound addRound n A x -
      WithLp.toLp 2 (fun r => ∑ k ∈ Finset.range n, A r k * x k)‖ ≤
        Real.sqrt (m : ℝ) * ((n : ℝ) * (P + R)) := by
  apply norm_le_coordinates _ _ (by positivity)
  intro r
  exact roundedDot_uniform_error mulRound addRound (A r) x P R n (hp r) (ha r)

example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    ‖roundedVector Q (0 : EucSpace 4) - 0‖ ≤ Real.sqrt (4 : ℝ) * 0 := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  refine ⟨Q, hQ, roundedVector_error hQ 0 (fun _ => 0) (fun _ => 0) 0 (by norm_num) ?_ ?_⟩
  · intro c
    exact ⟨zero_mem_grid (by decide) 23, zero_mem_grid (by decide) 23, le_rfl, le_rfl⟩
  · intro c
    norm_num

example := roundedMatrix_error (m := 2) id id 3 (fun r k => (r.val + k : ℝ))
  (fun _ => 1) 0 0 (by norm_num) (by norm_num) (by intro r k hk; simp) (by intro r k hk; simp)

end Transformer.SequenceModel.Attention.Execution
