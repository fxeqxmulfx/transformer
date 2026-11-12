module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Basic

/-!
# Feasibility repair without changing predictions

Source: `review/flat_spectrum.md`, CM4. Mixing a probability table with the uniform table
scales every centered complex moment. A single positive mixing coefficient can enforce
all spectral bounds while preserving strict class comparisons. This is an algebraic
certificate conditional on the norm bounds, not a certificate for floating-point SVD.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

open scoped Matrix.Norms.L2Operator

variable {m n C F : Type*} [Fintype m] [Fintype n] [DecidableEq n] [Fintype C]

/-- The numerical feasibility repair is one common mixture with uniform mass (CM4). -/
noncomputable def blend (α : ℝ) (P : Table m n C) : Table m n C :=
  α • P + (1 - α) • uniform

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- Centered moments scale exactly under the common mixture (CM4). -/
theorem moment_blend {φ : C → ℂ} (hφ : ∑ c, φ c = 0) (α : ℝ) (P : Table m n C) :
    moment φ (blend α P) = α • moment φ P := by
  rw [blend, moment_linear, moment_uniform_zero hφ, smul_zero, add_zero]

/-- A suitable common coefficient enforces every spectral bound while preserving the
probability constraints (CM4). -/
theorem feasible_blend [Nonempty C] {φ : F → C → ℂ} {R : F → ℝ} {P : Table m n C}
    (hP : Rows P) (hφ : ∀ f, ∑ c, φ f c = 0) {α : ℝ} (ha : 0 ≤ α) (ha1 : α ≤ 1)
    (hb : ∀ f, α * ‖moment (φ f) P‖ ≤ R f) : Feasible φ R (blend α P) := by
  refine ⟨rows_convex hP rows_uniform ha (sub_nonneg.mpr ha1) (by ring), ?_⟩
  intro f
  rw [moment_blend (hφ f), norm_smul, Real.norm_of_nonneg ha]
  exact hb f

omit [Fintype m] [Fintype n] [DecidableEq n] in
/-- A positive common coefficient preserves every strict class comparison, hence any
fixed tie-breaking argmax (CM4). -/
theorem blend_lt_iff {α : ℝ} (ha : 0 < α) (P : Table m n C) (i : m) (j : n) (c d : C) :
    blend α P i j c < blend α P i j d ↔ P i j c < P i j d := by
  change α * P i j c + (1 - α) * (Fintype.card C : ℝ)⁻¹ <
    α * P i j d + (1 - α) * (Fintype.card C : ℝ)⁻¹ ↔ _
  rw [add_lt_add_iff_right, mul_lt_mul_iff_right₀ ha]

example := moment_blend (m := Fin 1) (n := Fin 1)
  (φ := fun c : Bool => if c then (1 : ℂ) else -1) (by simp) (1 / 2) uniform

example := feasible_blend (m := Fin 1) (n := Fin 1) (P := uniform)
  (φ := fun (_ : Fin 1) (c : Bool) => if c then Complex.I else -Complex.I)
  (R := fun _ => 1) rows_uniform (by intro f; simp)
  (α := 1 / 2) (by norm_num) (by norm_num)
  (by intro f; rw [moment_uniform_zero (by simp), norm_zero]; norm_num)

example := blend_lt_iff (by norm_num : (0 : ℝ) < 1 / 2)
  (pointMass (fun (_ : Fin 1) (_ : Fin 1) => true)) 0 0 false true

end Transformer.Grokking.FlatSpectrum.CoupledMoments
