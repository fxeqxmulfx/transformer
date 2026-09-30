/-
# AdaFisher: Fisher blocks and empirical second moments

Gomes et al., arXiv:2405.16397v3, §2, equation `eq:fishermatrix`.
Finite weighted samples describe empirical expectations. Factorization of
an expectation is proved for the product sampling law, not assumed for
arbitrary coupled activations and sensitivities.
-/

import Mathlib.LinearAlgebra.Matrix.Kronecker
import Mathlib.Probability.Independence.Basic
import Mathlib.Tactic

open scoped BigOperators Matrix Kronecker

noncomputable section

namespace Transformer.AdaFisher

variable {a b n m : ℕ}

/-- Column-stacked layer gradient, §2: `vec(s hᵀ) = h ⊗ s`.
The first coordinate is the input column and the second the output row. -/
def layerGradient (h : Fin a → ℝ) (s : Fin b → ℝ) : Fin a × Fin b → ℝ :=
  fun p => s p.2 * h p.1

/-- Outer product used in §2, equation `eq:fishermatrix`. -/
def outer {ι κ : Type*} (u : ι → ℝ) (v : κ → ℝ) : Matrix ι κ ℝ :=
  fun i j => u i * v j

/-- Finite weighted second moment in §2. This is an uncentered moment. -/
def secondMoment (w : Fin n → ℝ) (x : Fin n → Fin a → ℝ) : Matrix (Fin a) (Fin a) ℝ :=
  fun i j => ∑ k, w k * x k i * x k j

/-- Empirical Fisher block before K-FAC factorization, §2. -/
def empiricalFisher (w : Fin n → ℝ) (h : Fin n → Fin a → ℝ)
    (s : Fin n → Fin b → ℝ) : Matrix (Fin a × Fin b) (Fin a × Fin b) ℝ :=
  fun p q => ∑ k, w k * layerGradient (h k) (s k) p * layerGradient (h k) (s k) q

/-- The outer product of a vectorized layer gradient is a Kronecker
product of outer products, §2, the equality preceding K-FAC's approximation. -/
theorem layerGradient_outer (h : Fin a → ℝ) (s : Fin b → ℝ) :
    outer (layerGradient h s) (layerGradient h s) = outer h h ⊗ₖ outer s s := by
  ext p q
  rcases p with ⟨pi, pj⟩
  rcases q with ⟨qi, qj⟩
  simp only [outer, layerGradient, Matrix.kronecker_apply]
  ring

/-- Fisher's quadratic form is nonnegative for nonnegative sample weights,
§2, equation `eq:fishermatrix`. No centeredness or independence is needed. -/
theorem empiricalFisher_quadratic_nonneg (w : Fin n → ℝ)
    (h : Fin n → Fin a → ℝ) (s : Fin n → Fin b → ℝ)
    (hw : ∀ k, 0 ≤ w k) (v : Fin a × Fin b → ℝ) :
    0 ≤ ∑ p, ∑ q, v p * empiricalFisher w h s p q * v q := by
  have heq : (∑ p, ∑ q, v p * empiricalFisher w h s p q * v q) =
      ∑ k, w k * (∑ p, v p * layerGradient (h k) (s k) p) ^ 2 := by
    simp only [empiricalFisher, Finset.mul_sum, Finset.sum_mul, pow_two]
    calc
      _ = ∑ p, ∑ k, ∑ q,
          v p * (w k * layerGradient (h k) (s k) p *
            layerGradient (h k) (s k) q) * v q := by
        apply Finset.sum_congr rfl
        intro p hp
        rw [Finset.sum_comm]
      _ = ∑ k, ∑ p, ∑ q,
          v p * (w k * layerGradient (h k) (s k) p *
            layerGradient (h k) (s k) q) * v q := Finset.sum_comm
      _ = _ := by
        apply Finset.sum_congr rfl
        intro k hk
        apply Finset.sum_congr rfl
        intro p hp
        apply Finset.sum_congr rfl
        intro q hq
        ring
  rw [heq]
  exact Finset.sum_nonneg fun k _ => mul_nonneg (hw k) (sq_nonneg _)

example : ∀ k : Fin 1, 0 ≤ (fun _ => (1 : ℝ)) k := by norm_num

/-- Exact K-FAC factorization under independent (product) finite sampling,
§2. The source's approximation becomes an equality under this law. -/
theorem productSampling_fisher (w : Fin n → ℝ) (z : Fin m → ℝ)
    (h : Fin n → Fin a → ℝ) (s : Fin m → Fin b → ℝ) :
    (fun p q => ∑ k, ∑ l, (w k * z l) *
      layerGradient (h k) (s l) p * layerGradient (h k) (s l) q) =
      secondMoment w h ⊗ₖ secondMoment z s := by
  ext p q
  rcases p with ⟨pi, pj⟩
  rcases q with ⟨qi, qj⟩
  simp only [secondMoment, Matrix.kronecker_apply, layerGradient,
    Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k hk
  apply Finset.sum_congr rfl
  intro l hl
  ring

/-- Independence alone does not diagonalize the uncentered second moment,
Appendix A.2, “Part 1: Diagonalization of KFs”. Constant activations are
independent, yet their off-diagonal second moment equals one. A zero-mean
assumption is needed to identify this matrix with a covariance matrix. -/
theorem independent_constants_secondMoment_not_diagonal :
    ProbabilityTheory.IndepFun (fun _ : ℝ => (1 : ℝ)) (fun _ : ℝ => (1 : ℝ))
      (MeasureTheory.Measure.dirac (0 : ℝ)) ∧
    secondMoment (fun _ : Fin 1 => (1 : ℝ))
      (fun _ : Fin 1 => fun _ : Fin 2 => (1 : ℝ)) 0 1 = 1 ∧
    secondMoment (fun _ : Fin 1 => (1 : ℝ))
      (fun _ : Fin 1 => fun _ : Fin 2 => (1 : ℝ)) 0 1 ≠ 0 := by
  refine ⟨ProbabilityTheory.indepFun_const_left _ _, ?_⟩
  norm_num [secondMoment, Fin.sum_univ_one]

end Transformer.AdaFisher
