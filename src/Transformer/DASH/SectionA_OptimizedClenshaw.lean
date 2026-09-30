/-
# DASH — the complete optimized matrix Clenshaw algorithm

arXiv:2602.02016v2, Appendix A, `algorithm:cbshv-clenshaw-matrix-optimized`.
The returned natural number counts matrix-matrix products, excluding scalar
multiplication, additions, and construction of the normalized argument.
-/

import Transformer.DASH.SectionA_MatrixChebyshev

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- Backward evaluation with the two highest coefficients initialized without
matrix products. Each remaining recurrence performs exactly one product.
Source: arXiv:2602.02016v2, Appendix A, optimized Clenshaw initialization/loop. -/
def optimizedClenshawState (S : Matrix (Fin n) (Fin n) ℝ) :
    List ℝ → (Matrix (Fin n) (Fin n) ℝ × Matrix (Fin n) (Fin n) ℝ) × ℕ
  | [] => ((0, 0), 0)
  | [c] => ((c • 1, 0), 0)
  | [c₁, c₂] => (((2 * c₂) • S + c₁ • 1, c₂ • 1), 0)
  | c₁ :: c₂ :: c₃ :: cs =>
      let b := optimizedClenshawState S (c₂ :: c₃ :: cs)
      ((2 • (S * b.1.1) - b.1.2 + c₁ • 1, b.1.1), b.2 + 1)

/-- Full optimized algorithm, including the final product and harmless
degree-zero/one cases omitted from the paper's pseudocode.
Source: arXiv:2602.02016v2, Appendix A, optimized Clenshaw output. -/
def optimizedClenshaw (S : Matrix (Fin n) (Fin n) ℝ) :
    List ℝ → Matrix (Fin n) (Fin n) ℝ × ℕ
  | [] => (0, 0)
  | [c] => (c • 1, 0)
  | [c₀, c₁] => (c₁ • S + c₀ • 1, 0)
  | c₀ :: c₁ :: c₂ :: cs =>
      let b := optimizedClenshawState S (c₁ :: c₂ :: cs)
      (S * b.1.1 - b.1.2 + c₀ • 1, b.2 + 1)

/-- The optimized loop constructs exactly the ordinary backward state,
not merely the same final value. Source: arXiv:2602.02016v2, Appendix A,
the two eliminated highest-degree recurrence steps. -/
theorem optimizedClenshawState_eq (S : Matrix (Fin n) (Fin n) ℝ) (cs : List ℝ) :
    (optimizedClenshawState S cs).1 =
      clenshawState S (cs.map (fun c => c • (1 : Matrix (Fin n) (Fin n) ℝ))) := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    cases cs with
    | nil => simp [optimizedClenshawState, clenshawState]
    | cons c' cs =>
      cases cs with
      | nil =>
        simp [optimizedClenshawState, clenshawState, two_mul, add_smul]
      | cons c'' cs =>
        change (2 • (S * (optimizedClenshawState S (c' :: c'' :: cs)).1.1) -
            (optimizedClenshawState S (c' :: c'' :: cs)).1.2 + c • 1,
            (optimizedClenshawState S (c' :: c'' :: cs)).1.1) =
          (2 * S * (clenshawState S ((c' :: c'' :: cs).map (fun a => a • 1))).1 -
            (clenshawState S ((c' :: c'' :: cs).map (fun a => a • 1))).2 + c • 1,
            (clenshawState S ((c' :: c'' :: cs).map (fun a => a • 1))).1)
        rw [ih]
        congr 2
        simp [two_mul, two_smul, add_mul]

/-- The optimized backward loop uses `max(length-2,0)` matrix products.
Source: arXiv:2602.02016v2, Appendix A, elimination of the first two products. -/
theorem optimizedClenshawState_count (S : Matrix (Fin n) (Fin n) ℝ) (cs : List ℝ) :
    (optimizedClenshawState S cs).2 = cs.length - 2 := by
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    cases cs with
    | nil => rfl
    | cons c' cs =>
      cases cs with
      | nil => rfl
      | cons c'' cs =>
        simp only [optimizedClenshawState, ih, List.length_cons]
        omega

/-- Every degree, including the two small-degree boundary cases, returns the
same matrix polynomial as standard Clenshaw. Source: arXiv:2602.02016v2,
Appendix A, the final optimized Clenshaw algorithm. -/
theorem optimizedClenshaw_eq (S : Matrix (Fin n) (Fin n) ℝ) (cs : List ℝ) :
    (optimizedClenshaw S cs).1 =
      clenshaw S (cs.map (fun c => c • (1 : Matrix (Fin n) (Fin n) ℝ))) := by
  cases cs with
  | nil => simp [optimizedClenshaw, clenshaw, clenshawState]
  | cons c cs =>
    cases cs with
    | nil => simp [optimizedClenshaw, clenshaw, clenshawState]
    | cons c' cs =>
      cases cs with
      | nil =>
        simp [optimizedClenshaw, clenshaw_optimized_final, clenshawState, two_mul]
      | cons c'' cs =>
        simp only [optimizedClenshaw, List.map_cons, clenshaw_optimized_final,
          optimizedClenshawState_eq]

/-- A coefficient list of length `d+1` costs `d-1` matrix products; subtraction
is truncated for degrees zero and one. Unlike the printed loop, the formal
algorithm also handles these degrees. Source: arXiv:2602.02016v2, Appendix A,
the claim of `d-1` matrix multiplications. -/
theorem optimizedClenshaw_count (S : Matrix (Fin n) (Fin n) ℝ) (cs : List ℝ) :
    (optimizedClenshaw S cs).2 = cs.length - 2 := by
  cases cs with
  | nil => rfl
  | cons c cs =>
    cases cs with
    | nil => rfl
    | cons c' cs =>
      cases cs with
      | nil => rfl
      | cons c'' cs =>
        simp only [optimizedClenshaw, optimizedClenshawState_count, List.length_cons]
        omega

/-- Optimized evaluation retains the ordinary scalar spectral polynomial.
Source: arXiv:2602.02016v2, Appendix A, optimized matrix Clenshaw. -/
theorem optimizedClenshaw_spectrum (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (cs : List ℝ) (hQ : Orthogonal Q) :
    (optimizedClenshaw (spectralMatrix Q s) cs).1 =
      spectralMatrix Q (fun i => clenshaw (s i) cs) := by
  rw [optimizedClenshaw_eq, clenshaw_spectrum Q s cs hQ]

/-- Spectral assumptions are satisfiable, arXiv:2602.02016v2, Appendix A. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
  simp [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
