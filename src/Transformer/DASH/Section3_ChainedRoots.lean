/-
# DASH — chaining the square-root solvers

arXiv:2602.02016v2, §3.3. A square-root call followed by an inverse
square-root call computes the inverse fourth root. Iteration convergence
is proved on the domain required by both calls.
-/

import Transformer.DASH.Section3_InverseRootCorrectness

open scoped Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The square-root spectrum is still inside NDB's convergence interval.
Source: arXiv:2602.02016v2, §3.3, the second NDB call for an inverse fourth root. -/
theorem squareRoot_ndb_domain (s : Fin n → ℝ) (hs : ∀ i, 0 < s i)
    (hupper : ∀ i, s i < 2) (i : Fin n) :
    0 < s i ^ (1 / 2 : ℝ) ∧ s i ^ (1 / 2 : ℝ) < 2 := by
  constructor
  · exact Real.rpow_pos_of_pos (hs i) _
  · rw [← Real.sqrt_eq_rpow]
    apply (Real.sqrt_lt (hs i).le (by norm_num : (0 : ℝ) ≤ 2)).2
    nlinarith [hupper i]

/-- The two-call spectral domain is satisfiable, arXiv:2602.02016v2, §3.3. -/
example : (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i < 2) := by norm_num

/-- NDB on the exact square root converges to the original inverse fourth
root. The first call's square root is its proved limit; no finite first-call
approximation is silently replaced by that limit.
Source: arXiv:2602.02016v2, §3.3, `(A^(1/2))^(-1/2)=A^(-1/4)`. -/
theorem ndb_inverse_fourth_root_convergence (Q : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → ℝ) (hQ : Orthogonal Q) (hs : ∀ i, 0 < s i)
    (hupper : ∀ i, s i < 2) :
    Filter.Tendsto (fun k => (ndbIterate (spectralPower Q s (1 / 2)) k).2)
      Filter.atTop (nhds (spectralPower Q s (-(1 / 4)))) := by
  have hdomain := squareRoot_ndb_domain s hs hupper
  have hlim := (ndb_matrix_convergence Q (fun i => s i ^ (1 / 2 : ℝ)) hQ
    (fun i => (hdomain i).1) (fun i => (hdomain i).2)).snd_nhds
  rw [spectralPower_nested Q s (1 / 2) (-(1 / 2)) (fun i => (hs i).le)] at hlim
  norm_num only [show (1 / 2 : ℝ) * -(1 / 2) = -(1 / 4) by norm_num] at hlim
  exact hlim

/-- Matrix chaining assumptions are satisfiable, arXiv:2602.02016v2, §3.3. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => (1 : ℝ)) i) ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i < 2) := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Repeated exact square-root spectra,
arXiv:2602.02016v2, §3.3, chaining calls for powers `1/2^k`. -/
def squareRootChain (s : Fin n → ℝ) : ℕ → Fin n → ℝ
  | 0 => s
  | k + 1 => fun i => squareRootChain s k i ^ (1 / 2 : ℝ)

/-- Chaining `k` square-root calls gives precisely the dyadic power `2^(-k)`.
Source: arXiv:2602.02016v2, §3.3, the limitation to `A^(±1/2^k)`. -/
theorem squareRootChain_eq (s : Fin n → ℝ) (hs : ∀ i, 0 ≤ s i) (k : ℕ) (i : Fin n) :
    squareRootChain s k i = s i ^ ((1 / 2 : ℝ) ^ k) := by
  induction k with
  | zero => simp [squareRootChain]
  | succ k ih =>
    change squareRootChain s k i ^ (1 / 2 : ℝ) = s i ^ ((1 / 2 : ℝ) ^ (k + 1))
    rw [ih, ← Real.rpow_mul (hs i), pow_succ]

/-- Nonnegative chaining spectra exist, arXiv:2602.02016v2, §3.3. -/
example : ∀ i : Fin 1, (0 : ℝ) ≤ (fun _ => (1 : ℝ)) i := by norm_num

end Transformer.DASH
