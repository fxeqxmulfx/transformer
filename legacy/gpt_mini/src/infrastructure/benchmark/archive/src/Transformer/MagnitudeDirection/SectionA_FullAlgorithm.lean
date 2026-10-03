/-
# Full row-and-column optimizer wrapper

arXiv:2606.25971v2, Appendix A, Algorithm 2. The wrapper implements the
complete printed step with a supplied gain map and derivative. Softplus
gives an invariant independent of the gain optimizer's output values.
-/

import Transformer.MagnitudeDirection.Section3_ScalarAlgorithm
import Transformer.MagnitudeDirection.Section4_PositiveGains

noncomputable section

namespace Transformer.MagnitudeDirection

/-- The storage visible to the optimizer, arXiv:2606.25971v2, Appendix A,
Algorithm 2. The model sees `weight`; raw gains are optimizer metadata. -/
structure FullState (m n : ℕ) where
  weight : Matrix (Fin m) (Fin n) ℝ
  rawRow : Fin m → ℝ
  rawCol : Fin n → ℝ

variable {m n : ℕ}

/-- Materialize gains and recover the direction, arXiv:2606.25971v2,
Appendix A, Algorithm 2, lines 1–2. -/
def fullDirection (phi : ℝ → ℝ) (s : FullState m n) : Matrix (Fin m) (Fin n) ℝ :=
  unfuse (fun i => phi (s.rawRow i)) (fun j => phi (s.rawCol j)) s.weight

/-- The full direction candidate, arXiv:2606.25971v2, Appendix A,
Algorithm 2, lines 6–7. -/
def fullCandidate (phi : ℝ → ℝ) (opt : MatrixStep m n)
    (s : FullState m n) (G : Matrix (Fin m) (Fin n) ℝ) (etaW : ℝ) :
    Matrix (Fin m) (Fin n) ℝ :=
  opt (fullDirection phi s)
    (directionGradient (fun i => phi (s.rawRow i)) (fun j => phi (s.rawCol j)) G) etaW

/-- The complete printed row/column step, arXiv:2606.25971v2, Appendix A,
Algorithm 2. The old direction and old gains determine all gradients.
Each gain callback has its own optimizer state and receives the gain LR. -/
def fullStep (phi dphi : ℝ → ℝ) (opt : MatrixStep m n)
    (rowOpt : GainStep m) (colOpt : GainStep n) (s : FullState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) (etaW etaG c : ℝ) : FullState m n :=
  let D := fullDirection phi s
  let row := fun i => phi (s.rawRow i)
  let col := fun j => phi (s.rawCol j)
  let nextRow := rowOpt s.rawRow (fun i => rowGradient col D G i * dphi (s.rawRow i)) etaG
  let nextCol := colOpt s.rawCol (fun j => colGradient row D G j * dphi (s.rawCol j)) etaG
  let nextD := matrixProject c (fullCandidate phi opt s G etaW)
  ⟨fuse (fun i => phi (nextRow i)) (fun j => phi (nextCol j)) nextD, nextRow, nextCol⟩

/-- For any smooth positive map and any base/gain optimizers, the next
recovered direction has the prescribed Frobenius norm whenever the direction
candidate is nonzero. Source: arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem fullStep_direction_norm (phi dphi : ℝ → ℝ) (opt : MatrixStep m n)
    (rowOpt : GainStep m) (colOpt : GainStep n) (s : FullState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) (etaW etaG c : ℝ)
    (hp : ∀ x, 0 < phi x) (hc : 0 ≤ c) (hC : fullCandidate phi opt s G etaW ≠ 0) :
    frobeniusNorm (fullDirection phi (fullStep phi dphi opt rowOpt colOpt s G etaW etaG c)) = c := by
  dsimp [fullDirection, fullStep]
  rw [unfuse_fuse _ _ _ (fun i => (hp _).ne') (fun j => (hp _).ne')]
  exact matrixProject_norm c _ hc hC

/-- Softplus and a stationary gradient step satisfy the invariant's premises,
arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
example : (∀ x, 0 < softplus x) ∧ (0 : ℝ) ≤ 1 ∧
    fullCandidate softplus (fun D G eta => D - eta • G)
      (⟨1, fun _ => Real.log (Real.exp 1 - 1), fun _ => Real.log (Real.exp 1 - 1)⟩ :
        FullState 1 1) 0 0 ≠ 0 := by
  refine ⟨softplus_pos, by norm_num, ?_⟩
  norm_num [fullCandidate, fullDirection, unfuse, softplus_unit_initialization]

/-- The default softplus implementation satisfies the sphere invariant without
any restrictions on finite updated raw gains.
Source: arXiv:2606.25971v2, Appendix A, Algorithm 2. -/
theorem softplus_fullStep_direction_norm (opt : MatrixStep m n)
    (rowOpt : GainStep m) (colOpt : GainStep n) (s : FullState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) (etaW etaG c : ℝ)
    (hc : 0 ≤ c) (hC : fullCandidate softplus opt s G etaW ≠ 0) :
    frobeniusNorm (fullDirection softplus
      (fullStep softplus softplusDerivative opt rowOpt colOpt s G etaW etaG c)) = c :=
  fullStep_direction_norm softplus softplusDerivative opt rowOpt colOpt s G etaW etaG c
    softplus_pos hc hC

/-- The softplus invariant has valid nonzero candidates, arXiv:2606.25971v2,
Appendix A, Algorithm 2. -/
example : (0 : ℝ) ≤ 1 ∧ fullCandidate softplus (fun D G eta => D - eta • G)
    (⟨1, fun _ => Real.log (Real.exp 1 - 1), fun _ => Real.log (Real.exp 1 - 1)⟩ :
      FullState 1 1) 0 0 ≠ 0 := by
  constructor
  · norm_num
  · norm_num [fullCandidate, fullDirection, unfuse, softplus_unit_initialization]

/-- Fused storage gives exactly the same linear-layer output as explicit
factor application. Source: arXiv:2606.25971v2, §3.1, “Fused weights”,
and Appendix G, “zero overhead to the architecture”. This is semantic
equality in real arithmetic, not a wall-clock throughput guarantee. -/
theorem fused_forward (row : Fin m → ℝ) (col x : Fin n → ℝ)
    (D : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) :
    (fuse row col D).mulVec x i = row i * D.mulVec (fun j => col j * x j) i := by
  simp only [Matrix.mulVec, dotProduct, fuse, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  ring

end Transformer.MagnitudeDirection
