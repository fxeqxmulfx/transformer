module

public import Transformer.Phasor.SuccessorDisplacement

/-!
# Gain interpolation for the tied successor read

Source: the residual-gain comparison in `experiments/phasor_value_transport/README.md`.
The direction read from the successor is fixed while the learned scalar gain
varies. These are exact score and margin results for the model at any position.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ} [NeZero p]

/-- At any token and depth, interpolating two gains interpolates the entire
score vector coordinate by coordinate, with all other weights held fixed.
Real interpolation also covers extrapolation beyond the two endpoints. -/
theorem Model.displacedSuccessorLogit_interpolate (M : Model D d H F p) (T : PairTables D p)
    (strength g h t eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) :
    M.displacedSuccessorLogit T strength ((1-t)*g+t*h) eps scale xs L i c =
      (1-t) * M.displacedSuccessorLogit T strength g eps scale xs L i c +
        t * M.displacedSuccessorLogit T strength h eps scale xs L i c := by
  simp only [displacedSuccessorLogit, inner_add_right, real_inner_smul_right]
  ring

/-- Gain changes have a uniform bound on the margin of two classes. The
constant uses the distance between their tied embeddings, for arbitrary
contexts and arbitrary strengths of the causal read. -/
theorem Model.displacedSuccessor_margin_gain_bound (M : Model D d H F p) (T : PairTables D p)
    (strength g h : ℝ) {eps : ℝ} (heps : 0 < eps) (hD : 0 < D)
    (scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (u v : Fin p) :
    let gap := fun gain => M.displacedSuccessorLogit T strength gain eps scale xs L i u -
      M.displacedSuccessorLogit T strength gain eps scale xs L i v
    |gap g - gap h| ≤ |g-h| * (‖M.embedding u - M.embedding v‖ * Real.sqrt D) := by
  let a := xs (i-1)
  let b := xs i
  let w := Perspective.softmaxWeight (M.logits eps scale xs L (i-1))
  let raw := (transportedValue M.embedding w (a*b) + (T.withSuccessor xs i).prefixRead .xor (0,0) xs i) +
    strength • (M.embedding (successorAddress xs i) - M.embedding b)
  let r := GPTMini.rmsNormEps eps raw
  let x := M.feature eps scale xs L i
  change |(⟪M.embedding u, x + g • r⟫ - ⟪M.embedding v, x + g • r⟫) -
    (⟪M.embedding u, x + h • r⟫ - ⟪M.embedding v, x + h • r⟫)| ≤ _
  rw [← inner_sub_left (M.embedding u) (M.embedding v) (x + g • r),
    ← inner_sub_left (M.embedding u) (M.embedding v) (x + h • r)]
  simp only [inner_add_right, real_inner_smul_right, add_sub_add_left_eq_sub,
    ← sub_mul, abs_mul]
  exact mul_le_mul_of_nonneg_left ((abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_norm_le eps heps hD raw) (norm_nonneg _)))
    (abs_nonneg _)

example (M : Model 64 16 4 86 68) (T : PairTables 64 68)
    (xs : ℕ → Fin 68) (L i : ℕ) (u v : Fin 68) :
    let gap := fun g => M.displacedSuccessorLogit T 1 g 1 1 xs L i u -
      M.displacedSuccessorLogit T 1 g 1 1 xs L i v
    |gap (3/4) - gap (1/2)| ≤ |(3/4 : ℝ)-1/2| * (‖M.embedding u - M.embedding v‖ * Real.sqrt 64) :=
  M.displacedSuccessor_margin_gain_bound T _ _ _ (by norm_num) (by decide) _ xs L i u v

end Transformer.Phasor
