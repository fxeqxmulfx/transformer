module

public import Transformer.SequenceModel.Basic
public import Transformer.GPTMini.RoPE

/-!
# Arbitrary channel-pair rotations and positional phases

Source: `review/flat_spectrum.md`, *General positional models*, G2;
`python/src/lab/infrastructure/nn/positions.py`, `rotate`. Angles are arbitrary
real functions of channel pair. RoPE chooses angle `position * frequency`;
no particular frequency grid or quarter-turn is built into the theory.
Adjacent pairs and GPTMini's split-half layout are related explicitly below.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel

open Landscape.Neural
open scoped BigOperators

variable {K : Type*}

/-- The full real plane rotation independently on every channel pair. -/
def rotate (angle : K → ℝ) (x : Channels K → ℝ) (i : Channels K) : ℝ :=
  if i.2 = 0 then x (i.1, 0) * Real.cos (angle i.1) - x (i.1, 1) * Real.sin (angle i.1)
  else x (i.1, 0) * Real.sin (angle i.1) + x (i.1, 1) * Real.cos (angle i.1)

/-- G2: zero angle is identity on the actual channel coordinates. -/
theorem rotate_zero (x : Channels K → ℝ) : rotate (fun _ => 0) x = x := by
  funext ⟨k, j⟩
  fin_cases j <;> simp [rotate]

/-- G2: the old model's operator is exactly one special angle choice. -/
theorem rotate_quarter (x : Channels K → ℝ) :
    rotate (fun _ => Real.pi / 2) x = rotate90 x := by
  funext ⟨k, j⟩
  fin_cases j <;> simp [rotate, rotate90]

/-- G2: composition adds arbitrary pairwise angles. -/
theorem rotate_add (a b : K → ℝ) (x : Channels K → ℝ) :
    rotate a (rotate b x) = rotate (fun k => a k + b k) x := by
  funext ⟨k, j⟩
  fin_cases j <;> simp [rotate, Real.sin_add, Real.cos_add] <;> ring

/-- G2: no information is lost by a rotation itself; negating its angles
recovers the original vector without a frequency or positivity restriction. -/
theorem rotate_inverse (a : K → ℝ) (x : Channels K → ℝ) :
    rotate (fun k => -a k) (rotate a x) = x := by
  rw [rotate_add]
  simpa only [neg_add_cancel] using rotate_zero x

/-- G2: each pair keeps its exact squared length. -/
theorem rotate_pair_sq (a : K → ℝ) (x : Channels K → ℝ) (k : K) :
    rotate a x (k, 0) ^ 2 + rotate a x (k, 1) ^ 2 = x (k, 0) ^ 2 + x (k, 1) ^ 2 := by
  simp only [rotate, ↓reduceIte, Fin.isValue, one_ne_zero]
  nlinarith [Real.sin_sq_add_cos_sq (a k)]

/-- G2: arbitrary phases preserve total squared length at every finite width. -/
theorem rotate_sum_sq [Fintype K] (a : K → ℝ) (x : Channels K → ℝ) :
    ∑ i, rotate a x i ^ 2 = ∑ i, x i ^ 2 := by
  simp only [Fintype.sum_prod_type, Fin.sum_univ_two]
  exact Finset.sum_congr rfl (fun k _ => rotate_pair_sq a x k)

/-- RoPE permits any frequency vector; the geometric grid is one instance. -/
def rope (frequency : K → ℝ) (position : ℝ) (x : Channels K → ℝ) : Channels K → ℝ :=
  rotate (fun k => position * frequency k) x

/-- G2: rotary query/key dot products use relative positions for any frequencies. -/
theorem rope_relative [Fintype K] (frequency : K → ℝ) (s t : ℝ)
    (x y : Channels K → ℝ) :
    ∑ i, rope frequency s x i * rope frequency t y i =
      ∑ i, x i * rope frequency (t - s) y i := by
  simp only [Fintype.sum_prod_type, Fin.sum_univ_two]
  apply Finset.sum_congr rfl
  intro k hk
  simp only [rope, rotate, ↓reduceIte, Fin.isValue, one_ne_zero,
    sub_mul, Real.cos_sub, Real.sin_sub]
  ring

/-- Read GPTMini's two-half rotary layout as an array of channel pairs. -/
def splitPairs (d : ℕ) (x : EucSpace d) (i : Channels (Fin (d / 2))) : ℝ :=
  if i.2 = 0 then x (GPTMini.ropeFst d i.1) else x (GPTMini.ropeSnd d i.1)

/-- G2: the arbitrary-angle operation agrees with GPTMini's genuine RoPE
on every rotated pair. An odd leftover coordinate is handled by GPTMini itself. -/
theorem rope_eq_gptMini (d : ℕ) (base position : ℝ) (x : EucSpace d) :
    rope (GPTMini.invFreq d base) position (splitPairs d x) =
      splitPairs d (GPTMini.applyRope d base position x) := by
  funext ⟨k, j⟩
  have h0 : GPTMini.ropeSplit d (GPTMini.ropeFst d k) = .inl (.inl k) :=
    (GPTMini.ropeSplit d).apply_symm_apply _
  have h1 : GPTMini.ropeSplit d (GPTMini.ropeSnd d k) = .inl (.inr k) :=
    (GPTMini.ropeSplit d).apply_symm_apply _
  fin_cases j <;>
    simp only [rope, rotate, splitPairs, ↓reduceIte, Fin.isValue, one_ne_zero,
      GPTMini.applyRope_apply, h0, h1, GPTMini.ropeCoord, Sum.elim_inl, Sum.elim_inr,
      GPTMini.ropeAngle]

/-- Arbitrary pair angles in the source's split-half layout. Any leftover
odd coordinate is unchanged, as in the existing general RoPE definition. -/
def rotateHead (d : ℕ) (angle : Fin (d / 2) → ℝ) (x : EucSpace d) : EucSpace d :=
  WithLp.toLp 2 (fun i => match GPTMini.ropeSplit d i with
    | .inl (.inl k) => rotate angle (splitPairs d x) (k, 0)
    | .inl (.inr k) => rotate angle (splitPairs d x) (k, 1)
    | .inr j => x ((GPTMini.ropeSplit d).symm (.inr j)))

/-- G2: the source half-split map with arbitrary angles preserves the norm. -/
theorem rotateHead_isometry (d : ℕ) (angle : Fin (d / 2) → ℝ) (x : EucSpace d) :
    ‖rotateHead d angle x‖ = ‖x‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  congr 1
  let f : ((Fin (d / 2) ⊕ Fin (d / 2)) ⊕ Fin (d - 2 * (d / 2))) → ℝ := fun j =>
    ‖(match j with
    | .inl (.inl k) => rotate angle (splitPairs d x) (k, 0)
    | .inl (.inr k) => rotate angle (splitPairs d x) (k, 1)
    | .inr j => x ((GPTMini.ropeSplit d).symm (.inr j)))‖ ^ 2
  change (∑ i, f (GPTMini.ropeSplit d i)) = _
  rw [GPTMini.sum_split' d f, GPTMini.sum_split d (fun i => ‖x i‖ ^ 2)]
  dsimp only [f]
  congr 1
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro k hk
  simp only [Real.norm_eq_abs, sq_abs]
  exact rotate_pair_sq angle (splitPairs d x) k

/-- G2: the supplied diff_gpt RoPE grid instantiates the arbitrary-angle map. -/
theorem rotateHead_eq_gptMini (d : ℕ) (base position : ℝ) (x : EucSpace d) :
    rotateHead d (GPTMini.ropeAngle d base position) x =
      GPTMini.applyRope d base position x := by
  ext i
  simp only [rotateHead, GPTMini.applyRope_apply]
  change (match GPTMini.ropeSplit d i with
    | .inl (.inl k) => rotate _ (splitPairs d x) (k, 0)
    | .inl (.inr k) => rotate _ (splitPairs d x) (k, 1)
    | .inr j => x ((GPTMini.ropeSplit d).symm (.inr j))) = _
  rcases GPTMini.ropeSplit d i with (k | k) | j <;>
    simp [rotate, splitPairs, GPTMini.ropeCoord]

example := rotate_add (fun k : Fin 3 => (k.val + 1 : ℝ) / 7)
  (fun k => (k.val + 2 : ℝ) / 11) (fun i => (i.1.val + i.2.val : ℝ))
example := rope_relative (fun k : Fin 4 => (k.val + 1 : ℝ) / 13) 2 17
  (fun i => (i.1.val + i.2.val : ℝ)) (fun i => (2 * i.1.val + i.2.val : ℝ))

end Transformer.SequenceModel
