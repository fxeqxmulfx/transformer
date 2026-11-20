module

public import Transformer.SequenceModel.Rotary

/-!
# Half-split head rotations form a group of isometries

`rotateHead d angle` (`Rotary`) turns channel pair `k` of a half-split head `x : EucSpace d` by
`angle k`, leaving an unpaired last coordinate alone. This module proves that these maps are a
group action on the head, which is what a phase on a whole stream needs:

* `splitPairs_rotateHead`: reading the pairs of a rotated head is rotating the pairs;
* `rotateHead_add`: two rotations compose by adding their angles; `rotateHead_zero_angles` zero
  angles are the identity, including at an odd width; `rotateHead_neg` undoes a rotation;
* `headPhase d φ`: the same angle `φ` on every pair, multiplication by `e^{iφ}` when the head is
  read as `d/2` complex coordinates `x_c + i x_{c + d/2}`; `headPhase_add`, `headPhase_norm`.

Source: the half-split layout of `reference/model.py` (`GPTMini.RoPE`); `rotateHead_zero_angles`
is `review/actual_network/sequence_models.md`, F5, moved here from `Attention.Finite.Phases`.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel

/-- The first coordinate of pair `k` after a rotation, as `rotate` on the pairs of `x`. -/
theorem rotateHead_apply_fst (d : ℕ) (angle : Fin (d / 2) → ℝ) (x : EucSpace d)
    (k : Fin (d / 2)) :
    rotateHead d angle x (GPTMini.ropeFst d k) = rotate angle (splitPairs d x) (k, 0) := by
  simp only [rotateHead, GPTMini.ropeFst, WithLp.ofLp_toLp, Equiv.apply_symm_apply]

/-- The second coordinate of pair `k` after a rotation, as `rotate` on the pairs of `x`. -/
theorem rotateHead_apply_snd (d : ℕ) (angle : Fin (d / 2) → ℝ) (x : EucSpace d)
    (k : Fin (d / 2)) :
    rotateHead d angle x (GPTMini.ropeSnd d k) = rotate angle (splitPairs d x) (k, 1) := by
  simp only [rotateHead, GPTMini.ropeSnd, WithLp.ofLp_toLp, Equiv.apply_symm_apply]

/-- The unpaired coordinates of an odd width are not rotated. -/
theorem rotateHead_apply_rest (d : ℕ) (angle : Fin (d / 2) → ℝ) (x : EucSpace d)
    (j : Fin (d - 2 * (d / 2))) :
    rotateHead d angle x ((GPTMini.ropeSplit d).symm (Sum.inr j)) =
      x ((GPTMini.ropeSplit d).symm (Sum.inr j)) := by
  simp only [rotateHead, WithLp.ofLp_toLp, Equiv.apply_symm_apply]

/-- Reading the channel pairs of a rotated head is rotating its channel pairs. -/
theorem splitPairs_rotateHead (d : ℕ) (angle : Fin (d / 2) → ℝ) (x : EucSpace d) :
    splitPairs d (rotateHead d angle x) = rotate angle (splitPairs d x) := by
  funext ⟨k, j⟩
  fin_cases j
  · simpa [splitPairs] using rotateHead_apply_fst d angle x k
  · simpa [splitPairs] using rotateHead_apply_snd d angle x k

/-- Composing two head rotations adds their angles pair by pair. -/
theorem rotateHead_add (d : ℕ) (a b : Fin (d / 2) → ℝ) (x : EucSpace d) :
    rotateHead d a (rotateHead d b x) = rotateHead d (fun k => a k + b k) x := by
  ext i
  have hi := (GPTMini.ropeSplit d).symm_apply_apply i
  rcases h : GPTMini.ropeSplit d i with (k | k) | j <;> rw [h] at hi <;> subst hi
  · rw [GPTMini.ropeSplit_symm_inl_inl, rotateHead_apply_fst, rotateHead_apply_fst,
      splitPairs_rotateHead, rotate_add]
  · rw [GPTMini.ropeSplit_symm_inl_inr, rotateHead_apply_snd, rotateHead_apply_snd,
      splitPairs_rotateHead, rotate_add]
  · rw [rotateHead_apply_rest, rotateHead_apply_rest, rotateHead_apply_rest]

/-- F5: zero phases are identity in the actual half-split layout, including
the unused coordinate at an odd width. -/
theorem rotateHead_zero_angles (d : ℕ) (x : EucSpace d) :
    rotateHead d (fun _ => 0) x = x := by
  ext i
  have hi := (GPTMini.ropeSplit d).symm_apply_apply i
  change (match GPTMini.ropeSplit d i with
    | .inl (.inl k) => rotate (fun _ => 0) (splitPairs d x) (k, 0)
    | .inl (.inr k) => rotate (fun _ => 0) (splitPairs d x) (k, 1)
    | .inr j => x ((GPTMini.ropeSplit d).symm (.inr j))) = x i
  rw [rotate_zero]
  rcases h : GPTMini.ropeSplit d i with (k | k) | k <;>
    simp only [h] at hi ⊢ <;> simp only [splitPairs, ↓reduceIte, Fin.isValue,
      one_ne_zero, GPTMini.ropeFst, GPTMini.ropeSnd, hi]

/-- Negating the angles undoes a head rotation. -/
theorem rotateHead_neg (d : ℕ) (a : Fin (d / 2) → ℝ) (x : EucSpace d) :
    rotateHead d (fun k => -a k) (rotateHead d a x) = x := by
  rw [rotateHead_add]
  simpa only [neg_add_cancel] using rotateHead_zero_angles d x

/-- Turning every channel pair by the same angle `φ`: multiplication by `e^{iφ}` in the complex
reading `x_c + i x_{c + d/2}` of the head. -/
def headPhase (d : ℕ) (φ : ℝ) (x : EucSpace d) : EucSpace d :=
  rotateHead d (fun _ => φ) x

/-- The phase `0` is the identity. -/
theorem headPhase_zero (d : ℕ) (x : EucSpace d) : headPhase d 0 x = x :=
  rotateHead_zero_angles d x

/-- Phases add. -/
theorem headPhase_add (d : ℕ) (a b : ℝ) (x : EucSpace d) :
    headPhase d a (headPhase d b x) = headPhase d (a + b) x :=
  rotateHead_add d (fun _ => a) (fun _ => b) x

/-- A phase preserves the norm. -/
theorem headPhase_norm (d : ℕ) (φ : ℝ) (x : EucSpace d) : ‖headPhase d φ x‖ = ‖x‖ :=
  rotateHead_isometry d _ x

/-- A quarter turn twice is a half turn, on every head of width `4`. -/
example (x : EucSpace 4) :
    headPhase 4 (Real.pi / 2) (headPhase 4 (Real.pi / 2) x) = headPhase 4 Real.pi x := by
  rw [headPhase_add]
  congr 1
  ring

end Transformer.SequenceModel
