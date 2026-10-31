module

public import Transformer.SequenceModel.Attention.Finite.Codes
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

/-!
# Active rotary phases realize bounded positional scores

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F5.
One actual half-split rotary pair carries nonzero normalized queries/keys.
The constructed phase table is fixed by position. This is an existence
result for arbitrary rotary phase tables, not the fixed integer-position
geometric frequency grid of a particular trained model.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity
open scoped RealInnerProductSpace

/-- F5: zero phases are identity in the actual half-split layout, including
the unused coordinate at an odd width. -/
theorem rotateHead_zero_angles (d : ℕ) (x : EucSpace d) :
    rotateHead d (fun _ => 0) x = x := by
  ext i
  have hi := (ropeSplit d).symm_apply_apply i
  change (match ropeSplit d i with
    | .inl (.inl k) => rotate (fun _ => 0) (splitPairs d x) (k, 0)
    | .inl (.inr k) => rotate (fun _ => 0) (splitPairs d x) (k, 1)
    | .inr j => x ((ropeSplit d).symm (.inr j))) = x i
  rw [rotate_zero]
  rcases h : ropeSplit d i with (k | k) | k <;>
    simp only [h] at hi ⊢ <;> simp only [splitPairs, ↓reduceIte, Fin.isValue,
      one_ne_zero, ropeFst, ropeSnd, hi]

/-- F5: the first coordinate of one actual rotation pair has the usual
cosine/sine expression, with no alternate coordinate convention. -/
theorem rotateHead_fst (d : ℕ) (angle : Fin (d / 2) → ℝ) (x : EucSpace d)
    (k : Fin (d / 2)) :
    rotateHead d angle x (ropeFst d k) =
      x (ropeFst d k) * Real.cos (angle k) - x (ropeSnd d k) * Real.sin (angle k) := by
  simp only [rotateHead, ropeFst, WithLp.ofLp_toLp, Equiv.apply_symm_apply,
    rotate, splitPairs, ↓reduceIte, Fin.isValue, one_ne_zero, ropeSnd]

/-- F5: one nonzero normalized Q/K pair makes the actual score a bounded cosine. -/
theorem headScore_pair (d : ℕ) (hd : 1 < d) {eps : ℝ} (heps : 0 < eps)
    (k : Fin (d / 2)) (angle : Fin (d / 2) → ℝ) :
    headScore d eps (fun _ => 0) angle
      (unnormalize d eps (unit (ropeFst d k)))
      (unnormalize d eps (unit (ropeFst d k))) = Real.cos (angle k) / Real.sqrt (d : ℝ) := by
  have hn : ‖unit (ropeFst d k)‖ ^ 2 < (d : ℝ) := by
    rw [norm_unit, one_pow]
    exact_mod_cast hd
  rw [headScore, rms_unnormalize d (by omega) eps heps _ hn,
    rotateHead_zero_angles]
  change ⟪EuclideanSpace.single (ropeFst d k) (1 : ℝ), rotateHead d angle (unit (ropeFst d k))⟫ / _ = _
  rw [EuclideanSpace.inner_single_left]
  change (1 : ℝ) * rotateHead d angle (unit (ropeFst d k)) (ropeFst d k) / _ = _
  rw [one_mul, rotateHead_fst]
  have hn : ropeFst d k ≠ ropeSnd d k := by
    intro h
    have he := congrArg (ropeSplit d) h
    simp only [ropeFst, ropeSnd, Equiv.apply_symm_apply, Sum.inl.injEq,
      Sum.inl_ne_inr] at he
  simp [unit, EuclideanSpace.single, Ne.symm hn]

/-- Position angle whose normalized score exponentiates to the chosen weight. -/
def weightPhase (d : ℕ) (w : ℝ) : ℝ := Real.arccos (Real.log w * Real.sqrt (d : ℝ))

/-- F5: every coefficient in the bounded interval is realized exactly;
there is no unbounded temperature or assumed sharp-attention limit. -/
theorem exp_headScore_weight (d : ℕ) (hd : 1 < d) {eps : ℝ} (heps : 0 < eps)
    (k : Fin (d / 2)) (w : ℝ) (hw : 1 < w) (hupper : w < Real.exp (1 / Real.sqrt (d : ℝ))) :
    Real.exp (headScore d eps (fun _ => 0) (fun _ => weightPhase d w)
      (unnormalize d eps (unit (ropeFst d k)))
      (unnormalize d eps (unit (ropeFst d k)))) = w := by
  have hs : 0 < Real.sqrt (d : ℝ) := Real.sqrt_pos.mpr (by exact_mod_cast (show 0 < d by omega))
  have hl : 0 < Real.log w := Real.log_pos hw
  have hu : Real.log w < 1 / Real.sqrt (d : ℝ) :=
    (Real.log_lt_iff_lt_exp (by linarith : 0 < w)).mpr hupper
  have hb : Real.log w * Real.sqrt (d : ℝ) ≤ 1 := by
    exact le_of_lt ((lt_div_iff₀ hs).mp hu)
  rw [headScore_pair d hd heps k, weightPhase,
    Real.cos_arccos (by linarith [mul_pos hl hs] : -1 ≤ Real.log w * Real.sqrt (d : ℝ)) hb,
    mul_div_cancel_right₀ _ hs.ne', Real.exp_log (by linarith : 0 < w)]

/-- Fixed phases for N operands followed by the fixed readout token. -/
def codePhases {N : ℕ} (d : ℕ) (w : EucSpace N) : Fin (N + 1) → Fin (d / 2) → ℝ :=
  Fin.lastCases (fun _ => 0) (fun j _ => weightPhase d (w j))

example := headScore_pair 4 (by decide) (eps := 1 / 8388608) (by norm_num) 0
  (fun k => (k.val + 1 : ℝ) / 7)
example := exp_headScore_weight 4 (by decide) (eps := 1 / 8388608) (by norm_num) 0
  (Real.exp (1 / 4)) (by exact Real.one_lt_exp_iff.mpr (by norm_num))
  (by
    apply Real.exp_lt_exp.mpr
    norm_num)

end Transformer.SequenceModel.Attention.Finite
