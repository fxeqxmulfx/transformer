module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Basic
public import Transformer.MHC.Section4_Stability

/-!
# Stable residual mixing does not make depth training convex

Source: `review/convex_depth.md`, CD4. This is a counterexample to a possible extension
of the convex CE theory, not to a claim of the mHC paper. Two learned residual blocks
violate the CE midpoint inequality. The same scalar action occurs on a disagreement
mode of strictly positive four-stream doubly stochastic mixers. The witness uses a
fixed stream-sensitive readout, not DiffGPT's complete normalized and tied output head.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open scoped Matrix.Norms.L2Operator
open Landscape.Neural (crossEntropy)
open Transformer.Residual (mix weightedSum)

/-- An actual scalar residual block with learned linear branch (CD4). -/
def residualBlock (t x : ℝ) : ℝ := x + (-2 * t) * x

/-- Two independently trainable residual blocks followed by a fixed binary readout
(CD4). The learned parameters remain the two original block coefficients. -/
noncomputable def twoBlockCE (p : ℝ × ℝ) : ℝ :=
  crossEntropy (![residualBlock p.2 (residualBlock p.1 1), 0] : Fin 2 → ℝ) 0

/-- The ordinary CE of this executed two-block network (CD4). -/
theorem twoBlockCE_formula (p : ℝ × ℝ) :
    twoBlockCE p = Real.log (1 + Real.exp (-(residualBlock p.2 (residualBlock p.1 1)))) := by
  rw [twoBlockCE, ConvexRecall.crossEntropy_relative]
  simp [Fin.sum_univ_two]

/-- A nonzero correct margin has smaller ordinary CE than a tied prediction (CD4). -/
theorem quarter_margin_better : Real.log (1 + Real.exp (-(1 / 4 : ℝ))) < Real.log 2 := by
  apply Real.log_lt_log (by positivity)
  have he : Real.exp (-(1 / 4 : ℝ)) < 1 := by
    rw [← Real.exp_zero, Real.exp_lt_exp]
    norm_num
  linarith

/-- Joint convexity fails strictly inside the bounded gate domain: both endpoints
have gates 1/4 or 3/4, while their midpoint has zero output margin (CD4). -/
theorem twoBlockCE_not_convex :
    ¬ ConvexOn ℝ {p : ℝ × ℝ | p.1 ∈ Set.Icc 0 1 ∧ p.2 ∈ Set.Icc 0 1} twoBlockCE := by
  intro hc
  have hconv := hc.2
  have h := hconv
    (x := ((1 / 4 : ℝ), (1 / 4 : ℝ))) (by norm_num)
    (y := ((3 / 4 : ℝ), (3 / 4 : ℝ))) (by norm_num)
    (a := (1 / 2 : ℝ)) (b := (1 / 2 : ℝ)) (by norm_num) (by norm_num) (by norm_num)
  have hm : (1 / 2 : ℝ) • ((1 / 4 : ℝ), (1 / 4 : ℝ)) +
      (1 / 2 : ℝ) • ((3 / 4 : ℝ), (3 / 4 : ℝ)) = ((1 / 2 : ℝ), (1 / 2 : ℝ)) := by
    ext <;> norm_num
  rw [hm] at h
  norm_num [twoBlockCE_formula, residualBlock] at h
  linarith [quarter_margin_better]

/-- Four-stream stochastic mixing between two stream pairs (CD4). Every entry is
strictly positive for interior t, including both nonconvexity witnesses. -/
noncomputable def fourMixer (t : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  fun i j => if i.val / 2 = j.val / 2 then (1 - t) / 2 else t / 2

/-- Both row and column normalization hold for every allowed gate (CD4), so the
obstruction survives the ideal mHC feasibility constraints of §4.1. -/
theorem fourMixer_stochastic (t : ℝ) (ht : t ∈ Set.Icc 0 1) :
    fourMixer t ∈ doublyStochastic ℝ (Fin 4) := by
  rw [mem_doublyStochastic_iff_sum]
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    unfold fourMixer
    split_ifs <;> linarith [ht.1, ht.2]
  · intro i
    fin_cases i <;> norm_num [fourMixer, Fin.sum_univ_succ] <;> ring
  · intro j
    fin_cases j <;> norm_num [fourMixer, Fin.sum_univ_succ] <;> ring

/-- Positivity is not a missing hypothesis in this example (CD4). -/
theorem fourMixer_pos (t : ℝ) (ht : t ∈ Set.Ioo 0 1) : ∀ i j, 0 < fourMixer t i j := by
  intro i j
  unfold fourMixer
  split_ifs <;> linarith [ht.1, ht.2]

/-- A nonzero disagreement of the four streams (CD4). -/
def disagreement : Fin 4 → ℝ := ![1, 1, -1, -1]

/-- Two actual mHC-style skip maps realize the residual counterexample's output
on this mode (CD4). Both maps may obey all ideal stochasticity constraints. -/
theorem fourMixer_two_steps (a b : ℝ) :
    mix (fourMixer b) (mix (fourMixer a) disagreement) 0 =
      residualBlock b (residualBlock a 1) := by
  simp [mix, weightedSum, fourMixer, disagreement, Fin.sum_univ_succ, residualBlock]
  ring

/-- The executed four-stream skip CE is exactly the nonconvex two-block objective
(CD4), with class logits `stream 0` and zero. No surrogate loss is substituted. -/
theorem fourMixer_loss_eq (p : ℝ × ℝ) :
    crossEntropy (![mix (fourMixer p.2) (mix (fourMixer p.1) disagreement) 0, 0] :
      Fin 2 → ℝ) 0 = twoBlockCE p := by
  rw [fourMixer_two_steps]
  rfl

/-- Interpolating the gate interpolates the effective matrix itself (CD4).
The counterexample does not arise from a nonlinear chart of the Birkhoff polytope. -/
theorem fourMixer_affine (s t a b : ℝ) (hab : a + b = 1) :
    fourMixer (a * s + b * t) = a • fourMixer s + b • fourMixer t := by
  ext i j
  change fourMixer (a * s + b * t) i j = a * fourMixer s i j + b * fourMixer t i j
  simp only [fourMixer]
  split_ifs <;> nlinarith

/-- The actual four-stream skip objective violates joint convexity (CD4), even
though each gate domain is convex and each effective skip matrix is stochastic. -/
theorem fourMixer_loss_not_convex :
    ¬ ConvexOn ℝ {p : ℝ × ℝ | p.1 ∈ Set.Icc 0 1 ∧ p.2 ∈ Set.Icc 0 1}
      (fun p => crossEntropy (![mix (fourMixer p.2) (mix (fourMixer p.1) disagreement) 0, 0] :
        Fin 2 → ℝ) 0) := by
  simpa only [fourMixer_loss_eq] using twoBlockCE_not_convex

/-- Every allowed witness obeys the ideal mHC spectral nonexpansion bound (CD4).
Signal stability and convexity in multiple trained blocks are distinct properties. -/
theorem fourMixer_norm_le (t : ℝ) (ht : t ∈ Set.Icc 0 1) : ‖fourMixer t‖ ≤ 1 := by
  exact Matrix.l2_opNorm_le_one_of_mem_doublyStochastic (fourMixer_stochastic t ht)

example := fourMixer_stochastic (1 / 4) (by norm_num)
example := fourMixer_stochastic (3 / 4) (by norm_num)
example := fourMixer_pos (1 / 4) (by norm_num)
example := fourMixer_pos (3 / 4) (by norm_num)
example := fourMixer_affine (1 / 4) (3 / 4) (1 / 2) (1 / 2) (by norm_num)
example := fourMixer_norm_le (1 / 4) (by norm_num)
example := fourMixer_norm_le (3 / 4) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexDepth
