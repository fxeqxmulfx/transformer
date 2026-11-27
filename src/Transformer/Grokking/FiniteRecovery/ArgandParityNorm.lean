module

public import Transformer.Grokking.FiniteRecovery.ArgandPhaseNorm
public import Transformer.Grokking.FlatSpectrum.ProductLoopGateParity
public import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-! A target-scale nuclear bound on the balanced binary modular-sum
cut. Its error term measures signed changes of the network's own logit
gap across rows. No off-diagonal labels or singular values are inputs. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Finset Matrix FlatSpectrum
open scoped BigOperators

private noncomputable def binaryPhase (x : ℝ) : ℝ := 2 * Real.sigmoid x - 1

private theorem binaryPhase_neg (x : ℝ) : binaryPhase (-x) = -binaryPhase x := by
  simp only [binaryPhase, Real.sigmoid_neg]
  ring

private theorem binaryPhase_abs_le_one (x : ℝ) : |binaryPhase x| ≤ 1 := by
  have h0 := Real.sigmoid_pos x
  have h1 := Real.sigmoid_lt_one x
  rw [abs_le]
  dsimp [binaryPhase]
  constructor <;> linarith

private theorem binaryPhase_dist (x y : ℝ) :
    |binaryPhase x - binaryPhase y| ≤ |x - y| / 2 := by
  have hL : LipschitzWith (1 / 4 : NNReal) Real.sigmoid := by
    apply lipschitzWith_of_nnNorm_deriv_le (fun s => (Real.hasDerivAt_sigmoid s).differentiableAt)
    intro s
    change ‖deriv Real.sigmoid s‖ ≤ ((1 / 4 : NNReal) : ℝ)
    rw [Real.deriv_sigmoid, Real.norm_of_nonneg (mul_nonneg (Real.sigmoid_pos _).le
      (sub_nonneg.mpr (Real.sigmoid_lt_one _).le))]
    norm_num only [NNReal.coe_div, NNReal.coe_one, NNReal.coe_ofNat]
    nlinarith [sq_nonneg (Real.sigmoid s - 1 / 2)]
  have h := hL.norm_sub_le x y
  simp only [Real.norm_eq_abs, NNReal.coe_div, NNReal.coe_one, NNReal.coe_ofNat] at h
  rw [show binaryPhase x - binaryPhase y = 2 * (Real.sigmoid x - Real.sigmoid y) by
    unfold binaryPhase; ring, abs_mul, abs_of_pos (by norm_num : 0 < (2 : ℝ))]
  linarith

variable {H d layers T : ℕ}

/-- The log-odds of the even residue against the odd one. -/
noncomputable def Model.parityGap (model : Model (ZMod 2) H d layers) (repeats : ℕ)
    (word : Word (ZMod 2) (T + 1)) : ℝ :=
  model.logits repeats word (Fin.last T) 0 - model.logits repeats word (Fin.last T) 1

private theorem phaseExpectation_two (p : ZMod 2 → ℝ) (hp : ∑ j, p j = 1) :
    phaseExpectation ZMod.stdAddChar p = ((2 * p 0 - 1 : ℝ) : ℂ) := by
  have hsum (g : ZMod 2 → ℂ) : (∑ j, g j) = g 0 + g 1 := Fin.sum_univ_two g
  have hchar : ZMod.stdAddChar (1 : ZMod 2) = -1 := by
    rw [stdAddChar_zmod_two, show (1 : ZMod 2).val = 1 by decide, pow_one]
  rw [phaseExpectation, hsum, AddChar.map_zero_eq_one, hchar, mul_one, mul_neg_one,
    ← sub_eq_add_neg, ← Complex.ofReal_sub]
  have hsum' : p 0 + p 1 = 1 := (Fin.sum_univ_two p).symm.trans hp
  exact congrArg Complex.ofReal (by linarith only [hsum'])

private theorem softmax_two_eq_sigmoid (x y : ℝ) :
    Real.exp x / (Real.exp x + Real.exp y) = Real.sigmoid (x - y) := by
  rw [Real.sigmoid_def, neg_sub, Real.exp_sub]
  field_simp [Real.exp_ne_zero, ne_of_gt (add_pos (Real.exp_pos x) (Real.exp_pos y))]

private theorem parityPhase_eq (model : Model (ZMod 2) H d layers) (repeats : ℕ)
    (word : Word (ZMod 2) (T + 1)) :
    phaseExpectation ZMod.stdAddChar (model.probability repeats word (Fin.last T)) =
      (binaryPhase (model.parityGap repeats word) : ℂ) := by
  rw [phaseExpectation_two _ (model.sum_probability repeats word (Fin.last T))]
  let z := model.logits repeats word (Fin.last T)
  have hsum : (∑ i : ZMod 2, Real.exp (z i)) = Real.exp (z 0) + Real.exp (z 1) :=
    Fin.sum_univ_two _
  change ((2 * (Real.exp (z 0) / ∑ j, Real.exp (z j)) - 1 : ℝ) : ℂ) =
    ((2 * Real.sigmoid (z 0 - z 1) - 1 : ℝ) : ℂ)
  rw [hsum, softmax_two_eq_sigmoid]

/-- A sign twist depending only on the sum of each half-word. -/
def rowParity {k : ℕ} (word : Word (ZMod 2) k) : ℝ :=
  if ∑ j, word j = 0 then 1 else -1

theorem rowParity_eq_character {k : ℕ} (word : Word (ZMod 2) k) :
    (rowParity word : ℂ) = ZMod.stdAddChar (∑ j, word j) := by
  have two (x : ZMod 2) : x = 0 ∨ x = 1 := by revert x; decide
  rcases two (∑ j, word j) with h | h
  · simp [rowParity, h]
  · simp [rowParity, h, stdAddChar_zmod_two, show (1 : ZMod 2).val = 1 by decide]

theorem nuclearNorm_modularSum_target {k s : ℕ} (e : Fin (s + 1) ≃ Word (ZMod 2) k) :
    nuclearNorm (targetKernel (enumeratedCut e) ZMod.stdAddChar (modularSum e)) = (s + 1 : ℕ) := by
  rw [targetKernel_modularSum, nuclearNorm_vecMulVec_of_norm_eq_one (by simp) (by simp), Fintype.card_fin]

/-- Given a reference row, dephase all logit gaps relative to that row.
The residual is zero for exact row sign equivariance. -/
noncomputable def Model.parityGapResidual {k s : ℕ}
    (model : Model (ZMod 2) H d layers) (repeats : ℕ) (e : Fin (s + 1) ≃ Word (ZMod 2) (k + 1)) : ℝ :=
  let cut := enumeratedCut e
  let z u v := model.parityGap (T := (k + 1) + k) repeats (cut (u,v))
  ∑ u, ∑ v, (z u v - rowParity (e u) * rowParity (e v) * z v v) ^ 2

/-- For every trained weight vector and any depth, the nuclear norm is
at most the truth's value N plus half the Euclidean discrepancy of the
signed readout gaps times √N. The diagonal anchors are all observed in
the staircase. The discrepancy uses predictions on unlabeled inputs. -/
theorem Model.nuclearNorm_modularSum_le {k s : ℕ} (model : Model (ZMod 2) H d layers)
    (repeats : ℕ) (e : Fin (s + 1) ≃ Word (ZMod 2) (k + 1)) :
    nuclearNorm (model.phaseKernel (T := (k + 1) + k) repeats (enumeratedCut e) ZMod.stdAddChar) ≤
      (s + 1 : ℕ) + √((s + 1 : ℕ) * model.parityGapResidual repeats e) / 2 := by
  let z u v := model.parityGap (T := (k + 1) + k) repeats (enumeratedCut e (u,v))
  let a u := rowParity (e u)
  let b v := a v * binaryPhase (z v v)
  let K := model.phaseKernel (T := (k + 1) + k) repeats (enumeratedCut e) ZMod.stdAddChar
  let Q := vecMulVec (fun u => (a u : ℂ)) (fun v => (b v : ℂ))
  have hsign (u) : a u = 1 ∨ a u = -1 := by unfold a rowParity; split <;> simp
  have ha (u) : ‖(a u : ℂ)‖ = 1 := by rcases hsign u with h | h <;> simp [h]
  have hb (v) : ‖(b v : ℂ)‖ ≤ 1 := by
    rw [Complex.norm_real, Real.norm_eq_abs, show b v = _ from rfl, abs_mul]
    rcases hsign v with h | h <;> simpa [h] using binaryPhase_abs_le_one (z v v)
  have hgap (u v) : ‖(K - Q) u v‖ ≤ |z u v - a u * a v * z v v| / 2 := by
    have hphase : K u v = (binaryPhase (z u v) : ℂ) := parityPhase_eq _ _ _
    rw [Matrix.sub_apply, hphase]
    change ‖(binaryPhase (z u v) : ℂ) - (a u : ℂ) * (↑(a v * binaryPhase (z v v)) : ℂ)‖ ≤ _
    rw [← Complex.ofReal_mul, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs]
    have h := binaryPhase_dist (z u v) (a u * a v * z v v)
    rcases hsign u with hu | hu <;> rcases hsign v with hv | hv <;>
      simpa [hu, hv, binaryPhase_neg] using h
  have hf : frobeniusSq (K - Q) ≤ model.parityGapResidual repeats e / 4 := by
    unfold Model.parityGapResidual
    change frobeniusSq (K - Q) ≤ (∑ u, ∑ v, (z u v - a u * a v * z v v) ^ 2) / 4
    simp only [Finset.sum_div, frobeniusSq]
    exact sum_le_sum fun u _ => sum_le_sum fun v _ => by
      simpa only [div_pow, sq_abs, show (2 : ℝ) ^ 2 = 4 by norm_num] using pow_le_pow_left₀ (norm_nonneg _) (hgap u v) 2
  have hb2 : (∑ v, ‖(b v : ℂ)‖ ^ 2) ≤ (s + 1 : ℕ) := by
    simpa using sum_le_sum (s := (univ : Finset (Fin (s + 1)))) fun v _ =>
      pow_le_pow_left₀ (norm_nonneg _) (hb v) 2
  have hhead := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hb2) (Real.sqrt_nonneg (s + 1 : ℕ))
  rw [Real.mul_self_sqrt (Nat.cast_nonneg _)] at hhead
  have ht := model.nuclearNorm_le_rowFactor (T := (k + 1) + k) repeats (enumeratedCut e)
    ZMod.stdAddChar (fun u => (a u : ℂ)) (fun v => (b v : ℂ)) ha
  simp only [Fintype.card_fin, min_self] at ht
  refine ht.trans (add_le_add hhead ?_)
  have hprod := mul_le_mul_of_nonneg_left hf (Nat.cast_nonneg (α := ℝ) (s + 1))
  calc
    √((s + 1 : ℕ) * frobeniusSq (K - Q)) ≤ √(((s + 1 : ℕ) * model.parityGapResidual repeats e) / 4) :=
      Real.sqrt_le_sqrt (by linarith only [hprod])
    _ = _ := by rw [Real.sqrt_div' _ (by norm_num)]; norm_num

/-- Observed staircase loss and a signed-gap discrepancy suffice for
strict classification of every hidden modular sum by the full model. -/
theorem modularSum_realizes_of_loss_and_gapResidual {k s : ℕ} (repeats : ℕ)
    (e : Fin (s + 1) ≃ Word (ZMod 2) (k + 1)) :
    let cut := enumeratedCut e
    let seen := observe (inputsAtCells cut (staircase s)) (modularSum e)
    ∃ ε > 0, ∀ model : Model (ZMod 2) H d layers,
      model.observedLoss (T := (k + 1) + k) repeats seen ≤ ε →
      (s + 1 : ℕ) * model.parityGapResidual repeats e ≤ 4 * ε ^ 2 →
      model.realizes (T := (k + 1) + k) repeats (modularSum e) := by
  let cut := enumeratedCut e
  obtain ⟨ε, hε, hcert⟩ := exists_observedLoss_nuclear_slack_for_realizes (H := H) (d := d)
    (layers := layers) (T := (k + 1) + k) repeats cut ZMod.stdAddChar ZMod.injective_stdAddChar
    (fun _ => by simp) (modularSum e) (inputsAtCells cut (staircase s))
    (by simpa only [observedCells_inputsAtCells] using modularSum_nuclear_recovers e)
  refine ⟨ε, hε, fun model hfit hg => hcert model hfit ?_⟩
  rw [show nuclearNorm (targetKernel cut ZMod.stdAddChar (modularSum e)) = (s + 1 : ℕ) from
    nuclearNorm_modularSum_target e]
  have hgap : √((s + 1 : ℕ) * model.parityGapResidual repeats e) ≤ 2 * ε :=
    Real.sqrt_le_iff.mpr ⟨by positivity, by linarith only [hg]⟩
  exact (model.nuclearNorm_modularSum_le repeats e).trans (add_le_add le_rfl (by linarith only [hgap]))

example (model : Model (ZMod 2) 4 16 1) :=
  model.nuclearNorm_modularSum_le (k := 3) (s := 15) 1 (wordEnumeration 2 4)
example (model : Model (ZMod 2) 4 16 1) :=
  model.nuclearNorm_modularSum_le (k := 7) (s := 255) 1 (wordEnumeration 2 8)
example (model : Model (ZMod 2) 4 16 1) :=
  model.nuclearNorm_modularSum_le (k := 15) (s := 65535) 1 (wordEnumeration 2 16)

end Transformer.Grokking.FiniteRecovery.Argand
