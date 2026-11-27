module

public import Transformer.Grokking.FiniteRecovery.DivisionEmbedding
public import Mathlib.Analysis.Real.Pi.Bounds

/-! The two operand positions in the five-token division prompt receive
different coefficients in a single native LeRoPE head with the initial
frequencies. A second head is uniform. These two mixtures suffice to
recover the numerator and denominator features separately. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2 Finset
open scoped ComplexConjugate

noncomputable def doubleGain (n : ℝ) : ℝ :=
  let g := Real.sqrt 64 / (n + Parity.offset)
  (Real.sqrt 64 / (g*n + Parity.offset))*g

theorem doubleGain_pos {n : ℝ} (hn : 0 ≤ n) : 0 < doubleGain n := by
  dsimp [doubleGain,Parity.offset]
  positivity

theorem rms_twice (x : Stream) :
    argandRMS Parity.offset (argandRMS Parity.offset x) = (doubleGain ‖x‖ : ℂ) • x := by
  let g : ℝ := Real.sqrt 64 / (‖x‖+Parity.offset)
  have hg : 0 < g := by dsimp [g,Parity.offset]; positivity
  change ((Real.sqrt 64 / (‖(g:ℂ) • x‖ + Parity.offset) : ℝ) : ℂ) • ((g:ℂ) • x) = _
  rw [norm_smul,Complex.norm_real,Real.norm_of_nonneg hg.le,smul_smul]
  rw [← Complex.ofReal_mul]
  rfl

noncomputable def posGain (j : Fin 5) : ℝ :=
  doubleGain (if j = 1 ∨ j = 3 then Real.sqrt 3 else 1)

theorem posGain_pos (j : Fin 5) : 0 < posGain j := by
  apply doubleGain_pos
  split <;> positivity

theorem normalized_prompt (χ : Character) (x : DivisionInput) (j : Fin 5) :
    let v := embedding χ (divisionPrompt x j)
    argandRMS Parity.offset (argandRMS Parity.offset v) = (posGain j : ℂ) • v := by
  dsimp only
  rw [rms_twice]
  have hn : ‖embedding χ (divisionPrompt x j)‖ =
      (if j = 1 ∨ j = 3 then Real.sqrt 3 else 1) := by
    fin_cases j
    · exact embedding_norm_small χ (t := 0) (by decide)
    · exact embedding_norm_residue χ (x.1.elim 0 Units.val)
    · exact embedding_norm_small χ (t := 10) (by decide)
    · exact embedding_norm_residue χ x.2.val
    · exact embedding_norm_small χ (t := 1) (by decide)
  rw [hn]
  rfl

noncomputable def rawHead (h : Fin 4) (j : Fin 5) (m : Fin 16) : ℂ :=
  if h = 1 ∧ m = 0 then (posGain j : ℂ) else 0

noncomputable def normalizedHead (h : Fin 4) (j : Fin 5) :=
  normalizeHead Parity.offset (rawHead h j)

noncomputable def posScore (h : Fin 4) (j : Fin 5) : ℂ :=
  let θ := leropeFrequency 10000 (0 : Fin 16 → ℝ)
  ((Real.log 5 / Real.sqrt 16 : ℝ) : ℂ) *
    ∑ m, conj (turn θ 4 (normalizedHead h 4) m) * turn θ j.val (normalizedHead h j) m

noncomputable def mix (h : Fin 4) (j : Fin 5) : ℂ :=
  let keep := (causalGeometry (0 : Fin 16 → ℝ)).visible (4 : Fin 5)
  ⟨maskedSoftmax keep (fun k => (posScore h k).re) j,
   maskedSoftmax keep (fun k => (posScore h k).im) j⟩

theorem mix_uniform (j : Fin 5) : mix 0 j = (⟨1/5,1/5⟩ : ℂ) := by
  have hz (k : Fin 5) : posScore 0 k = 0 := by
    simp [posScore,turn,normalizedHead,normalizeHead,rawHead]
  have hlast (k : Fin 5) : k ≤ 4 := by omega
  simp [mix,hz,maskedSoftmax,causalGeometry,hlast]

noncomputable def amplitude (j : Fin 5) : ℝ :=
  Real.sqrt 16 / (Real.sqrt (sqLen (rawHead 1 j)) + Parity.offset) * posGain j

theorem amplitude_pos (j : Fin 5) : 0 < amplitude j := by
  dsimp [amplitude,Parity.offset]
  positivity [posGain_pos j]

private theorem band (j : Fin 5) (m : Fin 16) :
    normalizedHead 1 j m = if m = 0 then (amplitude j : ℂ) else 0 := by
  by_cases hm : m = 0 <;> simp [normalizedHead,normalizeHead,rawHead,amplitude,hm]

private theorem real_score (j : Fin 5) : (posScore 1 j).re =
    Real.log 5 / Real.sqrt 16 * (amplitude 4 * amplitude j * Real.cos (j.val-4)) := by
  let θ := leropeFrequency 10000 (0 : Fin 16 → ℝ)
  have hθ : θ 0 = 1 := by simp [θ,leropeFrequency_zero,Transformer.RoPE.frequency]
  change (((Real.log 5 / Real.sqrt 16 : ℝ) : ℂ) * _).re = _
  rw [Complex.re_ofReal_mul]
  congr 1
  change score (turn θ 4 (normalizedHead 1 4)) (turn θ j.val (normalizedHead 1 j)) = _
  rw [score_turn,score,Finset.sum_eq_single 0]
  · simp only [turn,band,ite_true,hθ,mul_one,Complex.conj_ofReal,
      Complex.mul_re,Complex.ofReal_re,Complex.ofReal_im,mul_zero,sub_zero,
      Complex.exp_ofReal_mul_I_re]
    ring
  · intro m _ hm
    simp [band,hm]
  · simp

theorem mix_unequal : mix 1 1 ≠ mix 1 3 := by
  have h13 : amplitude 1 = amplitude 3 := by
    have hb : posGain 1 = posGain 3 := by simp [posGain]
    have hraw : rawHead 1 1 = rawHead 1 3 := by ext m; simp [rawHead,hb]
    simp [amplitude,hraw,hb]
  have hcos : Real.cos 3 < Real.cos 1 :=
    Real.strictAntiOn_cos ⟨by norm_num,by linarith [Real.pi_gt_three]⟩
      ⟨by norm_num,Real.pi_gt_three.le⟩ (by norm_num)
  have hscore : (posScore 1 1).re < (posScore 1 3).re := by
    rw [real_score,real_score,h13]
    norm_num only [Fin.val_one, Nat.cast_one, show ((3:Fin 5).val:ℝ) = 3 from rfl,
      show (1:ℝ)-4 = -3 by norm_num, show (3:ℝ)-4 = -1 by norm_num, Real.cos_neg]
    exact mul_lt_mul_of_pos_left (mul_lt_mul_of_pos_left hcos (mul_pos (amplitude_pos _) (amplitude_pos _)))
      (div_pos (Real.log_pos (by norm_num)) (by positivity))
  intro he
  have hre := congrArg Complex.re he
  let den := ∑ k : Fin 5, Real.exp (posScore 1 k).re
  have hd : 0 < den := Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
  have hlast (k : Fin 5) : k ≤ 4 := by omega
  have heq : Real.exp (posScore 1 1).re / den = Real.exp (posScore 1 3).re / den := by
    simpa [mix,maskedSoftmax,causalGeometry,hlast,den] using hre
  exact (div_lt_div_of_pos_right (Real.exp_lt_exp.mpr hscore) hd).ne heq

end Transformer.Grokking.FiniteRecovery.Argand.Division
