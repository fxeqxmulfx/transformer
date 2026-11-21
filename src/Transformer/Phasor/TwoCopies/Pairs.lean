module

public import Transformer.Phasor.TwoCopies.Slots
public import Transformer.LeRoPEBlock.ReadScaled

/-!
# Two copies: the DFT heads inside the main model's heads

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

RS4 and RS5 read keys with DFT heads, whose query is `1` and whose key is `1` turned by `t`, in the
pair layout. The main model reaches them. Every pair layout is the layout of a head vector
(`pairLayout_pairUnlayout`), so the DFT query and keys are head vectors `dftQuery`, `dftKey`,
neither of them zero. LeRoPE's scales `α` put its `d/2` frequencies at the DFT frequencies on
`2⌊d/2⌋ + 1` points (`leropeFrequencies_dftAlpha`). And the head RMS sends a multiple `m u` of a
vector to a positive multiple of `u` (`rmsNormEps_smul`, `headScale_pos`).
-/

@[expose] public section

open Matrix
open Transformer.MHC.PeriRMS Transformer.LeRoPEBlock Transformer.RoPE.RoFormer

namespace Transformer.Phasor.TwoCopies

variable {d : ℕ}

/-- The head vector with the pair layout `v`: `v (0, m)` at `ropeFst d m`, `v (1, m)` at
`ropeSnd d m`, and `0` at the leftover coordinate of an odd `d`. -/
noncomputable def pairUnlayout (d : ℕ) (v : Fin 2 × Fin (d / 2) → ℝ) : EucSpace d :=
  WithLp.toLp 2 fun i => match GPTMini.ropeSplit d i with
    | .inl (.inl m) => v (0, m)
    | .inl (.inr m) => v (1, m)
    | .inr _ => 0

theorem pairUnlayout_fst (d : ℕ) (v : Fin 2 × Fin (d / 2) → ℝ) (m : Fin (d / 2)) :
    pairUnlayout d v (GPTMini.ropeFst d m) = v (0, m) := by
  have h : GPTMini.ropeSplit d (GPTMini.ropeFst d m) = .inl (.inl m) :=
    (GPTMini.ropeSplit d).apply_symm_apply _
  rw [pairUnlayout, PiLp.toLp_apply, h]

theorem pairUnlayout_snd (d : ℕ) (v : Fin 2 × Fin (d / 2) → ℝ) (m : Fin (d / 2)) :
    pairUnlayout d v (GPTMini.ropeSnd d m) = v (1, m) := by
  have h : GPTMini.ropeSplit d (GPTMini.ropeSnd d m) = .inl (.inr m) :=
    (GPTMini.ropeSplit d).apply_symm_apply _
  rw [pairUnlayout, PiLp.toLp_apply, h]

/-- **Every pair layout is the layout of a head vector.** -/
theorem pairLayout_pairUnlayout (d : ℕ) (v : Fin 2 × Fin (d / 2) → ℝ) :
    pairLayout d (pairUnlayout d v) = v := by
  funext ⟨c, m⟩
  fin_cases c <;> simp [pairLayout, SequenceModel.splitPairs, pairUnlayout_fst, pairUnlayout_snd]

theorem pairLayout_smul (d : ℕ) (a : ℝ) (x : EucSpace d) :
    pairLayout d (a • x) = a • pairLayout d x := by
  funext ⟨c, m⟩
  simp only [pairLayout, SequenceModel.splitPairs, Pi.smul_apply, PiLp.smul_apply, smul_eq_mul]
  split_ifs <;> rfl

theorem pairLayout_zero (d : ℕ) : pairLayout d 0 = 0 := by
  simpa using pairLayout_smul d 0 0

/-- LeRoPE's scales that put the `d/2` frequencies at the DFT frequencies
`2π (l + 1)/(2⌊d/2⌋ + 1)`. -/
noncomputable def dftAlpha (d : ℕ) : Fin (d / 2) → ℝ :=
  fun m => Real.log (dftFreq (d / 2) m / RoPE.frequency 10000 d m)

/-- **LeRoPE's frequencies reach the DFT frequencies.** -/
theorem leropeFrequencies_dftAlpha (d : ℕ) :
    leropeFrequencies d (dftAlpha d) = dftFreq (d / 2) := by
  funext m
  have h1 : 0 < dftFreq (d / 2) m := by unfold dftFreq; positivity
  have h2 : 0 < RoPE.frequency 10000 d m := by unfold RoPE.frequency; positivity
  rw [leropeFrequencies, dftAlpha, RoPE.LeRoPE.learnedFrequency, Real.exp_log (div_pos h1 h2),
    div_mul_cancel₀ _ h2.ne']

/-- The factor that the head RMS gives a head vector `m u`. -/
noncomputable def headScale (d : ℕ) (eps m : ℝ) (u : EucSpace d) : ℝ :=
  rmsScale d eps (m ^ 2 * ‖u‖ ^ 2) * m

theorem rmsNormEps_smul (eps m : ℝ) (u : EucSpace d) :
    GPTMini.rmsNormEps eps (m • u) = headScale d eps m u • u := by
  rw [rmsNormEps_eq_smul, norm_smul, mul_pow, Real.norm_eq_abs, sq_abs, smul_smul, headScale]

theorem headScale_pos {eps m : ℝ} {u : EucSpace d} (hd : 0 < d) (heps : 0 ≤ eps) (hm : 0 < m)
    (hu : u ≠ 0) : 0 < headScale d eps m u := by
  have := norm_pos_iff.2 hu
  exact mul_pos (rmsScale_pos hd heps (by positivity)) hm

/-- The query of the DFT heads, `1` in every pair coordinate. -/
noncomputable def dftQuery (d : ℕ) : EucSpace d := pairUnlayout d 1

/-- The key of the DFT head turned by `t`. -/
noncomputable def dftKey (d t : ℕ) : EucSpace d := pairUnlayout d (rotary (dftFreq (d / 2)) t *ᵥ 1)

theorem one_ne_zero_pairs (hd : 2 ≤ d) : (1 : Fin 2 × Fin (d / 2) → ℝ) ≠ 0 := fun h => by
  simpa using congrFun h (0, ⟨0, by omega⟩)

theorem dftQuery_ne_zero (hd : 2 ≤ d) : dftQuery d ≠ 0 := fun h => one_ne_zero_pairs hd <| by
  rw [← pairLayout_pairUnlayout d 1, ← dftQuery, h, pairLayout_zero]

theorem dftKey_ne_zero (hd : 2 ≤ d) (t : ℕ) : dftKey d t ≠ 0 := fun h => one_ne_zero_pairs hd <| by
  have h1 := pairLayout_pairUnlayout d (rotary (dftFreq (d / 2)) t *ᵥ 1)
  rw [← dftKey, h, pairLayout_zero] at h1
  rw [← one_mulVec (1 : Fin 2 × Fin (d / 2) → ℝ), ← rotary_zero (dftFreq (d / 2)),
    ← neg_add_cancel (t : ℝ), rotary_add, ← mulVec_mulVec, ← h1, mulVec_zero]

/-- Main's head width `d = 32`: the DFT query and keys are not zero, and the head RMS keeps a
positive multiple of the query positive. -/
example (t : ℕ) : (1 : Fin 2 × Fin (32 / 2) → ℝ) ≠ 0 ∧ dftQuery 32 ≠ 0 ∧ dftKey 32 t ≠ 0 ∧
    0 < headScale 32 (1 / 8388608) 1 (dftQuery 32) :=
  ⟨one_ne_zero_pairs (by norm_num), dftQuery_ne_zero (by norm_num), dftKey_ne_zero (by norm_num) t,
    headScale_pos (by norm_num) (by norm_num) one_pos (dftQuery_ne_zero (by norm_num))⟩

end Transformer.Phasor.TwoCopies
