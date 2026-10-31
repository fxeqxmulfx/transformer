module

public import Transformer.SequenceModel.Attention.Grid.Separation
public import Transformer.SequenceModel.Attention.Finite.Head

/-!
# Actual causal XSA scores and context on the prescribed source grid

Source: `review/flat_spectrum.md`, *Prescribed RoPE capacity*, P1--P3.
Fixed integer positions use the source geometric frequencies. The ordinary
query matrix gain remains inside the positive-epsilon RMSNorm image. The
real softmax denominator retains self attention; XSA removes the final
label's value coordinate, uniformly for every possible final label.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Grid

open GPTMini Expressivity
open scoped BigOperators RealInnerProductSpace

/-- Source geometric RoPE at the unchanged integer token positions. -/
def integerAngles (d : ℕ) (base : ℝ) {T : ℕ} (j : Fin T) : Fin (d / 2) → ℝ :=
  ropeAngle d base (j.val : ℝ)

/-- P1: the first pair has frequency one, independently of the other pairs. -/
theorem first_frequency (d : ℕ) (hd : 1 < d) (base : ℝ) :
    invFreq d base ⟨0, by omega⟩ = 1 := by
  simp [invFreq]

/-- P3: a bounded query gain is realized by an actual raw query followed
by the source's positive-epsilon Q/K RMSNorm. -/
theorem rms_scaled_query (d : ℕ) (hd : 1 < d) {eps s : ℝ}
    (heps : 0 < eps) (hs : 0 < s) (hs1 : s < 1) (k : Fin (d / 2)) :
    rmsNormEps eps (unnormalize d eps (s • unit (ropeFst d k))) = s • unit (ropeFst d k) := by
  apply rms_unnormalize d (by omega) eps heps
  rw [norm_smul, norm_unit, mul_one, Real.norm_eq_abs, abs_of_pos hs]
  have hd' : (1 : ℝ) < d := by exact_mod_cast hd
  nlinarith

/-- P3: a first-pair query/key has the exact source relative-position score.
The gain is in the raw query weights, not an extra attention temperature. -/
theorem fixed_headScore (d : ℕ) (hd : 1 < d) (base t u : ℝ) {eps s : ℝ}
    (heps : 0 < eps) (hs : 0 < s) (hs1 : s < 1) :
    headScore d eps (ropeAngle d base t) (ropeAngle d base u)
      (unnormalize d eps (s • unit (ropeFst d ⟨0, by omega⟩)))
      (unnormalize d eps (unit (ropeFst d ⟨0, by omega⟩))) =
      s * Real.cos (u - t) / Real.sqrt (d : ℝ) := by
  let k : Fin (d / 2) := ⟨0, by omega⟩
  have hn : ‖unit (ropeFst d k)‖ ^ 2 < (d : ℝ) := by
    rw [norm_unit, one_pow]
    exact_mod_cast hd
  rw [headScore, rms_scaled_query d hd heps hs hs1,
    rms_unnormalize d (by omega) eps heps _ hn,
    rotateHead_eq_gptMini, rotateHead_eq_gptMini, applyRope_relative,
    real_inner_smul_left, ← rotateHead_eq_gptMini]
  congr 2
  change ⟪EuclideanSpace.single (ropeFst d k) (1 : ℝ),
    rotateHead d (ropeAngle d base (u - t)) (unit (ropeFst d k))⟫ = _
  rw [EuclideanSpace.inner_single_left]
  change (1 : ℝ) * rotateHead d _ _ (ropeFst d k) = _
  rw [one_mul, Finite.rotateHead_fst]
  have hne : ropeFst d k ≠ ropeSnd d k := by
    intro h
    have hh := congrArg (ropeSplit d) h
    simp only [ropeFst, ropeSnd, Equiv.apply_symm_apply, Sum.inl.injEq, Sum.inl_ne_inr] at hh
  simp only [unit, PiLp.single_apply, ↓reduceIte, Ne.symm hne,
    one_mul, zero_mul, sub_zero, ropeAngle, show invFreq d base k = 1 from first_frequency d hd base,
    mul_one]

/-- The fixed score slopes include the source head-dimension scaling. -/
def sourceSlopes (d N : ℕ) (j : Fin N) : ℝ := integerSlopes N j / Real.sqrt (d : ℝ)

/-- P3: dividing by the source scale preserves distinct integer slopes. -/
theorem sourceSlopes_injective (d : ℕ) (hd : 0 < d) (N : ℕ) :
    Function.Injective (sourceSlopes d N) := by
  intro i j h
  apply integerSlopes_injective N
  have hs : Real.sqrt (d : ℝ) ≠ 0 := (Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)).ne'
  exact (div_left_inj' hs).mp h

/-- P3: the last query's actual exponentiated scores on original tokens.
The final self term is retained, including when there is only one token. -/
theorem last_exp_score (d : ℕ) (hd : 1 < d) (base : ℝ) {eps s : ℝ}
    (heps : 0 < eps) (hs : 0 < s) (hs1 : s < 1) {N : ℕ} (j : Fin (N + 1)) :
    Real.exp (headScore d eps (integerAngles d base (Fin.last N)) (integerAngles d base j)
      (unnormalize d eps (s • unit (ropeFst d ⟨0, by omega⟩)))
      (unnormalize d eps (unit (ropeFst d ⟨0, by omega⟩)))) =
      Fin.lastCases (Real.exp (s / Real.sqrt (d : ℝ)))
        (fun i => exponentialWeights (sourceSlopes d N) s i) j := by
  rw [integerAngles, integerAngles, fixed_headScore d hd base _ _ heps hs hs1]
  refine Fin.lastCases ?_ (fun i => ?_) j
  · simp only [Fin.lastCases_last, sub_self, Real.cos_zero, mul_one]
  · simp only [Fin.lastCases_castSucc, Fin.val_castSucc, Fin.val_last,
      exponentialWeights, PiLp.toLp_apply, sourceSlopes, integerSlopes]
    rw [show (i.val : ℝ) - N = -((N : ℝ) - i.val) by ring, Real.cos_neg]
    congr 1
    ring

example := first_frequency 4 (by decide) 10000
example := rms_scaled_query 4 (by decide) (eps := 1 / 8388608) (s := 1 / 2)
  (by norm_num) (by norm_num) (by norm_num) 0
example := fixed_headScore 4 (by decide) 10000 9 3 (eps := 1 / 8388608) (s := 1 / 2)
  (by norm_num) (by norm_num) (by norm_num)
example := sourceSlopes_injective 4 (by decide) 97
example := last_exp_score 4 (by decide) 10000 (eps := 1 / 8388608) (s := 1 / 2)
  (by norm_num) (by norm_num) (by norm_num) (N := 7) 3

variable {N p d : ℕ}

/-- The real causal denominator, including the final token's self score. -/
def codeDenominator (d : ℕ) (s : ℝ) (N : ℕ) : ℝ :=
  (∑ j, exponentialWeights (sourceSlopes d N) s j) + Real.exp (s / Real.sqrt (d : ℝ))

/-- P3: the actual finite softmax denominator is strictly positive. -/
theorem codeDenominator_pos (d : ℕ) (s : ℝ) (N : ℕ) : 0 < codeDenominator d s N := by
  apply add_pos_of_nonneg_of_pos
  · exact Finset.sum_nonneg (fun j _ => (Real.exp_pos _).le)
  · exact Real.exp_pos _

/-- P3: exact final-row causal weights for unchanged source positions. -/
theorem last_weight (hd : 1 < d) (base : ℝ) {eps s : ℝ}
    (heps : 0 < eps) (hs : 0 < s) (hs1 : s < 1) (j : Fin (N + 1)) :
    causalWeight (fun a b => headScore d eps (integerAngles d base a) (integerAngles d base b)
      (unnormalize d eps (s • unit (ropeFst d ⟨0, by omega⟩)))
      (unnormalize d eps (unit (ropeFst d ⟨0, by omega⟩)))) (Fin.last N) j =
      Fin.lastCases (Real.exp (s / Real.sqrt (d : ℝ)))
        (fun i => exponentialWeights (sourceSlopes d N) s i) j / codeDenominator d s N := by
  have hI : Finset.Iic (Fin.last N) = Finset.univ := by ext i; simp [Fin.le_last]
  simp only [causalWeight, Fin.le_last, ↓reduceIte, hI, last_exp_score d hd base heps hs hs1]
  rw [Fin.sum_univ_castSucc]
  simp only [Fin.lastCases_castSucc, Fin.lastCases_last, codeDenominator]

/-- P3: the genuine rotary attention, clipped own-value subtraction and
fixed output bank give the ordered code for every possible last label.
For a one-token input, the source's first-position zero is respected. -/
theorem shifted_head_code (hd : 1 < d) (base : ℝ) {eps xeps s : ℝ}
    (heps : 0 < eps) (hx : xeps ≤ 1) (hs : 0 < s) (hs1 : s < 1)
    (e : (Fin p ⊕ Fin p) ↪ Fin d) (last : Fin p) (x : Fin N → Fin p) :
    shift e (headOutput eps xeps (integerAngles d base)
      (fun _ => unnormalize d eps (s • unit (ropeFst d ⟨0, by omega⟩)))
      (fun _ => unnormalize d eps (unit (ropeFst d ⟨0, by omega⟩)))
      (fun j => unit (e (.inl (Finite.withReadout last x j)))) (Fin.last N)) =
        (codeDenominator d s N)⁻¹ • Finite.contextCode e last (exponentialWeights (sourceSlopes d N) s) x := by
  by_cases hn : N = 0
  · subst N
    simp [headOutput, Finite.contextCode]
  · simp only [headOutput, Finite.withReadout, Fin.lastCases_last, Fin.val_last, hn,
      ↓reduceIte, normL2_unit _ xeps hx]
    rw [← Finite.removeOwn_apply, map_sum]
    simp only [map_smul, Finite.removeOwn_unit, last_weight hd base heps hs hs1]
    rw [Fin.sum_univ_castSucc]
    simp only [Fin.lastCases_castSucc, Fin.lastCases_last, ↓reduceIte, smul_zero, add_zero]
    rw [Finite.contextCode, Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    by_cases h : x j = last <;> simp [h, smul_smul, div_eq_mul_inv, mul_comm]

example := codeDenominator_pos 4 (1 / 2) 97
example := last_weight (d := 4) (by decide) 10000 (eps := 1 / 8388608) (s := 1 / 2)
  (by norm_num) (by norm_num) (by norm_num) (N := 3) 1
example := shifted_head_code (d := 4) (by decide) 10000
  (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (s := 1 / 2)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (finSumFinEquiv.toEmbedding : (Fin 2 ⊕ Fin 2) ↪ Fin 4) 1 ![1, 0, 1]

end Transformer.SequenceModel.Attention.Grid
