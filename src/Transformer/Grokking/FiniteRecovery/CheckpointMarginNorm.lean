module

public import Transformer.Grokking.FiniteRecovery.ArgandPhaseNorm
public import Transformer.Grokking.FiniteRecovery.DivisionEmbedding
public import Transformer.KERPLE.Section4_Attention
public import Mathlib.Analysis.Complex.ExponentialBounds

/-! A posteriori nuclear-norm bounds from a checkpoint's raw logit gaps.
The softmax is evaluated mathematically on those logits. The certificate
can obtain the target tokens from any identified class. For division the
nonzero character block has size 96 by 96 and target nuclear norm 96. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Finset Matrix FlatSpectrum Transformer.KERPLE

variable {A U V : Type*} [Fintype A]

/-- All competitors, including those outside the label code, enter the normalizer. -/
theorem phaseError_le_of_logit_gap (z : A → ℝ) (code : A → ℂ) (target : A) (γ : ℝ)
    (hcode : ∀ a, ‖code a‖ ≤ 1) (hgap : ∀ a ≠ target, γ ≤ z target - z a) :
    ‖phaseExpectation code (softmax z) - code target‖ ≤
      2 * ((Fintype.card A - 1 : ℕ) : ℝ) * Real.exp (-γ) := by
  classical
  have htarget := Real.exp_pos (z target)
  have hden := single_le_sum (s := (univ : Finset A))
    (fun a _ => (Real.exp_pos (z a)).le) (mem_univ target)
  have hpos := htarget.trans_le hden
  have hmass : ∑ a, softmax z a = 1 := by
    simp only [softmax, ← sum_div, div_self hpos.ne']
  have hwrong (a : A) (ha : a ≠ target) : softmax z a ≤ Real.exp (-γ) := by
    calc
      _ ≤ Real.exp (z a) / Real.exp (z target) :=
        div_le_div_of_nonneg_left (Real.exp_pos _).le htarget hden
      _ = _ := (Real.exp_sub _ _).symm
      _ ≤ _ := Real.exp_le_exp.mpr (by linarith [hgap a ha])
  have herr : 1 - softmax z target ≤ ((Fintype.card A - 1 : ℕ) : ℝ) * Real.exp (-γ) := by
    rw [← hmass, ← sum_erase_eq_sub (mem_univ target)]
    calc
      _ ≤ ∑ _a ∈ univ.erase target, Real.exp (-γ) :=
        sum_le_sum fun a ha => hwrong a (ne_of_mem_erase ha)
      _ = _ := by simp [nsmul_eq_mul]
  apply (phaseError_le_wrongMass (softmax z) code target
    (fun a => div_nonneg (Real.exp_pos _).le hpos.le) hmass hcode).trans
  rw [mul_assoc]
  exact mul_le_mul_of_nonneg_left herr (by norm_num)

/-- Rectangular table of expected token codes, indexed by the actual input pairs. -/
noncomputable def checkpointPhase (score : U → V → A → ℝ) (code : A → ℂ) : Matrix U V ℂ :=
  Matrix.of fun i j => phaseExpectation code (softmax (score i j))

variable [Fintype U] [Fintype V] [DecidableEq V]

/-- Distance to the entire target table in nuclear norm, for an arbitrary
finite readout and a uniform measured gap. -/
theorem checkpoint_phase_residual_le (score : U → V → A → ℝ) (code : A → ℂ)
    (target : U → V → A) (γ : ℝ) (hcode : ∀ a, ‖code a‖ ≤ 1)
    (hgap : ∀ i j a, a ≠ target i j → γ ≤ score i j (target i j) - score i j a) :
    nuclearNorm (checkpointPhase score code - Matrix.of fun i j => code (target i j)) ≤
      phaseTableSizeFactor (Fintype.card U) (Fintype.card V) *
        (2 * ((Fintype.card A - 1 : ℕ) : ℝ) * Real.exp (-γ)) := by
  apply nuclearNorm_le_of_entryBound _ (by positivity)
  intro i j
  exact phaseError_le_of_logit_gap _ _ _ _ hcode (hgap i j)

/-- Both sides of the nuclear-norm triangle inequality. -/
private theorem norm_difference_le (X Y : Matrix U V ℂ) :
    |nuclearNorm X - nuclearNorm Y| ≤ nuclearNorm (X-Y) := by
  have h₁ := nuclearNorm_add_le Y (X-Y)
  have h₂ := nuclearNorm_add_le X (Y-X)
  rw [add_sub_cancel] at h₁ h₂
  have hn : nuclearNorm (Y-X) = nuclearNorm (X-Y) := by
    rw [← neg_sub X Y, ← neg_one_smul ℂ (X-Y), nuclearNorm_smul, norm_neg, norm_one, one_mul]
  rw [hn] at h₂
  exact abs_le.mpr ⟨by linarith, by linarith⟩

namespace Division

/-- All unknown multiplicative row/column factors have the same bound.
The code is zero on tokens outside the unit group. The logits may come
from the real model or be the exact values of finite GPU outputs. -/
theorem factor_checkpoint_norm (χ : Character) (r c : DivisionUnit → DivisionUnit)
    (score : DivisionUnit → DivisionUnit → Fin 239 → ℝ) (γ : ℝ)
    (hgap : ∀ i j a, a ≠ residueToken (r i * c j).val →
      γ ≤ score i j (residueToken (r i * c j).val) - score i j a) :
    let X := checkpointPhase score (phase χ)
    let Y := Matrix.of fun i j => phase χ (residueToken (r i * c j).val)
    let bound := phaseTableSizeFactor 96 96 * (476 * Real.exp (-γ))
    nuclearNorm (X-Y) ≤ bound ∧ |nuclearNorm X - 96| ≤ bound := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  have hcode (t : Fin 239) : ‖phase χ t‖ ≤ 1 := by
    cases h : residue t with
    | none => simp [phase,h]
    | some a =>
      by_cases ha : a = 0
      · simp [phase,h,ha,χ.map_zero]
      · simp [phase,h,norm_mulChar_eq_one χ ha]
  let target (i j : DivisionUnit) := residueToken (r i * c j).val
  let Y : Matrix DivisionUnit DivisionUnit ℂ := Matrix.of fun i j => phase χ (target i j)
  have he : Y = vecMulVec (fun i => χ (r i).val) (fun j => χ (c j).val) := by
    ext i j
    simp [Y,target,vecMulVec_apply,phase_residue,map_mul]
  have hc : Fintype.card DivisionUnit = 96 :=
    (Fintype.card_congr divisionUnitIndex).trans (Fintype.card_fin _)
  have hY : nuclearNorm Y = 96 := by
    rw [he,nuclearNorm_vecMulVec_of_norm_eq_one
      (fun i => norm_mulChar_eq_one χ (r i).ne_zero)
      (fun j => norm_mulChar_eq_one χ (c j).ne_zero),hc]
    norm_num
  have h := checkpoint_phase_residual_le score _ target γ hcode hgap
  rw [hc,Fintype.card_fin] at h
  norm_num at h
  exact ⟨h,by rw [← hY]; exact (norm_difference_le _ Y).trans h⟩

/-- Margin 19 puts the complete learned unit table within three thousandths
of the target in nuclear norm. This also bounds the two scalar norms. -/
theorem division_checkpoint_margin_nineteen (χ : Character)
    (score : DivisionInput → Fin 239 → ℝ)
    (hgap : ∀ i j a, a ≠ divisionAnswer (some i,j) →
      19 ≤ score (some i,j) (divisionAnswer (some i,j)) - score (some i,j) a) :
    let X := checkpointPhase (fun i j => score (some i,j)) (phase χ)
    let Y := Matrix.of fun i j : DivisionUnit => phase χ (divisionAnswer (some i,j))
    nuclearNorm (X-Y) < 3/1000 ∧ |nuclearNorm X - 96| < 3/1000 := by
  have h := factor_checkpoint_norm χ id (fun j => j⁻¹) (fun i j => score (some i,j)) 19
    (fun i j => by simpa [divisionAnswer, div_eq_mul_inv] using hgap i j)
  have hs : phaseTableSizeFactor 96 96 < 941 := by
    unfold phaseTableSizeFactor
    apply (Real.sqrt_lt' (by norm_num)).mpr
    norm_num
  have he : (27/10 : ℝ)^19 < Real.exp 19 := by
    have hp : (27/10 : ℝ) < Real.exp 1 := lt_trans (by norm_num) Real.exp_one_gt_d9
    simpa using (pow_lt_pow_left₀ hp (by norm_num) (by decide : 19 ≠ 0)).trans_eq
      (Real.exp_nat_mul 1 19).symm
  have hsmall : phaseTableSizeFactor 96 96 * (476 * Real.exp (-19)) < 3/1000 := by
    rw [Real.exp_neg, ← div_eq_mul_inv]
    calc
      _ < 941 * (476 / Real.exp 19) := mul_lt_mul_of_pos_right hs (by positivity)
      _ < _ := by
        rw [← mul_div_assoc, div_lt_iff₀ (Real.exp_pos _)]
        linarith only [he]
  exact ⟨by simpa only [divisionAnswer,div_eq_mul_inv,id_eq] using h.1.trans_lt hsmall,
    h.2.trans_lt hsmall⟩

example (χ : Character) := division_checkpoint_margin_nineteen χ
  (fun x a => if a = divisionAnswer x then 19 else 0)
  (fun i j a ha => by simp [ha])

end Division
end Transformer.Grokking.FiniteRecovery.Argand
