module

public import Transformer.Grokking.FiniteRecovery.DivisionEmbedding

/-! The tied character coordinate and zero indicator strictly classify all
97 residues among 239 eligible tokens. The unused symbols have zero in
both coordinates. The arbitrary common component cancels in every score
gap, including that of the equals sign remaining in the residual stream. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2 FlatSpectrum
open scoped ComplexConjugate InnerProductSpace

private theorem tag_norm (χ : Character) (r : ZMod 97) :
    Complex.normSq (χ r) + (flag (residueToken r))^2 = 1 := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  by_cases hr : r = 0
  · subst r
    norm_num [flag,χ.map_zero]
  · have ht : residueToken r ≠ residueToken 0 := fun he => hr (residueToken_injective he)
    simp [flag,ht,Complex.normSq_eq_norm_sq,norm_mulChar_eq_one χ hr]

private theorem outside {t : Fin 239} (h : residue t = none) : flag t = 0 := by
  apply ite_eq_right
  intro he
  have := residue_some.mpr he.symm
  rw [h] at this
  contradiction

private theorem feature_bound (χ : Character) (t : Fin 239) :
    Complex.normSq (phase χ t) + (flag t)^2 ≤ 1 := by
  cases ht : residue t with
  | none => simp [phase,ht,outside ht]
  | some r =>
    rw [← residue_some.mp ht,phase_residue,tag_norm]

private theorem features_identify (χ : Character) (hinj : Function.Injective χ) (r : ZMod 97)
    (t : Fin 239) (hp : phase χ t = χ r) (hf : flag t = flag (residueToken r)) :
    t = residueToken r := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  cases ht : residue t with
  | none =>
    have hr : r = 0 := hinj (by simpa [phase,ht,χ.map_zero] using hp.symm)
    subst r
    simpa [flag] using hf
  | some b =>
    have htoken := residue_some.mp ht
    rw [← htoken,phase_residue] at hp
    exact htoken.symm.trans (congrArg residueToken (hinj hp))

noncomputable def answerScore (χ : Character) (r : ZMod 97) (t : Fin 239) : ℝ :=
  (χ r * conj (phase χ t)).re + flag t * flag (residueToken r)

theorem answerScore_strict (χ : Character) (hinj : Function.Injective χ) (r : ZMod 97)
    (t : Fin 239) (ht : t ≠ residueToken r) :
    answerScore χ r t < answerScore χ r (residueToken r) := by
  have he : 0 < Complex.normSq (χ r-phase χ t) + (flag t - flag (residueToken r))^2 := by
    by_cases hp : phase χ t = χ r
    · have hf : flag t ≠ flag (residueToken r) := fun h => ht (features_identify χ hinj r t hp h)
      exact add_pos_of_nonneg_of_pos (Complex.normSq_nonneg _) (sq_pos_of_ne_zero (sub_ne_zero.mpr hf))
    · exact add_pos_of_pos_of_nonneg (Complex.normSq_pos.mpr (sub_ne_zero.mpr (Ne.symm hp))) (sq_nonneg _)
  rw [Complex.normSq_sub] at he
  unfold answerScore
  rw [phase_residue,Complex.mul_conj,Complex.ofReal_re]
  nlinarith [tag_norm χ r,feature_bound χ t]

/-- Four coordinates visible to the tied classifier. The common coordinate
and all scratch coordinates are unrestricted. -/
def answerState (χ : Character) (r : ZMod 97) (v : Stream) : Prop :=
  ∃ s : ℝ, 0 < s ∧ v 1 = (s:ℂ)*χ r ∧ v 2 = 0 ∧
    v 3 = (s:ℂ)*(flag (residueToken r) : ℂ) ∧ v 4 = 0

theorem positive_scale {χ : Character} {r : ZMod 97} {v : Stream} (h : answerState χ r v)
    {a : ℝ} (ha : 0 < a) : answerState χ r ((a:ℂ) • v) := by
  obtain ⟨s,hs,h1,h2,h3,h4⟩ := h
  refine ⟨a*s,mul_pos ha hs,?_,?_,?_,?_⟩ <;>
    simp [PiLp.smul_apply,h1,h2,h3,h4,mul_assoc]

theorem strict_readout (χ : Character) (hinj : Function.Injective χ) (r : ZMod 97)
    (v : Stream) (h : answerState χ r v) (t : Fin 239) (ht : t ≠ residueToken r) :
    (⟪embedding χ t,v⟫_ℂ).re / 2 < (⟪embedding χ (residueToken r),v⟫_ℂ).re / 2 := by
  obtain ⟨s,hs,h1,h2,h3,h4⟩ := h
  have form a : (⟪embedding χ a,v⟫_ℂ).re = (v 0).re + s * answerScore χ r a := by
    rw [embedding_inner,h1,h2,h3,h4]
    have hphase : conj (phase χ a) * ((s:ℂ)*χ r) = (s:ℂ)*(χ r*conj (phase χ a)) := by ring
    rw [hphase]
    simp [answerScore,← Complex.ofReal_mul]
    ring
  rw [form,form]
  linarith [mul_lt_mul_of_pos_left (answerScore_strict χ hinj r t ht) hs]

end Transformer.Grokking.FiniteRecovery.Argand.Division
