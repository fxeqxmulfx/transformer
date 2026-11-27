module

public import Transformer.Grokking.FiniteRecovery.FactorRouting
public import Transformer.Grokking.FiniteRecovery.DivisionReadout

/-! The tied classifier compares all 239 tokens. On a unit numerator the
correct phase uniquely maximizes the real inner product. On the constant
row the centered indicator uniquely selects its arbitrary target token. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division.Factors

open Transformer.PhasorV2
open scoped InnerProductSpace ComplexConjugate

noncomputable def phaseValue (χ : Character) (r c : UnitFactor) (x : DivisionInput) : ℂ :=
  rowPhase χ r x.1 * χ (c x.2).toMul.val

noncomputable def evidence (χ : Character) (r c : UnitFactor) (z : Fin 239)
    (x : DivisionInput) (t : Fin 239) : ℝ :=
  (phaseValue χ r c x * conj (phase χ t)).re + tag z t * (zeroValue x.1)^2

theorem evidence_strict (χ : Character) (hinj : Function.Injective χ) (r c : UnitFactor) (z : Fin 239)
    (x : DivisionInput) (t : Fin 239) (ht : t ≠ factorTable unitToken r c z x) :
    evidence χ r c z x t < evidence χ r c z x (factorTable unitToken r c z x) := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  obtain ⟨a,b⟩ := x
  cases a with
  | none => simpa [evidence,phaseValue,rowPhase,zeroValue,factorTable] using tag_gap z t ht
  | some a =>
    let u := (r a+c b).toMul
    have he : phaseValue χ r c (some a,b) = χ u.val := by
      change χ (r a).toMul.val * χ (c b).toMul.val = χ ((r a).toMul.val * (c b).toMul.val)
      rw [map_mul]
    have hf : flag (residueToken u.val) = 0 := ite_eq_right fun e => u.ne_zero (residueToken_injective e)
    have hs := answerScore_strict χ hinj u.val t ht
    change evidence _ _ _ _ _ _ < evidence _ _ _ _ _ (residueToken u.val)
    simpa [evidence,he,zeroValue,answerScore,hf] using hs

/-- Only the common residual, the two output features and the three
operand scratch channels can be nonzero after the product sublayer. -/
def classifierState (χ : Character) (r c : UnitFactor) (x : DivisionInput) (v : Stream) : Prop :=
  ∃ s : ℝ, 0 < s ∧ v 1 = (s:ℂ)*phaseValue χ r c x ∧
    v 3 = (s:ℂ)*((zeroValue x.1)^2:ℝ) ∧ ∀ i : Fin 64,
    i ≠ 0 → i ≠ 1 → i ≠ 3 → i ≠ 5 → i ≠ 6 → i ≠ 7 → v i = 0

theorem scaled_state {χ : Character} {r c : UnitFactor} {x : DivisionInput} {v : Stream}
    (h : classifierState χ r c x v) {a : ℝ} (ha : 0 < a) :
    classifierState χ r c x ((a:ℂ) • v) := by
  obtain ⟨s,hs,h1,h3,hz⟩ := h
  refine ⟨a*s,mul_pos ha hs,?_,?_,?_⟩
  · simp [PiLp.smul_apply,h1,mul_assoc]
  · simp [PiLp.smul_apply,h3,mul_assoc]
  · intro i h0 h1 h3 h5 h6 h7
    simp [PiLp.smul_apply,hz i h0 h1 h3 h5 h6 h7]

theorem readout (χ : Character) (hinj : Function.Injective χ) (r c : UnitFactor) (z : Fin 239)
    (x : DivisionInput) (v : Stream) (h : classifierState χ r c x v) (t : Fin 239)
    (ht : t ≠ factorTable unitToken r c z x) :
    (⟪vocab χ r c z t,v⟫_ℂ).re/2 < (⟪vocab χ r c z (factorTable unitToken r c z x),v⟫_ℂ).re/2 := by
  obtain ⟨s,hs,h1,h3,hz⟩ := h
  have form (a : Fin 239) : (⟪vocab χ r c z a,v⟫_ℂ).re = (v 0).re + s*evidence χ r c z x a := by
    rw [vocab,Channels.inner,h1,h3]
    simp (disch := decide) only [hz,mul_zero,add_zero]
    have he : conj (phase χ a)*((s:ℂ)*phaseValue χ r c x) = (s:ℂ)*(phaseValue χ r c x*conj (phase χ a)) := by ring
    simp only [channels,he,Complex.add_re,Complex.re_ofReal_mul,Complex.ofReal_re,← Complex.ofReal_mul,evidence]
    ring
  rw [form,form]
  linarith [mul_lt_mul_of_pos_left (evidence_strict χ hinj r c z x t ht) hs]

end Transformer.Grokking.FiniteRecovery.Argand.Division.Factors
