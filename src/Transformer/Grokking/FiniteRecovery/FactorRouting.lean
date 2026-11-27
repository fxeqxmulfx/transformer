module

public import Transformer.Grokking.FiniteRecovery.FactorVocabulary

/-! The linear query projection compensates the two known token-norm
classes. It gives the ordinary positional amplitudes on every prompt,
including when a separator is itself the unknown constant-row output. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division.Factors

open Transformer.PhasorV2

noncomputable def numeralGain := doubleGain (Real.sqrt 6)
noncomputable def specialGain := doubleGain (Real.sqrt 3)

theorem numeralGain_pos : 0 < numeralGain := doubleGain_pos (Real.sqrt_nonneg _)
theorem specialGain_pos : 0 < specialGain := doubleGain_pos (Real.sqrt_nonneg _)

noncomputable def common : Stream →ₗ[ℂ] ℂ :=
  Parity.coord 0 + ((specialGain/numeralGain-1 : ℝ):ℂ) • Parity.coord 12 +
    ((doubleGain 1/specialGain-1 : ℝ):ℂ) • Parity.coord 11

theorem common_vector (p : Channels) :
    common p.vector = 1 + ((specialGain/numeralGain-1 : ℝ):ℂ)*p.numberKey +
      ((doubleGain 1/specialGain-1 : ℝ):ℂ)*p.specialKey := by
  change p.vector 0 + _ * p.vector 12 + _ * p.vector 11 = _
  norm_num [Channels.vector,EuclideanSpace.single,PiLp.single_apply]

noncomputable def input (χ : Character) (r c : UnitFactor) (z : Fin 239) (x : DivisionInput) (j : Fin 5) : Stream :=
  argandRMS Parity.offset (vocab χ r c z (divisionPrompt x j))

noncomputable def normalized (χ : Character) (r c : UnitFactor) (z : Fin 239)
    (x : DivisionInput) (j : Fin 5) : Stream := argandRMS Parity.offset (input χ r c z x j)

private theorem normalized_token (χ : Character) (r c : UnitFactor) (z : Fin 239)
    (x : DivisionInput) (j : Fin 5) :
    normalized χ r c z x j =
      (doubleGain ‖vocab χ r c z (divisionPrompt x j)‖ : ℂ) • vocab χ r c z (divisionPrompt x j) :=
  rms_twice _

private theorem common_number (χ : Character) (r c : UnitFactor) (z : Fin 239) (a : ZMod 97) :
    common (vocab χ r c z (residueToken a)) = ((specialGain/numeralGain:ℝ):ℂ) := by
  have hr : residue (residueToken a) = some a := residue_some.mpr rfl
  simp [vocab,common_vector,channels,number,hr]

private theorem common_small (χ : Character) (r c : UnitFactor) (z : Fin 239)
    {t : Fin 239} (ht : t.val < 22) :
    (doubleGain ‖vocab χ r c z t‖ : ℂ) * common (vocab χ r c z t) = (doubleGain 1:ℂ) := by
  rw [norm_small _ _ _ _ ht]
  rcases tag_values z t with h | h | h
  · simp [vocab,common_vector,channels,number,residue_small ht,h]
  all_goals
    have hp : (specialGain:ℂ) ≠ 0 := by exact_mod_cast specialGain_pos.ne'
    simp [vocab,common_vector,channels,number,residue_small ht,h,← show doubleGain (Real.sqrt 3) = specialGain from rfl]
    change (specialGain:ℂ) * (↑(doubleGain 1) / specialGain) = _
    field_simp

private theorem common_residue (χ : Character) (r c : UnitFactor) (z : Fin 239)
    (a : ZMod 97) (hn : ‖vocab χ r c z (residueToken a)‖ = Real.sqrt 6) :
    (doubleGain ‖vocab χ r c z (residueToken a)‖ : ℂ) * common (vocab χ r c z (residueToken a)) = (specialGain:ℂ) := by
  have hp : (numeralGain:ℂ) ≠ 0 := by exact_mod_cast numeralGain_pos.ne'
  rw [hn,common_number]
  change (numeralGain:ℂ) * (↑(specialGain/numeralGain : ℝ) : ℂ) = _
  push_cast
  field_simp

theorem normalized_common (χ : Character) (r c : UnitFactor) (z : Fin 239) (x : DivisionInput) (j : Fin 5) :
    common (normalized χ r c z x j) = (posGain j : ℂ) := by
  rw [normalized_token,map_smul,smul_eq_mul]
  fin_cases j
  · exact common_small χ r c z (t := 0) (by decide)
  · obtain ⟨a,b⟩ := x
    cases a with
    | none => exact common_residue χ r c z 0 (norm_zero _ _ _ _)
    | some a => exact common_residue χ r c z a.val (norm_unit _ _ _ _ _)
  · exact common_small χ r c z (t := 10) (by decide)
  · exact common_residue χ r c z x.2.val (norm_unit _ _ _ _ _)
  · exact common_small χ r c z (t := 1) (by decide)

noncomputable def features (m : Fin 16) : Stream →ₗ[ℂ] ℂ :=
  if m=0 then Parity.coord 2 else if m=1 then Parity.coord 4 else Parity.coord 8

theorem feature_vector (p : Channels) (m : Fin 16) :
    features m p.vector = if m=0 then p.row else if m=1 then p.column else (p.zeroFlag:ℂ) := by
  by_cases h0 : m=0 <;> by_cases h1 : m=1 <;>
    norm_num [features,h0,h1,Parity.coord,Channels.vector,EuclideanSpace.single,PiLp.single_apply]

private theorem features_small (χ : Character) (r c : UnitFactor) (z : Fin 239)
    (x : DivisionInput) (j : Fin 5) (ht : (divisionPrompt x j).val < 22) (m : Fin 16) :
    features m (normalized χ r c z x j) = 0 := by
  rw [normalized_token,map_smul]
  simp [vocab,feature_vector,channels,unitAt_small ht,(small_features χ ht).2]

noncomputable def rowPhase (χ : Character) (r : UnitFactor) (a : Option DivisionUnit) : ℂ :=
  a.elim 0 (fun u => χ (r u).toMul.val)

noncomputable def zeroValue (a : Option DivisionUnit) : ℝ := a.elim 1 (fun _ => 0)

/-- The numerator, denominator and zero indicator share one positive scale
after routing, for every choice of factors and constant-row output. -/
theorem routed (χ : Character) (r c : UnitFactor) (z : Fin 239) (x : DivisionInput) (i : Fin 64) :
    attentionWith (causalGeometry 0) (routeMap common features) Parity.offset 1 (normalized χ r c z x) (4 : Fin 5) i =
      (numeralGain:ℂ) * (if i=5 then rowPhase χ r x.1 else if i=6 then χ (c x.2).toMul.val
        else if i=7 then (zeroValue x.1 : ℂ) else 0) := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  rw [route_features _ _ _ (normalized_common _ _ _ _ _) (fun m =>
    ⟨features_small _ _ _ _ _ 0 (by change 0 < 22; decide) m,
      features_small _ _ _ _ _ 2 (by change 10 < 22; decide) m,
      features_small _ _ _ _ _ 4 (by change 1 < 22; decide) m⟩)]
  have hu (a : DivisionUnit) : unitAt (residueToken a.val) = some a := unitAt_some.mpr rfl
  have hf (a : DivisionUnit) : flag (residueToken a.val) = 0 :=
    ite_eq_right fun he => a.ne_zero (residueToken_injective he)
  obtain ⟨a,b⟩ := x
  cases a with
  | none =>
    have h1 : divisionPrompt (none,b) 1 = residueToken 0 := rfl
    have h3 : divisionPrompt (none,b) 3 = residueToken b.val := rfl
    simp only [normalized_token,h1,h3,norm_zero,norm_unit,map_smul,smul_eq_mul]
    simp [vocab,feature_vector,channels,unitAt_zero,hu,flag,rowPhase,zeroValue,numeralGain,mul_ite]
  | some a =>
    have h1 : divisionPrompt (some a,b) 1 = residueToken a.val := rfl
    have h3 : divisionPrompt (some a,b) 3 = residueToken b.val := rfl
    simp only [normalized_token,h1,h3,norm_unit,map_smul,smul_eq_mul]
    simp [vocab,feature_vector,channels,hu,hf,rowPhase,zeroValue,numeralGain,mul_ite]

end Transformer.Grokking.FiniteRecovery.Argand.Division.Factors
