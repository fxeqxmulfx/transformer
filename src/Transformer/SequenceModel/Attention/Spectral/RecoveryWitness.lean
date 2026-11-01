module

public import Transformer.SequenceModel.Attention.Spectral.Existence

/-!
# Concrete faithful-bank witnesses for the composed certificate

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J3--J7. A faithful character over F_3 gives nonconstant fifth-power division
and the true kernel satisfies the stated fitting and minimization premises.
Zero-numerator and exponent-zero examples are included in the lower modules.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Matrix Grokking.FlatSpectrum Expressivity

private def witnessChar : MulChar (ZMod 3) ℂ := Classical.choose (exists_injective_mulChar (p := 3))

private theorem witness_injective : Function.Injective witnessChar :=
  Classical.choose_spec (exists_injective_mulChar (p := 3))

private theorem witness_separates : ∀ a b : (ZMod 3)ˣ, a ≠ b →
    ∃ r : Unit, 0 < (fun _ : Unit => (1 : ℝ)) r ∧ (fun _ : Unit => witnessChar) r a ≠ witnessChar b :=
  fun a b hab => ⟨(), zero_lt_one, fun h => hab (Units.ext (@witness_injective a b h))⟩

example := witness_separates 1 (-1) (by decide)

example := recovered_unit_margin (fun _ : Unit => witnessChar) 5 (fun _ => zero_le_one)
  witness_separates Set.univ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩
  (fun _ => FeatureCodes.powerKernel witnessChar 5) (fun _ _ _ => rfl)
  (true_bank_minimizes _ _ _ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩)

example := recovered_power_margin (fun _ : Unit => witnessChar) 5 (fun _ => zero_le_one) (by simp)
  witness_separates Set.univ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩
  (fun _ => FeatureCodes.powerKernel witnessChar 5) (fun _ _ _ => rfl)
  (true_bank_minimizes _ _ _ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩)

example : ∃ m > 0, min 1 m ≤
    powerScores (fun _ : Unit => witnessChar) (fun _ => 1) 5
      (fun _ => FeatureCodes.powerKernel witnessChar 5) ![1, 2] (powerLabel 3 5 1 2) -
    powerScores (fun _ : Unit => witnessChar) (fun _ => 1) 5
      (fun _ => FeatureCodes.powerKernel witnessChar 5) ![1, 2] 0 := by
  obtain ⟨m, hm, hg⟩ := recovered_unit_margin (fun _ : Unit => witnessChar) 5
    (fun _ => zero_le_one) witness_separates Set.univ full_pattern_connected
    ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩ (fun _ => FeatureCodes.powerKernel witnessChar 5)
    (fun _ _ _ => rfl)
    (true_bank_minimizes _ _ _ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩)
  have hv : (0 : Fin 3) ≠ powerLabel 3 5 1 2 := by
    intro he
    have hn := powerLabel_nonzero (p := 3) 5 1 2 (by decide) (by decide)
    rw [← he] at hn
    exact hn (by decide)
  have h := powerScores_gap (fun _ : Unit => witnessChar) (fun _ => 1) 5 (by simp)
    (fun _ => FeatureCodes.powerKernel witnessChar 5) rfl m hg ![1, 2] (by decide) 0 hv
  exact ⟨m, hm, h⟩

example := exists_recovered_power_model (fun _ : Unit => witnessChar) 5
  (fun _ => zero_le_one) (by simp) witness_separates Set.univ full_pattern_connected
  ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩ (fun _ => FeatureCodes.powerKernel witnessChar 5)
  (fun _ _ _ => rfl)
  (true_bank_minimizes _ _ _ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩)
  0 1 (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)

example := exists_power_certificate (fun _ : Unit => witnessChar) 5
  (fun _ => zero_le_one) (by simp) witness_separates Set.univ full_pattern_connected
  ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩ (fun _ => FeatureCodes.powerKernel witnessChar 5)
  (fun _ _ _ => rfl)
  (true_bank_minimizes _ _ _ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩)
  0 1 (eps := 1 / 8388608) (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) (by norm_num)

end Transformer.SequenceModel.Attention.Spectral
