module

public import Transformer.Grokking.FiniteRecovery.OperandMixer

/-! The direct character embedding presents the numerator phase, conjugate
denominator phase and zero flag to the two-head operand router. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2

def bandIndex (m : Fin 16) : Fin 64 := ⟨m.val+1,by omega⟩

noncomputable def queryKey := selector (Parity.coord 0)
noncomputable def values : Stream →ₗ[ℂ] Heads 4 16 :=
  LinearMap.pi fun _ => LinearMap.pi fun m => Parity.coord (bandIndex m)
noncomputable def attentionMap := routeMap (Parity.coord 0) (fun m => Parity.coord (bandIndex m))

noncomputable def normalized (χ : Character) (x : DivisionInput) (j : Fin 5) : Stream :=
  (posGain j : ℂ) • embedding χ (divisionPrompt x j)

private theorem common (χ : Character) (x : DivisionInput) (j : Fin 5) :
    Parity.coord 0 (normalized χ x j) = (posGain j : ℂ) := by
  change (posGain j : ℂ) * embedding χ (divisionPrompt x j) 0 = _
  rw [(embedding_coords _ _).1,mul_one]

noncomputable def mixture (χ : Character) (x : DivisionInput) :=
  weightedValues fun j m => normalized χ x j (bandIndex m)

theorem native_mixture (χ : Character) (x : DivisionInput) :
    attentionWith (causalGeometry 0) attentionMap Parity.offset 1 (normalized χ x) (4 : Fin 5) =
      attentionMap.WO (mixture χ x) :=
  native_weightedValues _ _ _ (common χ x)

private theorem special_zero (χ : Character) {t : Fin 239} (ht : t.val < 22) (m : Fin 16) :
    embedding χ t (bandIndex m) = 0 := by
  obtain ⟨hp,hf⟩ := small_features χ ht
  have hi : bandIndex m ≠ 0 := by intro h; have := congrArg Fin.val h; simp [bandIndex] at this
  simp [embedding,hp,hf,EuclideanSpace.single,PiLp.single_apply,hi]

/-- The output projection writes operands at the equals sign before
the native residual addition and RMS. -/
theorem routed (χ : Character) (x : DivisionInput) (i : Fin 64) :
    attentionWith (causalGeometry 0) attentionMap Parity.offset 1 (normalized χ x) (4 : Fin 5) i =
      if i = 5 then normalized χ x 1 1 else if i = 6 then normalized χ x 3 2
      else if i = 7 then normalized χ x 1 3 else 0 := by
  have hz (j : Fin 5) (m : Fin 16) (ht : (divisionPrompt x j).val < 22) :
      Parity.coord (bandIndex m) (normalized χ x j) = 0 := by
    change (posGain j : ℂ) * embedding χ (divisionPrompt x j) (bandIndex m) = 0
    rw [special_zero χ ht,mul_zero]
  exact route_features _ _ _ (common χ x) (fun m =>
    ⟨hz 0 m (by change 0 < 22; decide),hz 2 m (by change 10 < 22; decide),hz 4 m (by change 1 < 22; decide)⟩) i

end Transformer.Grokking.FiniteRecovery.Argand.Division
