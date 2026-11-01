module

public import Transformer.SequenceModel.Attention.Spectral.Residues

/-!
# Recovered character scores on the original residue vocabulary

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J4. Unit inputs use the recovered feature bank and its fixed linear readout.
The zero numerator uses its elementary boundary rule. Denominator-zero rows
have arbitrary totalized scores and are excluded from classification claims.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open Matrix Grokking.FlatSpectrum Expressivity
open scoped BigOperators

variable {p : ℕ} [Fact p.Prime] {R : Type*} [Fintype R]

/-- Scores computed from the recovered bank, with explicit residue boundary cases.
No powered-division answer oracle is used on a nonzero input pair. -/
def powerScores (χ : R → MulChar (ZMod p) ℂ) (w : R → ℝ) (k : ℕ)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (x : Fin 2 → Fin p) (v : Fin p) : ℝ :=
  if ha : residue (x 0) = 0 then
    if v = zeroAnswer p k then 1 else -1
  else if hb : residue (x 1) = 0 then 0
  else if hv : residue v = 0 then -1
  else FeatureCodes.score w (FeatureCodes.unitCode χ)
    (fun r => X r (tokenUnit (x 0) ha) (tokenUnit (x 1) hb)) (tokenUnit v hv)

/-- J4: the zero row is its algebraic indicator, including exponent zero. -/
theorem powerScores_zero (χ : R → MulChar (ZMod p) ℂ) (w : R → ℝ) (k : ℕ)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (x : Fin 2 → Fin p)
    (ha : residue (x 0) = 0) (v : Fin p) :
    powerScores χ w k X x v = if v = zeroAnswer p k then 1 else -1 := by
  simp [powerScores, ha]

/-- J4: away from the zero boundary the scores use the actual recovered coordinates. -/
theorem powerScores_unit (χ : R → MulChar (ZMod p) ℂ) (w : R → ℝ) (k : ℕ)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (x : Fin 2 → Fin p)
    (ha : residue (x 0) ≠ 0) (hb : residue (x 1) ≠ 0) (v : Fin p) (hv : residue v ≠ 0) :
    powerScores χ w k X x v = FeatureCodes.score w (FeatureCodes.unitCode χ)
      (fun r => X r (tokenUnit (x 0) ha) (tokenUnit (x 1) hb)) (tokenUnit v hv) := by
  simp [powerScores, ha, hb, hv]

/-- J4: feature weights of total mass one give true unit answers score one. -/
theorem powerScores_target (χ : R → MulChar (ZMod p) ℂ) (w : R → ℝ) (k : ℕ)
    (hmass : ∑ r, w r = 1) (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ)
    (hX : X = fun r => FeatureCodes.powerKernel (χ r) k) (x : Fin 2 → Fin p)
    (ha : residue (x 0) ≠ 0) (hb : residue (x 1) ≠ 0) :
    powerScores χ w k X x (powerLabel p k (x 0) (x 1)) = 1 := by
  rw [powerScores_unit χ w k X x ha hb _ (powerLabel_nonzero k _ _ ha hb),
    tokenUnit_power k (x 0) (x 1) ha hb, hX]
  exact (FeatureCodes.codeScore_self w (FeatureCodes.unitCode χ) (FeatureCodes.unitCode_unit χ)
    ((tokenUnit (x 0) ha / tokenUnit (x 1) hb) ^ k)).trans hmass

/-- J4: the unit-feature margin and the explicit zero boundary give a
positive margin on every original numerator and every nonzero denominator. -/
theorem powerScores_gap (χ : R → MulChar (ZMod p) ℂ) (w : R → ℝ) (k : ℕ)
    (hmass : ∑ r, w r = 1) (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ)
    (hX : X = fun r => FeatureCodes.powerKernel (χ r) k) (m : ℝ)
    (hg : ∀ a b v : (ZMod p)ˣ, v ≠ (a / b) ^ k →
      m ≤ FeatureCodes.score w (FeatureCodes.unitCode χ) (fun r => X r a b) ((a / b) ^ k) -
        FeatureCodes.score w (FeatureCodes.unitCode χ) (fun r => X r a b) v)
    (x : Fin 2 → Fin p) (hb : residue (x 1) ≠ 0) (v : Fin p)
    (hv : v ≠ powerLabel p k (x 0) (x 1)) :
    min 1 m ≤ powerScores χ w k X x (powerLabel p k (x 0) (x 1)) - powerScores χ w k X x v := by
  by_cases ha : residue (x 0) = 0
  · have hz := powerLabel_zero k (x 0) (x 1) ha
    rw [powerScores_zero χ w k X x ha, powerScores_zero χ w k X x ha, hz]
    have hvz : v ≠ zeroAnswer p k := by simpa only [hz] using hv
    simp only [ite_eq_right hvz]
    exact (min_le_left 1 m).trans (by norm_num)
  · have ht := powerLabel_nonzero k (x 0) (x 1) ha hb
    by_cases hv0 : residue v = 0
    · rw [powerScores_target χ w k hmass X hX x ha hb]
      simp only [powerScores, dite_eq_right ha, dite_eq_right hb, dite_eq_left hv0]
      exact (min_le_left 1 m).trans (by norm_num)
    · have hunit : tokenUnit v hv0 ≠ (tokenUnit (x 0) ha / tokenUnit (x 1) hb) ^ k := by
        rw [← tokenUnit_power k (x 0) (x 1) ha hb]
        intro he
        exact hv (tokenUnit_injective _ _ _ _ he)
      rw [powerScores_unit χ w k X x ha hb _ ht, powerScores_unit χ w k X x ha hb v hv0,
        tokenUnit_power k (x 0) (x 1) ha hb]
      exact (min_le_right 1 m).trans (hg _ _ _ hunit)

/-- J4: spectral recovery yields a quantitative score certificate for all
residue numerators, without assuming the source model already predicts them. -/
theorem recovered_power_margin (χ : R → MulChar (ZMod p) ℂ) (k : ℕ)
    {w : R → ℝ} (hw : ∀ r, 0 ≤ w r) (hmass : ∑ r, w r = 1)
    (hsep : ∀ a b : (ZMod p)ˣ, a ≠ b → ∃ r, 0 < w r ∧ χ r a ≠ χ r b)
    (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)) (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (hfit : Fits χ k Ω X) (hmin : Minimizes χ k Ω X) :
    ∃ m > 0, ∀ (x : Fin 2 → Fin p), residue (x 1) ≠ 0 → ∀ v,
      v ≠ powerLabel p k (x 0) (x 1) →
      m ≤ powerScores χ w k X x (powerLabel p k (x 0) (x 1)) - powerScores χ w k X x v := by
  obtain ⟨m, hm, hg⟩ := recovered_unit_margin χ k hw hsep Ω hconn hmatch X hfit hmin
  refine ⟨min 1 m, lt_min (by norm_num) hm, ?_⟩
  exact powerScores_gap χ w k hmass X (recovered_bank χ k Ω hconn hmatch X hfit hmin) m hg

example := powerScores_zero (p := 3) (fun _ : Unit => 1) (fun _ => 1) 0
  (fun _ => FeatureCodes.powerKernel 1 0) ![0, 1] (by decide) 1
example := powerScores_unit (p := 3) (fun _ : Unit => 1) (fun _ => 1) 5
  (fun _ => FeatureCodes.powerKernel 1 5) ![1, 2] (by decide) (by decide) 1 (by decide)
example := powerScores_target (p := 3) (fun _ : Unit => 1) (fun _ => 1) 5 (by simp)
  (fun _ => FeatureCodes.powerKernel 1 5) rfl ![1, 2] (by decide) (by decide)

end Transformer.SequenceModel.Attention.Spectral
