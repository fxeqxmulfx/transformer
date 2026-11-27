module

public import Transformer.Grokking.FiniteRecovery.ArgandParityNorm
public import Transformer.Grokking.FiniteRecovery.ArgandPhaseEnergy

/-! Modular-sum nuclear control from the full model's normalized hidden
states and tied embeddings. The score discrepancy is bounded before the
softmax, so the estimate requires no missing labels. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Finset FlatSpectrum Transformer.Optimization
open scoped BigOperators InnerProductSpace Topology

variable {H d layers k s : ℕ}

/-- Sum of squared deviations of the full final states from the signed
diagonal states. Each diagonal word belongs to the staircase. -/
noncomputable def Model.parityStateResidual (model : Model (ZMod 2) H d layers)
    (repeats : ℕ) (e : Fin (s + 1) ≃ Word (ZMod 2) (k + 1)) : ℝ :=
  let h u v := model.hidden repeats (enumeratedCut e (u,v)) (Fin.last ((k + 1) + k))
  ∑ u, ∑ v, ‖h u v - ((rowParity (e u) * rowParity (e v) : ℝ) : ℂ) • h v v‖ ^ 2

/-- The parity gap reads just the projection onto the difference of
the two tied embedding rows. -/
theorem Model.parityGapResidual_le_states (model : Model (ZMod 2) H d layers)
    (repeats : ℕ) (e : Fin (s + 1) ≃ Word (ZMod 2) (k + 1)) :
    model.parityGapResidual repeats e ≤ (‖model.embedding 0 - model.embedding 1‖ / 2) ^ 2 *
      model.parityStateResidual repeats e := by
  let E := model.embedding 0 - model.embedding 1
  let h u v := model.hidden repeats (enumeratedCut e (u,v)) (Fin.last ((k + 1) + k))
  let z u v := model.parityGap (T := (k + 1) + k) repeats (enumeratedCut e (u,v))
  have hz (u v) : z u v = (⟪E, h u v⟫_ℂ).re / 2 := by
    simp only [z, Model.parityGap, Model.logits, Model.readout, E, h, inner_sub_left, Complex.sub_re]
    ring
  unfold Model.parityGapResidual Model.parityStateResidual
  dsimp only
  simp_rw [Finset.mul_sum]
  refine sum_le_sum fun u _ => sum_le_sum fun v _ => ?_
  let a := rowParity (e u) * rowParity (e v)
  let diff := h u v - (a : ℂ) • h v v
  have hread : z u v - a * z v v = (⟪E, diff⟫_ℂ).re / 2 := by
    simp only [hz, diff, inner_sub_right, inner_smul_right, Complex.sub_re, Complex.mul_re,
      Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]
    ring
  change (z u v - a * z v v) ^ 2 ≤ (‖E‖ / 2) ^ 2 * ‖diff‖ ^ 2
  rw [hread, ← mul_pow, ← sq_abs ((⟪E, diff⟫_ℂ).re / 2)]
  apply pow_le_pow_left₀ (abs_nonneg _)
  rw [abs_div, abs_of_pos (by norm_num : 0 < (2 : ℝ)), div_mul_eq_mul_div]
  exact div_le_div_of_nonneg_right ((Complex.abs_re_le_norm _).trans (norm_inner_le_norm _ _)) (by norm_num)

/-- At every training iterate and limit, the nuclear excess is bounded
by a hidden-state discrepancy and the learned embedding separation. -/
theorem Model.nuclearNorm_modularSum_le_states (model : Model (ZMod 2) H d layers)
    (repeats : ℕ) (e : Fin (s + 1) ≃ Word (ZMod 2) (k + 1)) :
    nuclearNorm (model.phaseKernel (T := (k + 1) + k) repeats (enumeratedCut e) ZMod.stdAddChar) ≤
      (s + 1 : ℕ) + ‖model.embedding 0 - model.embedding 1‖ *
        √((s + 1 : ℕ) * model.parityStateResidual repeats e) / 4 := by
  have hg := model.parityGapResidual_le_states repeats e
  let N : ℝ := (s + 1 : ℕ)
  let b := ‖model.embedding 0 - model.embedding 1‖ / 2
  have hN : 0 ≤ N := Nat.cast_nonneg _
  have hb : 0 ≤ b := by dsimp [b]; positivity
  have hs : N * model.parityGapResidual repeats e ≤ b ^ 2 * (N * model.parityStateResidual repeats e) :=
    (mul_le_mul_of_nonneg_left hg hN).trans_eq (by dsimp only [b]; ring)
  have h := Real.sqrt_le_sqrt hs
  rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq hb] at h
  refine (model.nuclearNorm_modularSum_le repeats e).trans (add_le_add le_rfl ?_)
  refine (div_le_div_of_nonneg_right h (by norm_num : (0 : ℝ) ≤ 2)).trans_eq ?_
  dsimp only [b, N]
  ring

/-- The true weight limit in the modular-sum experiment satisfies the
state-based estimate. Its coefficient uses only the observed prefix
energy and coupled decay, independent of the draw within that cap. -/
theorem modularSum_momentum_norm_le_states
    (realScoreInit : Bool) (repeats : ℕ) (e : Fin (s + 1) ≃ Word (ZMod 2) (k + 1))
    {lam : ℝ} (hlam : 0 < lam) (rate : ℕ → ℝ) (t₀ : ℕ) (cap : ℝ) :
    let cut := enumeratedCut e
    let batch := observedLastPositions (T := (k + 1) + k) (modularSum e) (inputsAtCells cut (staircase s))
    ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw : ComplexParameter (ZMod 2) H d layers → ℂ,
      let f := fullGradientObjective repeats batch lam
      let run := fullGradientMomentumRun realScoreInit repeats batch lam
        (fun t => if t < t₀ then rate t else η) draw
      let x := fun t => WithLp.toLp 2 (run t).weights
      fullMomentumEnergy f η (x t₀) (WithLp.toLp 2 (run t₀).velocity) ≤ cap →
        ∃ star : WeightSpace (ZMod 2) H d layers, Tendsto x atTop (𝓝 star) ∧ gradient f star = 0 ∧
          let model := Model.fromWeights (WithLp.ofLp star)
          nuclearNorm (model.phaseKernel (T := (k + 1) + k) repeats cut ZMod.stdAddChar) ≤ (s + 1 : ℕ) +
            √(2 * cap / lam) * √((s + 1 : ℕ) * model.parityStateResidual repeats e) / 2 := by
  let batch := observedLastPositions (T := (k + 1) + k) (modularSum e)
    (inputsAtCells (enumeratedCut e) (staircase s))
  obtain ⟨η₀, hη₀, hr⟩ := fullGradientMomentumRun_converges_below
    (H := H) (d := d) (layers := layers) realScoreInit repeats batch hlam rate t₀ cap
  refine ⟨η₀, hη₀, fun η hη hle draw hd => ?_⟩
  obtain ⟨star, hs, hstat, hval⟩ := hr η hη hle draw hd
  refine ⟨star, hs, hstat, ?_⟩
  let m := Model.fromWeights (WithLp.ofLp star)
  let R := √(2 * cap / lam)
  have hcost := (fullGradientObjective_coercive repeats batch lam star).trans hval
  have hR : ‖star‖ ≤ R := Real.le_sqrt_of_sq_le ((le_div_iff₀ hlam).mpr (by linarith only [hcost]))
  have he : ‖m.embedding 0 - m.embedding 1‖ ≤ 2 * R :=
    (norm_sub_le _ _).trans (by linarith only [hR, norm_embedding_le_weightSpace star 0,
      norm_embedding_le_weightSpace star 1])
  have hh := mul_le_mul_of_nonneg_right he (Real.sqrt_nonneg ((s + 1 : ℕ) * m.parityStateResidual repeats e))
  refine (m.nuclearNorm_modularSum_le_states repeats e).trans (add_le_add le_rfl ?_)
  change _ ≤ R * _ / 2
  linarith only [hh]

example (cap : ℝ) := modularSum_momentum_norm_le_states (H := 4) (d := 16) (layers := 1)
  (k := 3) (s := 15) true 1 (wordEnumeration 2 4) (show 0 < (1 / 10 : ℝ) by norm_num)
  (wsdRate 20000) 20000 cap

end Transformer.Grokking.FiniteRecovery.Argand
