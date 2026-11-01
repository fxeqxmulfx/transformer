module

public import Transformer.SequenceModel.Attention.Spectral.Certificate

/-!
# Satisfiable spectral premises and genuinely perturbed source executions

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J3--J7. Faithful characters and their true kernels instantiate every spectral
premise for all primes. The derived source model tolerates a strictly positive
readout error. This synthetic execution is not attributed to a CUDA kernel.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Matrix Grokking.FlatSpectrum Expressivity Execution

/-- J6: for every prime and exponent, the spectral construction yields
actual normalized source weights and a positive execution allowance.
One trace satisfying that allowance has a genuinely nonzero error at every
logit and still gives all powered-division class decisions correctly. -/
theorem exists_robust_power_source (p : ℕ) [Fact p.Prime] (k padding extraDepth : ℕ)
    {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps) (hx : xeps ≤ 1) :
    let cfg := Grid.sequenceConfig p (Fact.out : p.Prime).pos 1 padding extraDepth
    ∃ θ : ModelParams cfg, ∃ η > 0, ∀ (x : Fin 2 → Fin p), residue (x 1) ≠ 0 →
      let t := shiftedModel cfg θ eps xeps
        (sourceAngles cfg (fun _ => true) (fun j : Fin 2 => (j.val : ℝ))) x η
      ModelAccurate cfg θ eps xeps
        (sourceAngles cfg (fun _ => true) (fun j : Fin 2 => (j.val : ℝ))) x t η η ∧
      ∀ v, t.predict 1 v = decide (v = powerLabel p k (x 0) (x 1)) ∧
        t.logits 1 v - sourceLogit cfg θ eps xeps x v = η := by
  let cfg := Grid.sequenceConfig p (Fact.out : p.Prime).pos 1 padding extraDepth
  obtain ⟨χ, hinj⟩ := exists_injective_mulChar (p := p)
  have hsep : ∀ a b : (ZMod p)ˣ, a ≠ b →
      ∃ r : Unit, 0 < (fun _ : Unit => (1 : ℝ)) r ∧ (fun _ : Unit => χ) r a ≠ χ b :=
    fun a b hab => ⟨(), zero_lt_one, fun h => hab (Units.ext (hinj h))⟩
  obtain ⟨θ, g, m, η, _, hη, _, _, _, hg, hbudget, _⟩ :=
    exists_power_certificate (fun _ : Unit => χ) k (fun _ => zero_le_one) (by simp) hsep
      Set.univ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩
      (fun _ => FeatureCodes.powerKernel χ k) (fun _ _ _ => rfl)
      (true_bank_minimizes _ _ _ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩)
      padding extraDepth heps hxeps hx
  refine ⟨θ, η, hη, ?_⟩
  intro x hb
  have ht := shiftedModel_accurate cfg θ eps xeps
    (sourceAngles cfg (fun _ => true) (fun j : Fin 2 => (j.val : ℝ))) x η η hη.le hη.le
  refine ⟨ht, ?_⟩
  have hpred := executed_of_source_margin cfg θ heps hxeps x _ _ (hg x hb) _ η η
    hη.le hη.le ht hbudget
  intro v
  refine ⟨executed_tests_eq cfg _ 1 _ hpred v, ?_⟩
  simpa only [sourceLogit, score, decode_encode, show (Fin.last 1 : Fin 2) = 1 from by decide] using
    shiftedModel_logit_error cfg θ eps xeps _ x η 1 v

example := exists_robust_power_source 3 0 0 1 (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) (by norm_num)

section
local instance : Fact (Nat.Prime 97) := ⟨by decide⟩
example := exists_robust_power_source 97 5 0 3 (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) (by norm_num)
end

end Transformer.SequenceModel.Attention.Spectral
