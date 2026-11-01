module

public import Transformer.SequenceModel.Attention.Spectral.Powered
public import Transformer.SequenceModel.Attention.Spectral.Counting

/-!
# Spectral recovery, the normalized source transformer and RoPE counting

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J1--J7. This composes recovery of character scores with actual source weights,
a strictly positive execution allowance, and a fixed counting-formula family.
All numerators and natural exponents are included; denominators are nonzero.
Numerical range, precision and primitive execution contracts remain explicit.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Matrix Grokking.FlatSpectrum Expressivity Stability Execution CRASP
open scoped BigOperators

/-- J1--J7: recovered separating character scores compile into the normalized
source with exact geometric RoPE. The construction preserves the full score
table, has a positive source margin and a positive local execution tolerance,
and certifies every class decision by one fixed family of counting formulae.
The counting format must satisfy the displayed precision/range/error bounds;
the trace must satisfy the local operation and table contracts. -/
theorem exists_power_certificate {p : ℕ} [Fact p.Prime] {R : Type*} [Fintype R]
    (χ : R → MulChar (ZMod p) ℂ) (k : ℕ)
    {w : R → ℝ} (hw : ∀ r, 0 ≤ w r) (hmass : ∑ r, w r = 1)
    (hsep : ∀ a b : (ZMod p)ˣ, a ≠ b → ∃ r, 0 < w r ∧ χ r a ≠ χ r b)
    (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)) (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (hfit : Fits χ k Ω X) (hmin : Minimizes χ k Ω X)
    (padding extraDepth : ℕ) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) (hx : xeps ≤ 1) :
    let cfg := Grid.sequenceConfig p (Fact.out : p.Prime).pos 1 padding extraDepth
    let S := powerScores χ w k X
    ∃ θ : ModelParams cfg, ∃ g : (Fin 2 → Fin p) → ℝ, ∃ m η : ℝ,
      0 < m ∧ 0 < η ∧ (∀ v, ‖θ.embedding v‖ = amplitude cfg eps) ∧
      (∀ x, minimumGain cfg eps S ≤ g x) ∧
      (∀ x v, sourceLogit cfg θ eps xeps x v = g x * S x v) ∧
      (∀ x : Fin 2 → Fin p, residue (x 1) ≠ 0 → ∀ v, v ≠ powerLabel p k (x 0) (x 1) →
        minimumGain cfg eps S * m ≤ sourceLogit cfg θ eps xeps x (powerLabel p k (x 0) (x 1)) -
          sourceLogit cfg θ eps xeps x v) ∧
      ExecutionBelow cfg θ eps xeps η η (minimumGain cfg eps S * m) ∧
      ∀ bits frac, 1 / (2 : ℝ) ^ frac < Real.exp (-Real.sqrt (cfg.head_dim : ℝ)) →
        ∃ φ : Fin p → Rotary.Form (Fin p)
            (Rotary.Key (Rotary.State cfg bits frac) (Normalized.Coordinate cfg)),
          (∀ v, (φ v).depth ≤ cfg.n_layers) ∧
          ∀ (x : Fin 2 → Fin p), residue (x 1) ≠ 0 →
            ∀ (t : ModelTrace cfg 2) (δ τ : ℝ), 0 ≤ δ → 0 ≤ τ →
            ModelAccurate cfg θ eps xeps
              (sourceAngles cfg (fun _ => true) (fun j : Fin 2 => (j.val : ℝ))) x t δ τ →
            TraceInRange cfg θ eps xeps bits frac (fun _ => true) x →
            CountingBelow cfg θ eps xeps frac (minimumGain cfg eps S * m) →
            ExecutionBelow cfg θ eps xeps δ τ (minimumGain cfg eps S * m) →
            ∀ v, (φ v).sat (Rotary.kernel
              (Rotary.layers cfg θ eps xeps bits frac (fun _ => true))) (List.ofFn x) 2 =
                decide (v = powerLabel p k (x 0) (x 1)) ∧
              t.predict 1 v = decide (v = powerLabel p k (x 0) (x 1)) := by
  let cfg := Grid.sequenceConfig p (Fact.out : p.Prime).pos 1 padding extraDepth
  let S := powerScores χ w k X
  obtain ⟨θ, g, m, hm, he, hgain, hs, hgap⟩ := exists_recovered_power_model χ k hw hmass hsep
    Ω hconn hmatch X hfit hmin padding extraDepth heps hx
  have hwidth : 3 < cfg.d_model := lt_trans (by decide : 3 < 9)
    (Finite.sequence_room p (Fact.out : p.Prime).pos 2 padding).1
  have hγ : 0 < minimumGain cfg eps S * m := mul_pos (minimumGain_pos cfg hwidth heps S) hm
  obtain ⟨η, hη, herror⟩ := exists_execution_tolerance cfg θ eps xeps
    (minimumGain cfg eps S * m) hxeps.le hγ
  refine ⟨θ, g, m, η, hm, hη, he, hgain, hs, hgap, herror, ?_⟩
  intro bits frac hp
  exact uniform_margin_certificate cfg θ heps hxeps bits frac hp
    (fun x : Fin 2 → Fin p => residue (x 1) ≠ 0)
    (fun x => powerLabel p k (x 0) (x 1)) (minimumGain cfg eps S * m) hgap

end Transformer.SequenceModel.Attention.Spectral
