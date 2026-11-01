module

public import Transformer.SequenceModel.Attention.Spectral.Scores
public import Transformer.SequenceModel.Attention.Spectral.PoweredScores

/-!
# Powered-division recovery becomes a source transformer score computation

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J5. The transformer weights compile the recovered bank's real score table,
preserving every class score up to positive normalization. Spectral recovery
and joint separation supply correctness and a uniform margin, including zero
numerators. This is finite interpolation with an explicit width bound, not
an efficient character circuit or a theorem about optimizer convergence.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open GPTMini Matrix Grokking.FlatSpectrum Expressivity
open scoped BigOperators

variable {p : ℕ} [Fact p.Prime] {R : Type*} [Fintype R]

/-- J5: the spectral recovery premises construct actual source weights
whose complete tied-logit table is the recovered feature readout, up to
positive row scaling, with an explicit transferred positive margin. -/
theorem exists_recovered_power_model (χ : R → MulChar (ZMod p) ℂ) (k : ℕ)
    {w : R → ℝ} (hw : ∀ r, 0 ≤ w r) (hmass : ∑ r, w r = 1)
    (hsep : ∀ a b : (ZMod p)ˣ, a ≠ b → ∃ r, 0 < w r ∧ χ r a ≠ χ r b)
    (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)) (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω)
    (X : R → Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ) (hfit : Fits χ k Ω X) (hmin : Minimizes χ k Ω X)
    (padding extraDepth : ℕ) {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1) :
    let cfg := Grid.sequenceConfig p (Fact.out : p.Prime).pos 1 padding extraDepth
    let S := powerScores χ w k X
    ∃ θ : ModelParams cfg, ∃ g : (Fin 2 → Fin p) → ℝ, ∃ m : ℝ,
      0 < m ∧ (∀ v, ‖θ.embedding v‖ = amplitude cfg eps) ∧
      (∀ x, minimumGain cfg eps S ≤ g x) ∧
      (∀ x v, sourceLogit cfg θ eps xeps x v = g x * S x v) ∧
      ∀ (x : Fin 2 → Fin p), residue (x 1) ≠ 0 → ∀ v,
        v ≠ powerLabel p k (x 0) (x 1) →
        minimumGain cfg eps S * m ≤ sourceLogit cfg θ eps xeps x (powerLabel p k (x 0) (x 1)) -
          sourceLogit cfg θ eps xeps x v := by
  let cfg := Grid.sequenceConfig p (Fact.out : p.Prime).pos 1 padding extraDepth
  let S := powerScores χ w k X
  obtain ⟨m, hm, hgap⟩ := recovered_power_margin χ k hw hmass hsep Ω hconn hmatch X hfit hmin
  obtain ⟨θ, g, he, hg, hs⟩ := exists_finite_score_model p (Fact.out : p.Prime).pos 1
    padding extraDepth heps hx S
  have hwidth : 3 < cfg.d_model := lt_trans (by decide : 3 < 9)
    (Finite.sequence_room p (Fact.out : p.Prime).pos 2 padding).1
  refine ⟨θ, g, m, hm, he, hg, hs, ?_⟩
  intro x hb v hv
  exact source_margin_of_scores cfg θ eps xeps S g (minimumGain cfg eps S) m
    (minimumGain_pos cfg hwidth heps S) hm hg hs x _ v (hgap x hb v hv)

/-- J5: a faithful one-character bank exists for every prime and exponent.
Its character is chosen before the candidate recovered table. The resulting
source vocabulary contains every residue, and denominator zero is excluded. -/
theorem exists_faithful_power_model (p : ℕ) [Fact p.Prime] (k padding extraDepth : ℕ)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (Ω : Set ((ZMod p)ˣ × (ZMod p)ˣ)) (hconn : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm (ZMod p)ˣ, ∀ a, (a, σ a) ∈ Ω) :
    ∃ χ : MulChar (ZMod p) ℂ, ∀ X : Matrix (ZMod p)ˣ (ZMod p)ˣ ℂ,
      Fits (fun _ : Unit => χ) k Ω (fun _ => X) →
      Minimizes (fun _ : Unit => χ) k Ω (fun _ => X) →
      let cfg := Grid.sequenceConfig p (Fact.out : p.Prime).pos 1 padding extraDepth
      let S := powerScores (fun _ : Unit => χ) (fun _ => 1) k (fun _ => X)
      ∃ θ : ModelParams cfg, ∃ g : (Fin 2 → Fin p) → ℝ, ∃ m : ℝ,
        0 < m ∧ (∀ v, ‖θ.embedding v‖ = amplitude cfg eps) ∧
        (∀ x, minimumGain cfg eps S ≤ g x) ∧
        (∀ x v, sourceLogit cfg θ eps xeps x v = g x * S x v) ∧
        ∀ (x : Fin 2 → Fin p), residue (x 1) ≠ 0 → ∀ v,
          v ≠ powerLabel p k (x 0) (x 1) →
          minimumGain cfg eps S * m ≤ sourceLogit cfg θ eps xeps x (powerLabel p k (x 0) (x 1)) -
            sourceLogit cfg θ eps xeps x v := by
  obtain ⟨χ, hinj⟩ := exists_injective_mulChar (p := p)
  refine ⟨χ, ?_⟩
  intro X hfit hmin
  exact exists_recovered_power_model (fun _ : Unit => χ) k (fun _ => zero_le_one) (by simp)
    (fun a b hab => ⟨(), zero_lt_one, fun h => hab (Units.ext (hinj h))⟩)
    Ω hconn hmatch (fun _ => X) hfit hmin padding extraDepth heps hx

example := exists_faithful_power_model 3 0 0 1 (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) Set.univ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩

section
local instance : Fact (Nat.Prime 97) := ⟨by decide⟩
example := exists_faithful_power_model 97 5 0 3 (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) Set.univ full_pattern_connected ⟨Equiv.refl _, fun _ => Set.mem_univ _⟩
end

end Transformer.SequenceModel.Attention.Spectral
