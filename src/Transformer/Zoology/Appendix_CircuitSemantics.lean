/-
# Determinacy of well-formed arithmetic circuits

Arora et al., arXiv:2312.04927v1, Appendix `def: circuit-tuple` and
Theorem `thm: gen-ac`. The circuit model stores gates in topological order.
The existential evaluation relation is single-valued when every gate reads
only earlier nodes; this is needed before comparing its output to Coyote.
-/

import Transformer.Zoology.Appendix_CircuitDefs

namespace Transformer.Zoology

/-- A well-formed arithmetic circuit has at most one output for each input.
Source: Appendix `def: circuit-tuple`, DAG semantics. -/
theorem ArithmeticCircuit.computes_unique {inputs outputs : ℕ}
    (c : ArithmeticCircuit inputs outputs) (hwell : c.WellFormed)
    (x : Fin inputs → ℝ) (y₁ y₂ : Fin outputs → ℝ)
    (h₁ : c.Computes x y₁) (h₂ : c.Computes x y₂) : y₁ = y₂ := by
  obtain ⟨z₁, hz₁, hy₁⟩ := h₁
  obtain ⟨z₂, hz₂, hy₂⟩ := h₂
  have hz : ∀ i : Fin c.size, z₁ i = z₂ i := by
    intro i
    have hnat : ∀ m : ℕ, ∀ hm : m < c.size,
        z₁ ⟨m, hm⟩ = z₂ ⟨m, hm⟩ := by
      intro m
      induction m using Nat.strong_induction_on with
      | h m ih =>
          intro hm
          let node : Fin c.size := ⟨m, hm⟩
          have hpred : ∀ j : Fin c.size, j < node → z₁ j = z₂ j := by
            intro j hj
            have hjval : j.val < m := by
              change j.val < m at hj
              exact hj
            exact ih j.val hjval j.isLt
          have hv₁ := hz₁ node
          have hv₂ := hz₂ node
          cases hg : c.gate node with
          | input a =>
              simp only [hg, ArithmeticGate.value] at hv₁ hv₂
              exact hv₁.trans hv₂.symm
          | constant r =>
              simp only [hg, ArithmeticGate.value] at hv₁ hv₂
              exact hv₁.trans hv₂.symm
          | add a b =>
              have hab : a < node ∧ b < node := by
                simpa [ArithmeticCircuit.WellFormed, hg] using hwell node
              calc
                z₁ node = z₁ a + z₁ b := by
                  simpa [hg, ArithmeticGate.value] using hv₁
                _ = z₂ a + z₂ b := by rw [hpred a hab.1, hpred b hab.2]
                _ = z₂ node := by
                  symm
                  simpa [hg, ArithmeticGate.value] using hv₂
          | multiply a b =>
              have hab : a < node ∧ b < node := by
                simpa [ArithmeticCircuit.WellFormed, hg] using hwell node
              calc
                z₁ node = z₁ a * z₁ b := by
                  simpa [hg, ArithmeticGate.value] using hv₁
                _ = z₂ a * z₂ b := by rw [hpred a hab.1, hpred b hab.2]
                _ = z₂ node := by
                  symm
                  simpa [hg, ArithmeticGate.value] using hv₂
    exact hnat i.val i.isLt
  funext o
  calc
    y₁ o = z₁ (c.output o) := hy₁ o
    _ = z₂ (c.output o) := hz _
    _ = y₂ o := (hy₂ o).symm

/-- The hypotheses of circuit determinacy hold for the identity circuit.
Source: Appendix `def: circuit-tuple`, a one-gate example. -/
example : identityCircuit.WellFormed ∧
    identityCircuit.Computes (fun _ => 1) (fun _ => 1) := by
  constructor
  · intro i
    fin_cases i
    simp [identityCircuit]
  · exact (identityCircuit_computes _ _).mpr rfl

end Transformer.Zoology
