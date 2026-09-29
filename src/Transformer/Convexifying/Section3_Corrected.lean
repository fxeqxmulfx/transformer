/-
# Width-corrected convexification statements

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.2–§3.3 and Appendix A.
These theorems record the corrections justified by the paper's recovery
constructions and prove the resulting equivalences.
-/

import Transformer.Convexifying.Section3_Vector
import Transformer.Convexifying.Section3_Scaling
import Transformer.Convexifying.Section3_ScalarBridge
import Transformer.Convexifying.Section3_VectorRecovery

namespace Transformer.Convexifying

/-- **Corrected scalar equivalence.**  Equation (9) must not halve the loss,
and `h ≥ n` suffices to assign one head to every token row.  The paper's
Appendix A.4 gives a potentially smaller data-dependent bound `h ≥ h*`
with `h* ≤ N + 1`, but Theorem 1 states no width hypothesis.  This version
uses the simple uniform bound `n` and compares all objective sublevel sets.

Source: arXiv:2211.11052v1, §3.2, `theo:attn_multihead_convex_scalar`,
equations (8)–(9), and Appendix A.2/A.4. -/
theorem scalar_equivalence_sufficient_width {N h n d : ℕ}
    (X : Data N n d) (y : Vec N) (L : ℝ → ℝ → ℝ) (β : ℝ)
    (hn : 0 < n) (hh : n ≤ h) (hβ : 0 ≤ β) :
    ∀ r : ℝ,
      (∃ p : ScalarParameters h n d,
        p.Feasible ∧ scalarObjective X y L β p ≤ r) ↔
      (∃ Z : Fin n → Vec d, correctedConvexObjective X y L β Z ≤ r) := by
  intro r
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hh
  constructor
  · rintro ⟨p, hp, hcost⟩
    obtain ⟨ps, hps, hscaled⟩ :=
      ((scaling_equivalence X y L β hβ) r).mp ⟨p, hp, hcost⟩
    exact ⟨scalarConvexOf ps,
      (scalarConvexOf_objective_le X y L β ps hps hβ).trans hscaled⟩
  · rintro ⟨Z, hcost⟩
    let k₀ : Fin n := ⟨0, hn⟩
    let ps := scalarHeadsOf m k₀ Z
    have hps : ps.ScaledFeasible := scalarHeadsOf_feasible m k₀ Z
    have hscaled : scaledScalarObjective X y L β ps ≤ r := by
      rw [scalarHeadsOf_objective]
      exact hcost
    exact ((scaling_equivalence X y L β hβ) r).mpr ⟨ps, hps, hscaled⟩

/-- The scalar theorem's width and regularization hypotheses are jointly
satisfiable. -/
example : 0 < (1 : ℕ) ∧ (1 : ℕ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- **Corrected vector equivalence.**  The loss in equation (11) must
separate over output coordinates to match equation (12), and `h ≥ n c`
suffices to allocate a head to each token-output row.  Appendix A.2 in fact
constructs `h c` heads from `h` rows, despite Theorem 2 keeping `h` heads on
its left-hand side.  The source's unqualified assertion is refuted by
`vector_equivalence_false_for_one_head`.

Source: arXiv:2211.11052v1, §3.3,
`theo:attn_multihead_convex_vector`, equations (11)–(12), Appendix A.2/A.5. -/
theorem vector_equivalence_sufficient_width {N h n d c : ℕ}
    (X : Data N n d) (y : Fin N → Vec c)
    (L : Vec c → Vec c → ℝ) (scalarLoss : ℝ → ℝ → ℝ) (β : ℝ)
    (hL : IsSeparableLoss L scalarLoss)
    (hn : 0 < n) (hh : n * c ≤ h) (hβ : 0 ≤ β) :
    ∀ r : ℝ,
      (∃ p : VectorParameters h n d c,
        p.Feasible ∧ vectorObjective X y L β p ≤ r) ↔
      (∃ Z : Fin c → Fin n → Vec d,
        vectorConvexObjective X y scalarLoss β Z ≤ r) := by
  intro r
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hh
  constructor
  · rintro ⟨p, hp, hcost⟩
    exact ⟨vectorConvexOf p,
      (vectorConvexOf_objective_le X y L scalarLoss β p hL hp hβ).trans hcost⟩
  · rintro ⟨Z, hcost⟩
    let k₀ : Fin n := ⟨0, hn⟩
    let p := vectorHeadsOf m k₀ Z
    have hp : p.Feasible := vectorHeadsOf_feasible m k₀ Z
    have hobj : vectorObjective X y L β p ≤ r := by
      rw [vectorHeadsOf_objective m k₀ X y L scalarLoss β hL Z]
      exact hcost
    exact ⟨p, hp, hobj⟩

/-- The vector theorem's loss, width, and regularization hypotheses are
jointly satisfiable. -/
example : IsSeparableLoss oneOutputSquareLoss squareLoss ∧
    0 < (1 : ℕ) ∧ (1 : ℕ) * 1 ≤ 1 ∧ (0 : ℝ) ≤ 1 := by
  exact ⟨oneOutputSquareLoss_separable, by norm_num, by norm_num, by norm_num⟩

end Transformer.Convexifying
