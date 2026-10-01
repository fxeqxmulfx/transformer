/-
# `MAJ²`: the depth hierarchy

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E: `thm:ltc0_hierarchy`.

The paper concludes from `thm:TLC_depth` and the translations of
`CRASP.MajTwoEquiv` that the circuit depth hierarchy for `FO[<]`-uniform
`LTC⁰` is strict.  The last step is Theorem 3 of Behle & Lange, about
circuits, which this development does not model; what is stated here is the
half that lives in the logic — the `MAJ²` depth hierarchy is strict
(`majTwo_depth_hierarchy`).

The negative half uses the proved reverse inclusion of `thm:logical_inclusions`.
The positive half uses closed tests for the Boolean combination of subsequence
patterns defining `D_k`, constructed in `CRASP.MajTwoPiecewise`.
-/

import Transformer.CRASP.MajTwoPiecewise

namespace Transformer
namespace CRASP

/-- **Corollary `thm:ltc0_hierarchy`, negative half, from the inclusion.**
If every closed `MAJ²_k` language is a `TL[◁#,▷#]_k` language — the second
half of `thm:logical_inclusions`, `definable_of_closed_majTwo` — then no
closed `MAJ²_k` formula defines `D_{k+1}`.  This is the paper's argument:
`D_{k+1}` is not `TL[◁#,▷#]_k`-definable by `thm:TLC_depth`
(`definable_altPlusDouble`, proved).

The implication takes the inclusion as a hypothesis; the full hierarchy
below supplies it from `definable_of_closed_majTwo`.

Source: arXiv:2506.16055v3, Appendix E, proof of `thm:ltc0_hierarchy`. -/
theorem not_majTwo_altPlusDouble_of_inclusion (k : ℕ)
    (hincl : ∀ φ : Maj2 Bool, φ ∈ MajTwo Bool k → φ.Closed → Definable φ.lang k) :
    ∀ φ ∈ MajTwo Bool k, φ.Closed → φ.lang ≠ altPlusDouble (k + 1) :=
  fun φ hφ hc hlang => (definable_altPlusDouble k).2 (hlang ▸ hincl φ hφ hc)

/-- **Corollary `thm:ltc0_hierarchy`, logical half.**  The `MAJ²` depth
hierarchy is strict: `D_k` is `MAJ²_k`-definable while `D_{k+1}` is not.  By
Theorem 3 of Behle & Lange this is equivalent to the strictness of the circuit
depth hierarchy for `FO[<]`-uniform `LTC⁰`, which is the corollary as stated in
the paper; circuits are not modelled here.

The negative half follows from `definable_of_closed_majTwo`. For the positive
half the paper takes the `TL[◁#,▷#]_k` formula for `D_k` of `lem:piecewise_testable_depth`
and asserts that it has the form `#[ψ] > 0`, so that
`thm:tlc_to_majtwo_closed` makes its translation closed.  That form is not
available in general: at `k = 1`, `D_1 = a⁺` is not "some position satisfies
`ψ`" for a depth-`0` `ψ`, since position `1` of `a` and of `ab` agree on every
depth-`0` formula.  `D_1` is nevertheless defined by the closed `MAJ²_1`
formula `∃x[⊤] ∧ ¬∃x[Q_b(x)]`. The proof uses Boolean combinations of
closed middle-position tests for the `J`-expressions of `D_k`, each of width
at most `2k - 1`, as constructed in `exists_closed_majTwo_of_sublist`.

The paper does not bound `k`; `0 < k` is forced, not added.  At `k = 0` the
first half is false: no closed formula has depth `0`
(`Maj2.not_closed_of_depth_eq_zero`), so `MAJ²_0` defines no language, not
even `D_0 = {ε}`.

Source: arXiv:2506.16055v3, Appendix E, `thm:ltc0_hierarchy`. -/
theorem majTwo_depth_hierarchy (k : ℕ) (hk : 0 < k) :
    (∃ φ ∈ MajTwo Bool k, φ.Closed ∧ φ.lang = altPlusDouble k) ∧
      ∀ φ ∈ MajTwo Bool k, φ.Closed → φ.lang ≠ altPlusDouble (k + 1) := by
  have hpt : KPiecewiseTestable (2 * k - 1) (altPlusDouble k) :=
    kPiecewiseTestable_altPlus (2 * k - 1) (by omega)
  exact ⟨exists_closed_majTwo_of_piecewise_testable k hk _ hpt,
    not_majTwo_altPlusDouble_of_inclusion k fun φ hφ hc =>
      definable_of_closed_majTwo k φ hφ hc⟩

/-- The hypotheses of both hierarchy statements are satisfiable (Appendix E). -/
example (k : ℕ) : 0 < k + 1 ∧
    ∀ φ : Maj2 Bool, φ ∈ MajTwo Bool (k + 1) → φ.Closed → Definable φ.lang (k + 1) :=
  ⟨Nat.succ_pos k, fun φ hφ hc => definable_of_closed_majTwo (k + 1) φ hφ hc⟩

end CRASP
end Transformer
