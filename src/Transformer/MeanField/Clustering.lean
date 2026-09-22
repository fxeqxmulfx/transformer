/-
# Mean-Field Dynamics — clustering & rates (§4 of 2512.01868v4)

* `Theorem thm: clustering_finite`  — global clustering for `d ≥ 3` (revisits
                                       `thm: boumal`),
* `Corollary cor: d-ge-n`           — uniform random init when `d ≥ n`.

`Theorem thm: mfclust` is in `MeanField.GlobalRate`.

`Theorem thm: cone-collapse` is `Perspective.cone_collapse`
(in `Perspective.Section5_ExpRate`), proved there for every real `β` and all
`d`, `n` — the source assumes `β > 0`, `d ≥ 2`, `n ≥ 1` — and for the `Q, K`
variants as well; it is not restated here.

`thm: clustering_finite` is read against the uniform law
`Perspective.UniformTuple`, and is not proved.  `cor: d-ge-n` is
`d_ge_n_exponential`, proved: `n ≤ d` uniform particles lie in an open
hemisphere almost surely (`Perspective.ae_exists_openHemisphere`), which is
the hypothesis `thm: cone-collapse` runs on.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap
import Transformer.Perspective.Section2_GradientFlow
import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.Section5_HighD
import Transformer.Perspective.Section5_ExpRate
import Transformer.MeanField.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace MeanField

open MeanField Perspective

variable (d n : ℕ)

/-- **Theorem (thm: clustering_finite).** *Global clustering for `d ≥ 3`.*

(Markdahl–Thunberg–Boumal; Criscitiello–Boumal; Geshkovski et al.)

For both `eq: SA` and `eq: USA` with `n ≥ 2` particles in dimension `d ≥ 3`
and `β ≥ 0`, for almost every initial `(x_i(0))_{i ∈ [n]} ∈ (𝕊^{d-1})^n` the
trajectories exist globally and converge to a single cluster:

  `lim_{t → ∞} ‖x_i(t) - x_j(t)‖ = 0`.

"Almost every" is with respect to the uniform law on `(𝕊^{d-1})^n`,
`Perspective.UniformTuple`.  Against an arbitrary reference measure the claim
is false: the Dirac mass at an antipodal pair, a stationary configuration of
both dynamics, is a counterexample.  The two dynamics are one conjunction, as
the source states them in one theorem; for each, global existence is the first
half and convergence of every solution the second.

Not proved here.

Source: arXiv:2512.01868v4, §4, `thm:clustering_finite`. -/
theorem global_clustering (β : ℝ) (hd : 3 ≤ d) (hn : 2 ≤ n) (hβ : 0 ≤ β) :
    ∀ P : Measure (SphereTuple d n), Perspective.UniformTuple d n P →
    ∀ᵐ X₀ ∂P,
      ((∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ Perspective.SA d n β X) ∧
        ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
          ∀ i j : Idx n,
            Filter.Tendsto
              (fun t : ℝ => ‖(X t i : EucSpace d) - (X t j : EucSpace d)‖)
              Filter.atTop (nhds 0)) ∧
      ((∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ Perspective.USA d n β X) ∧
        ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.USA d n β X →
          ∀ i j : Idx n,
            Filter.Tendsto
              (fun t : ℝ => ‖(X t i : EucSpace d) - (X t j : EucSpace d)‖)
              Filter.atTop (nhds 0)) := by
  sorry

/-- The hypotheses of `global_clustering` are satisfiable: `d = 3`, `n = 2`,
`β = 0`. -/
example : 3 ≤ 3 ∧ 2 ≤ 2 ∧ (0 : ℝ) ≤ 0 := ⟨le_rfl, le_rfl, le_rfl⟩

/-- **Corollary (cor: d-ge-n).** *Uniform random initialization when
`d ≥ n`.*

If the initial tokens are sampled i.i.d. uniformly on `𝕊^{d-1}` and `d ≥ n`,
then they lie in some open hemisphere almost surely, and `thm: cone-collapse`
yields exponential convergence to a single cluster: there are `x⋆` and
`C, λ > 0` with `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}` for all `i` and `t ≥ 0`, for
`eq: SA` and for `eq: USA`.

**What the source says and what is changed here.**  The corollary inherits
the hypotheses `n ≥ 1`, `β > 0`, `d ≥ 2` of `thm: cone-collapse`; none is used
and all are dropped, as in `Perspective.d_infty_exponential`.  The source
justifies the hemisphere by "`n` points in dimension `d ≥ n` must lie in the
same hemisphere"; that holds almost surely and not for every configuration —
the antipodal pair `(x, -x)` lies in no open hemisphere and obeys no
exponential rate (`Perspective.antipodalPair_not_exponential`) — and the
almost-sure statement is `Perspective.ae_exists_openHemisphere`.

Source: arXiv:2512.01868v4, §4, `cor:d-ge-n` (with `thm:cone-collapse`). -/
theorem d_ge_n_exponential (β : ℝ) (hdn : n ≤ d) :
    ∀ P : Measure (SphereTuple d n), Perspective.UniformTuple d n P →
      ∀ᵐ X₀ ∂P,
        (∃ w : SSphere d, ∀ i : Idx n,
          0 < inner (𝕜 := ℝ) (X₀ i : EucSpace d) (w : EucSpace d)) ∧
        ExpConvergent d n (Perspective.SA d n β) X₀ ∧
        ExpConvergent d n (Perspective.USA d n β) X₀ := by
  intro P hP
  filter_upwards [ae_exists_openHemisphere d hdn P hP] with X₀ hX₀
  exact ⟨hX₀, (cone_collapse d n β X₀ hX₀).1, (cone_collapse d n β X₀ hX₀).2.1⟩

/-- The hypothesis of `d_ge_n_exponential` is satisfiable: `d = n = 1`; the
uniform law is quantified over. -/
example : 1 ≤ 1 := le_rfl

end MeanField
end Transformer
