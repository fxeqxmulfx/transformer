/-
# Mean-Field Dynamics — the pairing phase (§5 of 2512.01868v4)

* `eq: init_clust`      — a discrete multi-cluster initial datum
                          `μ_0 = Σ_j α_j δ_{x_j(0)}`,
* `Theorem thm: agazzi_merge` (Bruno, Pasqualotto & Agazzi) — in the
  low-temperature limit the unique closest pair merges first, on its own
  exponentially fast timescale, while every other cluster stands still.

The initial weights `α` enter through the dynamics, not through the
conclusion: `clusterSA` is `eq: SA` read at the measure `Σ_j α_j δ_{x_j(t)}`,
which is the mean-field field `Perspective.vectorField` evaluated there,
written out as a finite sum so that no probability-measure bundling is needed.

Self-attention is kept in `clusterSA`, as `eq: SA` has it; the `argmax` limit
of the survey's footnote forbids it, and `hardmaxPair` — where each of the two
merging clusters follows the *other* one — is that limit.

The `β → ∞` limit is uniform on a rescaled time interval `[0, T]`, so it is
written with a `δ`/`B` pair.  Every hypothesis that names a trajectory sits in
the conclusion rather than among the hypotheses: solutions of `eq: SA` are not
constructed here, so a satisfiability witness could not produce one.
-/

import Transformer.Basic
import Transformer.MeanField.Basic
import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators
open Real

namespace Transformer
namespace MeanField

variable (d : ℕ)

/-- **Equation (eq: SA) at a discrete multi-cluster configuration
(eq: init_clust).**

For `μ(t) = Σ_j α_j δ_{x_j(t)}` the mean-field field reads

  `ẋ_i = Proj_{x_i} ( (Σ_k α_k e^{β⟨x_i,x_k⟩})⁻¹ Σ_j α_j e^{β⟨x_i,x_j⟩} x_j )`.

Source: arXiv:2512.01868v4, §5, `eq: init_clust` with `eq: SA`. -/
def clusterSA (K : ℕ) (α : Idx K → ℝ) (β : ℝ) (X : ℝ → SphereTuple d K) : Prop :=
  ∀ t : ℝ, ∀ i : Idx K,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        ((∑ k : Idx K,
            α k * Real.exp (β * inner (𝕜 := ℝ)
              ((X t i : EucSpace d)) ((X t k : EucSpace d))))⁻¹ •
          ∑ j : Idx K,
            (α j * Real.exp (β * inner (𝕜 := ℝ)
              ((X t i : EucSpace d)) ((X t j : EucSpace d))))
              • ((X t j : EucSpace d)))) t

/-- **The limiting pairing dynamics.**

  `ẏ_ī = Proj_{y_ī}(y_j̄)`,  `ẏ_j̄ = Proj_{y_j̄}(y_ī)`,  `ẏ_k = 0` otherwise.

This is the `argmax` rule of the survey's footnote once `(ī, j̄)` is the
unique closest pair: each of the two attends to the other, and every other
cluster attends to one of them at an exponentially slower rate, which the
rescaling of time sends to zero.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge`. -/
def hardmaxPair (K : ℕ) (ibar jbar : Idx K) (Y : ℝ → SphereTuple d K) : Prop :=
  ∀ s : ℝ, ∀ k : Idx K,
    HasDerivAt (fun r => (Y r k : EucSpace d))
      (if k = ibar then proj d ((Y s ibar : EucSpace d)) ((Y s jbar : EucSpace d))
       else if k = jbar then proj d ((Y s jbar : EucSpace d)) ((Y s ibar : EucSpace d))
       else 0) s

/-- **Theorem (thm: agazzi_merge).** *The closest pair merges first.*

Start from a discrete multi-cluster configuration `eq: init_clust` whose
closest pair `(ī, j̄)` is unique, and rescale time by

  `dt = e^{β (1 - ⟨x_ī(0), x_j̄(0)⟩)} ds`.

Then, as `β → ∞`, the rescaled trajectories converge uniformly on any
interval `[0, T]` on which `⟨y_ī(s), y_j̄(s)⟩ ≤ 1 - ε` to the solution of
`hardmaxPair` started at the same configuration.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge` (Bruno, Pasqualotto &
Agazzi). -/
theorem agazzi_merge
    (K : ℕ) (α : Idx K → ℝ) (X₀ : SphereTuple d K)
    (hα : ∀ j : Idx K, 0 ≤ α j) (hsum : ∑ j : Idx K, α j = 1)
    (ibar jbar : Idx K) (hij : ibar ≠ jbar)
    (hmax : ∀ i j : Idx K, i ≠ j → (i, j) ≠ (ibar, jbar) → (i, j) ≠ (jbar, ibar) →
      inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((X₀ j : EucSpace d))
        < inner (𝕜 := ℝ) ((X₀ ibar : EucSpace d)) ((X₀ jbar : EucSpace d)))
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) :
    ∀ Y : ℝ → SphereTuple d K, Y 0 = X₀ → hardmaxPair d K ibar jbar Y →
      (∀ s ∈ Set.Icc (0 : ℝ) T,
        inner (𝕜 := ℝ) ((Y s ibar : EucSpace d)) ((Y s jbar : EucSpace d)) ≤ 1 - ε) →
      ∀ X : ℝ → ℝ → SphereTuple d K,
        (∀ β : ℝ, 0 < β → X β 0 = X₀ ∧ clusterSA d K α β (X β)) →
        ∀ δ : ℝ, 0 < δ → ∃ B : ℝ, ∀ β : ℝ, B < β →
          ∀ s ∈ Set.Icc (0 : ℝ) T, ∀ k : Idx K,
            ‖((X β (Real.exp (β * (1 - inner (𝕜 := ℝ)
                    ((X₀ ibar : EucSpace d)) ((X₀ jbar : EucSpace d)))) * s) k : EucSpace d))
                - ((Y s k : EucSpace d))‖ < δ := by
  sorry

/-- The hypotheses of `agazzi_merge` are satisfiable: two clusters of equal
mass.  With `K = 2` the only pair is `(0, 1)`, so the uniqueness requirement
is vacuous and any initial configuration will do. -/
example (X₀ : SphereTuple d 2) :
    (∀ j : Idx 2, 0 ≤ (fun _ : Idx 2 => (1 / 2 : ℝ)) j) ∧
      ∑ j : Idx 2, (fun _ : Idx 2 => (1 / 2 : ℝ)) j = 1 ∧
      (0 : Idx 2) ≠ 1 ∧
      (∀ i j : Idx 2, i ≠ j → (i, j) ≠ (0, 1) → (i, j) ≠ (1, 0) →
        inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((X₀ j : EucSpace d))
          < inner (𝕜 := ℝ) ((X₀ 0 : EucSpace d)) ((X₀ 1 : EucSpace d))) ∧
      (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 := by
  refine ⟨fun _ => by norm_num, by norm_num, by decide, fun i j hne h1 h2 => ?_,
    one_pos, one_pos⟩
  fin_cases i <;> fin_cases j <;> simp_all

/-- **Theorem (thm: agazzi_merge), the limiting picture.**

"All clusters remain stationary except for the closest pair `(ī, j̄)`, which
move along the unique geodesic connecting them and merge in finite rescaled
time."  The geodesic is unique exactly when the pair is not antipodal, which
is the hypothesis on `⟨y_ī(0), y_j̄(0)⟩`.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge`. -/
theorem hardmaxPair_merges (K : ℕ) (ibar jbar : Idx K) (hij : ibar ≠ jbar) :
    ∀ Y : ℝ → SphereTuple d K, hardmaxPair d K ibar jbar Y →
      inner (𝕜 := ℝ) ((Y 0 ibar : EucSpace d)) ((Y 0 jbar : EucSpace d)) ≠ -1 →
        (∀ k : Idx K, k ≠ ibar → k ≠ jbar → ∀ s : ℝ, Y s k = Y 0 k) ∧
        ∃ S : ℝ, 0 < S ∧ Y S ibar = Y S jbar := by
  sorry

/-- The hypothesis of `hardmaxPair_merges` is satisfiable: `0 ≠ 1` in
`Idx 2`. -/
example : (0 : Idx 2) ≠ 1 := by decide

end MeanField
end Transformer
