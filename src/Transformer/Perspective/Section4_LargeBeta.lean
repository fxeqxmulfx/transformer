module

public import Transformer.Basic
public import Transformer.Perspective.Section1_IPS
public import Transformer.Perspective.Section3_SmallBeta
public import Transformer.Perspective.Section5_HighD
public import Transformer.Perspective.CircleClustering

/-!
# §5 — A single cluster for large β

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §5 of the survey, which contains:

* `Theorem thm: beta.interval` — clustering when `β ≥ C(d) n²`, proved from the
  clustering at every `β ≥ 0`: `thm: boumal` for `d ≥ 3`, `clustering_circle` for
  `d = 2`.
-/

@[expose] public section

open scoped BigOperators
open MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (d : ℕ)

/-- **Theorem (thm: beta.interval).** *Cluster collapse at high β.*

Fix `d, n ≥ 2`. There exists a constant `C = C(d) > 0` (depending only on
`d`) such that whenever `β ≥ C(d) n²`, the conclusion of `thm: beta.tiny` holds
for both `SA` and `USA`:

For Lebesgue-almost any initial sequence `X₀ ∈ (𝕊^{d-1})^n`, there exists
`x⋆ ∈ 𝕊^{d-1}` such that `lim_{t→∞} x_i(t) = x⋆` for all `i ∈ [n]`, for the
solution of `SA` and for that of `USA` (`clusteringSetUSA`).

*Almost any* is not *any*: `antipodalPair_not_mem_clusteringSet` exhibits, for
every `β`, an initial sequence outside `𝒮_β` — an antipodal pair is stationary
for `SA`, so its two particles never meet. The exceptional set is therefore
non-empty at every `β`, however large, and the quantifier has to be the
almost-everywhere one, read against the uniform law as in §4.

**What the source says and what is changed here.**  `C` depends only on `d`, so `n` is
quantified after `C`; the statement had `n` fixed before `C`, which let `C` depend on
`n` as well.

**Proof.**  `C = 1` will do, since the conclusion holds for every `β ≥ 0`: for `d ≥ 3`
by `thm: boumal` (`boumal_clustering`), for `d = 2` by `clustering_circle`, through the
landscape of arXiv:2507.22857 on the circle. The survey notes at the start of §6 that
"the restrictions on `β` provided by Theorems `thm: beta.tiny` and `thm: beta.interval`
are specific to the two-dimensional case"; its own proof, Appendix B, shows that for
`β ≥ C n²` every critical point of `𝖤_β` other than a global maximum is a strict saddle.

Source: arXiv:2312.10794v5, §5, `thm: beta.interval`, and Appendix B. -/
theorem beta_interval (hd : 2 ≤ d) :
    ∃ C : ℝ, 0 < C ∧ ∀ n : ℕ, 2 ≤ n → ∀ β : ℝ, C * (n : ℝ)^2 ≤ β →
      ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
        ∀ᵐ X₀ ∂P, X₀ ∈ clusteringSet d n β ∧ X₀ ∈ clusteringSetUSA d n β := by
  refine ⟨1, one_pos, fun n hn β hβ P hP => ?_⟩
  have hβ0 : 0 ≤ β := le_trans (by positivity) hβ
  rcases (show d = 2 ∨ 3 ≤ d by omega) with rfl | hd3
  · exact clustering_circle n hβ0 P hP
  · exact boumal_clustering d n hd3 hn β hβ0 P hP

/-- The hypothesis of `beta_interval` holds for the circle, `d = 2`. -/
example := beta_interval 2 le_rfl

end Perspective
end Transformer
