/-
# The emergence of clusters in self-attention dynamics — the higher-dimensional
  remark

`r:higherdimclus` of arXiv:2305.05465v6, at the end of §7: the remark that
`t:boolean` does not extend to `d ≥ 2` because of "rare pathological
situations", supported by one example.

**What the source says.**  For `d = 2`, `n = 2` and the initial configuration
`x_1(0) = (1,ε)`, `x_2(0) = (1,-ε)`: "one can check that `x_i(t) → (1,0)` as
`t → +∞`, for `i = 1,2`, which means that a single cluster appears.  However,
the self-attention matrix converges toward the identity (which has rank 2)."

**The example is inconsistent, and the inconsistency is elementary.**  The
self-attention matrix `eq:P` is a continuous function of the configuration
alone.  If every token converges to one and the same point `p`, then every
score `⟨Qx_i, Kx_l⟩` converges to the same number `⟨Qp, Kp⟩`, so every entry
of `P(t)` converges to `1/n` — the rank-one matrix of uniform attention, not
the identity.  That is `tendsto_attentionMatrix_of_tendsto_common` below, and
`not_tendsto_id_of_tendsto_common` is the refutation it yields: no
configuration at all, whether or not it solves `eq:trans_dyn`, can have its
tokens converge to a common limit while its attention matrix converges to the
identity.  So the two assertions the source makes about its example cannot
both hold.

What this does *not* refute is the remark's conclusion — that the rank of the
limiting self-attention matrix need not equal the number of clusters.  It
refutes the example offered for it: whatever the solution from
`x_1(0) = (1,ε)`, `x_2(0) = (1,-ε)` does, it does not both cluster to a single
point and have attention converging to `I_2`.

Source: arXiv:2305.05465v6, `r:higherdimclus`.
-/

import Transformer.Clusters.Section1_Dynamics
import Mathlib.Analysis.InnerProductSpace.Continuous

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **A single cluster forces uniform attention.**  If all `n` tokens converge
to the same point, every entry of the self-attention matrix converges to
`1/n`, so `P(t)` converges to the rank-one matrix `(1/n)J`.

This is a statement about `eq:P` alone; no dynamics is involved.

Source: arXiv:2305.05465v6, `eq:P`, `r:higherdimclus`. -/
theorem tendsto_attentionMatrix_of_tendsto_common (Q K : ParamMatrix d)
    (X : ℝ → Idx n → EucSpace d) (p : EucSpace d)
    (h : ∀ i : Idx n, Tendsto (fun t => X t i) atTop (nhds p)) (i j : Idx n) :
    Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop (nhds ((n : ℝ)⁻¹)) := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hscore : ∀ l : Idx n,
      Tendsto (fun t => Real.exp (inner (𝕜 := ℝ) (Q (X t i)) (K (X t l)))) atTop
        (nhds (Real.exp (inner (𝕜 := ℝ) (Q p) (K p)))) := by
    intro l
    exact (Real.continuous_exp.tendsto _).comp
      (((Q.continuous.tendsto p).comp (h i)).inner ((K.continuous.tendsto p).comp (h l)))
  have hsum : Tendsto (fun t => ∑ l : Idx n, Real.exp (inner (𝕜 := ℝ) (Q (X t i)) (K (X t l))))
      atTop (nhds ((n : ℝ) * Real.exp (inner (𝕜 := ℝ) (Q p) (K p)))) := by
    have := tendsto_finsetSum (Finset.univ : Finset (Idx n)) (fun l _ => hscore l)
    simpa [Finset.sum_const, nsmul_eq_mul] using this
  have hlim : Real.exp (inner (𝕜 := ℝ) (Q p) (K p)) /
      ((n : ℝ) * Real.exp (inner (𝕜 := ℝ) (Q p) (K p))) = (n : ℝ)⁻¹ := by
    field_simp
  rw [← hlim]
  exact (hscore j).div hsum (by positivity)

/-- The hypothesis of `tendsto_attentionMatrix_of_tendsto_common` is
satisfiable: a configuration that does not move at all converges to a common
limit. -/
example (p : EucSpace d) :
    ∀ i : Idx n, Tendsto (fun _ : ℝ => (fun _ : Idx n => p) i) atTop (nhds p) :=
  fun _ => tendsto_const_nhds

/-- **Refutation of the example of `r:higherdimclus`.**  No configuration
whose tokens converge to a common limit has an off-diagonal attention entry
converging to `0`.

Source: arXiv:2305.05465v6, `r:higherdimclus`. -/
theorem not_tendsto_zero_of_tendsto_common (Q K : ParamMatrix d)
    (X : ℝ → Idx n → EucSpace d) (p : EucSpace d)
    (h : ∀ i : Idx n, Tendsto (fun t => X t i) atTop (nhds p)) (i j : Idx n) :
    ¬ Tendsto (fun t => attentionMatrix Q K (X t) i j) atTop (nhds 0) := by
  intro hzero
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have hn' : (0 : ℝ) < (n : ℝ) := Nat.cast_pos.mpr hn
  have := tendsto_nhds_unique
    (tendsto_attentionMatrix_of_tendsto_common Q K X p h i j) hzero
  exact absurd this (inv_pos.mpr hn').ne'

/-- **The two assertions of `r:higherdimclus` are incompatible.**  A
configuration whose tokens converge to one point cannot have its
self-attention matrix converge to the identity, as soon as there are two
distinct tokens.

Source: arXiv:2305.05465v6, `r:higherdimclus`. -/
theorem not_tendsto_id_of_tendsto_common (Q K : ParamMatrix d)
    (X : ℝ → Idx n → EucSpace d) (p : EucSpace d)
    (h : ∀ i : Idx n, Tendsto (fun t => X t i) atTop (nhds p))
    (i j : Idx n) (hij : i ≠ j) :
    ¬ ∀ a b : Idx n, Tendsto (fun t => attentionMatrix Q K (X t) a b) atTop
        (nhds (if a = b then (1 : ℝ) else 0)) := by
  intro hid
  have := hid i j
  rw [ite_eq_right hij] at this
  exact not_tendsto_zero_of_tendsto_common Q K X p h i j this

/-- The hypotheses of `not_tendsto_id_of_tendsto_common` are satisfiable: two
tokens pinned at the same point of `ℝ²`, which is the configuration the source
claims its example converges to. -/
example (p : EucSpace 2) :
    (∀ i : Idx 2, Tendsto (fun _ : ℝ => (fun _ : Idx 2 => p) i) atTop (nhds p)) ∧
      (0 : Idx 2) ≠ 1 :=
  ⟨fun _ => tendsto_const_nhds, by decide⟩

end Clusters
end Transformer
