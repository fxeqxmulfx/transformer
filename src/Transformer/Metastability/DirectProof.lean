/-
# Metastability — `thm: metastability` (§2 of arXiv:2410.06833v1), with its direct proof

The proof follows §2 ("A direct proof") of the paper, and is written once for `(SA)` and `(USA)`
through `IsAttnFlow`.  Its ingredients, in the order of the paper:

* Step 1, the particles stay in the caps `𝒮_q(2ε)` up to `T₂ = (ε/n) e^{(1-α)β}`:
  `cap_containment`, with the barrier principle of `PropagationBarrier` in place of a
  continuity argument;
* Steps 2–3, the collapse of a cap to `⟨x_i, x_j⟩ ≥ 1 - e^{-λβ}` by `T₁` and its propagation
  up to `T₂`: `collapse_dynamic`, where the scalar ODE of `lem: eminem` is replaced by the three
  explicit subsolutions of `CollapseComparison`, and `lem: collapsetime` by `phase_const`;
* the parameters: `γ > 0` and the two bounds of `eq: lambda.3` (`CollapseParams`);
* the assembly, `metastability_assemble`.
-/

import Transformer.Metastability.CapContainment
import Transformer.Metastability.CollapseParams
import Transformer.Metastability.CollapseTimeBounds
import Transformer.Metastability.DirectAssemble

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- **Theorem (thm: metastability).** *Dynamic metastability.*

Let `β > 1`, `ε ∈ (0, 1/16)`, and let `w_1, …, w_k` be centres such that every `x_i(0)` lies in
some cap `𝒮_q(ε)` and `γ(β) > 0` (that is, the configuration is `(β, ε)`-separated).  Let
`0 < λ` satisfy the two bounds of `eq: lambda.3`.  Then there are `T₂ > T₁ > 0`, with
`T₁ ≤ 2 n e^{8εβ} + e n λ β²/(β-1)` and `T₂ ≥ (ε/n) e^{(1-α)β}` (`MetastabilityTimes`), such that
for every solution of `(SA)` or of `(USA)` with this initial condition

1. if `x_i(0) ∈ 𝒮_q(ε)`, then `x_i(t) ∈ 𝒮_q(2ε)` for all `t ∈ [0, T₂]`;
2. for all `q` and `t ∈ [T₁, T₂]`,
   `max_{x_i(t), x_j(t) ∈ 𝒮_q(2ε)} ‖x_i(t) - x_j(t)‖² ≤ 2 e^{-λβ}`.

**What the source says and what is changed here.**

*`isSeparated` and the free `k`, `w`.*  The statement of the source takes a `(β, ε)`-separated
configuration — one for which *some* `k ≤ n` centres exist — and then speaks of `𝒮_q(2ε)` for
the centres of the separation.  The earlier version of this theorem had `isSeparated` and,
beside it, a `k` and a `w` unrelated to the witnesses of `isSeparated`.  The witnesses are now
the hypotheses: `k`, `w`, the cover `hcover` and `γ(β) > 0` (`hγ`).

*The first bound of `eq: lambda.3` has a sign error.*  The source prints
`e^{(1-α-β⁻¹ log((β-1)ε/(β²n²e)))β}(1 - e^{-γβ})`.  It is used to make
`2 n e^{8εβ} + e n λ β²/(β-1)` smaller than `T₂ = (ε/n) e^{(1-α)β}`.  Since
`(ε/n) e^{(1-α)β} e^{-γβ} = 2 n e^{8εβ}` by the definition of `γ`, this needs
`λ < e^{(1-α)β} · c · (1 - e^{-γβ})`, `c = (β-1)ε/(β²n²e) < 1`, i.e. a `+` in front of
`β⁻¹ log`.  With the printed `-` the bound is `c⁻²` times the corrected one, and the inequality
`T₁ < T₂` that the proof draws from it does not follow.  `hlam₁` carries the `+`.  (The proof's
`T_*(q)` also prints `e^{β(1-ρ_q)}` where `e^{β(ρ_q-1)}` is meant; the formal proof does not go
through `T_*`.)

*`Ω(1)`.*  `γ = Ω(1)` and `λ = Ω(1)` are statements about families indexed by `β`; at fixed
`β` they have no content.  They are not encoded; the conclusion does not use them, they serve
in the source only to make `T₂` exponentially long and `e^{-λβ}` exponentially small.

*One cap.*  For `k = 1` the maximum defining `α` is over the empty set, and `αDist` is `0`
(`alphaDist_one`); the proof uses only `α < 1`, which follows from `γ > 0`.

*Dimension and number of tokens.*  The source has `d, n ≥ 2` and `k ≤ n`.  Neither `d ≥ 2` nor
`k ≤ n` is used, and `n ≥ 2` is weakened to `n ≥ 1` (`n ≥ 1` is needed: `T₂` divides by `n`).

*The case `λ ≤ λ_*`.*  The source assumes `λ > λ_* = β⁻¹ log(1/(8ε))` "without loss of
generality, since the bound holds for all smaller `λ`".  That reduction does not preserve the
bound on `T₁`, which grows with `λ`.  Here the case `λ ≤ λ_*` is handled directly: then
`e^{-λβ} ≥ 8ε`, and two tokens of one cap `𝒮_q(2ε)` satisfy `‖x - y‖² ≤ 16ε ≤ 2 e^{-λβ}` by the
geometry of the cap (`inner_ge_of_mem_cap`) at every time.

*`(USA)`.*  The theorem is proved for `(SA)` (`Perspective.SA`) and for `(USA)` in the form
printed in the source, with `e^{β(⟨x_i,x_j⟩-1)}/n` weights (`unnormalizedSA`).  `T₂` is
exactly `(ε/n) e^{(1-α)β}`: the source's `T_esc ≥ ε/(n e^{-(1-α)β})` is an equality here.

*Solutions.*  A solution is a map `ℝ → (𝕊^{d-1})^n`; solutions of the Cauchy problem on
`[0, ∞)` extend to `ℝ` since the sphere is compact.  Only `t ≥ 0` enters the conclusion.

Source: arXiv:2410.06833v1, §2, `thm: metastability`, `eq: lambda.3`, and its proof
("A direct proof"). -/
theorem metastability {β ε : ℝ} (hβ : 1 < β) (hε : 0 < ε) (hε' : ε < 1 / 16) (hn : 1 ≤ n)
    (X₀ : SphereTuple d n) {k : ℕ} (w : Idx k → SSphere d)
    (hcover : ∀ i : Idx n, ∃ q : Idx k, X₀ i ∈ sphericalCap d (w q) ε)
    (hγ : 0 < γβ n β (αDist d k w ε) ε) {lam : ℝ} (hlam : 0 < lam)
    (hlam₁ : lam < Real.exp ((1 - αDist d k w ε + β⁻¹ * Real.log ((β - 1) * ε /
        (β ^ 2 * (n : ℝ) ^ 2 * Real.exp 1))) * β) *
      (1 - Real.exp (-(γβ n β (αDist d k w ε) ε * β))))
    (hlam₂ : lam < 1 - αDist d k w ε - Real.log (2 * (n : ℝ) ^ 2 /
        (1 - Real.exp (-(lamStar β ε * β)))) / β - Real.exp (-(lamStar β ε * β))) :
    ∃ T : MetastabilityTimes β ε (αDist d k w ε) n lam,
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ →
        (Perspective.SA d n β X ∨ unnormalizedSA d n β X) →
        -- (1) staying in the safety caps
        (∀ (i : Idx n) (q : Idx k), X₀ i ∈ sphericalCap d (w q) ε →
          ∀ t : ℝ, 0 ≤ t → t ≤ T.T2 → X t i ∈ sphericalCap d (w q) (2 * ε)) ∧
        -- (2) cap collapse: the pairwise distance within a cap is exponentially small
        ∀ (q : Idx k) (t : ℝ), T.T1 ≤ t → t ≤ T.T2 → ∀ i j : Idx n,
          X t i ∈ sphericalCap d (w q) (2 * ε) → X t j ∈ sphericalCap d (w q) (2 * ε) →
            ‖(X t i : EucSpace d) - (X t j : EucSpace d)‖ ^ 2 ≤ 2 * Real.exp (-(lam * β)) := by
  classical
  set α := αDist d k w ε with hα
  have hβ0 : 0 < β := by linarith
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hα1 : α < 1 := by linarith [eight_eps_lt_one_sub_alphaDist hβ0 hε hε' hn hγ]
  choose q hq using hcover
  have hFr : 2 * (n : ℝ) ^ 2 * Real.exp (-((1 - α) * β)) < collapseRate β (8 * ε) :=
    collapseRate_gt_of_gamma hβ0 hε hε' hγ hn
  have hFr0 : 0 < collapseRate β (8 * ε) := lt_of_le_of_lt (by positivity) hFr
  have hBT := collapse_bound_lt hβ hε hn hlam₁
  have hB0 : 0 < 2 * (n : ℝ) * Real.exp (8 * ε * β) + Real.exp 1 * n * lam * β ^ 2 / (β - 1) :=
    add_pos (by positivity) (div_pos (by positivity) (by linarith))
  set T₂ := ε / n * Real.exp ((1 - α) * β) with hT₂
  have hT₂c : ε / ((n : ℝ) * Real.exp (-((1 - α) * β))) = T₂ := by
    rw [hT₂, Real.exp_neg]
    field_simp
  -- item (1): the containment of Step 1
  have hcontX : ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → IsAttnFlow d n β X →
      ∀ (i : Idx n) (t : ℝ), 0 ≤ t → t ≤ T₂ → X t i ∈ sphericalCap d (w (q i)) (2 * ε) := by
    intro X hX0 hflow i t ht0 htT
    exact cap_containment hβ0.le hε (by linarith) hn hflow w q
      (fun i => by rw [hX0]; exact hq i) t ht0 (by rw [hT₂c]; exact htT) i
  have hnorm : ∀ x : SSphere d, ‖(x : EucSpace d)‖ = 1 := fun x => mem_sphere_zero_iff_norm.mp x.2
  by_cases hcase : Real.exp (-(lam * β)) < 8 * ε
  · -- `λ > λ_*`: the collapse of Steps 2–3
    have hlam₂' : lam < 1 - α - Real.log (2 * (n : ℝ) ^ 2 / (1 - 8 * ε)) / β - 8 * ε := by
      rw [exp_neg_lamStar_mul hβ0 hε] at hlam₂
      exact hlam₂
    have hFδ := collapseRate_gt_of_lam hβ0 hε hε' hn hcase hlam₂'
    have hT₁pos := collapseTime_pos hβ hn (Real.exp_pos _) hcase hFr0
    have hT₁ub : collapseTime n β (8 * ε) (Real.exp (-(lam * β)))
        ≤ 2 * n * Real.exp (8 * ε * β) + Real.exp 1 * n * lam * β ^ 2 / (β - 1) := by
      have h := collapseTime_le hβ hn hcase (by linarith : 8 * ε ≤ 1 / 2) hFr0
      rwa [show β * (8 * ε) = 8 * ε * β by ring] at h
    refine ⟨⟨collapseTime n β (8 * ε) (Real.exp (-(lam * β))), T₂, hT₁pos,
      hT₁ub.trans_lt hBT, hT₁ub, le_rfl⟩, ?_⟩
    intro X hX0 hXflow
    have hflow : IsAttnFlow d n β X :=
      hXflow.elim (isAttnFlow_of_SA hβ0.le) isAttnFlow_of_unnormalizedSA
    have hcont := hcontX X hX0 hflow
    refine metastability_assemble w hα1 hε q X₀ hq hcont ?_ hT₁pos.le
    intro t ht1 htT i j hij
    have hclass : ∀ a : Idx n, a ∈ Finset.univ.filter (fun l => q l = q i) ↔ q a = q i := by
      intro a; simp
    refine collapse_dynamic hβ hn (Real.exp_pos _) hcase (by linarith) hflow
      (Finset.univ.filter (fun l => q l = q i)) T₂ hFr hFδ (hT₁ub.trans hBT.le)
      ?_ ?_ t ⟨ht1, htT⟩ i ((hclass i).2 rfl) j ((hclass j).2 hij.symm)
    · intro s hs a ha b hb
      have hqa : q a = q i := (hclass a).1 ha
      have hqb : q b ≠ q i := fun h => hb ((hclass b).2 h)
      exact inner_le_alphaDist (by rw [hqa]; exact hqb.symm) (hcont a s hs.1 hs.2)
        (hcont b s hs.1 hs.2)
    · intro a ha b hb
      have hqa : q a = q i := (hclass a).1 ha
      have hqb : q b = q i := (hclass b).1 hb
      have h1 : 1 - ε ≤ ⟪(X₀ a : EucSpace d), (w (q i) : EucSpace d)⟫_ℝ := by
        have := hq a; rw [hqa] at this; exact this
      have h2 : 1 - ε ≤ ⟪(X₀ b : EucSpace d), (w (q i) : EucSpace d)⟫_ℝ := by
        have := hq b; rw [hqb] at this; exact this
      have h := inner_ge_of_mem_cap (hnorm (X₀ a)) (hnorm (X₀ b)) (hnorm (w (q i))) h1 h2
      rw [hX0]
      linarith
  · -- `λ ≤ λ_*`: `e^{-λβ} ≥ 8ε`, and two points of one cap `𝒮_q(2ε)` are `16ε`-close
    have hcase' : 8 * ε ≤ Real.exp (-(lam * β)) := not_lt.1 hcase
    refine ⟨⟨(2 * (n : ℝ) * Real.exp (8 * ε * β) + Real.exp 1 * n * lam * β ^ 2 / (β - 1)) / 2,
      T₂, half_pos hB0, by linarith, by linarith, le_rfl⟩, ?_⟩
    intro X hX0 hXflow
    have hflow : IsAttnFlow d n β X :=
      hXflow.elim (isAttnFlow_of_SA hβ0.le) isAttnFlow_of_unnormalizedSA
    have hcont := hcontX X hX0 hflow
    refine metastability_assemble w hα1 hε q X₀ hq hcont ?_ (half_pos hB0).le
    intro t ht1 htT i j hij
    have hi : 1 - 2 * ε ≤ ⟪(X t i : EucSpace d), (w (q i) : EucSpace d)⟫_ℝ :=
      hcont i t (by linarith [half_pos hB0]) htT
    have hj : 1 - 2 * ε ≤ ⟪(X t j : EucSpace d), (w (q i) : EucSpace d)⟫_ℝ := by
      have := hcont j t (by linarith [half_pos hB0]) htT
      rw [← hij] at this
      exact this
    have h := inner_ge_of_mem_cap (hnorm (X t i)) (hnorm (X t j)) (hnorm (w (q i))) hi hj
    linarith

end Metastability
end Transformer
