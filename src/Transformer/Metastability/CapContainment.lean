/-
# Metastability — Step 1 of the direct proof: the particles stay in their caps

Item 1 of `thm: metastability` in arXiv:2410.06833v1: a particle that starts in `𝒮_q(ε)`
is in `𝒮_q(2ε)` for all `t ∈ [0, T₂]`, `T₂ = (ε/n) e^{(1-α)β}`.

Let `q : Idx n → Idx k` assign to each token the cap of its initial position, and put
`E = e^{-(1-α)β}`.  For the finite family

  `f_j(t) = ⟨x_j(t), w_{q(j)}⟩ - (1 - ε) + n E t`

the barrier argument (`finite_nonneg_of_boundary_deriv_on`) applies.  Suppose all `f_i(t) ≥ 0`
at some `t ≤ T₂`.  Then `⟨x_i(t), w_{q(i)}⟩ ≥ 1 - ε - n E t ≥ 1 - 2ε` for every `i`, so every
token is in its cap `𝒮_{q(i)}(2ε)` and, by the definition of `α`, tokens of different caps
satisfy `⟨x_i, x_k⟩ ≤ α`: their attention weights are at most `E`.  If moreover `f_j(t) = 0`,
then `x_j` attains the minimum of `⟨x_k, w_{q(j)}⟩` over its own cap, so
`cap_velocity_ge` gives `d/dt ⟨x_j, w⟩ ≥ -(n-1) E` and `f_j' ≥ E > 0`: no `f_j` can cross zero.

The paper's leakage rate is `n E`; the count `(n-1)` — the token `j` is in its own cap — leaves
the margin `E` that makes the barrier strict, so the endpoint `T₂` needs no limiting argument.

Source: arXiv:2410.06833v1, §2, proof of `thm: metastability`, Step 1.
-/

import Transformer.Metastability.AttnComparison
import Transformer.Metastability.CapGeometry

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- **Cap containment** (item 1 of `thm: metastability`).  Let `X` be an attention flow
(`SA` or `USA`) and `q i` a cap containing `x_i(0)` in `𝒮_{q(i)}(ε)`.  With
`α = α(ε)`, `T₂ = ε / (n e^{-(1-α)β})`, every `x_i(t)` with `t ∈ [0, T₂]` lies in
`𝒮_{q(i)}(2ε)`.

No hypothesis on `γ` or on `α < 1` is needed here, only `0 < ε ≤ 1/2`, `β ≥ 0`, `n ≥ 1`.

Source: arXiv:2410.06833v1, §2, `thm: metastability` (1), Step 1 of its proof. -/
theorem cap_containment {β ε : ℝ} (hβ : 0 ≤ β) (hε : 0 < ε) (hε' : ε ≤ 1 / 2) (hn : 1 ≤ n)
    {X : ℝ → SphereTuple d n} (hX : IsAttnFlow d n β X) {k : ℕ} (w : Idx k → SSphere d)
    (q : Idx n → Idx k) (h0 : ∀ i, X 0 i ∈ sphericalCap d (w (q i)) ε) :
    ∀ t : ℝ, 0 ≤ t →
      t ≤ ε / ((n : ℝ) * Real.exp (-((1 - αDist d k w ε) * β))) →
        ∀ i, X t i ∈ sphericalCap d (w (q i)) (2 * ε) := by
  classical
  set α := αDist d k w ε with hα
  set E := Real.exp (-((1 - α) * β)) with hE
  have hEpos : 0 < E := Real.exp_pos _
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have hnE : 0 < (n : ℝ) * E := mul_pos hnpos hEpos
  set T₂ := ε / ((n : ℝ) * E) with hT₂
  have hT₂0 : 0 ≤ T₂ := (div_pos hε hnE).le
  have hnET : (n : ℝ) * E * T₂ = ε := by rw [hT₂]; field_simp
  obtain ⟨a, ha, hdiff⟩ := hX
  let f : Idx n → ℝ → ℝ := fun j t =>
    ⟪(X t j : EucSpace d), (w (q j) : EucSpace d)⟫_ℝ - (1 - ε) + (n : ℝ) * E * t
  have hderivW : ∀ (j : Idx n) (t : ℝ), HasDerivAt
      (fun s => ⟪(X s j : EucSpace d), (w (q j) : EucSpace d)⟫_ℝ)
      ⟪proj d (X t j : EucSpace d) (∑ l, a t j l • (X t l : EucSpace d)),
        (w (q j) : EucSpace d)⟫_ℝ t := by
    intro j t
    simpa using (hdiff t j).inner ℝ (hasDerivAt_const t (w (q j) : EucSpace d))
  have hderiv : ∀ (j : Idx n) (t : ℝ), HasDerivAt (f j)
      (⟪proj d (X t j : EucSpace d) (∑ l, a t j l • (X t l : EucSpace d)),
        (w (q j) : EucSpace d)⟫_ℝ + (n : ℝ) * E) t := by
    intro j t
    have h := ((hderivW j t).sub_const (1 - ε)).fun_add
      (((hasDerivAt_id t).const_mul ((n : ℝ) * E)))
    simpa [f] using h
  have hcont : ∀ j, ContinuousOn (f j) (Icc 0 T₂) := fun j =>
    (continuous_iff_continuousAt.2 fun t => (hderiv j t).continuousAt).continuousOn
  have hinit : (∀ j, 0 ≤ f j 0) ∧ (0 : ℝ) ≤ T₂ := by
    refine ⟨fun j => ?_, hT₂0⟩
    have h := h0 j
    simp only [f, sphericalCap, Set.mem_ofPred_eq] at h ⊢
    linarith
  have hboundary : ∀ t ∈ Ico 0 T₂, (∀ j, 0 ≤ f j t) → ∀ j, f j t = 0 → 0 < deriv (f j) t := by
    intro t ht hall j hj
    have htT : (n : ℝ) * E * t ≤ ε := by
      calc (n : ℝ) * E * t ≤ (n : ℝ) * E * T₂ :=
            mul_le_mul_of_nonneg_left ht.2.le hnE.le
        _ = ε := hnET
    have hcap : ∀ i, X t i ∈ sphericalCap d (w (q i)) (2 * ε) := fun i => by
      have h := hall i
      simp only [f] at h
      show 1 - 2 * ε ≤ _
      linarith
    have hnorm : ∀ l : Idx n, ‖(X t l : EucSpace d)‖ = 1 := fun l =>
      mem_sphere_zero_iff_norm.mp (X t l).2
    have hwt := ha t
    have hnn : ∀ l : Idx n, 0 ≤ a t j l := fun l =>
      le_trans (by have := Real.exp_pos
                        (β * (⟪(X t j : EucSpace d), (X t l : EucSpace d)⟫_ℝ - 1))
                   positivity) (hwt.lower j l)
    have hfarI : ∀ l : Idx n, l ∉ Finset.univ.filter (fun i => q i = q j) → a t j l ≤ E := by
      intro l hl
      have hql : q l ≠ q j := by simpa using hl
      exact attn_far_le hβ hwt (inner_le_alphaDist hql.symm (hcap j) (hcap l))
    have hjeq : ⟪(X t j : EucSpace d), (w (q j) : EucSpace d)⟫_ℝ = 1 - ε - (n : ℝ) * E * t := by
      simp only [f] at hj; linarith
    have hmin : ∀ l ∈ Finset.univ.filter (fun i => q i = q j),
        ⟪(X t j : EucSpace d), (w (q j) : EucSpace d)⟫_ℝ ≤
          ⟪(X t l : EucSpace d), (w (q j) : EucSpace d)⟫_ℝ := by
      intro l hl
      have hql : q l = q j := by simpa using hl
      have h := hall l
      simp only [f, hql] at h
      linarith
    have hvel := cap_velocity_ge (fun l => (X t l : EucSpace d)) (a t j)
      (Finset.univ.filter (fun i => q i = q j)) j (by simp)
      (w (q j) : EucSpace d) (mem_sphere_zero_iff_norm.mp (w (q j)).2) hnorm E hEpos.le hnn
      hfarI (by rw [hjeq]; linarith) hmin
    rw [(hderiv j t).deriv]
    have hn1 : ((n : ℝ) - 1) * E ≤ (n : ℝ) * E := by nlinarith
    nlinarith
  have hall := finite_nonneg_of_boundary_deriv_on f 0 T₂ hcont hinit hboundary
  intro t ht0 htT i
  have h := hall t ⟨ht0, htT⟩ i
  simp only [f] at h
  have htE : (n : ℝ) * E * t ≤ ε := by
    calc (n : ℝ) * E * t ≤ (n : ℝ) * E * T₂ := mul_le_mul_of_nonneg_left htT hnE.le
      _ = ε := hnET
  show 1 - 2 * ε ≤ _
  linarith

end Metastability
end Transformer
