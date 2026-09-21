/-
# Metastability — the mean-field theorem (§5 of 2410.06833v1)

`Theorem thm: metastability MF`, in the vocabulary of
`Metastability.MeanField`, with its hypotheses witnessed by the static
solution of `Metastability.MeanFieldStatic`.
-/

import Transformer.Metastability.MeanFieldStatic

open Real MeasureTheory

namespace Transformer
namespace Metastability

open Perspective

variable (d : ℕ)

/-- **Theorem (thm: metastability MF).** *Mean-field metastability.*

Let `β > 1`, `ε ∈ (0, 1/16)`, and let `μ_0 = (1/k) Σ_q ν_q` be a
`(β, ε)`-separated measure with caps `𝒮_q(ε)` around `w_1,…,w_k`.  Let `μ`
solve `eq: mean.field.pde` from `μ_0` and let `Φ` be the characteristic flow
of `v[μ(t)]` (`eq: flow.map`).  Then there are `T_2 > T_1 > 0` with

  `T_1 < (ε/k) e^{β(1-α-8ε)} < T_2`

such that, for every `q`:

1. `supp((Φ^t)_# ν_q) ⊂ 𝒮_q(2ε)` for every `t ∈ [0, T_2]`;

2. for every `t ∈ [T_1, T_2]` and every `0 < λ < γ`,
   `∫_{𝒮_q(2ε)} ‖Φ^t(x') - x(t)‖² dμ_0(x') ≤ e^{-λβ}`, where
   `x(t) ∈ argmin_{x ∈ Φ^t(𝒮_q(ε))} ⟨x, w_q⟩`.

`α` is `αDist` (`eq: alpha.dist MF`) and `γ` is `γβ` with `k` caps
(`eq: gamma.mf`).

**What the source says and what is changed here.**  The `argmin` need not be a
single point; the bound is asserted for every minimising selection `x`
(`IsCapArgmin`).  The side condition `γ(β) = Ω(1)` of
`def: init_measure_MF` is asymptotic in `β` and is not formalized.  The
source's `k ≤ n` refers to a number of particles that the mean-field setting
does not have; `0 < k` is kept.  The source's proof chooses `T_1` after
`λ` (through `T_*(q, λ)`), while its statement fixes `T_1` before `λ`; the
statement is what is written here.

Not proved here.

Source: arXiv:2410.06833v1, §5, `thm: metastability MF`. -/
theorem metastability_mf
    (β ε : ℝ) (k : ℕ) (w : Idx k → SSphere d) (ν : Idx k → ProbSphere d)
    (μ₀ : ProbSphere d) (μ : ℝ → ProbSphere d) (Φ : ℝ → SSphere d → SSphere d)
    (hβ : 1 < β) (hε : 0 < ε) (hε16 : ε < 1 / 16) (hk : 0 < k)
    (hν : ∀ q : Idx k, (ν q : Measure (SSphere d)).support ⊆ sphericalCap d (w q) ε)
    (hμ₀ : (μ₀ : Measure (SSphere d))
      = ((k : ℝ)⁻¹).toNNReal • ∑ q : Idx k, (ν q : Measure (SSphere d)))
    (hsep : 8 * ε < γβ k β (αDist d k w ε) ε)
    (hμ0 : μ 0 = μ₀) (hμ : meanFieldPDE d β μ) (hΦ : IsMFFlow d β μ Φ) :
    ∃ T₁ T₂ : ℝ, 0 < T₁ ∧ T₁ < T₂ ∧
      T₁ < ε / k * Real.exp (β * (1 - αDist d k w ε - 8 * ε)) ∧
      ε / k * Real.exp (β * (1 - αDist d k w ε - 8 * ε)) < T₂ ∧
      (∀ q : Idx k, ∀ t ∈ Set.Icc (0 : ℝ) T₂,
        (Measure.map (Φ t) (ν q : Measure (SSphere d))).support
          ⊆ sphericalCap d (w q) (2 * ε)) ∧
      ∀ q : Idx k, ∀ x : ℝ → SSphere d, IsCapArgmin d Φ (w q) ε x →
        ∀ t ∈ Set.Icc T₁ T₂, ∀ lam : ℝ, 0 < lam → lam < γβ k β (αDist d k w ε) ε →
          ∫ x' in sphericalCap d (w q) (2 * ε),
              ‖(Φ t x' : EucSpace d) - (x t : EucSpace d)‖ ^ 2
            ∂(μ₀ : Measure (SSphere d))
            ≤ Real.exp (-lam * β) := by
  sorry

/-- The hypotheses of `metastability_mf` are satisfiable, structural ones
included: on `𝕊^0`, one cap around `p`, `ν = μ_0 = δ_p` at rest, the identity
flow, `β = 1000` and `ε = 1/32` (`Metastability.MeanFieldStatic`). -/
example : let p := basePoint 0
    (1 : ℝ) < 1000 ∧ (0 : ℝ) < 1 / 32 ∧ (1 / 32 : ℝ) < 1 / 16 ∧ 0 < 1 ∧
      (∀ q : Idx 1, ((fun _ => diracProb 1 p) q : Measure (SSphere 1)).support
        ⊆ sphericalCap 1 ((fun _ => p) q) (1 / 32)) ∧
      ((diracProb 1 p : ProbSphere 1) : Measure (SSphere 1))
        = (((1 : ℕ) : ℝ)⁻¹).toNNReal •
            ∑ q : Idx 1, ((fun _ => diracProb 1 p) q : Measure (SSphere 1)) ∧
      8 * (1 / 32 : ℝ) < γβ 1 1000 (αDist 1 1 (fun _ => p) (1 / 32)) (1 / 32) ∧
      meanFieldPDE 1 1000 (fun _ => diracProb 1 p) ∧
      IsMFFlow 1 1000 (fun _ => diracProb 1 p) (fun _ x => x) := by
  intro p
  refine ⟨by norm_num, by norm_num, by norm_num, one_pos, fun _ y hy => ?_, ?_,
    separated_one_cap p, meanFieldPDE_const_diracProb 1000 p, isMFFlow_id_diracProb 1000 p⟩
  · rw [Interpolation.eq_of_mem_support_dirac hy]
    simp only [sphericalCap, Set.mem_ofPred_eq, real_inner_self_eq_norm_mul_norm,
      mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  · simp

end Metastability
end Transformer
