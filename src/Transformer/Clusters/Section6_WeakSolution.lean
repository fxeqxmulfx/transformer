/-
# The emergence of clusters in self-attention dynamics — solutions of the
  continuity equation

`e:conteq` and `d:solconti` of arXiv:2305.05465v6, §6: the mean-field form of
the dynamics and its notion of solution, with the first facts about
solutions that the proof of `c:wellposedtransformers` uses.

**What the source says and what is carried here.**

* `d:solconti` asks for `μ ∈ C⁰_comp(ℝ, 𝒫_c(ℝ^d))`, absolute continuity of
  `t ↦ ∫ g dμ_t` for every `g ∈ C_c^∞`, and the weak identity for almost
  every `t`.  It is carried as: each `μ_t` a probability measure; supports
  equi-contained in a ball on every compact time interval (the definition of
  `C⁰_comp`); `t ↦ ∫ g dμ_t` continuous; and the weak identity for **every**
  `t`.  Demanding the identity everywhere rather than almost everywhere is
  not a strengthening here: both sides are continuous in `t`, so an identity
  valid almost everywhere is valid everywhere, and absolute continuity then
  comes from the right-hand side being an integral in `t`.

* `C_c^∞(ℝ^d)` is `ContDiff ℝ (⊤ : ℕ∞)` with compact support.  The bare
  `ContDiff ℝ ⊤` is `C^ω`, analytic, and on `ℝ^d`, `d ≥ 1`, the only analytic
  function with compact support is `0`: with it the weak identity would test
  nothing.

* The source's weak identity writes `𝒳[μ_t]` inside the time integral, where
  `s` is the integration variable; that is a typographical slip — the
  vector field must be evaluated at the running time. It is carried as
  `𝒳[μ_s]`, as the proof and every derivation of `e:conteq` require.

* `∫_0^t` is the interval integral, which also gives the statement its meaning
  for `t < 0`; the source's solutions are defined on all of `ℝ`.

Reversing time turns a solution for `V` into one for `-V`
(`IsContEqSolution.comp_neg`), which reduces the source's statements on `ℝ`
to forward times.  Forward in time, a solution is the image of its initial
measure under the flow of its own field (`IsContEqSolution.eq_map_flow`): the
Lagrangian description that the proof of `c:wellposedtransformers` compares.

Source: arXiv:2305.05465v6, `e:conteq`, `d:solconti`.
-/

import Transformer.Clusters.Section6_KernelField
import Transformer.ContinuityEq.Lagrangian
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

open Set Metric Filter Topology MeasureTheory

namespace Transformer
namespace Clusters

open GlobalFlow

variable {d : ℕ}

/-- **Definition (d:solconti), a solution of `e:conteq`.**  A weakly
continuous curve of compactly supported probability measures, equi-supported
on compact time intervals, starting at `μ₀` and satisfying the weak
formulation of `∂_t μ + div(𝒳[μ]μ) = 0`.

Source: arXiv:2305.05465v6, `e:conteq`, `d:solconti`. -/
def IsContEqSolution (Q K V : ParamMatrix d) (μ₀ : Measure (EucSpace d))
    (μ : ℝ → Measure (EucSpace d)) : Prop :=
  (∀ t : ℝ, IsProbabilityMeasure (μ t)) ∧
    (∀ a b : ℝ, ∃ R : ℝ, 0 < R ∧ ∀ t ∈ Set.Icc a b, IsCarriedBy (μ t) R) ∧
      (∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
        Continuous fun t => ∫ x, g x ∂(μ t)) ∧
      μ 0 = μ₀ ∧
      ∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g → ∀ t : ℝ,
        ∫ x, g x ∂(μ t) = (∫ x, g x ∂μ₀) +
          ∫ s in (0 : ℝ)..t,
            ∫ x, inner (𝕜 := ℝ) (gradient g x) (attentionKernel Q K V (μ s) x) ∂(μ s)

/-- **The Dirac mass at the origin is a stationary solution.**  The attention
kernel vanishes there — the only value averaged is `V 0 = 0` — so nothing
moves.  This witnesses `d:solconti`.

Source: arXiv:2305.05465v6, `e:vectorfield`, `d:solconti`. -/
theorem isContEqSolution_dirac (Q K V : ParamMatrix d) :
    IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
      (fun _ => Measure.dirac (0 : EucSpace d)) := by
  refine ⟨fun _ => inferInstance, fun a b => ⟨1, one_pos, fun t _ => isCarriedBy_dirac 1 zero_le_one⟩,
    fun g _ _ => ?_, rfl, fun g _ _ t => ?_⟩
  · simpa using continuous_const
  · simp [attentionKernel_dirac_zero]

variable {Q K V : ParamMatrix d} {μ₀ : Measure (EucSpace d)} {μ : ℝ → Measure (EucSpace d)}

/-- A solution is carried near each time by a fixed ball. -/
theorem IsContEqSolution.eventually_carried (hμ : IsContEqSolution Q K V μ₀ μ) (t₀ : ℝ) :
    ∃ R, ∀ᶠ t in 𝓝 t₀, μ t (closedBall 0 R)ᶜ = 0 := by
  obtain ⟨R, -, hR⟩ := hμ.2.1 (t₀ - 1) (t₀ + 1)
  exact ⟨R, eventually_of_mem (Icc_mem_nhds (by linarith) (by linarith)) hR⟩

/-- `⟨∇g(x), v⟩ = Dg(x) v`. -/
theorem inner_gradient (g : EucSpace d → ℝ) (x v : EucSpace d) :
    inner (𝕜 := ℝ) (gradient g x) v = fderiv ℝ g x v := by
  rw [gradient, InnerProductSpace.toDual_symm_apply]

/-- `𝒳` is linear in `V`: reversing the value matrix reverses the field. -/
theorem attentionKernel_neg (Q K V : ParamMatrix d) (ν : Measure (EucSpace d))
    (x : EucSpace d) : attentionKernel Q K (-V) ν x = -attentionKernel Q K V ν x := by
  simp [attentionKernel, integral_neg]

/-- **Reversing time.**  If `μ` solves `e:conteq` for `V`, then `t ↦ μ_{-t}`
solves it for `-V`, from the same initial measure. -/
theorem IsContEqSolution.comp_neg (hμ : IsContEqSolution Q K V μ₀ μ) :
    IsContEqSolution Q K (-V) μ₀ fun t => μ (-t) := by
  obtain ⟨hprob, hcar, hcont, h0, hweak⟩ := hμ
  refine ⟨fun t => hprob (-t), fun a b => ?_, fun g hg hgc => (hcont g hg hgc).comp continuous_neg,
    by simpa using h0, fun g hg hgc t => ?_⟩
  · obtain ⟨R, hR, hRt⟩ := hcar (-b) (-a)
    exact ⟨R, hR, fun t ht => hRt (-t) ⟨by linarith [ht.2], by linarith [ht.1]⟩⟩
  · show ∫ x, g x ∂μ (-t) = ∫ x, g x ∂μ₀ + ∫ s in (0 : ℝ)..t,
        ∫ x, inner (𝕜 := ℝ) (gradient g x) (attentionKernel Q K (-V) (μ (-s)) x) ∂μ (-s)
    simp only [attentionKernel_neg, inner_neg_right, integral_neg, intervalIntegral.integral_neg]
    rw [hweak g hg hgc (-t), intervalIntegral.integral_comp_neg (fun s => ∫ x, inner (𝕜 := ℝ)
      (gradient g x) (attentionKernel Q K V (μ s) x) ∂μ s), neg_zero,
      intervalIntegral.integral_symm 0 (-t), neg_neg]

/-- **A solution is transported by the flow of its own field.**  On `[0, T]`,
`μ_t = (Φ_{0,t})_# μ₀`, where `Φ` is the flow of `w(τ, x) = 𝒳[μ_τ](x)` with the
curve held constant outside `[0, T]`: so held, `μ` is carried by one ball and
`w` is a flow field (`isFlowField_attentionKernel`), and `μ` solves the linear
continuity equation of `w` on `[0, T]` (`ContinuityEq.eq_map_flow`). -/
theorem IsContEqSolution.eq_map_flow (hμ : IsContEqSolution Q K V μ₀ μ) {T t : ℝ}
    (ht : t ∈ Icc 0 T) :
    μ t = μ₀.map (flow (fun τ x => attentionKernel Q K V (μ (max 0 (min τ T))) x) 0 t) := by
  obtain ⟨hprob, hcar, hcont, h0, hweak⟩ := hμ
  obtain ⟨R, hR, hRt⟩ := hcar 0 T
  have hT : 0 ≤ T := ht.1.trans ht.2
  set c : ℝ → ℝ := fun τ => max 0 (min τ T)
  have hcI : ∀ τ, c τ ∈ Icc 0 T := fun τ => ⟨le_max_left _ _, max_le hT (min_le_right _ _)⟩
  have hcid : ∀ τ ∈ Icc 0 T, c τ = τ := fun τ hτ => by
    show max 0 (min τ T) = τ
    rw [min_eq_left hτ.2, max_eq_right hτ.1]
  have : ∀ τ, IsProbabilityMeasure (μ (c τ)) := fun τ => hprob _
  have hsupp : ∀ τ, IsCarriedBy (μ (c τ)) R := fun τ => hRt _ (hcI τ)
  have hRc : ∀ τ₀, ∃ R', ∀ᶠ τ in 𝓝 τ₀, μ (c τ) (closedBall 0 R')ᶜ = 0 :=
    fun _ => ⟨R, Eventually.of_forall hsupp⟩
  have hcontc : ∀ g : EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) g → HasCompactSupport g →
      Continuous fun τ => ∫ x, g x ∂μ (c τ) := fun g hg hgc =>
    (hcont g hg hgc).comp (continuous_const.max (continuous_id.min continuous_const))
  have h0' : μ (c 0) = μ₀ := by rw [hcid 0 ⟨le_rfl, hT⟩, h0]
  have key := ContinuityEq.eq_map_flow (μ := fun τ => μ (c τ))
    (A := fun τ x => attentionKernelDeriv Q K V (μ (c τ)) x)
    (isFlowField_attentionKernel Q K V hR hsupp hcontc)
    (continuous_attentionKernelDeriv_curve Q K V hRc hcontc)
    (fun τ x => hasFDerivAt_attentionKernel Q K V R hR _ inferInstance (hsupp τ) x)
    hRc hcontc (T := T) (fun g hg hgc s hs => ?_) ht
  · simpa only [hcid t ht, h0'] using key
  show ∫ x, g x ∂μ (c s) = ∫ x, g x ∂μ (c 0) +
    ∫ σ in (0 : ℝ)..s, ∫ x, fderiv ℝ g x (attentionKernel Q K V (μ (c σ)) x) ∂μ (c σ)
  rw [hcid s hs, h0', hweak g hg hgc s]
  congr 1
  refine intervalIntegral.integral_congr fun σ hσ => ?_
  rw [uIcc_of_le hs.1] at hσ
  simp only [hcid σ ⟨hσ.1, hσ.2.trans hs.2⟩, inner_gradient]

/-- The hypotheses of the results above are satisfiable: the stationary
solution at the Dirac mass at the origin. -/
example (Q K V : ParamMatrix d) :
    IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
      (fun _ => Measure.dirac (0 : EucSpace d)) :=
  isContEqSolution_dirac Q K V

end Clusters
end Transformer
