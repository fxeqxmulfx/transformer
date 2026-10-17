/-
# The emergence of clusters in self-attention dynamics — stability and
  uniqueness of solutions of the continuity equation

`e:w2estimate` of arXiv:2305.05465v6, §6, for solutions carried by one ball,
and the uniqueness half of `c:wellposedtransformers`.

Two solutions `μ`, `ν` carried by `B̄(0, R)` on `[0, T]` are the images of
their initial measures under the flows of their own fields
(`IsContEqSolution.eq_map_flow`).  Along these flows the fields differ by
`‖𝒳[μ_τ](a) - 𝒳[ν_τ](b)‖ ≤ L ‖a - b‖ + C W₂(μ_τ, ν_τ)`
(`exists_norm_attentionKernel_sub_le`), and Grönwall's lemma on the transport
cost of every coupling (`ContinuityEq.W2_map_flow_le`) gives
`W₂(μ_t, ν_t) ≤ e^{(L + C) t} W₂(μ₀, ν₀)` (`exists_W2_le`): the source's proof
of `e:w2estimate`, with constants depending on the radius of the ball.  From
one initial measure the right side vanishes; `W₂` separates probability
measures carried by a ball (`eq_of_W2_eq_zero`), and reversing time
(`IsContEqSolution.comp_neg`) gives uniqueness on all of `ℝ`
(`IsContEqSolution.unique`).

Source: arXiv:2305.05465v6, `c:wellposedtransformers`, `e:w2estimate`.
-/

import Transformer.Clusters.Section6_WeakSolution
import Transformer.ContinuityEq.Stability

open Set Metric Filter Topology MeasureTheory

namespace Transformer
namespace Clusters

open GlobalFlow Wasserstein

variable {d : ℕ}

/-- **`W₂` separates probability measures carried by a ball**: `W₂(μ, ν) = 0`
forces `μ = ν`.  For smooth `g` with compact support, `g` is Lipschitz and
`|∫ g dμ - ∫ g dν| ≤ Lip(g) W₂(μ, ν) = 0` (`norm_integral_sub_le_W2`), and
such `g` separate the two measures (`SmoothApprox.ext_of_forall_contDiff`). -/
theorem eq_of_W2_eq_zero {μ ν : Measure (EucSpace d)} [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] {R : ℝ} (hμ : IsCarriedBy μ R) (hν : IsCarriedBy ν R)
    (h : W2 μ ν = 0) : μ = ν := by
  refine SmoothApprox.ext_of_forall_contDiff hμ hν fun g hg hgc => ?_
  obtain ⟨C, hC⟩ := hg.lipschitzWith_of_hasCompactSupport hgc (by simp)
  have hle := norm_integral_sub_le_W2 μ ν measurableSet_closedBall isBounded_closedBall hμ hν g
    (hg.continuous.integrable_of_hasCompactSupport hgc)
    (hg.continuous.integrable_of_hasCompactSupport hgc) C.coe_nonneg
    fun x _ y _ => by rw [← dist_eq_norm]; exact hC.dist_le_mul x y
  rwa [h, mul_zero, norm_le_zero_iff, sub_eq_zero] at hle

variable {Q K V : ParamMatrix d} {μ₀ ν₀ : Measure (EucSpace d)} {μ ν : ℝ → Measure (EucSpace d)}

/-- The initial measure of a solution is a probability measure. -/
theorem IsContEqSolution.isProbabilityMeasure (hμ : IsContEqSolution Q K V μ₀ μ) :
    IsProbabilityMeasure μ₀ :=
  hμ.2.2.2.1 ▸ hμ.1 0

/-- **Estimate (e:w2estimate) for solutions carried by one ball.**  Solutions
carried by `B̄(0, R)` on `[0, T]` satisfy `W₂(μ_t, ν_t) ≤ e^{C t} W₂(μ₀, ν₀)`
there, with `C` depending only on `Q`, `K`, `V`, `R` and `T`. -/
theorem exists_W2_le (Q K V : ParamMatrix d) {R : ℝ} (hR : 0 < R) (T : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ (μ₀ ν₀ : Measure (EucSpace d)) (μ ν : ℝ → Measure (EucSpace d)),
      IsContEqSolution Q K V μ₀ μ → IsContEqSolution Q K V ν₀ ν →
      (∀ t ∈ Icc 0 T, IsCarriedBy (μ t) R) → (∀ t ∈ Icc 0 T, IsCarriedBy (ν t) R) →
        ∀ t ∈ Icc 0 T, W2 (μ t) (ν t) ≤ Real.exp (C * t) * W2 μ₀ ν₀ := by
  set R' := R + ‖V‖ * R * |T|
  have hRR' : R ≤ R' := le_add_of_nonneg_right (by positivity)
  obtain ⟨C, hC, hCμ⟩ := exists_norm_attentionKernel_sub_le Q K V (R := R') (by positivity)
  set L := 2 * ‖(ContinuousLinearMap.adjoint Q).comp K‖ * ‖V‖ * R' ^ 2
  refine ⟨L + C, by positivity, fun μ₀ ν₀ μ ν hμ hν hμR hνR t ht => ?_⟩
  have hT : 0 ≤ T := ht.1.trans ht.2
  have : IsProbabilityMeasure μ₀ := hμ.isProbabilityMeasure
  have : IsProbabilityMeasure ν₀ := hν.isProbabilityMeasure
  set c : ℝ → ℝ := fun τ => max 0 (min τ T)
  have hcI : ∀ τ, c τ ∈ Icc 0 T := fun τ => ⟨le_max_left _ _, max_le hT (min_le_right _ _)⟩
  have hcid : ∀ τ ∈ Icc 0 T, c τ = τ := fun τ hτ => by
    show max 0 (min τ T) = τ
    rw [min_eq_left hτ.2, max_eq_right hτ.1]
  have : ∀ τ, IsProbabilityMeasure (μ (c τ)) := fun τ => hμ.1 _
  have : ∀ τ, IsProbabilityMeasure (ν (c τ)) := fun τ => hν.1 _
  have hw := isFlowField_attentionKernel Q K V hR (fun τ => hμR _ (hcI τ))
    fun g hg hgc => (hμ.2.2.1 g hg hgc).comp (continuous_const.max (continuous_id.min continuous_const))
  have hu := isFlowField_attentionKernel Q K V hR (fun τ => hνR _ (hcI τ))
    fun g hg hgc => (hν.2.2.1 g hg hgc).comp (continuous_const.max (continuous_id.min continuous_const))
  have hrepμ : ∀ τ ∈ Icc 0 T, μ τ =
      μ₀.map (flow (fun τ x => attentionKernel Q K V (μ (c τ)) x) 0 τ) := fun τ hτ =>
    hμ.eq_map_flow hτ
  have hrepν : ∀ τ ∈ Icc 0 T, ν τ =
      ν₀.map (flow (fun τ x => attentionKernel Q K V (ν (c τ)) x) 0 τ) := fun τ hτ =>
    hν.eq_map_flow hτ
  have h0 : ∀ τ ∈ Icc 0 T, R + ‖V‖ * R * τ ≤ R' := fun τ hτ => by
    have : ‖V‖ * R * τ ≤ ‖V‖ * R * |T| :=
      mul_le_mul_of_nonneg_left (hτ.2.trans (le_abs_self T)) (by positivity)
    linarith
  have hb := ContinuityEq.W2_map_flow_le hw hu (R := R)
    (hμ.2.2.2.1 ▸ hμR 0 ⟨le_rfl, hT⟩) (hν.2.2.2.1 ▸ hνR 0 ⟨le_rfl, hT⟩) (K := L) hC.le
    (fun τ hτ a ha b hb => ?_) ht
  · rwa [← hrepμ t ht, ← hrepν t ht] at hb
  rw [← hrepμ τ hτ, ← hrepν τ hτ]
  simp only [hcid τ hτ]
  exact hCμ (μ τ) (ν τ) (hμ.1 τ) (hν.1 τ) ((hμR τ hτ).mono hRR') ((hνR τ hτ).mono hRR') a b
    (closedBall_subset_closedBall (h0 τ hτ) hb)

/-- **Uniqueness forward in time**: two solutions from one initial measure
agree at every `t ≥ 0`. -/
theorem IsContEqSolution.eq_of_nonneg (hμ : IsContEqSolution Q K V μ₀ μ)
    (hν : IsContEqSolution Q K V μ₀ ν) {t : ℝ} (ht : 0 ≤ t) : μ t = ν t := by
  obtain ⟨Rμ, hRμ, hμR⟩ := hμ.2.1 0 t
  obtain ⟨Rν, -, hνR⟩ := hν.2.1 0 t
  have hμR' : ∀ s ∈ Icc 0 t, IsCarriedBy (μ s) (max Rμ Rν) := fun s hs =>
    (hμR s hs).mono (le_max_left _ _)
  have hνR' : ∀ s ∈ Icc 0 t, IsCarriedBy (ν s) (max Rμ Rν) := fun s hs =>
    (hνR s hs).mono (le_max_right _ _)
  obtain ⟨C, -, hC⟩ := exists_W2_le Q K V (lt_max_of_lt_left hRμ) t
  have : IsProbabilityMeasure μ₀ := hμ.isProbabilityMeasure
  have : IsProbabilityMeasure (μ t) := hμ.1 t
  have : IsProbabilityMeasure (ν t) := hν.1 t
  refine eq_of_W2_eq_zero (hμR' t ⟨ht, le_rfl⟩) (hνR' t ⟨ht, le_rfl⟩)
    (le_antisymm ?_ (W2_nonneg _ _))
  simpa [W2_self] using hC μ₀ μ₀ μ ν hμ hν hμR' hνR' t ⟨ht, le_rfl⟩

/-- **Uniqueness (c:wellposedtransformers)**: two solutions of `e:conteq` from
one initial measure are equal, at negative times by reversing time. -/
theorem IsContEqSolution.unique (hμ : IsContEqSolution Q K V μ₀ μ)
    (hν : IsContEqSolution Q K V μ₀ ν) : μ = ν := by
  funext t
  rcases le_total 0 t with ht | ht
  · exact hμ.eq_of_nonneg hν ht
  · simpa using hμ.comp_neg.eq_of_nonneg hν.comp_neg (neg_nonneg.2 ht)

/-- The hypotheses of the results above are satisfiable: the stationary
solution at the Dirac mass at the origin, carried by the unit ball, against
itself, with `R = 1 > 0` and `t = 0 ≥ 0`. -/
example (Q K V : ParamMatrix d) :
    IsContEqSolution Q K V (Measure.dirac (0 : EucSpace d))
        (fun _ => Measure.dirac (0 : EucSpace d)) ∧
      (∀ _ : ℝ, IsCarriedBy (Measure.dirac (0 : EucSpace d)) 1) ∧
      Wasserstein.W2 (Measure.dirac (0 : EucSpace d)) (Measure.dirac 0) = 0 ∧
      (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 :=
  ⟨isContEqSolution_dirac Q K V, fun _ => isCarriedBy_dirac 1 zero_le_one, W2_self _, one_pos,
    le_rfl⟩

end Clusters
end Transformer
