/-
# The emergence of clusters in self-attention dynamics — convergence to the origin

§8 of arXiv:2305.05465v6, `t:cas-Idintro`: at `V = -I_d`, `QᵀK = I_d`, every
token tends to the origin.

**What the source says and what is carried here.**  The source concludes by
compactness: a limit point `x*` of `x(t_k)` that is not stationary would give,
through the stability estimate `e:w2estimate` of the flow, a lower bound on
`‖ẋ_i‖` over the intervals `[t_k, t_k + T₀]`, against `e:finiteinegral`.  The
flow of `e:-Iddyn` is not needed for that.  By `l:cas1circle` the tokens stay
in a ball of radius `R`, and the drift is a convex combination of tokens, so
every token is `R`-Lipschitz in time (`norm_sub_le_of_negIdDynamics`).  The
total squared drift `G` is then uniformly continuous along the trajectory; it
is bounded below by some `η > 0` wherever `‖x_i‖ ≥ ε`, because the only zero of
`G` is the origin (`l:stationary`) and the set is compact; so each time
`‖x_i‖ ≥ ε`, `G ≥ η/2` for a fixed length of time.  Since `G` is integrable
(`e:finiteinegral`), that happens only finitely far out.

Source: arXiv:2305.05465v6, `t:cas-Idintro` and its proof in §8.
-/

import Transformer.Clusters.Section8_Bounded
import Transformer.Clusters.Section8_Energy
import Transformer.Clusters.Section8_Stationary
import Mathlib.Topology.UniformSpace.HeineCantor
import Mathlib.Topology.Order.Compact

open scoped BigOperators
open Real Filter Topology MeasureTheory

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- The drift of `e:-Iddyn` is minus a convex combination of the tokens, so it
is no longer than the longest token. -/
theorem norm_negIdDrift_le (Q K : ParamMatrix d) (Y : Idx n → EucSpace d) {R : ℝ}
    (hY : ∀ j, ‖Y j‖ ≤ R) (i : Idx n) : ‖negIdDrift Q K Y i‖ ≤ R := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  rw [negIdDrift, norm_neg]
  calc ‖∑ j, attentionMatrix Q K Y i j • Y j‖ ≤ ∑ j, attentionMatrix Q K Y i j * R := by
        refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun j _ => ?_)
        rw [norm_smul, Real.norm_of_nonneg (attentionMatrix_nonneg Q K Y i j)]
        exact mul_le_mul_of_nonneg_left (hY j) (attentionMatrix_nonneg Q K Y i j)
    _ = R := by rw [← Finset.sum_mul, sum_attentionMatrix hn, one_mul]

/-- The drift of `e:-Iddyn` depends continuously on the configuration. -/
theorem continuous_negIdDrift (Q K : ParamMatrix d) (i : Idx n) :
    Continuous fun Y : Idx n → EucSpace d => negIdDrift Q K Y i := by
  have hs : ∀ l, Continuous fun Y : Idx n → EucSpace d =>
      Real.exp (inner (𝕜 := ℝ) (Q (Y i)) (K (Y l))) := fun l => by fun_prop
  unfold negIdDrift attentionMatrix Perspective.softmaxWeight
  refine (continuous_finsetSum _ fun j _ => ?_).neg
  refine Continuous.smul (M := ℝ) ?_ (continuous_apply (A := fun _ : Idx n => EucSpace d) j)
  exact (hs j).div (continuous_finsetSum _ fun l _ => hs l) fun Y =>
    (Finset.sum_pos (fun l _ => Real.exp_pos _) ⟨i, Finset.mem_univ _⟩).ne'

/-- **The tokens are Lipschitz in time.**  Inside the ball of `l:cas1circle`
the speed is at most `R`. -/
theorem norm_sub_le_of_negIdDynamics (Q K : ParamMatrix d) (X : ℝ → Idx n → EucSpace d)
    (hX : NegIdDynamics Q K X) {R : ℝ} (hR : ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖X t i‖ ≤ R)
    (i : Idx n) {t s : ℝ} (ht : 0 ≤ t) (hts : t ≤ s) : ‖X s i - X t i‖ ≤ R * (s - t) :=
  norm_image_sub_le_of_norm_deriv_le_segment' (f := fun s => X s i)
    (fun x _ => (hX x i).hasDerivWithinAt)
    (fun x hx => norm_negIdDrift_le Q K (X x) (fun j => hR j x (ht.trans hx.1)) i) s
    ⟨hts, le_rfl⟩

/-- **Theorem (t:cas-Idintro).**  Let `V = -I_d` and `QᵀK = I_d`.  Then for
any initial sequence of tokens and any `i ∈ [n]`, `‖x_i(t)‖ → 0` as
`t → +∞`.

Source: arXiv:2305.05465v6, `t:cas-Idintro`. -/
theorem negId_tendsto_zero (Q K : ParamMatrix d) (hQK : IsIdentityQK Q K)
    (X : ℝ → Idx n → EucSpace d) (hX : NegIdDynamics Q K X) (i : Idx n) :
    Tendsto (fun t => ‖X t i‖) atTop (nhds 0) := by
  obtain ⟨R, hR0, hR⟩ := exists_bound_negIdDynamics Q K hQK X hX
  set G : (Idx n → EucSpace d) → ℝ := fun Y => ∑ k, ‖negIdDrift Q K Y k‖ ^ 2
  have hG : Continuous G :=
    continuous_finsetSum _ fun k _ => (continuous_negIdDrift Q K k).norm.pow 2
  have hint : IntegrableOn (fun t => G (X t)) (Set.Ioi 0) :=
    integrable_finsetSum _ fun k _ =>
      (integrableOn_sq_norm_negIdDrift Q K hQK X hX k).mono_set Set.Ioi_subset_Ici_self
  have hG0 : ∀ Y, G Y = 0 → IsStationaryConfig Q K Y := fun Y h k => by
    have hz := (Finset.sum_eq_zero_iff_of_nonneg
      (fun k _ => sq_nonneg ‖negIdDrift Q K Y k‖)).1 h k (Finset.mem_univ k)
    have : negIdDrift Q K Y k = 0 := by simpa using hz
    simpa [negIdDrift] using this
  have hGnn : ∀ Y, 0 ≤ G Y := fun Y => Finset.sum_nonneg fun k _ => sq_nonneg _
  clear_value G
  have hXc : Continuous X :=
    continuous_pi fun k => continuous_iff_continuousAt.2 fun t => (hX t k).continuousAt
  have hGc : Continuous fun s => G (X s) := hG.comp hXc
  rw [tendsto_order]
  refine ⟨fun a ha => Eventually.of_forall fun t => ha.trans_le (norm_nonneg _), fun ε _ => ?_⟩
  by_contra hne
  have hfreq : ∃ᶠ t in atTop, ε ≤ ‖X t i‖ := by
    simpa only [not_eventually, not_lt] using hne
  set S : Set (Idx n → EucSpace d) := Set.univ.pi fun _ => Metric.closedBall 0 R
  have hS : IsCompact S := isCompact_univ_pi fun _ => isCompact_closedBall _ _
  have hK : IsCompact (S ∩ {Y | ε ≤ ‖Y i‖}) :=
    hS.inter_right (isClosed_le continuous_const (continuous_apply i).norm)
  have hmemS : ∀ t, 0 ≤ t → X t ∈ S := fun t ht =>
    Set.mem_univ_pi.2 fun k => mem_closedBall_zero_iff.2 (hR k t ht)
  obtain ⟨t₀, ht₀, ht₀0⟩ := (hfreq.and_eventually (eventually_ge_atTop 0)).exists
  obtain ⟨Y₀, hY₀, hmin⟩ := hK.exists_isMinOn ⟨X t₀, hmemS t₀ ht₀0, ht₀⟩ hG.continuousOn
  -- The squared drift is bounded below where `‖x_i‖ ≥ ε`: `l:stationary`.
  have hη : 0 < G Y₀ := by
    refine lt_of_le_of_ne (hGnn Y₀) fun h => ?_
    have h0 := eq_zero_of_isStationaryConfig Q K hQK Y₀ (hG0 Y₀ h.symm) i
    have h2 : ε ≤ ‖Y₀ i‖ := hY₀.2
    rw [h0, norm_zero] at h2
    linarith
  set η := G Y₀
  obtain ⟨δ', hδ', hU⟩ := Metric.uniformContinuousOn_iff.1
    (hS.uniformContinuousOn_of_continuous hG.continuousOn) (η / 2) (half_pos hη)
  set δ := δ' / (2 * R)
  have hδ : 0 < δ := div_pos hδ' (by linarith)
  -- The integral of `G` over `[t, t + δ]` tends to `0`: `e:finiteinegral`.
  have hdiff : Tendsto (fun t => (∫ s in (0)..(t + δ), G (X s)) - ∫ s in (0)..t, G (X s))
      atTop (𝓝 0) := by
    have hH := intervalIntegral_tendsto_integral_Ioi (μ := volume) 0 hint tendsto_id
    simpa using (hH.comp (tendsto_atTop_add_const_right _ δ tendsto_id)).sub hH
  obtain ⟨t, ⟨ht, ht0⟩, hsmall⟩ := ((hfreq.and_eventually (eventually_ge_atTop 0)).and_eventually
    (hdiff.eventually (gt_mem_nhds (mul_pos hδ (half_pos hη))))).exists
  rw [intervalIntegral.integral_interval_sub_left (hGc.intervalIntegrable _ _)
    (hGc.intervalIntegrable _ _)] at hsmall
  have hlow : ∀ s ∈ Set.Icc t (t + δ), η / 2 ≤ G (X s) := by
    rintro s ⟨hs1, hs2⟩
    have hd : dist (X t) (X s) < δ' := by
      rw [dist_pi_lt_iff hδ']
      intro k
      rw [dist_comm, dist_eq_norm]
      have h1 := norm_sub_le_of_negIdDynamics Q K X hX hR k ht0 hs1
      have h2 : R * (s - t) ≤ R * δ := mul_le_mul_of_nonneg_left (by linarith) hR0.le
      have h3 : R * δ = δ' / 2 := by simp only [δ]; field_simp
      linarith
    have h4 := hU (X t) (hmemS t ht0) (X s) (hmemS s (by linarith)) hd
    rw [Real.dist_eq] at h4
    have hmt : η ≤ G (X t) := hmin ⟨hmemS t ht0, ht⟩
    linarith [(abs_lt.1 h4).2]
  have hI := intervalIntegral.integral_mono_on (by linarith : t ≤ t + δ)
    (intervalIntegrable_const (μ := volume)) (hGc.intervalIntegrable _ _) hlow
  simp only [intervalIntegral.integral_const, smul_eq_mul, add_sub_cancel_left] at hI
  linarith

/-- The hypotheses of `negId_tendsto_zero` are satisfiable: `Q = K = I_d` and
the configuration sitting at the origin. -/
example : IsIdentityQK (1 : ParamMatrix d) 1 ∧
    NegIdDynamics (n := n) (1 : ParamMatrix d) 1 (fun _ _ => (0 : EucSpace d)) :=
  ⟨isIdentityQK_one d, (transformerDynamics_neg_one_iff _ _ _).mp (transformerDynamics_zero 1 1 _)⟩

end Clusters
end Transformer
