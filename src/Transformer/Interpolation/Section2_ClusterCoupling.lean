/-
# Clustered laws near atomic targets

ArXiv:2411.04551v3, §2.2, the end of the proof of `prop: compression`:
once most of the mass lies in small balls `𝓑_k^i` around the atoms
`x_k^i`, the law is close to `μ_1^i = Σ_k α_k^i δ_{x_k^i}`. The source
bounds `W_1` through Kantorovich–Rubinstein duality and concludes by
the equivalence of the Wasserstein distances on the sphere; here an
explicit coupling bounds `W_2` directly.

Let pairwise disjoint sets `E_k`, all of whose points lie within `s` of
`x_k`, carry each at least `α_k - e`. The coupling sends the fraction
`min(α_k, ν(E_k)) / ν(E_k)` of the mass of `E_k` to `x_k` and couples
the remaining masses, of total at most `M e`, independently. Its cost
is at most `s² + 4 M e`.
-/

import Transformer.Interpolation.Wasserstein
import Mathlib.MeasureTheory.Measure.Sub

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Two unit vectors are at distance at most `2`. -/
theorem dist_le_two_of_sphere (x y : SSphere d) : dist x y ≤ 2 := by
  rw [Subtype.dist_eq, dist_eq_norm]
  have h := norm_sub_le (x : EucSpace d) (y : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp y.2] at h
  linarith

/-- **Clustered mass bounds the distance to the atomic target.** If
pairwise disjoint measurable sets `E_k` within distance `s` of `x_k`
carry at least `α_k - e` each, then `ν` is within `√(s² + 4 M e)` of
`Σ_k α_k δ_{x_k}` in `W_2`. Source: arXiv:2411.04551v3, §2.2, the end of
the proof of `prop: compression`, with an explicit coupling in place of
the duality bound of `W_1`. -/
theorem W2_le_of_cluster_masses {M : ℕ} (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (x : Fin M → SSphere d) (α : Fin M → ℝ≥0) (hα : ∑ k, α k = 1)
    (E : Fin M → Set (SSphere d)) (hE : ∀ k, MeasurableSet (E k))
    (hdisj : Pairwise (Function.onFun Disjoint E)) (s e : ℝ) (he : 0 ≤ e)
    (hEs : ∀ k, ∀ y ∈ E k, dist y (x k) ≤ s)
    (hmass : ∀ k, (α k : ℝ≥0∞) ≤ ν (E k) + ENNReal.ofReal e) :
    W2 d ν (∑ k, α k • Measure.dirac (x k)) ≤ Real.sqrt (s ^ 2 + 4 * M * e) := by
  classical
  set β : Fin M → ℝ≥0∞ := fun k => min (α k : ℝ≥0∞) (ν (E k)) with hβ_def
  set c : Fin M → ℝ≥0∞ := fun k => β k / ν (E k) with hc_def
  have hcβ (k : Fin M) : c k * ν (E k) = β k := by
    by_cases h0 : ν (E k) = 0
    · simp [hc_def, hβ_def, h0]
    · exact ENNReal.div_mul_cancel h0 (measure_ne_top ν _)
  have hc1 (k : Fin M) : c k ≤ 1 :=
    ENNReal.div_le_of_le_mul (by rw [one_mul]; exact min_le_right _ _)
  have hβα (k : Fin M) : β k ≤ α k := min_le_left _ _
  have hαsum : ∑ k, (α k : ℝ≥0∞) = 1 := by
    rw [← ENNReal.ofNNReal_finsetSum, hα, ENNReal.coe_one]
  set μ₁ : Measure (SSphere d) := ∑ k, c k • ν.restrict (E k) with hμ₁_def
  have hμ₁ : μ₁ ≤ ν := by
    refine Measure.le_iff.mpr fun t ht => ?_
    have hU : ν (⋃ k, t ∩ E k) = ∑ k, ν (t ∩ E k) := by
      rw [measure_iUnion (fun k l hkl => (hdisj hkl).mono inter_subset_right inter_subset_right)
        (fun k => ht.inter (hE k)), tsum_fintype]
    calc μ₁ t = ∑ k, c k * ν (t ∩ E k) := by
          simp [hμ₁_def, Measure.coe_finsetSum, Measure.restrict_apply ht]
      _ ≤ ∑ k, ν (t ∩ E k) := Finset.sum_le_sum fun k _ => mul_le_of_le_one_left' (hc1 k)
      _ = ν (⋃ k, t ∩ E k) := hU.symm
      _ ≤ ν t := measure_mono (iUnion_subset fun k => inter_subset_left)
  have hμ₁univ : μ₁ univ = ∑ k, β k := by
    simp only [hμ₁_def, Measure.coe_finsetSum, Finset.sum_apply, Measure.smul_apply,
      Measure.restrict_apply_univ, smul_eq_mul]
    exact Finset.sum_congr rfl fun k _ => hcβ k
  set m : ℝ≥0∞ := ∑ k, ((α k : ℝ≥0∞) - β k) with hm_def
  have hmβ : m + ∑ k, β k = 1 := by
    rw [hm_def, ← Finset.sum_add_distrib, ← hαsum]
    exact Finset.sum_congr rfl fun k _ => tsub_add_cancel_of_le (hβα k)
  have hβtop : ∑ k, β k ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (hmβ ▸ le_add_self)
  have hm_top : m ≠ ⊤ := ne_top_of_le_ne_top ENNReal.one_ne_top (hmβ ▸ le_self_add)
  have : IsFiniteMeasure μ₁ := ⟨by rw [hμ₁univ]; exact hβtop.lt_top⟩
  set ν₂ : Measure (SSphere d) := ν - μ₁ with hν₂_def
  have hν₂add : ν₂ + μ₁ = ν := Measure.sub_add_cancel_of_le hμ₁
  have hν₂univ : ν₂ univ = m := by
    have h1 : ν₂ univ + μ₁ univ = 1 := by rw [← Measure.add_apply, hν₂add, measure_univ]
    rw [hμ₁univ] at h1
    exact (ENNReal.add_left_inj hβtop).mp (h1.trans hmβ.symm)
  set τ₂ : Measure (SSphere d) := ∑ k, ((α k : ℝ≥0∞) - β k) • Measure.dirac (x k) with hτ₂_def
  have hτ₂univ : τ₂ univ = m := by
    simp [hτ₂_def, hm_def, Measure.coe_finsetSum]
  have : IsFiniteMeasure ν₂ := ⟨by rw [hν₂univ]; exact hm_top.lt_top⟩
  have : IsFiniteMeasure τ₂ := ⟨by rw [hτ₂univ]; exact hm_top.lt_top⟩
  have hcancel (ρ : Measure (SSphere d)) (hρ : ρ univ = m) : m⁻¹ • m • ρ = ρ := by
    by_cases hm0 : m = 0
    · rw [hm0] at hρ ⊢
      rw [Measure.measure_univ_eq_zero.mp hρ, smul_zero, smul_zero]
    · rw [smul_smul, ENNReal.inv_mul_cancel hm0 hm_top, one_smul]
  have hpair (k : Fin M) : Measurable fun y : SSphere d => (y, x k) :=
    measurable_id.prodMk measurable_const
  set γ₁ : Measure (SSphere d × SSphere d) :=
    ∑ k, c k • (ν.restrict (E k)).map (fun y => (y, x k)) with hγ₁_def
  set γ₂ : Measure (SSphere d × SSphere d) := m⁻¹ • ν₂.prod τ₂ with hγ₂_def
  have hcoup : IsCoupling d ν (∑ k, α k • Measure.dirac (x k)) (γ₁ + γ₂) := by
    constructor
    · have h1 : Measure.map Prod.fst γ₁ = μ₁ := by
        rw [hγ₁_def, Measure.map_finset_sum measurable_fst.aemeasurable]
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [Measure.map_smul _ measurable_fst.aemeasurable,
          Measure.map_map measurable_fst (hpair k)]
        simp [Function.comp_def]
      have h2 : Measure.map Prod.fst γ₂ = ν₂ := by
        rw [hγ₂_def, Measure.map_smul _ measurable_fst.aemeasurable, Measure.map_fst_prod,
          hτ₂univ, hcancel ν₂ hν₂univ]
      rw [Measure.map_add _ _ measurable_fst, h1, h2, add_comm, hν₂add]
    · rw [Measure.map_add _ _ measurable_snd, hγ₁_def, Measure.map_finset_sum
        measurable_snd.aemeasurable, hγ₂_def, Measure.map_smul _ measurable_snd.aemeasurable,
        Measure.map_snd_prod, hν₂univ, hcancel τ₂ hτ₂univ, hτ₂_def, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [Measure.map_smul _ measurable_snd.aemeasurable, Measure.map_map measurable_snd (hpair k),
        ENNReal.smul_def, ← tsub_add_cancel_of_le (hβα k), add_smul, add_comm (_ • _),
        tsub_add_cancel_of_le (hβα k)]
      simp only [Function.comp_def, Measure.map_const, Measure.restrict_apply_univ, smul_smul,
        hcβ k]
  have hf : Measurable fun p : SSphere d × SSphere d => dist p.1 p.2 ^ 2 :=
    (continuous_dist.pow 2).measurable
  have hcost₁ : ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂γ₁ ≤ ENNReal.ofReal (s ^ 2) := by
    rw [hγ₁_def, lintegral_finsetSum_measure]
    calc ∑ k, ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2)
            ∂(c k • (ν.restrict (E k)).map (fun y => (y, x k)))
        ≤ ∑ k, ENNReal.ofReal (s ^ 2) * β k := by
          refine Finset.sum_le_sum fun k _ => ?_
          rw [lintegral_smul_measure, lintegral_map hf.ennreal_ofReal (hpair k), smul_eq_mul,
            ← hcβ k, mul_comm (c k), mul_comm (c k), ← mul_assoc]
          refine mul_le_mul_left ?_ _
          calc ∫⁻ y, ENNReal.ofReal (dist y (x k) ^ 2) ∂ν.restrict (E k)
              ≤ ∫⁻ _y, ENNReal.ofReal (s ^ 2) ∂ν.restrict (E k) := by
                refine lintegral_mono_ae ((ae_restrict_mem (hE k)).mono fun y hy => ?_)
                exact ENNReal.ofReal_le_ofReal (pow_le_pow_left₀ dist_nonneg (hEs k y hy) 2)
            _ = ENNReal.ofReal (s ^ 2) * ν (E k) := by
                rw [lintegral_const, Measure.restrict_apply_univ]
      _ = ENNReal.ofReal (s ^ 2) * ∑ k, β k := (Finset.mul_sum _ _ _).symm
      _ ≤ ENNReal.ofReal (s ^ 2) * 1 := mul_le_mul_right (hmβ ▸ le_add_self) _
      _ = ENNReal.ofReal (s ^ 2) := mul_one _
  have hcost₂ : ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂γ₂ ≤ ENNReal.ofReal 4 * m := by
    rw [hγ₂_def, lintegral_smul_measure, smul_eq_mul]
    have hle : ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂ν₂.prod τ₂ ≤
        ENNReal.ofReal 4 * (m * m) := by
      calc ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂ν₂.prod τ₂
          ≤ ∫⁻ _p, ENNReal.ofReal 4 ∂ν₂.prod τ₂ := by
            refine lintegral_mono fun p => ENNReal.ofReal_le_ofReal ?_
            have h2 := dist_le_two_of_sphere p.1 p.2
            nlinarith [dist_nonneg (x := p.1) (y := p.2)]
        _ = ENNReal.ofReal 4 * (m * m) := by
            rw [lintegral_const, ← univ_prod_univ, Measure.prod_prod, hν₂univ, hτ₂univ]
    by_cases hm0 : m = 0
    · have hz : ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂ν₂.prod τ₂ = 0 := by
        rw [hm0, mul_zero, mul_zero] at hle
        exact le_antisymm hle (zero_le)
      rw [hz, mul_zero]
      exact zero_le
    · calc m⁻¹ * ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂ν₂.prod τ₂
          ≤ m⁻¹ * (ENNReal.ofReal 4 * (m * m)) := mul_le_mul_right hle _
        _ = ENNReal.ofReal 4 * m := by
            rw [mul_comm, mul_assoc, mul_assoc, ENNReal.mul_inv_cancel hm0 hm_top, mul_one]
  have hmM : m ≤ (M : ℝ≥0∞) * ENNReal.ofReal e := by
    calc m ≤ ∑ _k : Fin M, ENNReal.ofReal e := by
          refine Finset.sum_le_sum fun k _ => ?_
          rcases min_choice (α k : ℝ≥0∞) (ν (E k)) with h | h
          · rw [hβ_def]
            simp only [h, tsub_self]
            exact zero_le
          · rw [hβ_def]
            simp only [h]
            exact tsub_le_iff_left.mpr (hmass k)
      _ = (M : ℝ≥0∞) * ENNReal.ofReal e := by simp
  have hcost : ∫ p, dist p.1 p.2 ^ 2 ∂(γ₁ + γ₂) ≤ s ^ 2 + 4 * M * e := by
    rw [integral_eq_lintegral_of_nonneg_ae (ae_of_all _ fun p => by positivity)
      hf.aestronglyMeasurable]
    refine ENNReal.toReal_le_of_le_ofReal (by positivity) ?_
    rw [lintegral_add_measure]
    calc ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂γ₁ + ∫⁻ p, ENNReal.ofReal (dist p.1 p.2 ^ 2) ∂γ₂
        ≤ ENNReal.ofReal (s ^ 2) + ENNReal.ofReal 4 * ((M : ℝ≥0∞) * ENNReal.ofReal e) :=
          add_le_add hcost₁ (hcost₂.trans (mul_le_mul_right hmM _))
      _ = ENNReal.ofReal (s ^ 2 + 4 * M * e) := by
          rw [ENNReal.ofReal_add (by positivity) (by positivity), mul_assoc,
            ENNReal.ofReal_mul (by norm_num : (0 : ℝ) ≤ 4), ENNReal.ofReal_mul (by positivity),
            ENNReal.ofReal_natCast]
  exact (W2_le_of_coupling d _ _ _ hcoup).trans (Real.sqrt_le_sqrt hcost)

/-- The hypotheses of `W2_le_of_cluster_masses` are satisfiable: one
atom of weight one at a point, the whole sphere as its set, radius two
and no defect. -/
example (z : SSphere 2) :
    (∑ _k : Fin 1, (1 : ℝ≥0)) = 1 ∧ MeasurableSet (univ : Set (SSphere 2)) ∧
      (∀ y ∈ (univ : Set (SSphere 2)), dist y z ≤ 2) ∧
      ((1 : ℝ≥0) : ℝ≥0∞) ≤ (Measure.dirac z) (univ : Set (SSphere 2)) + ENNReal.ofReal 0 :=
  ⟨by simp, MeasurableSet.univ, fun y _ => dist_le_two_of_sphere y z, by simp⟩

end Transformer.Interpolation
