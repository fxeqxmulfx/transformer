/-
# Width decay for compact consensus families

Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. A common
lower bound on two probability-weighted interaction kernels controls
the difference between the top and bottom velocities. Subtracting the
common kernel makes the mean terms cancel, without requiring atoms.

For a compact family of scalar curves, a contraction bound at each
maximal pair then controls every pairwise gap exponentially. Apply the
compact maximum principle to the gaps multiplied by `exp(κ t)`.
Only continuity on future times and derivatives of the actual curves
are used; derivatives of the extrema are never assumed.
-/

import Transformer.Perspective.CompactMaxCurve
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open Real Set MeasureTheory
open scoped Topology

namespace Transformer.Perspective

variable {ι : Type*}

/-- The common kernel floor contracts the gap between an
extremal pair by the full probability mass, including nonatomic laws.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem consensus_extreme_deriv_sub_le [MeasurableSpace ι]
    (ρ : Measure ι) [IsProbabilityMeasure ρ] (a b g : ι → ℝ) (κ M m : ℝ)
    (ha : ∀ᵐ j ∂ρ, κ ≤ a j) (hb : ∀ᵐ j ∂ρ, κ ≤ b j)
    (hM : ∀ᵐ j ∂ρ, g j ≤ M) (hm : ∀ᵐ j ∂ρ, m ≤ g j) (hg : Integrable g ρ)
    (hia : Integrable (fun j => a j * (g j - M)) ρ)
    (hib : Integrable (fun j => b j * (g j - m)) ρ) :
    (∫ j, a j * (g j - M) ∂ρ) - (∫ j, b j * (g j - m) ∂ρ) ≤ -κ * (M - m) := by
  have hupper := integral_mono_ae hia ((hg.sub (integrable_const M)).const_mul κ)
    (by filter_upwards [ha, hM] with j haj hMj
        exact mul_le_mul_of_nonpos_right haj (sub_nonpos.mpr hMj))
  have hlower := integral_mono_ae ((hg.sub (integrable_const m)).const_mul κ) hib
    (by filter_upwards [hb, hm] with j hbj hmj
        exact mul_le_mul_of_nonneg_right hbj (sub_nonneg.mpr hmj))
  simp only [Pi.sub_apply] at hupper hlower
  rw [integral_const_mul, integral_sub hg (integrable_const M), integral_const,
    probReal_univ, one_smul] at hupper
  rw [integral_const_mul, integral_sub hg (integrable_const m), integral_const,
    probReal_univ, one_smul] at hlower
  linarith

variable [TopologicalSpace ι] [CompactSpace ι]

/-- A local contraction bound at attaining maximum and minimum curves
gives an exponential bound on every gap in a compact family.
Auxiliary for arXiv:2411.04551v3, §2, `prop: targets.atoms`. -/
theorem compact_width_decay_of_deriv_bounds
    (g g' : ι → ℝ → ℝ) (κ D : ℝ)
    (hc : ContinuousOn (fun q : ι × ℝ => g q.1 q.2) (univ ×ˢ Ici 0))
    (hd : ∀ i r, 0 ≤ r → HasDerivAt (g i) (g' i r) r)
    (hcontract : ∀ r, 0 ≤ r → ∀ i k,
      (∀ j, g j r ≤ g i r) → (∀ j, g k r ≤ g j r) →
        g' i r - g' k r ≤ -κ * (g i r - g k r))
    (hD : ∀ i k, g i 0 - g k 0 ≤ D) (T : ℝ) (hT : 0 ≤ T) (i k : ι) :
    g i T - g k T ≤ D * exp (-κ * T) := by
  classical
  let F (p : ι × ι) (r : ℝ) := exp (κ * r) * (g p.1 r - g p.2 r)
  let F' (p : ι × ι) (r : ℝ) := exp (κ * r) * κ * (g p.1 r - g p.2 r) +
    exp (κ * r) * (g' p.1 r - g' p.2 r)
  have hc₁ : ContinuousOn (fun q : (ι × ι) × ℝ => g q.1.1 q.2) (univ ×ˢ Icc 0 T) :=
    hc.comp ((continuous_fst.fst.prodMk continuous_snd).continuousOn)
      (fun _ hq => ⟨mem_univ _, hq.2.1⟩)
  have hc₂ : ContinuousOn (fun q : (ι × ι) × ℝ => g q.1.2 q.2) (univ ×ˢ Icc 0 T) :=
    hc.comp ((continuous_fst.snd.prodMk continuous_snd).continuousOn)
      (fun _ hq => ⟨mem_univ _, hq.2.1⟩)
  have hcF : ContinuousOn (fun q : (ι × ι) × ℝ => F q.1 q.2) (univ ×ˢ Icc 0 T) :=
    ((continuous_const.mul continuous_snd).rexp.continuousOn).mul (hc₁.sub hc₂)
  have hex (r : ℝ) (hr : r ∈ Icc 0 T) : ∃ p : ι × ι, ∀ q, F q r ≤ F p r := by
    have hp : Continuous (fun p : ι × ι => F p r) := by
      rw [← continuousOn_univ]
      exact hcF.comp ((continuous_id.prodMk continuous_const).continuousOn)
        (fun _ _ => ⟨mem_univ _, hr⟩)
    obtain ⟨p, _, hp⟩ := (isCompact_univ : IsCompact (univ : Set (ι × ι))).exists_isMaxOn
      ⟨(i, k), mem_univ _⟩ hp.continuousOn
    exact ⟨p, fun q => hp (mem_univ q)⟩
  let pick (r : ℝ) : ι × ι := if hr : r ∈ Icc 0 T then (hex r hr).choose else (i, k)
  let R (r : ℝ) := F (pick r) r
  have hmax (r : ℝ) (hr : r ∈ Icc 0 T) (p : ι × ι) : F p r ≤ R r := by
    dsimp only [R, pick]
    rw [dite_eq_left hr]
    exact (hex r hr).choose_spec p
  have hatt (r : ℝ) : ∃ p, R r = F p r := ⟨pick r, rfl⟩
  have hFd (p : ι × ι) (r : ℝ) (hr : r ∈ Icc 0 T) : HasDerivAt (F p) (F' p r) r := by
    have he : HasDerivAt (fun r : ℝ => exp (κ * r)) (exp (κ * r) * κ) r := by
      convert ((hasDerivAt_id r).const_mul κ).exp using 1 <;> simp
    exact (he.mul ((hd p.1 r hr.1).sub (hd p.2 r hr.1))).congr_deriv (by rfl)
  have hnp (r : ℝ) (hr : r ∈ Icc 0 T) (p : ι × ι) (hp : F p r = R r) : F' p r ≤ 0 := by
    have he : 0 < exp (κ * r) := exp_pos _
    have htop (j : ι) : g j r ≤ g p.1 r := by
      have hh := (hmax r hr (j, p.2)).trans_eq hp.symm
      change exp (κ * r) * (g j r - g p.2 r) ≤ exp (κ * r) * (g p.1 r - g p.2 r) at hh
      nlinarith
    have hbot (j : ι) : g p.2 r ≤ g j r := by
      have hh := (hmax r hr (p.1, j)).trans_eq hp.symm
      change exp (κ * r) * (g p.1 r - g j r) ≤ exp (κ * r) * (g p.1 r - g p.2 r) at hh
      nlinarith
    have hh := mul_le_mul_of_nonneg_left (hcontract r hr.1 p.1 p.2 htop hbot) he.le
    dsimp only [F']
    nlinarith
  have hmono := compact_max_curve_le_of_deriv_nonpos F F' R 0 T hT hcF hFd hmax
    (fun r _ => hatt r) hnp
  obtain ⟨p, hp⟩ := hatt 0
  have hR0 : R 0 ≤ D := by
    rw [hp]
    simpa only [F, mul_zero, exp_zero, one_mul] using hD p.1 p.2
  have hF := (hmax T ⟨hT, le_rfl⟩ (i, k)).trans (hmono.trans hR0)
  change exp (κ * T) * (g i T - g k T) ≤ D at hF
  rw [← le_div_iff₀' (exp_pos _), div_eq_mul_inv, ← exp_neg] at hF
  simpa only [neg_mul] using hF

/-- A genuine probability and nonzero bounded scalar and kernel data
jointly satisfy all integral-comparison hypotheses. -/
example : ∃ ρ : Measure Unit, IsProbabilityMeasure ρ ∧
    (∀ᵐ j : Unit ∂ρ, (1 : ℝ) ≤ (fun _ : Unit => (1 : ℝ)) j) ∧
    (∀ᵐ j : Unit ∂ρ, (1 : ℝ) ≤ (fun _ : Unit => (2 : ℝ)) j) ∧
    (∀ᵐ j : Unit ∂ρ, (fun _ : Unit => (2 : ℝ)) j ≤ 3) ∧
    (∀ᵐ j : Unit ∂ρ, (1 : ℝ) ≤ (fun _ : Unit => (2 : ℝ)) j) ∧
    Integrable (fun _ : Unit => (2 : ℝ)) ρ ∧
    Integrable (fun _ : Unit => (1 : ℝ) * (2 - 3)) ρ ∧
    Integrable (fun _ : Unit => (2 : ℝ) * (2 - 1)) ρ := by
  refine ⟨Measure.dirac (), inferInstance, ae_of_all _ (by norm_num), ae_of_all _ (by norm_num),
    ae_of_all _ (by norm_num), ae_of_all _ (by norm_num),
    integrable_const _, integrable_const _, integrable_const _⟩

/-- An infinite compact family of actual exponentially contracting
curves realizes all continuity, derivative, extremal contraction and
initial-width hypotheses jointly. -/
example : ∃ (g g' : Icc (0 : ℝ) 1 → ℝ → ℝ),
    Continuous (fun q : Icc (0 : ℝ) 1 × ℝ => g q.1 q.2) ∧
    (∀ i r, HasDerivAt (g i) (g' i r) r) ∧
    (∀ r i k, g' i r - g' k r ≤ -(1 : ℝ) * (g i r - g k r)) ∧
    (∀ i k, g i 0 - g k 0 ≤ (1 : ℝ)) ∧ (0 : ℝ) ≤ 1 := by
  let g (i : Icc (0 : ℝ) 1) (r : ℝ) := (i : ℝ) * exp (-r)
  let g' (i : Icc (0 : ℝ) 1) (r : ℝ) := -(i : ℝ) * exp (-r)
  refine ⟨g, g', ?_, ?_, ?_, ?_, zero_le_one⟩
  · exact (continuous_subtype_val.comp continuous_fst).mul (continuous_snd.neg.rexp)
  · intro i r
    convert ((hasDerivAt_id r).neg.exp).const_mul (i : ℝ) using 1 <;> simp [g, g']
  · intro r i k
    dsimp only [g, g']
    linarith
  · intro i k
    dsimp only [g]
    simp only [neg_zero, exp_zero, mul_one]
    linarith [i.2.2, k.2.1]

end Transformer.Perspective
