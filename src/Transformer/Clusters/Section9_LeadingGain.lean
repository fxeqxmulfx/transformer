/-
# Leading attention gain vanishes under subspace positivity

ArXiv:2305.05465v6, §9, the proof of `l:3hyperplanes11`, using
`eq:phistarvar` and `e:majorizeweights`. A metric representing the
attention form on the leading subspace is sufficient for the
corrected-energy argument. Positivity away from that subspace is unused.

Both exponentially small score errors are integrated and subtracted
from the leading squared norm. The resulting energy is monotone on
future times, bounded by the initial projected hull, and convergent.
Its derivative dominates the maximal leading attention gap.

The actual projected dynamics makes the gap uniformly continuous.
The proved derivative-domination principle then forces the gap to
tend to zero, with no assumption that token positions converge.
-/

import Transformer.Clusters.Section9_LeadingEnergy
import Transformer.Clusters.Section10_CorrectedEnergy
import Transformer.Clusters.Section10_ProjectedGain

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- A representing leading form supplies the actual monotone corrected
energy and a derivative bound for every leading attention gap.
Source: arXiv:2305.05465v6, §9, `eq:phistarvar`, `e:majorizeweights`,
and §10, the energy argument, restricted to the leading form. -/
theorem RescaledDynamics.exists_corrected_energy_of_leading_form
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hform : IsAttentionFormOn Q K A F) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧
      (∀ i : Idx n, Differentiable ℝ (correctedProjectedEnergy A proj Z lam δ D i)) ∧
      (∀ i : Idx n, MonotoneOn (correctedProjectedEnergy A proj Z lam δ D i) (Set.Ici 0)) ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
        2 * lam * inner (𝕜 := ℝ) (A (proj (Z t i)))
          (A (proj (Z t j) - proj (Z t i))) ≤
          deriv (correctedProjectedEnergy A proj Z lam δ D i) t := by
  obtain ⟨δ, D, hδ, hD, henergy⟩ :=
    hZ.exists_projected_energy_bound_of_leading_form Q K V proj A F G lam hV hproj hform Z
  have hdiff (i : Idx n) : Differentiable ℝ (correctedProjectedEnergy A proj Z lam δ D i) :=
    fun t => (hasDerivAt_correctedProjectedEnergy A proj Z lam δ hδ D i t _
      (hZ.hasDerivAt_norm_sq_projected Q K V proj A F G lam hV hproj Z t i)).differentiableAt
  have hbound (t : ℝ) (ht : 0 ≤ t) (i j : Idx n) :
      2 * lam * inner (𝕜 := ℝ) (A (proj (Z t i)))
        (A (proj (Z t j) - proj (Z t i))) ≤
        deriv (correctedProjectedEnergy A proj Z lam δ D i) t := by
    obtain ⟨v, hv, hb⟩ := henergy t ht i
    rw [(hasDerivAt_correctedProjectedEnergy A proj Z lam δ hδ D i t v hv).deriv]
    have hj := hb j
    rw [div_eq_mul_inv, ← Real.exp_neg] at hj
    linarith
  refine ⟨δ, D, hδ, hD, hdiff, ?_, hbound⟩
  intro i
  apply monotoneOn_of_deriv_nonneg (convex_Ici (0 : ℝ)) (hdiff i).continuous.continuousOn
    (hdiff i).differentiableOn
  intro t ht
  have ht0 : 0 ≤ t := interior_subset ht
  simpa only [sub_self, map_zero, inner_zero_right, mul_zero] using hbound t ht0 i i

example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionFormOn (1 : ParamMatrix 1) 1 1 ⊤ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1,
    isAttentionFormOn_of_isAttentionRoot 1 1 1 ⊤ (isAttentionRoot_id 1),
    rescaledDynamics_one_const 1 1 _⟩

/-- Every actual token has a convergent corrected leading energy whose
derivative still dominates all its leading gaps.
Source: arXiv:2305.05465v6, §9, the bounded leading coordinate in `l:fj`,
combined with the energy argument used for `l:3hyperplanes11`. -/
theorem RescaledDynamics.exists_convergent_energy_of_leading_form
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hform : IsAttentionFormOn Q K A F) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧
      (∀ i : Idx n, Differentiable ℝ (correctedProjectedEnergy A proj Z lam δ D i)) ∧
      (∀ i : Idx n, ∃ L : ℝ,
        Tendsto (correctedProjectedEnergy A proj Z lam δ D i) atTop (nhds L)) ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
        2 * lam * inner (𝕜 := ℝ) (A (proj (Z t i)))
          (A (proj (Z t j) - proj (Z t i))) ≤
          deriv (correctedProjectedEnergy A proj Z lam δ D i) t := by
  obtain ⟨δ, D, hδ, hD, hdiff, hmono, hbound⟩ :=
    hZ.exists_corrected_energy_of_leading_form Q K V proj A F G lam hV hproj hform Z
  obtain ⟨_, R, _, _, hR, _, hnorm, _⟩ :=
    hZ.exists_bound_attention_normalization Q K V proj A F G lam hV hproj Z
  refine ⟨δ, D, hδ, hD, hdiff, ?_, hbound⟩
  intro i
  have hupper (t : ℝ) (ht : 0 ≤ t) :
      correctedProjectedEnergy A proj Z lam δ D i t ≤ R ^ 2 :=
    (correctedProjectedEnergy_le_norm_sq A proj Z lam δ D hV.2.2.2.1.le
      hδ hD.le i t).trans
      ((pow_le_pow_iff_left₀ (norm_nonneg _) hR.le two_ne_zero).mpr (hnorm t ht i))
  obtain ⟨L, hL, _⟩ := exists_tendsto_of_monotoneOn (hmono i) hupper
  exact ⟨L, hL⟩

example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionFormOn (1 : ParamMatrix 1) 1 1 ⊤ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1,
    isAttentionFormOn_of_isAttentionRoot 1 1 1 ⊤ (isAttentionRoot_id 1),
    rescaledDynamics_one_const 1 1 _⟩

/-- **The actual maximal leading gain vanishes under positivity on the
leading subspace alone.** Source: arXiv:2305.05465v6, §9, the drift
argument for `l:3hyperplanes11`. Uniform continuity of the gain comes
from the actual projected dynamics and compactness of its initial hull. -/
theorem RescaledDynamics.tendsto_attentionGain_zero_of_leading_form
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hform : IsAttentionFormOn Q K A F) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (i : Idx n) :
    Tendsto (fun t => attentionGain A (fun j => proj (Z t j)) i) atTop (nhds 0) := by
  obtain ⟨δ, D, _, _, hdiff, hlim, hbound⟩ :=
    hZ.exists_convergent_energy_of_leading_form Q K V proj A F G lam hV hproj hform Z
  obtain ⟨L, hL⟩ := hlim i
  apply tendsto_zero_of_deriv_domination
    (fun t => correctedProjectedEnergy A proj Z lam δ D i t / (2 * lam))
    (fun t => deriv (correctedProjectedEnergy A proj Z lam δ D i) t / (2 * lam))
    (fun t => attentionGain A (fun j => proj (Z t j)) i) (L / (2 * lam))
    (fun t => ((hdiff i t).hasDerivAt).div_const (2 * lam))
  · intro t ht
    unfold attentionGain
    apply (Finset.sup'_le_iff _ _).mpr
    intro j _
    apply (le_div_iff₀ (mul_pos (show (0 : ℝ) < 2 by norm_num) hV.2.2.2.1)).mpr
    simpa only [mul_comm] using hbound t ht i j
  · exact fun t _ => attentionGain_nonneg A (fun j => proj (Z t j)) i
  · exact hZ.uniformContinuousOn_projected_attentionGain Q K V proj A F G lam hV hproj Z i
  · exact hL.div_const (2 * lam)

example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionFormOn (1 : ParamMatrix 1) 1 1 ⊤ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1,
    isAttentionFormOn_of_isAttentionRoot 1 1 1 ⊤ (isAttentionRoot_id 1),
    rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters
