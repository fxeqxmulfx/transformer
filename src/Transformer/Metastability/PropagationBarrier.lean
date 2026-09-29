/-
# A finite-family barrier principle for propagation

If finitely many continuously varying quantities start nonnegative and each
has positive derivative whenever it reaches zero while the others remain
nonnegative, none can cross below zero. This avoids differentiating their
minimum, which may have corners.
-/

import Mathlib.Analysis.Calculus.DerivativeTest
import Mathlib.Topology.Order.IntermediateValue

open scoped Topology

namespace Transformer
namespace Metastability

open Filter Set

/-- A finite collection of scalar constraints, each continuous on `[a, b]`, remains
nonnegative on `[a, b]` when its vector field points strictly inward at every active
boundary constraint.  Only continuity on the interval is needed, so a constraint may be
given by a formula with singularities outside `[a, b]`. -/
theorem finite_nonneg_of_boundary_deriv_on {ι : Type*} [Fintype ι]
    (f : ι → ℝ → ℝ) (a b : ℝ)
    (hf : ∀ i, ContinuousOn (f i) (Icc a b))
    (ha : (∀ i, 0 ≤ f i a) ∧ a ≤ b)
    (hboundary : ∀ (t : ℝ), t ∈ Ico a b → (∀ i, 0 ≤ f i t) →
      ∀ i, f i t = 0 → 0 < deriv (f i) t) :
    ∀ t ∈ Icc a b, ∀ i, 0 ≤ f i t := by
  let s : Set ℝ := {t | ∀ i, 0 ≤ f i t}
  have hs : IsClosed (s ∩ Icc a b) := by
    have h : IsClosed (Icc a b ∩ ⋂ i, (Icc a b ∩ (f i) ⁻¹' Ici 0)) :=
      isClosed_Icc.inter (isClosed_iInter fun i =>
        (hf i).preimage_isClosed_of_isClosed isClosed_Icc isClosed_Ici)
    convert h using 1
    ext t
    simp only [s, mem_inter_iff, mem_ofPred_eq, mem_iInter, mem_preimage, mem_Ici]
    exact ⟨fun h => ⟨h.2, fun i => ⟨h.2, h.1 i⟩⟩, fun h => ⟨fun i => (h.2 i).2, h.1⟩⟩
  have hstep : ∀ t ∈ s ∩ Ico a b, s ∈ 𝓝[>] t := by
    intro t ht
    have htI : t ∈ Ico a b := ht.2
    have hti : ∀ i, ∀ᶠ u in 𝓝[>] t, 0 ≤ f i u := by
      intro i
      rcases (ht.1 i).eq_or_lt with hzero | hpos
      · have hderiv := hboundary t htI ht.1 i hzero.symm
        have hsign := eventually_nhdsWithin_sign_eq_of_deriv_pos hderiv hzero.symm
        filter_upwards [hsign.filter_mono nhdsWithin_le_nhds, self_mem_nhdsWithin] with u hu htu
        have htu' : t < u := htu
        have hsignu : SignType.sign (f i u) = SignType.pos := by
          simpa [sign_pos (sub_pos.mpr htu')] using hu
        exact (sign_eq_one_iff.mp hsignu).le
      · have hc : ContinuousWithinAt (f i) (Ioi t) t :=
          ((hf i).continuousWithinAt (Ico_subset_Icc_self htI)).mono_of_mem_nhdsWithin
            (Icc_mem_nhdsGT_of_mem htI)
        exact (hc.tendsto.eventually_const_lt hpos).mono (fun _ h => h.le)
    exact (Filter.eventually_all.mpr hti : ∀ᶠ u in 𝓝[>] t, ∀ i, 0 ≤ f i u)
  exact fun t ht i =>
    hs.Icc_subset_of_forall_mem_nhdsWithin ha.1 hstep ht i

/-- A finite collection of scalar constraints remains nonnegative when its
vector field points strictly inward at every active boundary constraint. -/
theorem finite_nonneg_of_boundary_deriv {ι : Type*} [Fintype ι]
    (f : ι → ℝ → ℝ) (a b : ℝ)
    (hf : ∀ i, Continuous (f i))
    (ha : (∀ i, 0 ≤ f i a) ∧ a ≤ b)
    (hboundary : ∀ (t : ℝ), t ∈ Ico a b → (∀ i, 0 ≤ f i t) →
      ∀ i, f i t = 0 → 0 < deriv (f i) t) :
    ∀ t ∈ Icc a b, ∀ i, 0 ≤ f i t :=
  finite_nonneg_of_boundary_deriv_on f a b (fun i => (hf i).continuousOn) ha hboundary

end Metastability
end Transformer
