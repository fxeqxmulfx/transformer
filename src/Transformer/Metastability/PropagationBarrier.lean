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

/-- A finite collection of scalar constraints remains nonnegative when its
vector field points strictly inward at every active boundary constraint. -/
theorem finite_nonneg_of_boundary_deriv {ι : Type*} [Fintype ι]
    (f : ι → ℝ → ℝ) (a b : ℝ)
    (hf : ∀ i, Continuous (f i))
    (ha : (∀ i, 0 ≤ f i a) ∧ a ≤ b)
    (hboundary : ∀ (t : ℝ), t ∈ Ico a b → (∀ i, 0 ≤ f i t) →
      ∀ i, f i t = 0 → 0 < deriv (f i) t) :
    ∀ t ∈ Icc a b, ∀ i, 0 ≤ f i t := by
  let s : Set ℝ := {t | ∀ i, 0 ≤ f i t}
  have hs : IsClosed s := by
    have h : IsClosed (⋂ i, (f i) ⁻¹' Ici 0) :=
      isClosed_iInter (fun i => isClosed_Ici.preimage (hf i))
    convert h using 1
    ext t
    simp [s]
  have hstep : ∀ t ∈ s ∩ Ico a b, s ∈ 𝓝[>] t := by
    intro t ht
    have hti : ∀ i, ∀ᶠ u in 𝓝[>] t, 0 ≤ f i u := by
      intro i
      rcases (ht.1 i).eq_or_lt with hzero | hpos
      · have hderiv := hboundary t ht.2 ht.1 i hzero.symm
        have hsign := eventually_nhdsWithin_sign_eq_of_deriv_pos hderiv hzero.symm
        filter_upwards [hsign.filter_mono nhdsWithin_le_nhds, self_mem_nhdsWithin] with u hu htu
        have htu' : t < u := htu
        have hsignu : SignType.sign (f i u) = SignType.pos := by
          simpa [sign_pos (sub_pos.mpr htu')] using hu
        exact (sign_eq_one_iff.mp hsignu).le
      · have hc : Tendsto (f i) (𝓝[>] t) (𝓝 (f i t)) :=
          (hf i).continuousAt.continuousWithinAt
        exact (hc.eventually_const_lt hpos).mono (fun _ h => h.le)
    exact (Filter.eventually_all.mpr hti : ∀ᶠ u in 𝓝[>] t, ∀ i, 0 ≤ f i u)
  have hmain : Icc a b ⊆ s :=
    (hs.inter isClosed_Icc).Icc_subset_of_forall_mem_nhdsWithin ha.1 hstep
  exact fun t ht i => hmain ht i

end Metastability
end Transformer
