/-
# Real analytic germs: Noetherian

Adapted from Bochao Kong's LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, Noetherian/Ruckert.lean.
The coefficient field and analytic germs are real. Only the required
dependency closure is retained. Apache-2.0 license:
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.RealAnalyticGerms.NoetherianRemainder
import Transformer.RealAnalyticGerms.ZeroDimension
import Transformer.RealAnalyticGerms.PreparedAssociate
import Mathlib.RingTheory.Finiteness.Ideal

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open Transformer.AnalyticPreparation

/-- **Rückert's basis theorem.**  The local ring of analytic germs at the
origin of every finite-dimensional real affine space is Noetherian.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticGerm_isNoetherian_core :
    ∀ n : ℕ, IsNoetherianRing (AnalyticGerm n)
  | 0 => analyticGerm_isNoetherian_zero
  | n + 1 => by
      let : IsNoetherianRing (AnalyticGerm n) :=
        analyticGerm_isNoetherian_core n
      rw [isNoetherianRing_iff_ideal_fg]
      intro I
      by_cases hI : I = ⊥
      · subst I
        exact Submodule.fg_bot
      · obtain ⟨f, hfI, hf_ne⟩ :=
          Submodule.exists_mem_ne_zero_of_ne_bot hI
        obtain ⟨L, d, H, a, u, hH, hcoord, hH0, horder, hprep⟩ :=
          exists_regularized_weierstrassPreparation hf_ne
        let e : AnalyticGerm (n + 1) ≃+* AnalyticGerm (n + 1) :=
          coordinatePullback L
        let J : Ideal (AnalyticGerm (n + 1)) := I.map e.toRingHom
        let p : AnalyticGerm (n + 1) :=
          preparedPolynomialGerm a hprep.1
        have hcoord_mem : coordinatePullback L f ∈ J := by
          exact Ideal.mem_map_of_mem e.toRingHom hfI
        have hassoc : Associated (coordinatePullback L f) p := by
          exact coordinatePullback_associated_preparedPolynomialGerm
            L H a u hcoord hprep
        have hpJ : p ∈ J :=
          (Ideal.mem_iff_of_associated hassoc).mp hcoord_mem
        have hJfg : J.FG :=
          Ideal.fg_of_remainder_kernel J p hpJ
            (preparedGermDivisionRemainderLinearMap
              a hprep.1 hprep.2.1)
            (preparedGermDivisionRemainderLinearMap_ker_le
              a hprep.1 hprep.2.1)
        have hback :
            (J.map e.symm.toRingHom).FG :=
          hJfg.map e.symm.toRingHom
        simpa [J, e, Ideal.map_map] using hback

/-- The Noetherian structure supplied by Rückert's theorem.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
instance analyticGerm_instIsNoetherianRing (n : ℕ) :
    IsNoetherianRing (AnalyticGerm n) :=
  analyticGerm_isNoetherian_core n

end Transformer.RealAnalyticGerms
