/-
Copyright (c) 2026 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in
third_party/hex-real-roots-mathlib/LICENSE.
Authors: Kim Morrison

Adapted from leanprover/hex-real-roots-mathlib,
revision 53ce31466dc5ff520463249d470119c1e0006e22.
-/

import Mathlib.Topology.Connected.TotallyDisconnected
import Mathlib.Topology.Instances.Sign

/-!
# Signs of continuous nonvanishing functions

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

open Filter Topology

namespace Transformer.Sturm

variable {α : Type*} [Zero α] [TopologicalSpace α] [LinearOrder α] [OrderTopology α]

/-- The sign of a continuous function is continuous wherever the function is nonzero.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem continuousOn_sign {β : Type*} [TopologicalSpace β] {f : β → α} {s : Set β}
    (hf : ContinuousOn f s) (h0 : ∀ x ∈ s, f x ≠ 0) :
    ContinuousOn (fun x => SignType.sign (f x)) s := by
  refine (continuousOn_of_forall_continuousAt fun y hy => ?_).comp' hf (Set.mapsTo_image _ _)
  obtain ⟨x, hx, rfl⟩ := hy
  exact continuousAt_sign_of_ne_zero (h0 x hx)

/-- A continuous nonvanishing function has constant sign on a preconnected set.

Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem sign_eq_of_continuousOn {β : Type*} [TopologicalSpace β] {f : β → α}
    {s : Set β} (hs : IsPreconnected s) (hf : ContinuousOn f s)
    (h0 : ∀ x ∈ s, f x ≠ 0) {x y : β} (hx : x ∈ s) (hy : y ∈ s) :
    SignType.sign (f x) = SignType.sign (f y) :=
  hs.constant (continuousOn_sign hf h0) hx hy

end Transformer.Sturm
