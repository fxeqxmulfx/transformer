/-
# Scalar convergence from eventual signed crossings

ArXiv:2411.04990v2, Appendix A and §5, scalar convergence
for the smooth frozen specialization of `thm:fixed_centers`.
The printed general C1 convergence lemma is false, so we keep
each premise of the specialized scalar argument explicit.

A strictly positive derivative at a fixed level prevents a
subsequent downward crossing after its starting time. Thus a
curve is eventually on one side of that level. The negative
derivative case follows by reversing the curve and the level.
If this alternative holds on a dense set of levels, an
eventually bounded curve converges. Its limit is the supremum
of its eventual lower levels, with both bounds proved directly.

Finally a countable exceptional set of levels has dense
complement in the real line. Hence eventual strict derivative
signs at all other levels give scalar convergence. Applying
this to the actual frozen row still requires boundedness and
the actual field limits; those hypotheses are not presumed.
For the eventual equilibrium step, we also prove that an
eventually bounded curve cannot have a nonzero derivative limit.
-/

import Transformer.Causal.Section4_ScalarLimits
import Mathlib.Topology.Algebra.Module.Cardinality
import Mathlib.Analysis.SpecialFunctions.Trigonometric.ArctanDeriv

open Real Set Filter

namespace Transformer.Causal

/-- An eventual positive derivative at a level makes an actual differentiable scalar curve eventually stay on one side of that level.
Source: arXiv:2411.04990v2, Appendix A and §5, the scalar crossing argument for the smooth frozen specialization. -/
theorem frozenScalar_eventually_side_of_positive_level (r r' : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (r' t) t) (a T : ℝ)
    (hb : ∀ t, T ≤ t → r t = a → 0 < r' t) :
    (∀ᶠ t in atTop, a ≤ r t) ∨ ∀ᶠ t in atTop, r t ≤ a := by
  by_cases hex : ∃ s, T ≤ s ∧ a ≤ r s
  · obtain ⟨s, hTs, hs⟩ := hex
    exact Or.inl (eventually_atTop.mpr ⟨s, csaScalar_forward_barrier r r' hd a T s hTs hs hb⟩)
  · push Not at hex
    exact Or.inr (eventually_atTop.mpr ⟨T, fun t ht => (hex t ht).le⟩)

/-- An eventual negative derivative at a level gives the same eventual-side alternative for an actual differentiable scalar curve.
Source: arXiv:2411.04990v2, Appendix A and §5, the reversed scalar crossing argument for the smooth frozen specialization. -/
theorem frozenScalar_eventually_side_of_negative_level (r r' : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (r' t) t) (a T : ℝ)
    (hb : ∀ t, T ≤ t → r t = a → r' t < 0) :
    (∀ᶠ t in atTop, a ≤ r t) ∨ ∀ᶠ t in atTop, r t ≤ a := by
  have h := frozenScalar_eventually_side_of_positive_level (fun t => -r t) (fun t => -r' t)
    (fun t => (hd t).neg) (-a) T (fun t ht he => by
      have hn := hb t ht (by linarith)
      linarith)
  rcases h with hlo | hhi
  · exact Or.inr (hlo.mono (fun _ ht => neg_le_neg_iff.mp ht))
  · exact Or.inl (hhi.mono (fun _ ht => neg_le_neg_iff.mp ht))

/-- An eventually bounded real curve with an eventual-side alternative at every level in a dense set has an actual real limit.
Source: arXiv:2411.04990v2, Appendix A and §5, the order-completeness step of the scalar convergence argument for the smooth frozen specialization. -/
theorem frozenScalar_tendsto_of_dense_levels (r : ℝ → ℝ) (A B : ℝ)
    (hbound : ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B) (s : Set ℝ) (hs : Dense s)
    (hlevels : ∀ a ∈ s, (∀ᶠ t in atTop, a ≤ r t) ∨ ∀ᶠ t in atTop, r t ≤ a) :
    ∃ l : ℝ, Tendsto r atTop (nhds l) := by
  let L : Set ℝ := {a | ∀ᶠ t in atTop, a ≤ r t}
  have hL : L.Nonempty := ⟨A, hbound.mono (fun _ ht => ht.1)⟩
  have hB : BddAbove L := by
    refine ⟨B, ?_⟩
    intro a ha
    obtain ⟨t, ht, hbt⟩ := (ha.and hbound).exists
    exact ht.trans hbt.2
  refine ⟨sSup L, tendsto_order.mpr ⟨?_, ?_⟩⟩
  · intro a ha
    obtain ⟨b, hb, hab⟩ := exists_lt_of_lt_csSup hL ha
    exact hb.mono (fun _ ht => hab.trans_le ht)
  · intro b hb
    obtain ⟨a, has, hla, hab⟩ := hs.exists_between hb
    rcases hlevels a has with hlo | hhi
    · have hle := le_csSup hB (show a ∈ L from hlo)
      exact False.elim ((not_lt_of_ge hle) hla)
    · exact hhi.mono (fun _ ht => ht.trans_lt hab)

/-- Eventual strict derivative signs at a dense set of fixed levels give convergence of an eventually bounded actual scalar curve.
Source: arXiv:2411.04990v2, Appendix A and §5, the differentiable scalar step of the smooth frozen convergence argument. -/
theorem frozenScalar_tendsto_of_signed_levels (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (A B : ℝ) (hbound : ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B) (s : Set ℝ) (hs : Dense s)
    (hlevels : ∀ a ∈ s, ∃ T : ℝ, (∀ t, T ≤ t → r t = a → 0 < r' t) ∨
      ∀ t, T ≤ t → r t = a → r' t < 0) : ∃ l : ℝ, Tendsto r atTop (nhds l) := by
  apply frozenScalar_tendsto_of_dense_levels r A B hbound s hs
  intro a ha
  obtain ⟨T, hb | hb⟩ := hlevels a ha
  · exact frozenScalar_eventually_side_of_positive_level r r' hd a T hb
  · exact frozenScalar_eventually_side_of_negative_level r r' hd a T hb

/-- A countable exceptional level set suffices for the actual scalar convergence argument when all remaining fixed levels have eventual strict derivative sign.
Source: arXiv:2411.04990v2, Appendix A and §5, the smooth frozen specialization; boundedness and signed crossings remain explicit premises. -/
theorem frozenScalar_tendsto_of_countable_levels (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (A B : ℝ) (hbound : ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B) (C : Set ℝ) (hC : C.Countable)
    (hlevels : ∀ a, a ∉ C → ∃ T : ℝ, (∀ t, T ≤ t → r t = a → 0 < r' t) ∨
      ∀ t, T ≤ t → r t = a → r' t < 0) : ∃ l : ℝ, Tendsto r atTop (nhds l) := by
  exact frozenScalar_tendsto_of_signed_levels r r' hd A B hbound Cᶜ (hC.dense_compl ℝ) hlevels

/-- Any eventual constant lower bound on the derivative of an eventually bounded actual scalar curve must be nonpositive.
Source: arXiv:2411.04990v2, Appendix A and §5, the bounded-drift argument used to identify the smooth frozen limit as critical. -/
theorem frozenScalar_eventual_derivative_lower_nonpos (r r' : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (r' t) t) (A B : ℝ)
    (hbound : ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B) (c : ℝ) (hc : ∀ᶠ t in atTop, c ≤ r' t) : c ≤ 0 := by
  by_contra h
  have hc0 : 0 < c := lt_of_not_ge h
  obtain ⟨T, hT⟩ := eventually_atTop.mp (hbound.and hc)
  have hAB : A ≤ B := (hT T le_rfl).1.1.trans (hT T le_rfl).1.2
  let u := T + (B - A + 1) / c
  have hTu : T ≤ u := by
    have hp : 0 ≤ (B - A + 1) / c := div_nonneg (by linarith) hc0.le
    dsimp [u]
    linarith
  have hlin := csaScalar_linear_growth r r' hd c T (fun t ht => (hT t ht).2) u hTu
  have hid : c * (u - T) = B - A + 1 := by dsimp [u]; field_simp; ring
  rw [hid] at hlin
  linarith [(hT u hTu).1.2, (hT T le_rfl).1.1]

/-- The actual derivative limit of an eventually bounded differentiable real curve is nonpositive whenever that limit exists.
Source: arXiv:2411.04990v2, Appendix A and §5, the one-sided bounded-drift argument for the actual smooth frozen limit. -/
theorem frozenScalar_derivative_limit_nonpos (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (A B : ℝ) (hbound : ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B)
    (v : ℝ) (hv : Tendsto r' atTop (nhds v)) : v ≤ 0 := by
  by_contra h
  have hv0 : 0 < v := lt_of_not_ge h
  have he : ∀ᶠ t in atTop, v / 2 ≤ r' t := (hv.eventually_const_lt (show v / 2 < v by linarith)).mono
    (fun _ ht => ht.le)
  have hn := frozenScalar_eventual_derivative_lower_nonpos r r' hd A B hbound (v / 2) he
  linarith

/-- Any existing actual derivative limit of an eventually bounded differentiable real curve is zero.
Source: arXiv:2411.04990v2, Appendix A and §5, the criticality step for the actual smooth frozen specialization. -/
theorem frozenScalar_derivative_limit_eq_zero (r r' : ℝ → ℝ) (hd : ∀ t, HasDerivAt r (r' t) t)
    (A B : ℝ) (hbound : ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B)
    (v : ℝ) (hv : Tendsto r' atTop (nhds v)) : v = 0 := by
  have hn := frozenScalar_derivative_limit_nonpos r r' hd A B hbound v hv
  have hp := frozenScalar_derivative_limit_nonpos (fun t => -r t) (fun t => -r' t)
    (fun t => (hd t).neg) (-B) (-A)
    (hbound.mono (fun t ht => ⟨neg_le_neg ht.2, neg_le_neg ht.1⟩)) (-v) hv.neg
  linarith

/-- A genuine nonconstant bounded curve with positive derivative jointly realizes the scalar boundedness and signed-crossing hypotheses.
Source: arXiv:2411.04990v2, Appendix A and §5, the auxiliary real scalar convergence argument. -/
example : (∀ t : ℝ, HasDerivAt Real.arctan (1 / (1 + t ^ 2)) t) ∧
    (∀ᶠ t : ℝ in atTop, -(Real.pi / 2) ≤ Real.arctan t ∧ Real.arctan t ≤ Real.pi / 2) ∧
    (∀ t : ℝ, (0 : ℝ) ≤ t → Real.arctan t = 0 → 0 < 1 / (1 + t ^ 2)) := by
  exact ⟨Real.hasDerivAt_arctan, Eventually.of_forall (fun t => ⟨(Real.arctan_mem_Ioo t).1.le,
    (Real.arctan_mem_Ioo t).2.le⟩), fun t _ _ => by positivity⟩

/-- The actual nonconstant increasing arctangent curve realizes the convergence statement with signed crossings at every real level.
Source: arXiv:2411.04990v2, Appendix A and §5, the explicit real scalar convergence premises used for the smooth frozen specialization. -/
example : ∃ l : ℝ, Tendsto Real.arctan atTop (nhds l) := by
  apply frozenScalar_tendsto_of_countable_levels Real.arctan (fun t => 1 / (1 + t ^ 2)) Real.hasDerivAt_arctan
    (-(Real.pi / 2)) (Real.pi / 2)
    (Eventually.of_forall (fun t => ⟨(Real.arctan_mem_Ioo t).1.le, (Real.arctan_mem_Ioo t).2.le⟩)) {0} (countable_singleton 0)
  intro a ha
  exact ⟨0, Or.inl (fun t ht he => by positivity)⟩

/-- Reversing the same genuine nonconstant curve realizes the negative-derivative crossing premises.
Source: arXiv:2411.04990v2, Appendix A and §5, the reversed scalar argument for the smooth frozen specialization. -/
example : (∀ t : ℝ, HasDerivAt (fun u => -Real.arctan u) (-(1 / (1 + t ^ 2))) t) ∧
    ∀ t : ℝ, (0 : ℝ) ≤ t → -Real.arctan t = 0 → -(1 / (1 + t ^ 2)) < 0 := by
  exact ⟨fun t => (Real.hasDerivAt_arctan t).neg, fun t _ _ => neg_neg_of_pos (by positivity)⟩

/-- An actual constant scalar equilibrium jointly realizes boundedness, an eventual derivative lower bound, and the existing derivative limit premises.
Source: arXiv:2411.04990v2, Appendix A and §5, the scalar equilibrium case of the smooth frozen specialization. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 t) ∧
    (∀ᶠ _ : ℝ in atTop, (-1 : ℝ) ≤ (0 : ℝ) ∧ (0 : ℝ) ≤ 1) ∧
    (∀ᶠ _ : ℝ in atTop, (0 : ℝ) ≤ 0) ∧ Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0) :=
  ⟨fun t => hasDerivAt_const t 0, Eventually.of_forall (fun _ => ⟨by norm_num, zero_le_one⟩),
    Eventually.of_forall (fun _ => le_rfl), tendsto_const_nhds⟩

end Transformer.Causal
