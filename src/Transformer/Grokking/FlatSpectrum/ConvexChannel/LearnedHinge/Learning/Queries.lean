module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning.Basic

/-!
# Actual complex CE derivatives along certified feasible queries

Source: `review/convex/moments.md`, CM82. Two feasible endpoints and the existing
convex domain provide an entire neighbourhood, not just two isolated probes.
The actual recovered complex loss there has the derived ordinary CE derivative.
Finite differences at a nonzero radius are not asserted to equal that derivative.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning

open scoped BigOperators Topology

/-- Every intermediate query between two feasible endpoint queries stays
in a convex domain, CM82. This applies to full arrays and degenerate domains. -/
theorem query_interval {V : Type*} [AddCommGroup V] [Module ℝ V] {S : Set V}
    (hS : Convex ℝ S) {p v : V} {radius : ℝ} (hr : 0 < radius) (hp : p ∈ S)
    (hplus : p + radius • v ∈ S) (hminus : p + (-radius) • v ∈ S)
    {amount : ℝ} (ha : |amount| ≤ radius) : p + amount • v ∈ S := by
  have hab := abs_le.mp ha
  by_cases hn : 0 ≤ amount
  · have ht : amount / radius ∈ Set.Icc (0 : ℝ) 1 :=
      ⟨div_nonneg hn hr.le, (div_le_iff₀ hr).mpr (by simpa using hab.2)⟩
    have hm := hS.add_smul_mem hp hplus ht
    simpa only [smul_smul, div_mul_cancel₀ amount hr.ne'] using hm
  · have hneg : 0 ≤ -amount := by linarith
    have ht : -amount / radius ∈ Set.Icc (0 : ℝ) 1 :=
      ⟨div_nonneg hneg hr.le, (div_le_iff₀ hr).mpr (by linarith)⟩
    have hm := hS.add_smul_mem hp hminus ht
    have he : (-amount / radius) * (-radius) = amount := by
      field_simp
    simpa only [smul_smul, he] using hm

variable {E X J C : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable [Fintype J] [Fintype C] [DecidableEq C]

/-- The proposed slope is the derivative of the actual complete complex
loss along every certified two-sided feasible direction, CM82. Every factor
is recovered from the perturbed joint point; no outside-domain forward is used. -/
theorem complexLoss_hasDerivAt (input : X → E →ₗ[ℝ] ℂ) (region : Set X)
    (mask : J → X × Bool → Bool) {p v : Parameters J C E} {radius : ℝ}
    (hr : 0 < radius) (hp : p ∈ complexDomain input region mask)
    (hplus : p + radius • v ∈ complexDomain input region mask)
    (hminus : p + (-radius) • v ∈ complexDomain input region mask)
    {x : X} (hx : x ∈ region) (target : C) :
    HasDerivAt (fun amount : ℝ => complexLoss (input x) target (p + amount • v))
      (complexSlope (input x) (fun j k => mask j (x, k)) target p v) 0 := by
  have hd := ConvexDepth.rowCE_hasDerivAt (complexLogits p (input x))
    (logitDirection (input x) (fun j k => mask j (x, k)) v) target
  rw [← ceSlope_eq] at hd
  apply hd.congr_of_eventuallyEq
  filter_upwards [eventually_abs_sub_lt (0 : ℝ) hr] with amount ha
  have hb : |amount| ≤ radius := by simpa using ha.le
  have hm := query_interval (complexDomain_convex input region mask) hr hp hplus hminus hb
  unfold complexLoss
  rw [logits_shift input region mask amount hp hm hx]
  rfl

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Learning
