module

public import Transformer.Causal.Section4_TwoTokenLeading

/-!
# Positive leading components along every original causal row

ArXiv:2411.04990v2, §4, `thm2`, with the original equation of §2.
An original leading covector gives a scalar triangular cooperative
system. A common strictly positive initial covector stays positive
at every forward time, for every original token. Its forcing is
the actual predecessor sum and its diagonal coefficient includes
the original nonlinear radial projection and self attention weight.
No positivity of these components is assumed along the trajectory.
-/

@[expose] public section

open Real Set
open scoped BigOperators

namespace Transformer.Causal

variable {d n : ℕ}

/-- A nonnegative actual forcing preserves a positive initial scalar
component on the forward half-line. Source: arXiv:2411.04990v2, §§3–4,
the integrating factor of `lemma1` applied to the leading rows of `thm2`. -/
theorem scalar_pos_of_nonneg_forcing (h f r : ℝ → ℝ) (hh : Continuous h)
    (hr : ∀ t, HasDerivAt r (f t + h t * r t) t)
    (hf : ∀ t, 0 ≤ t → 0 ≤ f t) (h0 : 0 < r 0) (t : ℝ) (ht : 0 ≤ t) : 0 < r t := by
  let H : ℝ → ℝ := fun u => ∫ s in (0 : ℝ)..u, h s
  have hH (u : ℝ) : HasDerivAt H (h u) u :=
    intervalIntegral.integral_hasDerivAt_right (hh.intervalIntegrable _ _)
      (hh.stronglyMeasurableAtFilter _ _) hh.continuousAt
  have hd (u : ℝ) : HasDerivAt (fun s => r s * Real.exp (-H s))
      (f u * Real.exp (-H u)) u := by
    convert (hr u).mul ((hH u).neg.exp) using 1
    simp only [Pi.neg_apply]
    ring
  have hm : MonotoneOn (fun s => r s * Real.exp (-H s)) (Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
      (fun u _ => (hd u).continuousAt.continuousWithinAt)
      (fun u _ => (hd u).differentiableAt.differentiableWithinAt)
    intro u hu
    rw [interior_Ici] at hu
    rw [(hd u).deriv]
    exact mul_nonneg (hf u hu.le) (Real.exp_pos _).le
  have hle := hm (by simp) ht ht
  have hp : 0 < r t * Real.exp (-H t) := by
    simp only [H, intervalIntegral.integral_same, neg_zero, Real.exp_zero, mul_one] at hle
    exact h0.trans_le hle
  exact (mul_pos_iff_of_pos_right (Real.exp_pos _)).mp hp

/-- The original last row's common leading coefficient, including its
actual self weight and full radial average. Source: arXiv:2411.04990v2,
§4, `thm2`, the last row of §2, `eqn:csa`. -/
noncomputable def csaLastLeadingCoefficient (β : ℝ) (Q K V : ParamMatrix d) (lam : ℝ)
    (X : ℝ → SphereTuple d (n + 1)) (t : ℝ) : ℝ :=
  lam * csaTokenWeight β Q K (fun j => (X t j : EucSpace d)) (Fin.last n) (Fin.last n) -
    csaRadialCoefficient β Q K V X t (Fin.last n)

/-- The common leading coefficient is continuous along the actual
original flow. Source: arXiv:2411.04990v2, §4, the smooth field of `thm2`. -/
theorem continuous_csaLastLeadingCoefficient (β : ℝ) (Q K V : ParamMatrix d) (lam : ℝ)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) :
    Continuous (csaLastLeadingCoefficient β Q K V lam X) := by
  have hx (i : Idx (n + 1)) : Continuous (fun t => (X t i : EucSpace d)) :=
    continuous_iff_continuousAt.mpr fun t =>
      ((CSA_iff_csaAmbientField β Q K V X).mp hX t i).continuousAt
  have hall : Continuous (fun t => fun i => (X t i : EucSpace d)) := continuous_pi hx
  exact (continuous_const.mul
    ((contDiff_csaTokenWeight β Q K (Fin.last n) (Fin.last n)).continuous.comp hall)).sub
    ((hx (Fin.last n)).inner ((contDiff_csaAverage β Q K V (Fin.last n)).continuous.comp hall))

/-- The actual last leading component has the full actual predecessor
forcing and its common original diagonal coefficient.
Source: arXiv:2411.04990v2, §4, `thm2`, and §2, `eqn:csa`. -/
theorem hasDerivAt_csaLast_leading (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ v (X s (Fin.last n) : EucSpace d))
      (lam * ∑ j : Idx n, csaTokenWeight β Q K (fun k => (X t k : EucSpace d))
        (Fin.last n) j.castSucc * inner ℝ v (X t j.castSucc : EucSpace d) +
          csaLastLeadingCoefficient β Q K V lam X t *
            inner ℝ v (X t (Fin.last n) : EucSpace d)) t := by
  have h := hasDerivAt_csaLeadingOverlap β Q K V L lam heig hinv v hv X hX (Fin.last n) t
  apply h.congr_deriv
  simp only [Fin.sum_univ_castSucc, csaLastLeadingCoefficient, csaRadialCoefficient]
  ring

/-- A common positive original leading covector stays positive on all
tokens at every forward time. Positivity is proved by the original
causal prefix induction. Source: arXiv:2411.04990v2, §4, a pointwise
leading-hemisphere case of `thm2`; this initialization condition is explicit. -/
theorem csaLeading_pos_of_initial_pos (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 ≤ lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L) :
    ∀ (m : ℕ) (X : ℝ → SphereTuple d m), CSA d m β Q K V X →
      (∀ i, 0 < inner ℝ v (X 0 i : EucSpace d)) →
        ∀ i t, 0 ≤ t → 0 < inner ℝ v (X t i : EucSpace d) := by
  intro m
  induction m with
  | zero => intro X hX hp i; exact i.elim0
  | succ m ih =>
    intro X hX hp i
    have hpre := ih (fun t j => X t j.castSucc) (CSA_castSucc β Q K V X hX)
      (fun j => hp j.castSucc)
    refine Fin.lastCases ?_ (fun j => hpre j) i
    intro t ht
    apply scalar_pos_of_nonneg_forcing (csaLastLeadingCoefficient β Q K V lam X)
      (fun s => lam * ∑ j : Idx m, csaTokenWeight β Q K (fun k => (X s k : EucSpace d))
        (Fin.last m) j.castSucc * inner ℝ v (X s j.castSucc : EucSpace d))
      (fun s => inner ℝ v (X s (Fin.last m) : EucSpace d))
      (continuous_csaLastLeadingCoefficient β Q K V lam X hX)
      (hasDerivAt_csaLast_leading β Q K V L lam heig hinv v hv X hX) ?_ (hp _) t ht
    intro s hs
    apply mul_nonneg hlam
    exact Finset.sum_nonneg fun j _ => mul_nonneg
      (csaTokenWeight_nonneg β Q K _ _ _) (hpre j s hs).le

/-- Actual nonzero consensus realizes the spectral, positive-covector
and full original flow hypotheses at every finite tuple size.
Source: arXiv:2411.04990v2, §4, the leading consensus in `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 2) (m : ℕ) :
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (basePoint 1 : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    CSA 2 m β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    (∀ i : Idx m, 0 < inner ℝ (basePoint 1 : EucSpace 2)
      ((fun _ : Idx m => basePoint 1) i : EucSpace 2)) := by
  refine ⟨fun v _ => by simp, fun z hz => hz, Submodule.mem_top,
    CSA_const_consensus β Q K (basePoint 1), fun _ => ?_⟩
  rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (basePoint 1).2]
  norm_num

/-- A positive constant component with zero forcing satisfies all
forward integrating-factor premises together. Source: arXiv:2411.04990v2,
§4, the scalar sign comparison for the leading rows of `thm2`. -/
example : Continuous (fun _ : ℝ => (0 : ℝ)) ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) (0 + 0 * 1) t) ∧
    (∀ t : ℝ, (0 : ℝ) ≤ t → (0 : ℝ) ≤ 0) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by
  exact ⟨continuous_const, fun t => by simpa using hasDerivAt_const t (1 : ℝ),
    fun _ _ => le_rfl, one_pos, zero_le_one⟩

end Transformer.Causal
