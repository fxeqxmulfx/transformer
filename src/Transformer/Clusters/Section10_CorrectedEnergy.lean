/-
# An actual monotone energy for the leading component

ArXiv:2305.05465v6, §10, Step 2' and `e:derivativenorm123`.
Subtracting the integrated softmax and spectral-gap errors from
`||A π_F z_i||²` gives a nondecreasing energy on future times.
Its derivative dominates every leading attention gap, including the
self-token gap, which is zero.

The corrected operator order and leading factor are those established
from the actual ODE in `Section10_ProjectedEnergy`. The error constants
are supplied by the proved score estimate, rather than assumed.
-/

import Transformer.Clusters.Section10_ProjectedEnergy
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- The leading squared norm with both exponential errors integrated
and subtracted. Source: arXiv:2305.05465v6, §10, Step 2', corrected
from `π_F A` to `A π_F` and retaining the leading speed `λ`. -/
noncomputable def correctedProjectedEnergy (A proj : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (lam δ D : ℝ) (i : Idx n) (t : ℝ) : ℝ :=
  ‖A (proj (Z t i))‖ ^ 2 - (n : ℝ) * Real.exp (-(2 * lam * t)) -
    (4 * lam * D / δ) * Real.exp (-δ * t)

/-- The integrated errors have exactly the derivative needed to cancel
the lower-bound losses. Source: arXiv:2305.05465v6, §10, Step 2'.
The squared-norm derivative is explicit; the actual ODE supplies it below. -/
theorem hasDerivAt_correctedProjectedEnergy (A proj : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (lam δ : ℝ) (hδ : 0 < δ) (D : ℝ)
    (i : Idx n) (t v : ℝ) (hv : HasDerivAt (fun s => ‖A (proj (Z s i))‖ ^ 2) v t) :
    HasDerivAt (correctedProjectedEnergy A proj Z lam δ D i)
      (v + 2 * lam * (n : ℝ) * Real.exp (-(2 * lam * t)) +
        4 * lam * D * Real.exp (-δ * t)) t := by
  have hfirst : HasDerivAt (fun s : ℝ => Real.exp (-(2 * lam * s)))
      (Real.exp (-(2 * lam * t)) * (-(2 * lam))) t := by
    simpa only [id_eq, mul_one] using ((hasDerivAt_id t).const_mul (2 * lam)).fun_neg.exp
  have hsecond : HasDerivAt (fun s : ℝ => Real.exp (-δ * s))
      (Real.exp (-δ * t) * (-δ)) t := by
    simpa only [id_eq, mul_one, neg_mul] using ((hasDerivAt_id t).const_mul δ).fun_neg.exp
  have h := (hv.fun_sub (hfirst.const_mul (n : ℝ))).fun_sub
    (hsecond.const_mul (4 * lam * D / δ))
  apply h.congr_deriv
  field_simp [hδ.ne']
  ring

/-- A positive decay rate and the full squared-norm derivative input
coexist for a stationary, possibly nonzero token.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, Step 2'. -/
example (z : EucSpace 1) : (0 : ℝ) < 1 ∧
    HasDerivAt (fun _ : ℝ => ‖(1 : ParamMatrix 1) ((1 : ParamMatrix 1) z)‖ ^ 2) 0 0 :=
  ⟨one_pos, hasDerivAt_const 0 _⟩

/-- Subtracting the positive corrections keeps the energy below the
leading squared norm. Source: arXiv:2305.05465v6, §10, Step 2'; this
transfers the projected-hull bound to the corrected energy. -/
theorem correctedProjectedEnergy_le_norm_sq (A proj : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (lam δ D : ℝ) (hlam : 0 ≤ lam)
    (hδ : 0 < δ) (hD : 0 ≤ D) (i : Idx n) (t : ℝ) :
    correctedProjectedEnergy A proj Z lam δ D i t ≤ ‖A (proj (Z t i))‖ ^ 2 := by
  have hfirst : 0 ≤ (n : ℝ) * Real.exp (-(2 * lam * t)) := by positivity
  have hsecond : 0 ≤ (4 * lam * D / δ) * Real.exp (-δ * t) := by positivity
  unfold correctedProjectedEnergy
  linarith

/-- All scalar signs needed for the energy upper bound coexist.
Source: arXiv:2305.05465v6, §10, Step 2'. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 :=
  ⟨zero_le_one, one_pos, zero_le_one⟩

/-- **The actual projected dynamics supplies a monotone corrected energy
whose derivative dominates every leading gap.**
Source: arXiv:2305.05465v6, §10, Step 2' and `e:derivativenorm123`, with
the corrected order `A π_F` and the leading factor `λ`.
Differentiability holds on the whole line; monotonicity and the derivative
lower bound are on the future interval where the spectral error is bounded. -/
theorem RescaledDynamics.exists_corrected_projected_energy
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧
      (∀ i : Idx n, Differentiable ℝ (correctedProjectedEnergy A proj Z lam δ D i)) ∧
      (∀ i : Idx n, MonotoneOn (correctedProjectedEnergy A proj Z lam δ D i) (Set.Ici 0)) ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
        2 * lam * inner (𝕜 := ℝ) (A (proj (Z t i)))
          (A (proj (Z t j) - proj (Z t i))) ≤
          deriv (correctedProjectedEnergy A proj Z lam δ D i) t := by
  obtain ⟨δ, D, hδ, hD, henergy⟩ :=
    hZ.exists_projected_energy_deriv_lower_bound Q K V proj A F G lam hV hproj hA Z
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

/-- The root, paranormal splitting, projection, and actual ODE inputs
are simultaneously realized by nonzero identity-value tokens.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, Step 2'. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 _⟩

/-- **Each token has a convergent corrected energy that still dominates
all its leading attention gaps.**
Source: arXiv:2305.05465v6, §10, the bounded-energy argument following
Step 2'. The actual projected hull bounds the energy above on future
times; no convergence of token positions is assumed here. -/
theorem RescaledDynamics.exists_convergent_corrected_projected_energy
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
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
    hZ.exists_corrected_projected_energy Q K V proj A F G lam hV hproj hA Z
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

/-- All inputs, including the attention root and the coupled ODE, hold
for a genuine nonzero stationary rescaled configuration.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, Step 2'. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters
