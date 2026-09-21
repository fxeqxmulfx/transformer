/-
# §4 — The Gronwall bound behind `th:beta_small`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`eq: youareawizardharry` — the Grönwall bound of §4 read at `t = m`, below
`1/8`, for `β` small — together with the one-particle lemmas: at `n = 1` a
lone token stands still under both dynamics.

`e:approxsphere`, the Grönwall bound itself, is proved in
`Transformer.Perspective.Beta0Gronwall`.
-/

import Transformer.Perspective.Beta0Gronwall
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- **A lone token does not move under `eq: SA`.**

At `n = 1` the attention average is the token itself — the single weight is
`e^{β}` and it is divided by the partition function `Z = e^{β}` — so the drift
is `Proj_x x = 0` and every solution is constant.  This is the converse of
`SA_const_consensus` in the one-particle case.

Source: arXiv:2312.10794v5, §2, `eq: SA`. -/
theorem const_of_SA_one (β : ℝ) (X : ℝ → SphereTuple d 1)
    (hX : Perspective.SA d 1 β X) (t : ℝ) (i : Idx 1) :
    (X t i : EucSpace d) = (X 0 i : EucSpace d) := by
  have hzero : ∀ s : ℝ, HasDerivAt (fun r => (X r i : EucSpace d)) 0 s := by
    intro s
    obtain rfl : i = 0 := Subsingleton.elim i 0
    have hx : ‖(X s (0 : Idx 1) : EucSpace d)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp (X s 0).2
    have hxx : inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
        ((X s (0 : Idx 1) : EucSpace d)) = 1 := by
      rw [real_inner_self_eq_norm_mul_norm, hx]; ring
    have hZ : partitionSA d 1 β X s 0 = Real.exp β := by
      simp [partitionSA]
    have hsum : (∑ j : Idx 1,
        Real.exp (β * inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
            ((X s j : EucSpace d))) • ((X s j : EucSpace d)))
        = Real.exp β • ((X s (0 : Idx 1) : EucSpace d)) := by
      simp
    have hval : proj d ((X s (0 : Idx 1) : EucSpace d))
        ((partitionSA d 1 β X s 0)⁻¹ • ∑ j : Idx 1,
          Real.exp (β * inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
              ((X s j : EucSpace d))) • ((X s j : EucSpace d))) = 0 := by
      rw [hZ, hsum, smul_smul, inv_mul_cancel₀ (Real.exp_ne_zero β), one_smul,
        proj, hxx, one_smul, sub_self]
    exact (hX s 0).congr_deriv hval
  exact is_const_of_deriv_eq_zero (fun s => (hzero s).differentiableAt)
    (fun s => (hzero s).deriv) t 0

/-- The hypothesis of `const_of_SA_one` is satisfiable: the constant curve
solves `eq: SA` (`SA_const_consensus`). -/
example (β : ℝ) : Perspective.SA 1 1 β (fun _ _ => basePoint 0) :=
  SA_const_consensus 1 1 one_pos β (basePoint 0)

/-- **A lone token does not move under `e:Snonres0` either.**

At `n = 1` the mean of the tuple is the token itself, so the drift is again
`Proj_x x = 0`.

Source: arXiv:2312.10794v5, §4, `e:Snonres0`. -/
theorem const_of_beta0Dynamics_one (X : ℝ → SphereTuple d 1)
    (hX : beta0Dynamics d 1 X) (t : ℝ) (i : Idx 1) :
    (X t i : EucSpace d) = (X 0 i : EucSpace d) := by
  have hzero : ∀ s : ℝ, HasDerivAt (fun r => (X r i : EucSpace d)) 0 s := by
    intro s
    obtain rfl : i = 0 := Subsingleton.elim i 0
    have hx : ‖(X s (0 : Idx 1) : EucSpace d)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp (X s 0).2
    have hxx : inner (𝕜 := ℝ) ((X s (0 : Idx 1) : EucSpace d))
        ((X s (0 : Idx 1) : EucSpace d)) = 1 := by
      rw [real_inner_self_eq_norm_mul_norm, hx]; ring
    have hval : proj d ((X s (0 : Idx 1) : EucSpace d))
        ((((1 : ℕ) : ℝ)⁻¹) • ∑ j : Idx 1, ((X s j : EucSpace d))) = 0 := by
      simp only [Nat.cast_one, inv_one, one_smul, Fin.sum_univ_one]
      rw [proj, hxx, one_smul, sub_self]
    exact (hX s 0).congr_deriv hval
  exact is_const_of_deriv_eq_zero (fun s => (hzero s).differentiableAt)
    (fun s => (hzero s).deriv) t 0

/-- The hypothesis of `const_of_beta0Dynamics_one` is satisfiable: the constant
curve solves `e:Snonres0`. -/
example : beta0Dynamics 1 1 (fun _ _ => basePoint 0) := by
  intro t i
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1))
      ((basePoint 0 : EucSpace 1)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  have hval : proj 1 ((basePoint 0 : EucSpace 1))
      ((((1 : ℕ) : ℝ)⁻¹) • ∑ _j : Idx 1, ((basePoint 0 : EucSpace 1))) = 0 := by
    simp only [Nat.cast_one, inv_one, one_smul, Fin.sum_univ_one]
    rw [proj, hxx, one_smul, sub_self]
  exact (hasDerivAt_const t _).congr_deriv hval.symm

/-- **Equation (eq: youareawizardharry).**  For any `m`, there exists `β_m > 0`
such that for `β ∈ [0, β_m]`,

  `‖x_i^β(m) - x_i^0(m)‖ ≤ 1/8`.

Proved from the Grönwall bound `e:approxsphere`
(`Perspective.solutions_close_at_small_beta`, constant `C = 2 e²`):
`β_m = 1/(8 C e^{3m})` turns its bound at `t = m` into `1/8`.

Source: arXiv:2312.10794v5, §4, `eq: youareawizardharry`. -/
theorem distance_bound_at_time_m (m : ℕ) :
    ∃ βm : ℝ, 0 < βm ∧
      ∀ β : ℝ, 0 ≤ β → β ≤ βm →
        ∀ Xβ X0t : ℝ → SphereTuple d n,
          Xβ 0 = X0t 0 →
          Perspective.SA d n β Xβ → beta0Dynamics d n X0t →
          ∀ i : Idx n,
            ‖((Xβ (m : ℝ) i : EucSpace d))
              - ((X0t (m : ℝ) i : EucSpace d))‖ ≤ (1/8 : ℝ) := by
  obtain ⟨C, hC, hclose⟩ := solutions_close_at_small_beta d n
  have hE : (0 : ℝ) < Real.exp (3 * m) := Real.exp_pos _
  refine ⟨1 / (8 * C * Real.exp (3 * m)), by positivity, ?_⟩
  intro β hβ0 hβm Xβ X0t hinit hSA h0 i
  have h := hclose β hβ0 Xβ X0t hinit hSA h0 (m : ℝ) (Nat.cast_nonneg m) i
  have hmono : C * β * Real.exp (3 * m)
      ≤ C * (1 / (8 * C * Real.exp (3 * m))) * Real.exp (3 * m) := by
    gcongr
  refine h.trans (hmono.trans (le_of_eq ?_))
  field_simp

end Perspective
end Transformer
