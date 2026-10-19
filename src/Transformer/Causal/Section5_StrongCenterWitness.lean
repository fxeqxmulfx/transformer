module

public import Transformer.Causal.Metastability

/-!
# An actual strong center with a nonempty predecessor set

ArXiv:2411.04990v2, §5, `lemma:meta`, `eq:renyi`, and `eq:time_bd`.
The two antipodal tokens form a stationary solution of the actual CSA
equation. At `β = c = 1`, `ε = 1/4`, and `T = 1/100`, the second token
satisfies every strong-center hypothesis and has one preceding token.
This witnesses the velocity and barrier application as well as the
original metastability lemma.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

/-- Two antipodal actual circle tokens, parametrized by their first token.
Source: arXiv:2411.04990v2, `eqn:csa`, at `Q = K = V = I₂`. -/
noncomputable def antipodalCircleTuple (u : SSphere 2) : SphereTuple 2 2 :=
  ![u, ⟨-(u : EucSpace 2), by
    simp only [mem_sphere_zero_iff_norm, norm_neg, mem_sphere_zero_iff_norm.mp u.2]⟩]

/-- The actual two-token antipodal configuration is stationary for CSA
at every temperature. Source: arXiv:2411.04990v2, `eqn:csa`.
Each attention average is radial, so its tangent projection vanishes. -/
theorem csa_const_antipodal (β : ℝ) (u : SSphere 2) :
    CSA 2 2 β (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) (fun _ => antipodalCircleTuple u) := by
  have hu : ‖(u : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp u.2
  intro t k
  have hproj : proj 2 ((antipodalCircleTuple u) k : EucSpace 2)
      ((∑ j : Idx 2, if (j : ℕ) ≤ (k : ℕ) then
          Real.exp (β * inner (𝕜 := ℝ)
            ((antipodalCircleTuple u) k : EucSpace 2)
            ((antipodalCircleTuple u) j : EucSpace 2)) else 0)⁻¹ •
        ∑ j : Idx 2, if (j : ℕ) ≤ (k : ℕ) then
          Real.exp (β * inner (𝕜 := ℝ)
            ((antipodalCircleTuple u) k : EucSpace 2)
            ((antipodalCircleTuple u) j : EucSpace 2)) •
              ((antipodalCircleTuple u) j : EucSpace 2) else 0) = 0 := by
    fin_cases k <;>
      simp [antipodalCircleTuple, Fin.sum_univ_two, proj, inner_smul_right,
        inner_add_right, hu, smul_add, smul_smul]
    module
  simpa only [ContinuousLinearMap.id_apply, hproj] using
    hasDerivAt_const t ((antipodalCircleTuple u) k : EucSpace 2)

/-- The two actual antipodal tokens have chord distance exactly two.
Source: arXiv:2411.04990v2, §5, the chord separation in `eq:renyi`. -/
theorem antipodalCircleTuple_gap (u : SSphere 2) :
    ‖((antipodalCircleTuple u) 1 : EucSpace 2) -
      ((antipodalCircleTuple u) 0 : EucSpace 2)‖ = 2 := by
  change ‖-(u : EucSpace 2) - (u : EucSpace 2)‖ = 2
  have heq : -(u : EucSpace 2) - (u : EucSpace 2) = (-2 : ℝ) • (u : EucSpace 2) := by module
  rw [heq, norm_smul, mem_sphere_zero_iff_norm.mp u.2]
  norm_num

/-- **Every hypothesis of the strong-center comparison is satisfiable
with a preceding token.** The original parameters are `β = c = 1`,
`ε = 1/4`, `T = 1/100`; both actual tokens are antipodal and stationary.
Source: arXiv:2411.04990v2, §5, `lemma:meta`, `eq:c`, `eq:renyi`, and
`eq:time_bd`. -/
theorem strong_center_witness : ∃ X : ℝ → SphereTuple 2 2,
    CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 / 100 ∧ (0 : ℕ) < 1 ∧
    (∀ i : Idx 2, i < 1 →
      (1 : ℝ) * (1 + 2 * (1 / 4)) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) <
        ‖(X 0 1 : EucSpace 2) - (X 0 i : EucSpace 2)‖) ∧
    Real.sqrt 1 * Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) < 1 ∧
    (1 : ℝ) / 100 * 1 * h_pot 1 (1 * (1 : ℝ) ^ (-(1 / 2 : ℝ))) <
      (1 : ℝ) / 4 * 1 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  let X : ℝ → SphereTuple 2 2 := fun _ => antipodalCircleTuple (basePoint 1)
  obtain ⟨τ, hτ0, hτπ, hcos, _, hτ1, _⟩ := h_pot_unimodal 1 le_rfl
  have hpeak : Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) < 1 := by
    rw [← hcos, Real.arccos_cos hτ0.le hτπ.le]
    simpa only [Real.one_rpow] using hτ1
  have hh : h_pot 1 1 ≤ 1 := by
    have he : Real.exp (1 * (Real.cos 1 - 1)) ≤ 1 :=
      Real.exp_le_one_iff.mpr (by linarith [Real.cos_le_one (x := 1)])
    have hprod := mul_le_mul_of_nonneg_left (Real.sin_le_one (x := 1))
      (Real.exp_pos (1 * (Real.cos 1 - 1))).le
    have hprod' : h_pot 1 1 ≤ Real.exp (1 * (Real.cos 1 - 1)) := by
      simpa only [h_pot, mul_one] using hprod
    exact hprod'.trans he
  refine ⟨X, csa_const_antipodal 1 (basePoint 1), by norm_num, by norm_num,
    by norm_num, by norm_num, ?_, by simpa only [Real.sqrt_one, one_mul] using hpeak, ?_⟩
  · intro i hi
    have hi0 : i = 0 := by fin_cases i <;> simp_all
    rw [hi0]
    change (1 : ℝ) * (1 + 2 * (1 / 4)) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) <
      ‖((antipodalCircleTuple (basePoint 1)) 1 : EucSpace 2) -
        ((antipodalCircleTuple (basePoint 1)) 0 : EucSpace 2)‖
    rw [antipodalCircleTuple_gap]
    norm_num
  · norm_num only [Real.one_rpow, mul_one, one_mul]
    linarith

/-- The hypotheses of `strong_center_displacement_at_scale` hold simultaneously,
including genuine separation from one predecessor. -/
example : ∃ X : ℝ → SphereTuple 2 2,
    CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X ∧
    (∀ i : Idx 2, i < 1 →
      (1 : ℝ) * (1 + 2 * (1 / 4)) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) <
        ‖(X 0 1 : EucSpace 2) - (X 0 i : EucSpace 2)‖) ∧
    (∀ t ∈ Set.Icc (0 : ℝ) (1 / 100),
      ‖(X t 1 : EucSpace 2) - (X 0 1 : EucSpace 2)‖ < 1 / 4) := by
  obtain ⟨X, hX, hβ, hε, _, _, hsep, hc, hT⟩ := strong_center_witness
  refine ⟨X, hX, hsep, fun t ht => ?_⟩
  have h := strong_center_displacement_at_scale 1 1 (1 / 4) hβ hβ hε X hX 1 hsep hc (1 / 100)
    (by simpa using hT)
    t ht.1 ht.2
  simpa only [Real.one_rpow, mul_one] using h

/-- The hypotheses of `strong_center_displacement_le` hold simultaneously
for the same actual CSA trajectory, at radius `r = 1`. -/
example : ∃ X : ℝ → SphereTuple 2 2,
    CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X ∧
    (∀ t ∈ Set.Icc (0 : ℝ) (1 / 100),
      ‖(X t 1 : EucSpace 2) - (X 0 1 : EucSpace 2)‖ ≤ h_pot 1 1 * t) := by
  obtain ⟨X, hX, hβ, hε, htime, hk, hsep, hc, hT⟩ := strong_center_witness
  have hrπ : (1 : ℝ) ≤ π := by linarith [Real.pi_gt_three]
  have hrpeak : Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) ≤ 1 := by
    simpa only [Real.sqrt_one, one_mul] using hc.le
  refine ⟨X, hX, ?_⟩
  have h := strong_center_displacement_le 1 hβ 1 (1 / 4) hβ hε hrπ hrpeak X hX 1 hk
    (by simpa only [Real.one_rpow, mul_one] using hsep) (1 / 100) htime
    (by simpa using hT)
  simpa using h

/-- The original `lemma_meta` also has a witness with a nonempty
preceding-token range. Its constant index sequence selects the second
token; the quantified separation conditions hold at every index.
Source: arXiv:2411.04990v2, §5, `lemma:meta`. -/
example : ∃ X : ℝ → SphereTuple 2 2,
    CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X ∧
    ∀ t ∈ Set.Icc (0 : ℝ) (1 / 100),
      ‖(X t 1 : EucSpace 2) - (X 0 1 : EucSpace 2)‖ < 1 / 4 := by
  obtain ⟨X, hX, hβ, hε, _, _, hsep, hc, hT⟩ := strong_center_witness
  refine ⟨X, hX, fun t ht => ?_⟩
  have h := lemma_meta 2 1 1 (1 / 4) hβ hβ hε X hX (fun _ => 1)
    (fun _ => by decide) (fun _ i hi => hsep i hi) hc 0 (1 / 100) (by simpa using hT) t ht.1 ht.2
  norm_num at h ⊢
  exact h

end Transformer.Causal
