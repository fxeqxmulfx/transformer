/-
# Original causal projected interactions and the fixed leader

ArXiv:2411.04990v2, §4, proof of `thm1`, and §2, `eqn:csa`.
The original field is the sum of its actual normalized projected
interactions. At identity values its own current-token projection
vanishes, leaving exactly the original predecessor forces.

The first original row sees only its current token, independently of
the original query and key matrices. Its equation is the manuscript's
single-token ODE. At identity values the actual spherical leader has
zero velocity and remains at its original initial position. These
identities retain the original exponential normalization and every
masked term, so subsequent overlap estimates concern the actual flow.
-/

import Transformer.Causal.Section4_FlowPrefix
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual original causal field is exactly the sum of its projected original interactions.
Source: arXiv:2411.04990v2, §4, the first display in the proof of `thm1`. -/
theorem csaAmbientField_eq_sum (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) : csaAmbientField β Q K V Y i =
      ∑ j, csaTokenWeight β Q K Y i j • proj d (Y i) (V (Y j)) := by
  simp only [csaAmbientField, csaAverage, proj, inner_sum, real_inner_smul_right,
    Finset.sum_smul, smul_sub, smul_smul, Finset.sum_sub_distrib]

/-- The current-token term of the original identity-value field vanishes on the original sphere.
Source: arXiv:2411.04990v2, §4, the predecessor-only formula for `thm1`. -/
theorem csaAmbientField_id_eq_erase (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) (hi : ‖Y i‖ = 1) :
    csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) Y i =
      ∑ j ∈ Finset.univ.erase i, csaTokenWeight β Q K Y i j • proj d (Y i) (Y j) := by
  rw [csaAmbientField_eq_sum, ← Finset.add_sum_erase Finset.univ
    (fun j => csaTokenWeight β Q K Y i j • proj d (Y i) ((ContinuousLinearMap.id ℝ _) (Y j)))
    (Finset.mem_univ i)]
  have hp : proj d (Y i) (Y i) = 0 := by simpa only [one_smul] using proj_smul_self hi 1
  simp only [ContinuousLinearMap.id_apply, hp, smul_zero, zero_add]

/-- The first original row partition is its single original self exponential.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, the fixed leader in `thm1`. -/
theorem csaPartition_first (β : ℝ) (Q K : ParamMatrix d) (Y : Idx (n + 1) → EucSpace d) :
    csaPartition β Q K Y 0 = Real.exp (β * inner ℝ (Q (Y 0)) (K (Y 0))) := by
  simp [csaPartition]

/-- The actual first original row is exactly its own coordinate probability vector.
Source: arXiv:2411.04990v2, §4, the leader row in the proof of `thm1`. -/
theorem csaTokenWeight_first (β : ℝ) (Q K : ParamMatrix d) (Y : Idx (n + 1) → EucSpace d)
    (j : Idx (n + 1)) : csaTokenWeight β Q K Y 0 j = (Pi.single 0 1 : Idx (n + 1) → ℝ) j := by
  by_cases hj : j = 0
  · subst j
    simp [csaTokenWeight, csaPartition_first]
  · have hv : ¬ (j : ℕ) ≤ 0 := by
      intro h
      apply hj
      apply Fin.eq_of_val_eq
      exact Nat.eq_zero_of_le_zero h
    simp [csaTokenWeight, hv, hj]

/-- The actual first original row average is the value image of its own original leader token.
Source: arXiv:2411.04990v2, §4, the first-token equation in `thm1`. -/
theorem csaAverage_first (β : ℝ) (Q K V : ParamMatrix d) (Y : Idx (n + 1) → EucSpace d) :
    csaAverage β Q K V Y 0 = V (Y 0) := by
  simp [csaAverage, csaTokenWeight_first]

/-- The original first-token field is exactly the original single-token projected value field.
Source: arXiv:2411.04990v2, §3 and §4, the leader equation in the proof of `thm1`. -/
theorem csaAmbientField_first (β : ℝ) (Q K V : ParamMatrix d) (Y : Idx (n + 1) → EucSpace d) :
    csaAmbientField β Q K V Y 0 = proj d (Y 0) (V (Y 0)) := by
  rw [csaAmbientField, csaAverage_first]

/-- Every actual original leader solves the original one-token value equation, with arbitrary original scores.
Source: arXiv:2411.04990v2, §3, `lemma1`, and §4, proof of `thm1`. -/
theorem hasDerivAt_csaLeader (β : ℝ) (Q K V : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K V X) (t : ℝ) :
    HasDerivAt (fun s => (X s 0 : EucSpace d)) (proj d (X t 0 : EucSpace d) (V (X t 0 : EucSpace d))) t := by
  have h := (CSA_iff_csaAmbientField β Q K V X).mp hX t 0
  rwa [csaAmbientField_first] at h

/-- The actual first original token satisfies the manuscript's full single-token ODE predicate.
Source: arXiv:2411.04990v2, §3, `lemma1`, as used in §4, `thm1`. -/
theorem csaLeader_singleTokenODE (β : ℝ) (Q K V : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K V X) : singleTokenODE d V (fun t => (X t 0 : EucSpace d)) :=
  fun t => hasDerivAt_csaLeader β Q K V X hX t

/-- The actual original leader has zero velocity at identity values.
Source: arXiv:2411.04990v2, §4, the fixed first token in `thm1`. -/
theorem hasDerivAt_csaLeader_id (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X) (t : ℝ) :
    HasDerivAt (fun s => (X s 0 : EucSpace d)) 0 t := by
  have h := hasDerivAt_csaLeader β Q K _ X hX t
  have hp : proj d (X t 0 : EucSpace d) (X t 0 : EucSpace d) = 0 := by
    simpa only [one_smul] using proj_smul_self (mem_sphere_zero_iff_norm.mp (X t 0).2) 1
  simpa only [ContinuousLinearMap.id_apply, hp] using h

/-- The actual original leader stays exactly at its original initial token for every real time.
Source: arXiv:2411.04990v2, §4, `x₁(t)=x₁(0)` in the proof of `thm1`. -/
theorem csaLeader_constant (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X) (t : ℝ) : X t 0 = X 0 0 := by
  apply Subtype.ext
  exact is_const_of_deriv_eq_zero (fun s => (hasDerivAt_csaLeader_id β Q K X hX s).differentiableAt)
    (fun s => (hasDerivAt_csaLeader_id β Q K X hX s).deriv) t 0

/-- The constructed original causal flow keeps its actual first token fixed at identity values.
Source: arXiv:2411.04990v2, §4, the conditioning leader of `thm1`. -/
theorem csaFlow_leader (β : ℝ) (Q K : ParamMatrix d) (X₀ : SphereTuple d (n + 1)) (t : ℝ) :
    csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t 0 = X₀ 0 := by
  rw [csaLeader_constant β Q K _ (csaFlow_spec β Q K _ X₀).2 t, csaFlow_zero]

/-- The actual original leader-anchored overlap derivative is the manuscript's causal interaction sum.
Source: arXiv:2411.04990v2, §4, the final displayed derivative in the proof of `thm1`.
The anchor is the actual fixed first initial token and every original masked row weight remains. -/
theorem hasDerivAt_csaOverlap_id (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X) (i : Idx (n + 1)) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ (X s i : EucSpace d) (X 0 0 : EucSpace d))
      (∑ j, csaTokenWeight β Q K (fun l => (X t l : EucSpace d)) i j *
        (inner ℝ (X t j : EucSpace d) (X 0 0 : EucSpace d) -
          inner ℝ (X t i : EucSpace d) (X t j : EucSpace d) *
            inner ℝ (X t i : EucSpace d) (X 0 0 : EucSpace d))) t := by
  have h := ((CSA_iff_csaAmbientField β Q K _ X).mp hX t i).inner ℝ
    (hasDerivAt_const t (X 0 0 : EucSpace d))
  simpa! only [inner_zero_right, zero_add, csaAmbientField_eq_sum, sum_inner,
    real_inner_smul_left, ContinuousLinearMap.id_apply, proj, inner_sub_left] using h

/-- The original causal velocity is tangent at each original unit token, for every original value matrix.
Source: arXiv:2411.04990v2, §2, the projection in `eqn:csa`. -/
theorem inner_csaAmbientField_eq_zero (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) (hi : ‖Y i‖ = 1) :
    inner ℝ (Y i) (csaAmbientField β Q K V Y i) = 0 :=
  inner_proj_eq_zero hi (csaAverage β Q K V Y i)

/-- With a single original token and identity values the entire actual causal flow is stationary.
Source: arXiv:2411.04990v2, §4, the base leader of the induction proving `thm1`. -/
theorem csaFlow_singleton_id (β : ℝ) (Q K : ParamMatrix d) (X₀ : SphereTuple d 1) (t : ℝ) :
    csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t = X₀ := by
  funext i
  rw [Subsingleton.elim i (0 : Idx 1), csaFlow_leader]

/-- A genuine nonzero original consensus trajectory jointly satisfies every leader and self-projection premise.
Source: arXiv:2411.04990v2, §4, the consensus equilibria in the proof of `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    ‖(basePoint 1 : EucSpace 2)‖ = 1 :=
  ⟨CSA_const_consensus β Q K (basePoint 1), mem_sphere_zero_iff_norm.mp (basePoint 1).2⟩

end Transformer.Causal
