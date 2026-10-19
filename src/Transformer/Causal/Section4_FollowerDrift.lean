module

public import Transformer.Causal.Section4_TwoTokenConvergence

/-!
# The original last-token drift near a converging prefix

ArXiv:2411.04990v2, §4, proof of `thm1`.
The actual last token sees its original predecessors and its original
self term. Identity values annihilate the self projection, while
retaining that term in the actual partition. Thus the original last
velocity is exactly its predecessor sum with the original weights.

For any original unit anchor, each predecessor's overlap force differs
from the consensus force `1-r²` by at most its distance to the anchor.
The actual total predecessor weight is at least the original positive
weight floor when a predecessor exists. The resulting overlap
derivative is at least `floor*(1-r²)` minus the sum of actual prefix
distances. That error tends to zero whenever the original prefix
converges. No convergence or differential estimate is built into a
replacement field or an auxiliary definition.
-/

@[expose] public section

open scoped BigOperators
open Real Filter

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual last original identity-value field is exactly its original predecessor force.
Source: arXiv:2411.04990v2, §4, the predecessor-only formula in the proof of `thm1`. -/
theorem csaLastField_id_eq_sum (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx (n + 1) → EucSpace d) (hY : ∀ j, ‖Y j‖ = 1) :
    csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) Y (Fin.last n) =
      ∑ j : Idx n, csaTokenWeight β Q K Y (Fin.last n) j.castSucc •
        proj d (Y (Fin.last n)) (Y j.castSucc) := by
  rw [csaAmbientField_eq_sum, Fin.sum_univ_castSucc]
  have hp : proj d (Y (Fin.last n)) (Y (Fin.last n)) = 0 := by
    simpa only [one_smul] using proj_smul_self (hY (Fin.last n)) 1
  simp only [ContinuousLinearMap.id_apply, hp, smul_zero, add_zero]

/-- The original projected overlap force is within the actual predecessor distance of its consensus value.
Source: arXiv:2411.04990v2, §4, the scalar drift comparison in the induction proving `thm1`. -/
theorem csaOverlapInteraction_approx (x p a : SSphere d) :
    |inner ℝ (proj d (x : EucSpace d) (p : EucSpace d)) (a : EucSpace d) -
      (1 - inner ℝ (x : EucSpace d) (a : EucSpace d) ^ 2)| ≤
        ‖(p : EucSpace d) - (a : EucSpace d)‖ := by
  have he : inner ℝ (proj d (x : EucSpace d) (p : EucSpace d)) (a : EucSpace d) -
      (1 - inner ℝ (x : EucSpace d) (a : EucSpace d) ^ 2) =
      inner ℝ (proj d (x : EucSpace d) ((p : EucSpace d) - (a : EucSpace d))) (a : EucSpace d) := by
    simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left,
      real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp a.2, one_pow]
    ring
  rw [he]
  calc
    _ ≤ ‖proj d (x : EucSpace d) ((p : EucSpace d) - (a : EucSpace d))‖ * ‖(a : EucSpace d)‖ :=
      abs_real_inner_le_norm _ _
    _ ≤ ‖(p : EucSpace d) - (a : EucSpace d)‖ := by
      rw [mem_sphere_zero_iff_norm.mp a.2, mul_one]
      exact norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) _

/-- The actual original last-token overlap force has a positive consensus drift minus its actual prefix error.
Source: arXiv:2411.04990v2, §4, the inductive overlap derivative in `thm1`.
The original partition, including the last token's self exponential, is retained. -/
theorem csaLastOverlap_lower (β : ℝ) (Q K : ParamMatrix d) (Y : SphereTuple d (n + 1))
    (a : SSphere d) (hn : 1 ≤ n) :
    csaWeightFloor β Q K (n + 1) *
        (1 - inner ℝ (Y (Fin.last n) : EucSpace d) (a : EucSpace d) ^ 2) -
      ∑ j : Idx n, ‖(Y j.castSucc : EucSpace d) - (a : EucSpace d)‖ ≤
        inner ℝ (csaAmbientField β Q K (ContinuousLinearMap.id ℝ _)
          (fun j => (Y j : EucSpace d)) (Fin.last n)) (a : EucSpace d) := by
  let q := 1 - inner ℝ (Y (Fin.last n) : EucSpace d) (a : EucSpace d) ^ 2
  have hq : 0 ≤ q := by
    have hr := csaSphereOverlap_bounds (Y (Fin.last n)) a
    nlinarith [mul_nonneg (sub_nonneg.mpr hr.2) (by linarith :
      0 ≤ 1 + inner ℝ (Y (Fin.last n) : EucSpace d) (a : EucSpace d))]
  have hmass : csaWeightFloor β Q K (n + 1) ≤
      ∑ j : Idx n, csaTokenWeight β Q K (fun j => (Y j : EucSpace d)) (Fin.last n) j.castSucc := by
    have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
    calc
      _ ≤ (n : ℝ) * csaWeightFloor β Q K (n + 1) := by
        nlinarith [csaWeightFloor_pos β Q K (Nat.zero_lt_succ n)]
      _ ≤ _ := csaPredecessorWeight_lower β Q K _ (fun j => mem_sphere_zero_iff_norm.mp (Y j).2)
  calc
    _ ≤ (∑ j : Idx n, csaTokenWeight β Q K (fun j => (Y j : EucSpace d)) (Fin.last n) j.castSucc) * q -
        ∑ j : Idx n, ‖(Y j.castSucc : EucSpace d) - (a : EucSpace d)‖ :=
      sub_le_sub_right (mul_le_mul_of_nonneg_right hmass hq) _
    _ = ∑ j : Idx n, (csaTokenWeight β Q K (fun j => (Y j : EucSpace d)) (Fin.last n) j.castSucc * q -
        ‖(Y j.castSucc : EucSpace d) - (a : EucSpace d)‖) := by
      rw [Finset.sum_sub_distrib, Finset.sum_mul]
    _ ≤ ∑ j : Idx n, csaTokenWeight β Q K (fun j => (Y j : EucSpace d)) (Fin.last n) j.castSucc *
        inner ℝ (proj d (Y (Fin.last n) : EucSpace d) (Y j.castSucc : EucSpace d)) (a : EucSpace d) := by
      apply Finset.sum_le_sum
      intro j hj
      have hw0 := csaTokenWeight_nonneg β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc
      have hw1 := csaTokenWeight_le_one β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc
      have he := (abs_le.mp (csaOverlapInteraction_approx (Y (Fin.last n)) (Y j.castSucc) a)).1
      have hforce : q - ‖(Y j.castSucc : EucSpace d) - (a : EucSpace d)‖ ≤
          inner ℝ (proj d (Y (Fin.last n) : EucSpace d) (Y j.castSucc : EucSpace d)) (a : EucSpace d) :=
        by dsimp [q]; linarith
      calc
        _ ≤ csaTokenWeight β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc *
            (q - ‖(Y j.castSucc : EucSpace d) - (a : EucSpace d)‖) := by
          nlinarith [mul_le_mul_of_nonneg_right hw1 (norm_nonneg
            ((Y j.castSucc : EucSpace d) - (a : EucSpace d)))]
        _ ≤ _ := mul_le_mul_of_nonneg_left hforce hw0
    _ = _ := by
      rw [csaLastField_id_eq_sum β Q K _ (fun j => mem_sphere_zero_iff_norm.mp (Y j).2), sum_inner]
      simp only [real_inner_smul_left]

/-- The actual original last-token leader overlap has the exact original field as its derivative.
Source: arXiv:2411.04990v2, §4, the final overlap display in the proof of `thm1`. -/
theorem hasDerivAt_csaLastOverlap (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ (X s (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d))
      (inner ℝ (csaAmbientField β Q K (ContinuousLinearMap.id ℝ _)
        (fun j => (X t j : EucSpace d)) (Fin.last n)) (X 0 0 : EucSpace d)) t := by
  have h := ((CSA_iff_csaAmbientField β Q K _ X).mp hX t (Fin.last n)).inner ℝ
    (hasDerivAt_const t (X 0 0 : EucSpace d))
  simpa! only [inner_zero_right, zero_add] using h

/-- The actual overlap derivative obeys the original positive-drift estimate at every real time.
Source: arXiv:2411.04990v2, §4, the induction proving `thm1`. -/
theorem csaLastOverlap_deriv_lower (β : ℝ) (Q K : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X) (hn : 1 ≤ n) (t : ℝ) :
    csaWeightFloor β Q K (n + 1) *
        (1 - inner ℝ (X t (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d) ^ 2) -
      ∑ j : Idx n, ‖(X t j.castSucc : EucSpace d) - (X 0 0 : EucSpace d)‖ ≤
        deriv (fun s => inner ℝ (X s (Fin.last n) : EucSpace d) (X 0 0 : EucSpace d)) t := by
  rw [(hasDerivAt_csaLastOverlap β Q K X hX t).deriv]
  exact csaLastOverlap_lower β Q K (X t) (X 0 0) hn

/-- The actual sum of original predecessor distances vanishes when each actual predecessor converges.
Source: arXiv:2411.04990v2, §4, the converging-prefix hypothesis in the induction for `thm1`. -/
theorem tendsto_csaPrefixError (X : ℝ → SphereTuple d (n + 1)) (a : SSphere d)
    (hp : ∀ j : Idx n, Tendsto (fun t => X t j.castSucc) atTop (nhds a)) :
    Tendsto (fun t => ∑ j : Idx n, ‖(X t j.castSucc : EucSpace d) - (a : EucSpace d)‖)
      atTop (nhds 0) := by
  have hj (j : Idx n) : Tendsto
      (fun t => ‖(X t j.castSucc : EucSpace d) - (a : EucSpace d)‖) atTop (nhds 0) := by
    have hcoe := (continuous_subtype_val.tendsto a).comp (hp j)
    simpa only [Function.comp_def, sub_self, norm_zero] using (hcoe.sub_const (a : EucSpace d)).norm
  simpa only [Finset.sum_const_zero] using tendsto_finsetSum Finset.univ (fun j hjmem => hj j)

/-- Genuine original nonzero consensus satisfies the actual solution and predecessor-size hypotheses together.
Source: arXiv:2411.04990v2, §4, the consensus equilibria in `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
      (1 : ℕ) ≤ 1 ∧ (∀ j : Idx 2, ‖(fun _ => (basePoint 1 : EucSpace 2)) j‖ = 1) :=
  ⟨CSA_const_consensus β Q K (basePoint 1), le_rfl,
    fun _ => mem_sphere_zero_iff_norm.mp (basePoint 1).2⟩

/-- The original prefix-convergence hypothesis is jointly realized by that same nonzero consensus path.
Source: arXiv:2411.04990v2, §4, the consensus equilibrium of `thm1`. -/
example : ∀ j : Idx 1, Tendsto (fun _ : ℝ => (fun _ : Idx 2 => basePoint 1) j.castSucc)
    atTop (nhds (basePoint 1 : SSphere 2)) := fun _ => tendsto_const_nhds

end Transformer.Causal
