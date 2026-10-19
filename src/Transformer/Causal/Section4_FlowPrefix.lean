module

public import Transformer.Causal.Section4_SphereFlow

/-!
# Time translation and exact prefix consistency of the original causal flow

ArXiv:2411.04990v2, §4, proof of `thm1`.
The original autonomous causal equation is invariant under translating
time, and its proved uniqueness gives the flow composition identity.
Every fixed-time flow map is therefore injective, with the opposite
time supplying its inverse on each actual trajectory.

Removing the last token leaves every earlier original row partition,
weight and projected field unchanged. Consequently every prefix of
an original solution solves the smaller original causal system, and
the deterministic constructed flow respects those prefixes exactly.
This is the triangular dependence used to condition on earlier tokens.
The proof retains the actual exponential scores and current-token term;
no prefix consistency or flow property is built into a definition.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {d n : ℕ}

/-- Time translates of original causal trajectories solve the same original autonomous equation.
Source: arXiv:2411.04990v2, §4, the autonomous system in the proof of `thm1`. -/
theorem CSA_time_shift (β : ℝ) (Q K V : ParamMatrix d) (X : ℝ → SphereTuple d n)
    (hX : CSA d n β Q K V X) (s : ℝ) : CSA d n β Q K V (fun t => X (t + s)) := by
  intro t i
  have h := (hX (t + s) i).scomp t ((hasDerivAt_id t).add_const s)
  simpa! only [Function.comp_def, id_eq, one_smul] using h

/-- The actual original causal flow satisfies its exact time-composition identity.
Source: arXiv:2411.04990v2, §4, the deterministic autonomous flow of `thm1`. -/
theorem csaFlow_add (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d n) (s t : ℝ) :
    csaFlow β Q K V (csaFlow β Q K V X₀ s) t = csaFlow β Q K V X₀ (t + s) := by
  have h := csaFlow_eq_of_CSA β Q K V (fun t => csaFlow β Q K V X₀ (t + s))
    (CSA_time_shift β Q K V _ (csaFlow_spec β Q K V X₀).2 s)
  simpa only [zero_add] using congrFun h t

/-- Original causal flow maps at every real time are injective on the actual initial state space.
Source: arXiv:2411.04990v2, §4, invertibility of the original autonomous flow in `thm1`. -/
theorem csaFlow_injective (β : ℝ) (Q K V : ParamMatrix d) (s : ℝ) :
    Function.Injective (fun X₀ : SphereTuple d n => csaFlow β Q K V X₀ s) := by
  intro X₀ Y₀ h
  have he := congrArg (fun Z => csaFlow β Q K V Z (-s)) h
  simpa only [csaFlow_add, neg_add_cancel, csaFlow_zero] using he

/-- The original causal flow maps are onto, with opposite-time flow providing a preimage.
Source: arXiv:2411.04990v2, §4, invertibility of the original flow used in `thm1`. -/
theorem csaFlow_surjective (β : ℝ) (Q K V : ParamMatrix d) (s : ℝ) :
    Function.Surjective (fun X₀ : SphereTuple d n => csaFlow β Q K V X₀ s) := by
  intro X₀
  refine ⟨csaFlow β Q K V X₀ (-s), ?_⟩
  change csaFlow β Q K V (csaFlow β Q K V X₀ (-s)) s = X₀
  rw [csaFlow_add, add_neg_cancel, csaFlow_zero]

/-- Each fixed-time original causal flow map is a bijection on the original sphere-tuple state space.
Source: arXiv:2411.04990v2, §4, the invertible flow used in the proof of `thm1`. -/
theorem csaFlow_bijective (β : ℝ) (Q K V : ParamMatrix d) (s : ℝ) :
    Function.Bijective (fun X₀ : SphereTuple d n => csaFlow β Q K V X₀ s) :=
  ⟨csaFlow_injective β Q K V s, csaFlow_surjective β Q K V s⟩

/-- An earlier original row partition is unchanged by removing the last original token.
Source: arXiv:2411.04990v2, §4, independence of a causal prefix in the proof of `thm1`. -/
theorem csaPartition_castSucc (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx (n + 1) → EucSpace d) (i : Idx n) :
    csaPartition β Q K Y i.castSucc = csaPartition β Q K (fun j => Y j.castSucc) i := by
  unfold csaPartition
  rw [Fin.sum_univ_castSucc]
  simp only [Fin.val_castSucc, Fin.val_last, ite_eq_right (not_le.mpr i.isLt), add_zero]

/-- Removing the last original token leaves every retained original row weight unchanged.
Source: arXiv:2411.04990v2, §4, the causal prefix dependence of `thm1`. -/
theorem csaTokenWeight_castSucc (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx (n + 1) → EucSpace d) (i j : Idx n) :
    csaTokenWeight β Q K Y i.castSucc j.castSucc =
      csaTokenWeight β Q K (fun l => Y l.castSucc) i j := by
  simp only [csaTokenWeight, Fin.val_castSucc, csaPartition_castSucc]

/-- The last original token has zero original weight in every earlier row.
Source: arXiv:2411.04990v2, §2, the original mask in `eqn:csa`. -/
theorem csaTokenWeight_last_above (β : ℝ) (Q K : ParamMatrix d)
    (Y : Idx (n + 1) → EucSpace d) (i : Idx n) :
    csaTokenWeight β Q K Y i.castSucc (Fin.last n) = 0 := by
  simp only [csaTokenWeight, Fin.val_last, Fin.val_castSucc, ite_eq_right (not_le.mpr i.isLt)]

/-- The actual original value average of every earlier row is exactly the smaller original system's average.
Source: arXiv:2411.04990v2, §4, the triangular original field of `thm1`. -/
theorem csaAverage_castSucc (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx (n + 1) → EucSpace d) (i : Idx n) :
    csaAverage β Q K V Y i.castSucc = csaAverage β Q K V (fun j => Y j.castSucc) i := by
  rw [csaAverage, Fin.sum_univ_castSucc]
  simp only [csaTokenWeight_castSucc, csaTokenWeight_last_above, zero_smul, add_zero, csaAverage]

/-- The original projected field respects the actual original causal prefix.
Source: arXiv:2411.04990v2, §4, the triangular ODE used in `thm1`. -/
theorem csaAmbientField_castSucc (β : ℝ) (Q K V : ParamMatrix d)
    (Y : Idx (n + 1) → EucSpace d) (i : Idx n) :
    csaAmbientField β Q K V Y i.castSucc = csaAmbientField β Q K V (fun j => Y j.castSucc) i := by
  rw [csaAmbientField, csaAverage_castSucc]
  rfl

/-- Every prefix of an actual original solution solves the smaller original causal equation.
Source: arXiv:2411.04990v2, §4, conditioning on a causal prefix in the proof of `thm1`. -/
theorem CSA_castSucc (β : ℝ) (Q K V : ParamMatrix d) (X : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K V X) : CSA d n β Q K V (fun t j => X t j.castSucc) := by
  apply (CSA_iff_csaAmbientField β Q K V _).mpr
  intro t i
  have h := (CSA_iff_csaAmbientField β Q K V X).mp hX t i.castSucc
  rwa [csaAmbientField_castSucc] at h

/-- The constructed original deterministic flow has exactly the smaller flow as its original prefix.
Source: arXiv:2411.04990v2, §4, prefix independence used by the induction in `thm1`. -/
theorem csaFlow_castSucc (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d (n + 1))
    (t : ℝ) (i : Idx n) :
    csaFlow β Q K V X₀ t i.castSucc = csaFlow β Q K V (fun j => X₀ j.castSucc) t i := by
  have h := csaFlow_eq_of_CSA β Q K V (fun t j => csaFlow β Q K V X₀ t j.castSucc)
    (CSA_castSucc β Q K V _ (csaFlow_spec β Q K V X₀).2)
  have he := congrFun (congrFun h t) i
  simpa only [csaFlow_zero] using he.symm

/-- Original causal solutions with the same original initial prefix agree on that entire prefix at every time.
Source: arXiv:2411.04990v2, §4, the conditional-prefix argument in the proof of `thm1`. -/
theorem CSA_prefix_eq_of_initial_prefix (β : ℝ) (Q K V : ParamMatrix d)
    (X Y : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X)
    (hY : CSA d (n + 1) β Q K V Y) (h0 : ∀ j : Idx n, X 0 j.castSucc = Y 0 j.castSucc) :
    ∀ t : ℝ, ∀ j : Idx n, X t j.castSucc = Y t j.castSucc := by
  have h := CSA_unique β Q K V (fun t j => X t j.castSucc) (fun t j => Y t j.castSucc)
    (CSA_castSucc β Q K V X hX) (CSA_castSucc β Q K V Y hY) (funext h0)
  exact fun t j => congrFun (congrFun h t) j

/-- Changing only the last original initial token leaves the entire earlier original flow unchanged.
Source: arXiv:2411.04990v2, §4, independence of the preceding-token system in `thm1`. -/
theorem csaFlow_prefix_congr (β : ℝ) (Q K V : ParamMatrix d) (X₀ Y₀ : SphereTuple d (n + 1))
    (h0 : ∀ j : Idx n, X₀ j.castSucc = Y₀ j.castSucc) (t : ℝ) (i : Idx n) :
    csaFlow β Q K V X₀ t i.castSucc = csaFlow β Q K V Y₀ t i.castSucc := by
  rw [csaFlow_castSucc, csaFlow_castSucc, funext h0]

/-- An actual nonzero consensus trajectory supplies the complete original dynamical inputs
for time translation, uniqueness, prefix restriction and the flow identities at once. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 3 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    (∀ j : Idx 2, (fun _ : Idx 3 => basePoint 1) j.castSucc =
      (fun _ : Idx 3 => basePoint 1) j.castSucc) :=
  ⟨CSA_const_consensus β Q K (basePoint 1), CSA_const_consensus β Q K (basePoint 1), fun _ => rfl⟩

end Transformer.Causal
