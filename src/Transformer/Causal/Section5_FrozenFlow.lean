/-
# Global actual flows for the original frozen-token angular system

ArXiv:2411.04990v2, §5, Theorem `thm:fixed_centers`.
The actual original angular field is smooth and bounded in maximum
norm by one. To apply the proved linear-growth existence theorem,
adjoin a scalar coordinate `s` and use the auxiliary field
`(phi,s) ↦ (s • frozenVelocity(phi),0)`. Its norm is at most the
norm of `(phi,s)`, and it is smooth. From an initial scalar one,
the scalar derivative is zero, so it stays exactly one and the
first component solves precisely the original angular equation.

This constructs a genuine global original trajectory from every
original angular tuple. Local Lipschitz regularity of the original
field gives uniqueness. The resulting deterministic original flow
retains all actual frozen centers, original center masses, original
causal interactions, and the original partition. No convergence or
stability property is assumed in constructing it.
-/

import Transformer.Causal.Section5_FrozenField
import Transformer.Landscape.GlobalFlow.Existence
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Linear

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n m : ℕ}

/-- An auxiliary field whose scalar-one slice is exactly the actual original frozen-token equation.
Source: arXiv:2411.04990v2, §5, global original dynamics of `thm:fixed_centers`.
The extra coordinate has zero derivative and is proved to stay one. -/
noncomputable def frozenLiftedField (β : ℝ) (θ a : Idx m → ℝ)
    (p : (Idx n → ℝ) × ℝ) : (Idx n → ℝ) × ℝ := (p.2 • frozenVelocity n m β θ a p.1, 0)

/-- The auxiliary field is smooth because the actual original frozen-token field is smooth.
Source: arXiv:2411.04990v2, §5, the original smooth normalized dynamics in `thm:fixed_centers`. -/
theorem contDiff_frozenLiftedField (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    ContDiff ℝ (⊤ : ℕ∞) (frozenLiftedField (n := n) β θ a) := by
  have hf := (contDiff_frozenVelocity (n := n) β θ a ha).comp
    (contDiff_fst : ContDiff ℝ (⊤ : ℕ∞) (Prod.fst : (Idx n → ℝ) × ℝ → Idx n → ℝ))
  exact (contDiff_snd.smul hf).prodMk contDiff_const

/-- The auxiliary field has genuine linear growth, from the proved bound on the actual original angular velocity.
Source: arXiv:2411.04990v2, §5, global existence for the original dynamics in `thm:fixed_centers`. -/
theorem norm_frozenLiftedField_le (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : (Idx n → ℝ) × ℝ) : ‖frozenLiftedField β θ a p‖ ≤ ‖p‖ := by
  rw [frozenLiftedField, Prod.norm_def, norm_smul, norm_zero, Prod.norm_def]
  apply max_le
  · calc
      ‖p.2‖ * ‖frozenVelocity n m β θ a p.1‖ ≤ ‖p.2‖ * 1 :=
        mul_le_mul_of_nonneg_left (norm_frozenVelocity_le_one β θ a ha p.1) (norm_nonneg _)
      _ ≤ max ‖p.1‖ ‖p.2‖ := by rw [mul_one]; exact le_max_right _ _
  · exact (norm_nonneg p.2).trans (le_max_right _ _)

/-- Every original lifted angular initial tuple has a genuine global solution of the original frozen-token equation.
Source: arXiv:2411.04990v2, §5, the global original flow in `thm:fixed_centers`. -/
theorem exists_frozenODE (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (φ₀ : Idx n → ℝ) :
    ∃ φ : ℝ → Idx n → ℝ, φ 0 = φ₀ ∧ ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t := by
  have hLip := ((contDiff_frozenLiftedField (n := n) β θ a ha).of_le (by simp)).locallyLipschitz
  obtain ⟨Γ, hΓ0, hΓ⟩ := Landscape.GlobalFlow.exists_global hLip zero_le_one
    (fun p => by simpa only [one_mul] using norm_frozenLiftedField_le β θ a ha p) (φ₀, (1 : ℝ))
  have hsnd (t : ℝ) : HasDerivAt (fun s => (Γ s).2) 0 t := by
    have h := (ContinuousLinearMap.snd ℝ (Idx n → ℝ) ℝ).hasFDerivAt.comp_hasDerivAt t (hΓ t)
    change HasDerivAt (fun s => (Γ s).2) 0 t at h
    exact h
  have hs (t : ℝ) : (Γ t).2 = 1 := by
    have h := is_const_of_deriv_eq_zero (fun s => (hsnd s).differentiableAt)
      (fun s => (hsnd s).deriv) t 0
    simpa only [hΓ0] using h
  refine ⟨fun t => (Γ t).1, congrArg Prod.fst hΓ0, ?_⟩
  intro t
  have h := (ContinuousLinearMap.fst ℝ (Idx n → ℝ) ℝ).hasFDerivAt.comp_hasDerivAt t (hΓ t)
  change HasDerivAt (fun s => (Γ s).1) ((Γ t).2 • frozenVelocity n m β θ a (Γ t).1) t at h
  simpa only [hs t, one_smul] using h

/-- Two actual original frozen-token solutions with the same actual initial tuple coincide on all real times.
Source: arXiv:2411.04990v2, §5, uniqueness of the original flow of `thm:fixed_centers`. -/
theorem frozenODE_unique (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ ψ : ℝ → Idx n → ℝ)
    (hφ : ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t)
    (hψ : ∀ t, HasDerivAt ψ (frozenVelocity n m β θ a (ψ t)) t) (h0 : φ 0 = ψ 0) : φ = ψ :=
  Landscape.GlobalFlow.eq_of_hasDerivAt (locallyLipschitz_frozenVelocity β θ a ha) hφ hψ h0

/-- Every original angular initial tuple has exactly one global original frozen-token trajectory.
Source: arXiv:2411.04990v2, §5, the original deterministic dynamics in `thm:fixed_centers`. -/
theorem exists_unique_frozenODE (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (φ₀ : Idx n → ℝ) :
    ∃! φ : ℝ → Idx n → ℝ, φ 0 = φ₀ ∧ ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t := by
  obtain ⟨φ, h0, hφ⟩ := exists_frozenODE β θ a ha φ₀
  exact ⟨φ, ⟨h0, hφ⟩, fun ψ hψ => frozenODE_unique β θ a ha ψ φ hψ.2 hφ (hψ.1.trans h0.symm)⟩

/-- The actual original global frozen-token flow, chosen only after its original ODE existence proof.
Source: arXiv:2411.04990v2, §5, the original flow in `thm:fixed_centers`. -/
noncomputable def frozenFlow (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (φ₀ : Idx n → ℝ) :
    ℝ → Idx n → ℝ := Classical.choose (exists_frozenODE β θ a ha φ₀)

/-- The actual original chosen flow has its specified original initial tuple and solves the original ODE at every real time.
Source: arXiv:2411.04990v2, §5, the original dynamics in `thm:fixed_centers`. -/
theorem frozenFlow_spec (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (φ₀ : Idx n → ℝ) :
    frozenFlow β θ a ha φ₀ 0 = φ₀ ∧ ∀ t, HasDerivAt (frozenFlow β θ a ha φ₀)
      (frozenVelocity n m β θ a (frozenFlow β θ a ha φ₀ t)) t :=
  Classical.choose_spec (exists_frozenODE β θ a ha φ₀)

/-- The actual original frozen-token flow starts from its actual original angular tuple.
Source: arXiv:2411.04990v2, §5, the initialization in `thm:fixed_centers`. -/
theorem frozenFlow_zero (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (φ₀ : Idx n → ℝ) :
    frozenFlow β θ a ha φ₀ 0 = φ₀ := (frozenFlow_spec β θ a ha φ₀).1

/-- Every actual original solution is exactly the original deterministic flow from its original initial tuple.
Source: arXiv:2411.04990v2, §5, the original all-solutions quantifier in `thm:fixed_centers`. -/
theorem frozenFlow_eq_of_ODE (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : ℝ → Idx n → ℝ) (hφ : ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t) :
    frozenFlow β θ a ha (φ 0) = φ :=
  frozenODE_unique β θ a ha _ φ (frozenFlow_spec β θ a ha (φ 0)).2 hφ (frozenFlow_zero β θ a ha (φ 0))

/-- Translating time preserves the exact original autonomous frozen-token ODE.
Source: arXiv:2411.04990v2, §5, the autonomous original system of `thm:fixed_centers`. -/
theorem frozenODE_time_shift (β : ℝ) (θ a : Idx m → ℝ) (φ : ℝ → Idx n → ℝ)
    (hφ : ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t) (s : ℝ) :
    ∀ t, HasDerivAt (fun u => φ (u + s)) (frozenVelocity n m β θ a (φ (t + s))) t := by
  intro t
  simpa! only [Function.comp_def, id_eq, one_smul] using
    (hφ (t + s)).scomp t ((hasDerivAt_id t).add_const s)

/-- The actual original frozen-token flow composes by adding its original real times.
Source: arXiv:2411.04990v2, §5, the autonomous original flow of `thm:fixed_centers`. -/
theorem frozenFlow_add (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ₀ : Idx n → ℝ) (s t : ℝ) :
    frozenFlow β θ a ha (frozenFlow β θ a ha φ₀ s) t = frozenFlow β θ a ha φ₀ (t + s) := by
  have h := frozenFlow_eq_of_ODE β θ a ha (fun u => frozenFlow β θ a ha φ₀ (u + s))
    (frozenODE_time_shift β θ a _ (frozenFlow_spec β θ a ha φ₀).2 s)
  simpa only [zero_add] using congrFun h t

/-- Every real-time original frozen-token flow map is injective on the original angular tuples.
Source: arXiv:2411.04990v2, §5, the original deterministic flow in `thm:fixed_centers`. -/
theorem frozenFlow_injective (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (s : ℝ) :
    Function.Injective (fun φ₀ : Idx n → ℝ => frozenFlow β θ a ha φ₀ s) := by
  intro φ₀ ψ₀ he
  have h := congrArg (fun x => frozenFlow β θ a ha x (-s)) he
  simpa only [frozenFlow_add, neg_add_cancel, frozenFlow_zero] using h

/-- The actual original flow restricted to its preceding tokens is exactly the smaller original frozen-token flow.
Source: arXiv:2411.04990v2, §5, original prefix independence in the induction for `thm:fixed_centers`. -/
theorem frozenFlow_castSucc (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ₀ : Idx (n + 1) → ℝ) (t : ℝ) (j : Idx n) :
    frozenFlow β θ a ha φ₀ t j.castSucc = frozenFlow β θ a ha (fun l => φ₀ l.castSucc) t j := by
  have h := frozenFlow_eq_of_ODE β θ a ha (fun u l => frozenFlow β θ a ha φ₀ u l.castSucc)
    (frozenODE_castSucc β θ a _ (frozenFlow_spec β θ a ha φ₀).2)
  simpa only [frozenFlow_zero] using (congrFun (congrFun h t) j).symm

/-- Actual positive frozen mass and an actual global original trajectory jointly realize the existence and uniqueness premises.
Source: arXiv:2411.04990v2, §5, the original frozen-token system of `thm:fixed_centers`. -/
example (β : ℝ) : (∀ j : Idx 1, (0 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j) ∧
    ∃ φ : ℝ → Idx 2 → ℝ, φ 0 = 0 ∧
      ∀ t, HasDerivAt φ (frozenVelocity 2 1 β (fun _ => 0) (fun _ => 1) (φ t)) t :=
  ⟨fun _ => zero_le_one, exists_frozenODE β (fun _ => 0) (fun _ => 1) (fun _ => zero_le_one) 0⟩

end Transformer.Causal
