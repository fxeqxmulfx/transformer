/-
# Attention's forward pass and Frank-Wolfe — linear leader branches

The proof of `lem:singleLeader` in arXiv:2508.09628v1, §2.1, needs to allow
singular branches: two tokens can merge on an open set. Each fixed sequence
of leader indices nevertheless defines a linear map on the initial tuple,
and that map fixes every consensus tuple. There are only finitely many such
maps at each time. These facts replace the source's incorrect assertion that
every piecewise affine update preserves sets of positive measure.
-/

import Transformer.FrankWolfe.Section1_Models

open scoped BigOperators

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- A branch of `eq:hardmax.dynamics.V_ae` with the leader index `σ i` fixed
for each token. Source: arXiv:2508.09628v1, §2.1–§2.2. -/
def leaderStepLinearMap (P : ParamMatrix d) (σ : Idx n → Idx n) :
    (Idx n → EucSpace d) →ₗ[ℝ] (Idx n → EucSpace d) where
  toFun X i := X i + P (X (σ i) - X i)
  map_add' X Y := by
    funext i
    simp only [Pi.add_apply]
    rw [show X (σ i) + Y (σ i) - (X i + Y i) =
      (X (σ i) - X i) + (Y (σ i) - Y i) by abel, map_add]
    abel
  map_smul' c X := by
    funext i
    simp only [Pi.smul_apply, RingHom.id_apply]
    rw [← smul_sub, map_smul, smul_add]

/-- Selecting each token itself gives the identity branch, for every value
map. Source: arXiv:2508.09628v1, `eq:hardmax.dynamics.V_ae`. -/
@[simp] theorem leaderStepLinearMap_id (P : ParamMatrix d) :
    leaderStepLinearMap P (id : Idx n → Idx n) = LinearMap.id := by
  ext X i
  simp [leaderStepLinearMap]

/-- All compositions of `t` leader branches of `eq:hardmax.dynamics.V_ae`.
No invertibility of these maps is asserted or needed.
Source: arXiv:2508.09628v1, §2.1, `lem:singleLeader`. -/
noncomputable def leaderBranches (P : ℕ → ParamMatrix d) (t : ℕ) :
    Finset ((Idx n → EucSpace d) →ₗ[ℝ] (Idx n → EucSpace d)) :=
  open Classical in
  match t with
  | 0 => {LinearMap.id}
  | t + 1 => (leaderBranches P t).biUnion fun L =>
      Finset.univ.image fun σ : Idx n → Idx n => (leaderStepLinearMap (P t) σ).comp L

/-- Any possible leader selection extends a branch by one step.
Source: arXiv:2508.09628v1, §2.1, `lem:singleLeader`. -/
theorem mem_leaderBranches_succ (P : ℕ → ParamMatrix d) (t : ℕ)
    (L : (Idx n → EucSpace d) →ₗ[ℝ] (Idx n → EucSpace d))
    (hL : L ∈ leaderBranches P t) (σ : Idx n → Idx n) :
    (leaderStepLinearMap (P t) σ).comp L ∈ leaderBranches P (t + 1) := by
  classical
  exact Finset.mem_biUnion.mpr ⟨L, hL,
    Finset.mem_image.mpr ⟨σ, Finset.mem_univ _, rfl⟩⟩

/-- The branch-membership hypothesis holds at time zero, including for the
zero value map. Source: arXiv:2508.09628v1, §2.1. -/
example : (LinearMap.id : (Idx 1 → EucSpace 1) →ₗ[ℝ] (Idx 1 → EucSpace 1)) ∈
    leaderBranches (fun _ => (0 : ParamMatrix 1)) 0 := by
  simp [leaderBranches]

/-- Every branch fixes consensus configurations: this survives even when the
branch is singular and merges initially distinct tokens.
Source: arXiv:2508.09628v1, §2.1, `lem:singleLeader`. -/
theorem leaderBranches_consensus (P : ℕ → ParamMatrix d) (t : ℕ)
    (L : (Idx n → EucSpace d) →ₗ[ℝ] (Idx n → EucSpace d))
    (hL : L ∈ leaderBranches P t) (v : EucSpace d) :
    L (fun _ => v) = fun _ => v := by
  classical
  induction t generalizing L with
  | zero =>
      simp only [leaderBranches, Finset.mem_singleton] at hL
      subst L
      rfl
  | succ t ih =>
      obtain ⟨M, hM, h⟩ := Finset.mem_biUnion.mp hL
      obtain ⟨σ, _, rfl⟩ := Finset.mem_image.mp h
      rw [LinearMap.comp_apply, ih M hM]
      funext i
      simp [leaderStepLinearMap]

/-- Consensus preservation has a concrete, nonempty instance: one token and
the identity branch at time zero. Source: arXiv:2508.09628v1, §2.1. -/
example : (LinearMap.id : (Idx 1 → EucSpace 1) →ₗ[ℝ] (Idx 1 → EucSpace 1)) ∈
    leaderBranches (fun _ => (0 : ParamMatrix 1)) 0 ∧
    (LinearMap.id : (Idx 1 → EucSpace 1) →ₗ[ℝ] (Idx 1 → EucSpace 1))
      (fun _ => (0 : EucSpace 1)) = fun _ => 0 := by
  exact ⟨by simp [leaderBranches], rfl⟩

end FrankWolfe
end Transformer
