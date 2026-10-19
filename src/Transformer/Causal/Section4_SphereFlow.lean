module

public import Transformer.Causal.Section4_ParticleField
public import Transformer.Perspective.Section2_SphereFlow
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# The actual causal attention flow exists globally and is unique

ArXiv:2411.04990v2, §2, `eqn:csa`, and §4, proof of `thm1`.
The proved smooth ambient extension has linear growth. Its global
solution preserves the unit norms because each squared-norm defect
solves a homogeneous scalar ODE with zero initial value. The extension
factor is consequently one, and this is a solution of the original
causal equation for the original query, key and value matrices.

Every spherical solution also solves the same smooth ambient equation,
so local uniqueness gives global uniqueness. The deterministic flow
is then selected from this proved construction. It includes arbitrary
real temperatures, all original matrices, and empty token tuples.
No existence or invariance premise is included in its definition.
-/

@[expose] public section

open scoped BigOperators
open Real Set Filter Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- Original unit token norms are preserved by the constructed causal ambient ODE.
Source: arXiv:2411.04990v2, §2, the tangent projection in `eqn:csa`.
Each actual norm defect satisfies a homogeneous scalar equation, rather than being assumed zero. -/
theorem norm_eq_one_of_csaGlobalField (β : ℝ) (Q K V : ParamMatrix d)
    (Y : ℝ → Idx n → EucSpace d)
    (hY : ∀ t : ℝ, HasDerivAt Y (csaGlobalField β Q K V (Y t)) t)
    (h0 : ∀ i : Idx n, ‖Y 0 i‖ = 1) (t : ℝ) (i : Idx n) : ‖Y t i‖ = 1 := by
  have hc : Continuous Y := continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  let v (s : ℝ) := csaAverage β Q K V (Y s) i
  let a (s : ℝ) := -2 * Perspective.saExtensionScale (Y s) * inner ℝ (Y s i) (v s)
  have hv : Continuous v := (contDiff_csaAverage β Q K V i).continuous.comp hc
  have ha : Continuous a :=
    (continuous_const.mul (Perspective.contDiff_saExtensionScale.continuous.comp hc)).mul
      (((continuous_apply i).comp hc).inner hv)
  have hd (s : ℝ) : HasDerivAt (fun r => ‖Y r i‖ ^ 2 - 1)
      (a s * (‖Y s i‖ ^ 2 - 1)) s := by
    convert ((hasDerivAt_pi.mp (hY s)) i).norm_sq.sub_const 1 using 1
    simp only [csaGlobalField, Pi.smul_apply, csaAmbientField, proj,
      real_inner_smul_right, inner_sub_right, real_inner_self_eq_norm_sq]
    dsimp [a, v]
    ring
  have hz := Perspective.scalar_eq_zero_of_linearODE a (fun s => ‖Y s i‖ ^ 2 - 1) ha hd
    (by rw [h0 i]; norm_num) t
  nlinarith [norm_nonneg (Y t i)]

/-- Every original spherical initial tuple has an actual global causal attention trajectory.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, the global flow used in `thm1`.
The actual spherical constraint is a proved invariant of the ambient construction. -/
theorem exists_CSA (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d n) :
    ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ CSA d n β Q K V X := by
  obtain ⟨Y, hY0, hY⟩ := Landscape.GlobalFlow.exists_global (locallyLipschitz_csaGlobalField β Q K V)
    (C := ((n : ℝ) + 1) * ‖V‖) (by positivity) (norm_csaGlobalField_le β Q K V)
    (fun i => (X₀ i : EucSpace d))
  have hu (t : ℝ) (i : Idx n) : ‖Y t i‖ = 1 :=
    norm_eq_one_of_csaGlobalField β Q K V Y hY
      (fun j => by rw [hY0]; exact mem_sphere_zero_iff_norm.mp (X₀ j).2) t i
  let X (t : ℝ) (i : Idx n) : SSphere d := ⟨Y t i, mem_sphere_zero_iff_norm.mpr (hu t i)⟩
  refine ⟨X, ?_, (CSA_iff_csaAmbientField β Q K V X).mpr ?_⟩
  · funext i
    exact Subtype.ext (congrFun hY0 i)
  · intro t i
    have hd := hY t
    rw [csaGlobalField_eq_of_unit β Q K V (Y t) (hu t)] at hd
    exact (hasDerivAt_pi.mp hd) i

/-- Original spherical causal attention trajectories with the same initial tuple coincide.
Source: arXiv:2411.04990v2, §4, the deterministic autonomous flow in the proof of `thm1`. -/
theorem CSA_unique (β : ℝ) (Q K V : ParamMatrix d) (X Y : ℝ → SphereTuple d n)
    (hX : CSA d n β Q K V X) (hY : CSA d n β Q K V Y) (h0 : X 0 = Y 0) : X = Y := by
  have hpath (W : ℝ → SphereTuple d n) (hW : CSA d n β Q K V W) (t : ℝ) :
      HasDerivAt (fun s i => (W s i : EucSpace d))
        (csaGlobalField β Q K V (fun i => (W t i : EucSpace d))) t := by
    rw [csaGlobalField_eq_of_unit β Q K V _
      (fun j => mem_sphere_zero_iff_norm.mp (W t j).2)]
    exact hasDerivAt_pi.mpr ((CSA_iff_csaAmbientField β Q K V W).mp hW t)
  have he := Landscape.GlobalFlow.eq_of_hasDerivAt (locallyLipschitz_csaGlobalField β Q K V)
    (hpath X hX) (hpath Y hY) (by rw [h0])
  funext t i
  exact Subtype.ext (congrFun (congrFun he t) i)

/-- Every original initial tuple specifies exactly one global causal attention solution.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, the flow in `thm1`. -/
theorem exists_unique_CSA (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d n) :
    ∃! X : ℝ → SphereTuple d n, X 0 = X₀ ∧ CSA d n β Q K V X := by
  obtain ⟨X, hX0, hX⟩ := exists_CSA β Q K V X₀
  exact ⟨X, ⟨hX0, hX⟩, fun Y hY => CSA_unique β Q K V Y X hY.2 hX (hY.1.trans hX0.symm)⟩

/-- The deterministic original causal attention flow, chosen from its proved global existence theorem.
Source: arXiv:2411.04990v2, §4, proof of `thm1`; uniqueness makes this choice canonical. -/
noncomputable def csaFlow (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d n) :
    ℝ → SphereTuple d n := Classical.choose (exists_CSA β Q K V X₀)

/-- The constructed original causal flow starts at its actual prescribed tuple and solves the actual equation.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, `thm1`. -/
theorem csaFlow_spec (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d n) :
    csaFlow β Q K V X₀ 0 = X₀ ∧ CSA d n β Q K V (csaFlow β Q K V X₀) :=
  Classical.choose_spec (exists_CSA β Q K V X₀)

/-- The exact original initial-value identity of the causal flow.
Source: arXiv:2411.04990v2, §4, the autonomous flow in the proof of `thm1`. -/
theorem csaFlow_zero (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d n) :
    csaFlow β Q K V X₀ 0 = X₀ := (csaFlow_spec β Q K V X₀).1

/-- Every original causal solution is the actual deterministic flow from its initial tuple.
Source: arXiv:2411.04990v2, §4, the flow used in `thm1`. -/
theorem csaFlow_eq_of_CSA (β : ℝ) (Q K V : ParamMatrix d) (X : ℝ → SphereTuple d n)
    (hX : CSA d n β Q K V X) : csaFlow β Q K V (X 0) = X :=
  CSA_unique β Q K V _ X (csaFlow_spec β Q K V (X 0)).2 hX (csaFlow_zero β Q K V (X 0))

/-- An actual original consensus tuple is a stationary solution at identity values and arbitrary original scores.
Source: arXiv:2411.04990v2, §4, the consensus critical points in the proof of `thm1`. -/
theorem CSA_const_consensus (β : ℝ) (Q K : ParamMatrix d) (x : SSphere d) :
    CSA d n β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => x) := by
  apply (CSA_iff_csaAmbientField β Q K _ _).mpr
  intro t i
  rw [csaAmbientField_consensus_id]
  exact hasDerivAt_const t (x : EucSpace d)

/-- The original deterministic flow keeps each actual consensus tuple stationary.
Source: arXiv:2411.04990v2, §4, the consensus equilibria of `thm1`. -/
theorem csaFlow_consensus (β : ℝ) (Q K : ParamMatrix d) (x : SSphere d) :
    csaFlow (n := n) β Q K (ContinuousLinearMap.id ℝ _) (fun _ => x) = fun _ _ => x :=
  csaFlow_eq_of_CSA β Q K _ _ (CSA_const_consensus β Q K x)

/-- Every actual causal trajectory is continuous as an original sphere-valued token tuple.
Source: arXiv:2411.04990v2, §4, the original autonomous flow of `thm1`. -/
theorem continuous_csaFlow_path (β : ℝ) (Q K V : ParamMatrix d) (X₀ : SphereTuple d n) :
    Continuous (csaFlow β Q K V X₀) := by
  apply continuous_pi_iff.mpr
  intro i
  apply continuous_induced_rng.mpr
  exact continuous_iff_continuousAt.mpr fun t => ((csaFlow_spec β Q K V X₀).2 t i).continuousAt

/-- Nonzero original consensus tokens jointly satisfy the ambient ODE, unit initial norms,
the original sphere equation and the initial-value agreement used by uniqueness. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => fun _ : Idx 2 => (basePoint 1 : EucSpace 2))
      (csaGlobalField β Q K (ContinuousLinearMap.id ℝ _) (fun _ : Idx 2 => (basePoint 1 : EucSpace 2))) t) ∧
    (∀ i : Idx 2, ‖(fun _ : Idx 2 => (basePoint 1 : EucSpace 2)) i‖ = 1) ∧
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    (fun _ : ℝ => fun _ : Idx 2 => basePoint 1) 0 = (fun _ : ℝ => fun _ : Idx 2 => basePoint 1) 0 := by
  have hu : ∀ i : Idx 2, ‖(fun _ : Idx 2 => (basePoint 1 : EucSpace 2)) i‖ = 1 :=
    fun _ => mem_sphere_zero_iff_norm.mp (basePoint 1).2
  refine ⟨?_, hu, CSA_const_consensus β Q K (basePoint 1), rfl⟩
  intro t
  rw [csaGlobalField_eq_of_unit β Q K _ _ hu, csaAmbientField_consensus_id]
  exact hasDerivAt_const t _

end Transformer.Causal
