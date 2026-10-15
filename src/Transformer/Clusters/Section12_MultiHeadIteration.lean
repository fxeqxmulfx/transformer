/-
# Exact construction and scalar coordinates of the original multi-head iteration

ArXiv:2305.05465v6, §12, `sec:conclusion`, the multi-head display,
and §3, `r:discreterescaling`.
The original finite-step equation defines a trajectory from every
initial configuration and determines that trajectory uniquely.
No limit assertion is built into this construction.

In dimension one its coordinate equation is exactly the finite sum
of the original scalar head averages, with the actual separate
query-key and value coefficients. The converse coordinate identity
also recovers the original vector equation. In the singleton case,
the constructed solution is the actual geometric progression with
factor `1+Δt Σ_h v_h`, providing nonzero original trajectories.
-/

import Transformer.Clusters.Section12_MultiHeadScalar

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {d n H : ℕ}

/-- The original multi-head update, with every original head matrix retained.
Source: arXiv:2305.05465v6, §12, `sec:conclusion`, the displayed iteration. -/
def MultiHeadTransformer (Δt : ℝ) (Q K V : Idx H → ParamMatrix d)
    (X : ℕ → Idx n → EucSpace d) : Prop :=
  ∀ (k : ℕ) (i : Idx n),
    X (k + 1) i = X k i + Δt • ∑ h : Idx H, ∑ j : Idx n,
      attentionMatrix (Q h) (K h) (X k) i j • V h (X k j)

/-- One actual finite step of the original multi-head update.
Source: arXiv:2305.05465v6, §12, the displayed iteration. -/
noncomputable def multiHeadUpdate (Δt : ℝ) (Q K V : Idx H → ParamMatrix d)
    (Y : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  Y i + Δt • ∑ h, ∑ j, attentionMatrix (Q h) (K h) Y i j • V h (Y j)

/-- The actual original trajectory, constructed by finite recursion on the layer index.
Source: arXiv:2305.05465v6, §12, the displayed multi-head update. -/
noncomputable def multiHeadTrajectory (Δt : ℝ) (Q K V : Idx H → ParamMatrix d)
    (Y : Idx n → EucSpace d) : ℕ → Idx n → EucSpace d
  | 0 => Y
  | k + 1 => multiHeadUpdate Δt Q K V (multiHeadTrajectory Δt Q K V Y k)

/-- The constructed trajectory has the prescribed original initial state.
Source: arXiv:2305.05465v6, §12, the original initial-value iteration. -/
theorem multiHeadTrajectory_zero (Δt : ℝ) (Q K V : Idx H → ParamMatrix d) (Y : Idx n → EucSpace d) :
    multiHeadTrajectory Δt Q K V Y 0 = Y := rfl

/-- Finite recursion satisfies the actual multi-head equation at every layer.
Source: arXiv:2305.05465v6, §12, the displayed multi-head iteration. -/
theorem multiHeadTrajectory_solution (Δt : ℝ) (Q K V : Idx H → ParamMatrix d) (Y : Idx n → EucSpace d) :
    MultiHeadTransformer Δt Q K V (multiHeadTrajectory Δt Q K V Y) := by
  intro k i
  rfl

/-- The source's discrete initial-value problem has an actual trajectory for every input.
Source: arXiv:2305.05465v6, §12, `sec:conclusion`. -/
theorem exists_multiHeadTrajectory (Δt : ℝ) (Q K V : Idx H → ParamMatrix d) (Y : Idx n → EucSpace d) :
    ∃ X, X 0 = Y ∧ MultiHeadTransformer Δt Q K V X :=
  ⟨multiHeadTrajectory Δt Q K V Y, multiHeadTrajectory_zero _ _ _ _ _, multiHeadTrajectory_solution _ _ _ _ _⟩

/-- Every solution of the original finite-step equation equals the actual recursive trajectory.
Source: arXiv:2305.05465v6, §12, the multi-head update as an initial-value iteration. -/
theorem multiHeadTransformer_eq_trajectory (Δt : ℝ) (Q K V : Idx H → ParamMatrix d)
    (X : ℕ → Idx n → EucSpace d) (hX : MultiHeadTransformer Δt Q K V X) :
    X = multiHeadTrajectory Δt Q K V (X 0) := by
  funext k
  induction k with
  | zero => rfl
  | succ k ih =>
      funext i
      rw [hX k i, ih]
      rfl

/-- Identical original initial states give identical actual multi-head trajectories.
Source: arXiv:2305.05465v6, §12, the deterministic finite-step update. -/
theorem multiHeadTransformer_unique (Δt : ℝ) (Q K V : Idx H → ParamMatrix d)
    (X Y : ℕ → Idx n → EucSpace d) (hX : MultiHeadTransformer Δt Q K V X)
    (hY : MultiHeadTransformer Δt Q K V Y) (hinit : X 0 = Y 0) : X = Y := by
  calc
    X = multiHeadTrajectory Δt Q K V (X 0) := multiHeadTransformer_eq_trajectory Δt Q K V X hX
    _ = multiHeadTrajectory Δt Q K V (Y 0) := by rw [hinit]
    _ = Y := (multiHeadTransformer_eq_trajectory Δt Q K V Y hY).symm

/-- Each actual original step has exactly the scalar head-sum coordinate.
Source: arXiv:2305.05465v6, §12, the head sum and §7's scalar reduction. -/
theorem multiHeadUpdate_coord (Δt : ℝ) (Q K V : Idx H → ParamMatrix 1)
    (Y : Idx n → EucSpace 1) (i : Idx n) :
    multiHeadUpdate Δt Q K V Y i 0 = Y i 0 + Δt *
      multiHeadScalarDrift (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1))
        (fun h => (V h unit1) 0) (fun j => Y j 0) (Y i 0) := by
  change (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 1)) (multiHeadUpdate Δt Q K V Y i) = _
  unfold multiHeadUpdate
  simp only [map_add, map_smul, map_sum, smul_eq_mul]
  congr 2
  unfold multiHeadScalarDrift
  apply Finset.sum_congr rfl
  intro h hh
  rw [← attention_scalarHeadAverage, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  change attentionMatrix (Q h) (K h) Y i j * (V h (Y j)) 0 = _
  rw [coord_apply_one]
  ring

/-- The exact one-dimensional coordinate equation holds along every original solution.
Source: arXiv:2305.05465v6, §12, the original finite-step equation. -/
theorem multiHeadTransformer_coord {Δt : ℝ} {Q K V : Idx H → ParamMatrix 1}
    {X : ℕ → Idx n → EucSpace 1} (hX : MultiHeadTransformer Δt Q K V X) (k : ℕ) (i : Idx n) :
    X (k + 1) i 0 = X k i 0 + Δt *
      multiHeadScalarDrift (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1))
        (fun h => (V h unit1) 0) (fun j => X k j 0) (X k i 0) := by
  exact (congrArg (fun z : EucSpace 1 => z 0) (hX k i)).trans (multiHeadUpdate_coord Δt Q K V (X k) i)

/-- Equality of the actual scalar coordinates recovers the original vector iteration.
Source: arXiv:2305.05465v6, §12, the original update in dimension one. -/
theorem multiHeadTransformer_iff_coord (Δt : ℝ) (Q K V : Idx H → ParamMatrix 1)
    (X : ℕ → Idx n → EucSpace 1) :
    MultiHeadTransformer Δt Q K V X ↔ ∀ k i, X (k + 1) i 0 = X k i 0 + Δt *
      multiHeadScalarDrift (fun h => inner (𝕜 := ℝ) (Q h unit1) (K h unit1))
        (fun h => (V h unit1) 0) (fun j => X k j 0) (X k i 0) := by
  constructor
  · exact fun hX k i => multiHeadTransformer_coord hX k i
  · intro hX k i
    apply (EuclideanSpace.equiv (Fin 1) ℝ).injective
    ext l
    fin_cases l
    change X (k + 1) i 0 = multiHeadUpdate Δt Q K V (X k) i 0
    exact (hX k i).trans (multiHeadUpdate_coord Δt Q K V (X k) i).symm

/-- A single original token follows the actual geometric progression for all heads.
Source: arXiv:2305.05465v6, §12, the original iteration with one token. -/
theorem multiHeadTransformer_singleton_power (Δt : ℝ) (Q K V : Idx H → ParamMatrix 1) (z : EucSpace 1) :
    MultiHeadTransformer Δt Q K V (fun k (_ : Idx 1) => (1 + Δt * ∑ h, (V h unit1) 0) ^ k • z) := by
  apply (multiHeadTransformer_iff_coord Δt Q K V _).mpr
  intro k i
  simp only [PiLp.smul_apply, smul_eq_mul, multiHeadScalarDrift, scalarHeadAverage_const one_pos,
    ← Finset.sum_mul, pow_succ]
  ring

/-- Two actual positive heads and a nonzero token realize the original equation
and its exact coordinate identity, with strict geometric growth. -/
example : ∃ X Y : ℕ → Idx 1 → EucSpace 1,
    MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) X ∧
    MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) Y ∧
    X 0 = Y 0 ∧ X 0 0 ≠ 0 ∧ X 1 0 = (3 : ℝ) • unit1 := by
  let X := fun k : ℕ => fun _ : Idx 1 => (3 : ℝ) ^ k • unit1
  have hX : MultiHeadTransformer (H := 2) 1 (fun _ => (1 : ParamMatrix 1)) (fun _ => 1) (fun _ => 1) X := by
    convert multiHeadTransformer_singleton_power (H := 2) 1 (fun _ => (1 : ParamMatrix 1))
      (fun _ => 1) (fun _ => 1) unit1 using 1
    norm_num [X, unit1]
  refine ⟨X, X, hX, hX, rfl, ?_, ?_⟩
  · simpa only [X, pow_zero, one_smul] using (show unit1 ≠ 0 from fun h => by simpa using congrArg norm h)
  · simp only [X, pow_one]

end Transformer.Clusters
