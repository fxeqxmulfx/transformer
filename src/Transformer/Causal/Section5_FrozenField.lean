module

public import Transformer.Causal.Section5_FrozenWeights
public import Mathlib.Analysis.Calculus.ContDiff.RCLike
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Smoothness and prefix consistency of the original frozen-token field

ArXiv:2411.04990v2, §5, Theorem `thm:fixed_centers`.
The actual partition and both original interaction numerators are
finite sums of smooth trigonometric exponentials. The actual
partition is strictly positive for nonnegative original center
masses, so the original normalized field is smooth and locally
Lipschitz on the full original space of lifted angular states.

The original causal mask makes the original first `n` velocities
independent of the appended last token. The exact prefix identities
retain every frozen center, every original center mass, and the
original normalization. Consequently the actual preceding portion
of every original frozen-token solution solves exactly the smaller
original frozen-token system. Actual nonzero center masses with
coincident original angles provide joint solution witnesses.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n m : ℕ}

/-- The actual original frozen partition is smooth on the original full angular state space.
Source: arXiv:2411.04990v2, §5, the partition in `thm:fixed_centers`. -/
theorem contDiff_frozenZ (β : ℝ) (θ a : Idx m → ℝ) (k : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun φ : Idx n → ℝ => frozenZ n m β θ a φ k) := by
  unfold frozenZ
  apply ContDiff.add
  · apply ContDiff.sum
    intro j hj
    fun_prop
  · apply ContDiff.sum
    intro j hj
    fun_prop

/-- Each actual original causal particle weight is smooth with the actual positive original partition.
Source: arXiv:2411.04990v2, §5, the original normalized interactions in `thm:fixed_centers`. -/
theorem contDiff_frozenParticleWeight (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (k j : Idx n) :
    ContDiff ℝ (⊤ : ℕ∞) (fun φ : Idx n → ℝ => frozenParticleWeight β θ a φ k j) := by
  unfold frozenParticleWeight
  apply ContDiff.div _ (contDiff_frozenZ β θ a k) (fun φ => (frozenZ_pos β θ a ha φ k).ne')
  by_cases h : j ≤ k
  · simp only [ite_eq_left h]
    fun_prop
  · simp only [ite_eq_right h]
    exact contDiff_const

/-- Each original frozen-center weight is smooth with its original center mass and original partition.
Source: arXiv:2411.04990v2, §5, the original frozen-center force in `thm:fixed_centers`. -/
theorem contDiff_frozenCenterWeight (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (k : Idx n) (j : Idx m) :
    ContDiff ℝ (⊤ : ℕ∞) (fun φ : Idx n → ℝ => frozenCenterWeight β θ a φ k j) := by
  unfold frozenCenterWeight
  apply ContDiff.div _ (contDiff_frozenZ β θ a k) (fun φ => (frozenZ_pos β θ a ha φ k).ne')
  fun_prop

/-- The actual full original frozen-token vector field is smooth at every original angular state.
Source: arXiv:2411.04990v2, §5, the smooth specialized model of `thm:fixed_centers`. -/
theorem contDiff_frozenVelocity (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    ContDiff ℝ (⊤ : ℕ∞) (frozenVelocity n m β θ a) := by
  apply contDiff_pi.mpr
  intro k
  unfold frozenVelocity
  apply ContDiff.mul
  · exact (contDiff_frozenZ β θ a k).inv (fun φ => (frozenZ_pos β θ a ha φ k).ne')
  · apply ContDiff.add
    · apply ContDiff.sum
      intro j hj
      fun_prop
    · apply ContDiff.sum
      intro j hj
      fun_prop

/-- The actual original frozen-token field is locally Lipschitz on the full original angular space.
Source: arXiv:2411.04990v2, §5, uniqueness for the original dynamics of `thm:fixed_centers`. -/
theorem locallyLipschitz_frozenVelocity (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    LocallyLipschitz (frozenVelocity n m β θ a) :=
  ((contDiff_frozenVelocity β θ a ha).of_le (by simp)).locallyLipschitz

/-- Appending a last original particle leaves every actual original preceding partition exactly unchanged.
Source: arXiv:2411.04990v2, §5, the triangular original system in `thm:fixed_centers`. -/
theorem frozenZ_castSucc (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx (n + 1) → ℝ) (k : Idx n) :
    frozenZ (n + 1) m β θ a φ k.castSucc = frozenZ n m β θ a (fun j => φ j.castSucc) k := by
  unfold frozenZ
  simp only [Finset.sum_filter]
  rw [Fin.sum_univ_castSucc]
  have hlast : ¬ Fin.last n ≤ k.castSucc := not_le.mpr (Fin.castSucc_lt_last k)
  simp only [Fin.castSucc_le_castSucc_iff, ite_eq_right hlast, add_zero]

/-- The actual original preceding causal particle weights are independent of the appended original last token.
Source: arXiv:2411.04990v2, §5, prefix independence in `thm:fixed_centers`. -/
theorem frozenParticleWeight_castSucc (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx (n + 1) → ℝ) (k j : Idx n) :
    frozenParticleWeight β θ a φ k.castSucc j.castSucc =
      frozenParticleWeight β θ a (fun l => φ l.castSucc) k j := by
  unfold frozenParticleWeight
  simp only [Fin.castSucc_le_castSucc_iff, frozenZ_castSucc]

/-- The appended last original particle has zero original causal weight in every earlier original row.
Source: arXiv:2411.04990v2, §5, the original causal mask in `thm:fixed_centers`. -/
theorem frozenParticleWeight_last_above (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx (n + 1) → ℝ) (k : Idx n) :
    frozenParticleWeight β θ a φ k.castSucc (Fin.last n) = 0 := by
  unfold frozenParticleWeight
  rw [ite_eq_right (not_le.mpr (Fin.castSucc_lt_last k)), zero_div]

/-- The original frozen-center weights in every preceding row are exactly those of the smaller original system.
Source: arXiv:2411.04990v2, §5, the original frozen forcing in `thm:fixed_centers`. -/
theorem frozenCenterWeight_castSucc (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx (n + 1) → ℝ) (k : Idx n) (j : Idx m) :
    frozenCenterWeight β θ a φ k.castSucc j = frozenCenterWeight β θ a (fun l => φ l.castSucc) k j := by
  unfold frozenCenterWeight
  rw [frozenZ_castSucc]

/-- The actual original frozen-token field has exactly its smaller original field as its causal prefix.
Source: arXiv:2411.04990v2, §5, the original inductive system of `thm:fixed_centers`. -/
theorem frozenVelocity_castSucc (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx (n + 1) → ℝ) (k : Idx n) :
    frozenVelocity (n + 1) m β θ a φ k.castSucc = frozenVelocity n m β θ a (fun j => φ j.castSucc) k := by
  rw [frozenVelocity_eq_weights, Fin.sum_univ_castSucc, frozenVelocity_eq_weights]
  simp only [frozenParticleWeight_castSucc, frozenParticleWeight_last_above, zero_mul,
    add_zero, frozenCenterWeight_castSucc]

/-- Every actual original preceding trajectory solves exactly the smaller original frozen-token ODE.
Source: arXiv:2411.04990v2, §5, the induction proving `thm:fixed_centers`. -/
theorem frozenODE_castSucc (β : ℝ) (θ a : Idx m → ℝ) (φ : ℝ → Idx (n + 1) → ℝ)
    (hφ : ∀ t, HasDerivAt φ (frozenVelocity (n + 1) m β θ a (φ t)) t) :
    ∀ t, HasDerivAt (fun s j => φ s j.castSucc)
      (frozenVelocity n m β θ a (fun j => φ t j.castSucc)) t := by
  intro t
  apply hasDerivAt_pi.mpr
  intro j
  have h := hasDerivAt_pi.mp (hφ t) j.castSucc
  simpa only [frozenVelocity_castSucc] using h

/-- Actual coincident original angles and positive frozen mass jointly realize smoothness and original solution premises.
Source: arXiv:2411.04990v2, §5, the one-center consensus equilibrium in `thm:fixed_centers`. -/
example (β : ℝ) : (∀ j : Idx 1, (0 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j) ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (fun _ : Idx 2 => (0 : ℝ)))
      (frozenVelocity 2 1 β (fun _ => 0) (fun _ => 1) (fun _ => 0)) t) := by
  refine ⟨fun _ => zero_le_one, ?_⟩
  have hv : frozenVelocity 2 1 β (fun _ => 0) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity]
  intro t
  simpa only [hv] using hasDerivAt_const t (fun _ : Idx 2 => (0 : ℝ))

end Transformer.Causal
