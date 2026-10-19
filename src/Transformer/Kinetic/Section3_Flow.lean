module

public import Transformer.Kinetic.Section3_Weights
public import Transformer.Landscape.GlobalFlow.Existence
public import Mathlib.Analysis.ODE.Gronwall
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# The flow map of the causal dynamics

ArXiv:2605.09213v1, §1, `eq:gpt-dyn-alibi`, as used in the proof of
`lem:correl`: the trajectories are functions of the initial angles.

The field `gptVec` is globally Lipschitz and vanishes at `0`, so every initial
configuration has exactly one solution on `ℝ` (`Landscape.GlobalFlow.exists_global`,
`Landscape.GlobalFlow.eq_of_hasDerivAt`). `flowMap t x` is its value at time `t`, and
every `IsGPTFlow` trajectory is `t ↦ flowMap t (θ 0)` (`eq_flowMap`). For
`t ≥ 0` the map `x ↦ flowMap t x` is Lipschitz, by Grönwall, hence measurable.

The field only sees the angles modulo `2π`: moving one initial angle by whole
turns moves that token's trajectory by the same turns and leaves the others
unchanged (`flowMap_add_single`).
-/

@[expose] public section

open Real

namespace Transformer
namespace Kinetic

/-- The field of `eq:gpt-dyn-alibi` as a map of the configuration. -/
noncomputable def gptVec (lam β : ℝ) (N : ℕ) (θ : Idx N → ℝ) : Idx N → ℝ :=
  fun j => gptField lam β N θ j

theorem gptVec_zero (lam β : ℝ) (N : ℕ) : gptVec lam β N 0 = 0 := by
  funext j
  simp [gptVec, gptField, wBetaDeriv]

/-- One component of the field moves by at most `L₁ (|δθ_j| + max_k |δθ_k|)`
over the predecessors, when `w_β'` is `L₁`-Lipschitz. -/
theorem abs_gptField_sub_le (lam β : ℝ) (N : ℕ) {L₁ : ℝ}
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (θ θ' : Idx N → ℝ)
    (j : Idx N) :
    |gptField lam β N θ' j - gptField lam β N θ j| ≤
      ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j),
        alibiWeight lam N j k * (L₁ * (|θ' j - θ j| + |θ' k - θ k|)) := by
  rw [gptField_eq_sum, gptField_eq_sum, ← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
  rw [← mul_sub, abs_mul, abs_of_nonneg (alibiWeight_nonneg lam N j k)]
  refine mul_le_mul_of_nonneg_left ((hL _ _).trans ?_) (alibiWeight_nonneg lam N j k)
  refine mul_le_mul_of_nonneg_left ?_ ((abs_nonneg _).trans ((hL 0 1).trans_eq (by simp)))
  calc |θ' j - θ' k - (θ j - θ k)| = |(θ' j - θ j) - (θ' k - θ k)| := by ring_nf
    _ ≤ |θ' j - θ j| + |θ' k - θ k| := abs_sub _ _

/-- The field is `2L₁`-Lipschitz for the sup norm. -/
theorem norm_gptVec_sub_le (lam β : ℝ) (N : ℕ) {L₁ : ℝ} (hL₁ : 0 ≤ L₁)
    (hL : ∀ x y, |wBetaDeriv β x - wBetaDeriv β y| ≤ L₁ * |x - y|) (θ θ' : Idx N → ℝ) :
    ‖gptVec lam β N θ' - gptVec lam β N θ‖ ≤ 2 * L₁ * ‖θ' - θ‖ := by
  refine (pi_norm_le_iff_of_nonneg (by positivity)).2 fun j => ?_
  rw [Pi.sub_apply, Real.norm_eq_abs]
  have hc : ∀ k, |θ' k - θ k| ≤ ‖θ' - θ‖ := fun k => by
    simpa [Real.norm_eq_abs] using norm_le_pi_norm (θ' - θ) k
  refine (abs_gptField_sub_le lam β N hL θ θ' j).trans ?_
  calc ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j),
        alibiWeight lam N j k * (L₁ * (|θ' j - θ j| + |θ' k - θ k|))
      ≤ ∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j),
          alibiWeight lam N j k * (2 * L₁ * ‖θ' - θ‖) :=
        Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (by nlinarith [hc j, hc k])
          (alibiWeight_nonneg lam N j k)
    _ = (∑ k ∈ Finset.univ.filter (fun k : Idx N => k < j), alibiWeight lam N j k) *
          (2 * L₁ * ‖θ' - θ‖) := by rw [Finset.sum_mul]
    _ ≤ 1 * (2 * L₁ * ‖θ' - θ‖) :=
        mul_le_mul_of_nonneg_right (sum_alibiWeight_le_one lam N j) (by positivity)
    _ = 2 * L₁ * ‖θ' - θ‖ := one_mul _

theorem exists_lipschitzWith_gptVec (lam β : ℝ) (N : ℕ) :
    ∃ K : NNReal, LipschitzWith K (gptVec lam β N) := by
  obtain ⟨L₁, -, hL₁, -, -, hL, -⟩ := exists_lipschitz_wBetaDeriv β
  refine ⟨⟨2 * L₁, by positivity⟩, LipschitzWith.of_dist_le_mul fun θ θ' => ?_⟩
  rw [dist_eq_norm, dist_eq_norm]
  exact norm_gptVec_sub_le lam β N hL₁ hL θ' θ

/-- A curve solves `eq:gpt-dyn-alibi` iff it solves `θ' = gptVec θ`. -/
theorem isGPTFlow_iff (lam β : ℝ) (N : ℕ) (θ : ℝ → Idx N → ℝ) :
    IsGPTFlow lam β N θ ↔ ∀ t, HasDerivAt θ (gptVec lam β N (θ t)) t := by
  simp only [IsGPTFlow, hasDerivAt_pi, gptVec]

theorem exists_isGPTFlow (lam β : ℝ) (N : ℕ) (x : Idx N → ℝ) :
    ∃ θ : ℝ → Idx N → ℝ, θ 0 = x ∧ IsGPTFlow lam β N θ := by
  obtain ⟨K, hK⟩ := exists_lipschitzWith_gptVec lam β N
  obtain ⟨θ, h0, hθ⟩ := Landscape.GlobalFlow.exists_global hK.locallyLipschitz K.2
    (fun y => by simpa [gptVec_zero, dist_eq_norm] using hK.dist_le_mul y 0) x
  exact ⟨θ, h0, (isGPTFlow_iff lam β N θ).2 hθ⟩

/-- The solution of `eq:gpt-dyn-alibi` from `x`, at time `t`. -/
noncomputable def flowMap (lam β : ℝ) (N : ℕ) (t : ℝ) (x : Idx N → ℝ) : Idx N → ℝ :=
  Classical.choose (exists_isGPTFlow lam β N x) t

theorem flowMap_zero (lam β : ℝ) (N : ℕ) (x : Idx N → ℝ) : flowMap lam β N 0 x = x :=
  (Classical.choose_spec (exists_isGPTFlow lam β N x)).1

theorem isGPTFlow_flowMap (lam β : ℝ) (N : ℕ) (x : Idx N → ℝ) :
    IsGPTFlow lam β N (fun t => flowMap lam β N t x) :=
  (Classical.choose_spec (exists_isGPTFlow lam β N x)).2

theorem hasDerivAt_flowMap (lam β : ℝ) (N : ℕ) (x : Idx N → ℝ) (t : ℝ) (j : Idx N) :
    HasDerivAt (fun s => flowMap lam β N s x j) (gptField lam β N (flowMap lam β N t x) j) t :=
  isGPTFlow_flowMap lam β N x t j

/-- **Uniqueness.** Every solution is the flow of its initial configuration. -/
theorem eq_flowMap (lam β : ℝ) (N : ℕ) {θ : ℝ → Idx N → ℝ} (hθ : IsGPTFlow lam β N θ) (t : ℝ) :
    θ t = flowMap lam β N t (θ 0) := by
  obtain ⟨K, hK⟩ := exists_lipschitzWith_gptVec lam β N
  have h := Landscape.GlobalFlow.eq_of_hasDerivAt hK.locallyLipschitz ((isGPTFlow_iff lam β N θ).1 hθ)
    ((isGPTFlow_iff lam β N _).1 (isGPTFlow_flowMap lam β N (θ 0)))
    (by simp [flowMap_zero])
  exact congrFun h t

/-- For `t ≥ 0` the flow is Lipschitz in the initial configuration. -/
theorem exists_dist_flowMap_le (lam β : ℝ) (N : ℕ) :
    ∃ K : NNReal, ∀ t, 0 ≤ t → ∀ x y : Idx N → ℝ,
      dist (flowMap lam β N t x) (flowMap lam β N t y) ≤ dist x y * Real.exp (K * t) := by
  obtain ⟨K, hK⟩ := exists_lipschitzWith_gptVec lam β N
  refine ⟨K, fun t ht x y => ?_⟩
  have hd : ∀ z : Idx N → ℝ, ∀ s, HasDerivAt (fun s => flowMap lam β N s z)
      (gptVec lam β N (flowMap lam β N s z)) s :=
    fun z => (isGPTFlow_iff lam β N _).1 (isGPTFlow_flowMap lam β N z)
  have hc : ∀ z : Idx N → ℝ, Continuous fun s => flowMap lam β N s z :=
    fun z => continuous_iff_continuousAt.2 fun s => (hd z s).continuousAt
  have := dist_le_of_trajectories_ODE (v := fun _ => gptVec lam β N) (a := 0) (b := t)
    (δ := dist x y) (fun _ => hK)
    (hc x).continuousOn (fun s _ => (hd x s).hasDerivWithinAt) (hc y).continuousOn
    (fun s _ => (hd y s).hasDerivWithinAt) (le_of_eq (by simp [flowMap_zero])) t ⟨ht, le_rfl⟩
  simpa using this

theorem continuous_flowMap (lam β : ℝ) (N : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    Continuous (flowMap lam β N t) := by
  obtain ⟨K, hK⟩ := exists_dist_flowMap_le lam β N
  refine (LipschitzWith.of_dist_le_mul (K := ⟨Real.exp (K * t), (Real.exp_pos _).le⟩)
    fun x y => ?_).continuous
  rw [mul_comm]
  exact hK t ht x y

theorem measurable_flowMap (lam β : ℝ) (N : ℕ) {t : ℝ} (ht : 0 ≤ t) (j : Idx N) :
    Measurable fun x => flowMap lam β N t x j :=
  ((continuous_apply j).comp (continuous_flowMap lam β N ht)).measurable

/-- The field is blind to whole turns of one angle. -/
theorem gptVec_add_single (lam β : ℝ) (N : ℕ) (θ : Idx N → ℝ) (l : Idx N) (n : ℤ) :
    gptVec lam β N (θ + Pi.single l (n * (2 * π))) = gptVec lam β N θ := by
  funext j
  simp only [gptVec, gptField]
  congr 1
  refine Finset.sum_congr rfl fun k hk => ?_
  have hkj : k ≠ j := (Finset.mem_filter.1 hk).2.ne
  congr 1
  simp only [Pi.add_apply]
  by_cases hj : j = l
  · subst hj
    rw [Pi.single_eq_same, Pi.single_eq_of_ne hkj, add_zero,
      show θ j + n * (2 * π) - θ k = θ j - θ k + n * (2 * π) by ring]
    exact (wBetaDeriv_periodic β).int_mul n _
  · by_cases hk' : k = l
    · subst hk'
      rw [Pi.single_eq_of_ne hj, Pi.single_eq_same, add_zero,
        show θ j - (θ k + n * (2 * π)) = θ j - θ k + (-n : ℤ) * (2 * π) by push_cast; ring]
      exact (wBetaDeriv_periodic β).int_mul (-n) _
    · rw [Pi.single_eq_of_ne hj, Pi.single_eq_of_ne hk']
      simp

/-- **Whole turns.** Moving one initial angle by `n` turns moves that token's
trajectory by `n` turns and no other. -/
theorem flowMap_add_single (lam β : ℝ) (N : ℕ) (x : Idx N → ℝ) (l : Idx N) (n : ℤ) (t : ℝ) :
    flowMap lam β N t (x + Pi.single l (n * (2 * π))) =
      flowMap lam β N t x + Pi.single l (n * (2 * π)) := by
  have hθ : IsGPTFlow lam β N (fun s => flowMap lam β N s x + Pi.single l (n * (2 * π))) := by
    rw [isGPTFlow_iff]
    intro s
    rw [gptVec_add_single]
    exact ((isGPTFlow_iff lam β N _).1 (isGPTFlow_flowMap lam β N x) s).add_const _
  have := eq_flowMap lam β N hθ t
  simp only [flowMap_zero] at this
  exact this.symm

/-- The hypotheses of `abs_gptField_sub_le` and `norm_gptVec_sub_le` hold for the
Lipschitz constant of `exists_lipschitz_wBetaDeriv`. -/
example (lam β : ℝ) (θ θ' : Idx 2 → ℝ) :=
  have hw := (exists_lipschitz_wBetaDeriv β).choose_spec.choose_spec
  And.intro (abs_gptField_sub_le lam β 2 hw.2.2.2.1 θ θ' 1)
    (norm_gptVec_sub_le lam β 2 hw.1 hw.2.2.2.1 θ θ')

/-- The hypotheses of `eq_flowMap`, `continuous_flowMap` and `measurable_flowMap` hold for
the flow itself and `t = 1`. -/
example (lam β : ℝ) (x : Idx 2 → ℝ) :=
  And.intro (eq_flowMap lam β 2 (isGPTFlow_flowMap lam β 2 x) 1) <|
  And.intro (continuous_flowMap lam β 2 (t := 1) zero_le_one)
    (measurable_flowMap lam β 2 (t := 1) zero_le_one 0)

end Kinetic
end Transformer
