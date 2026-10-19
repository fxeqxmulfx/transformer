module

public import Transformer.Causal.Section4_AnisotropicMargins
public import Transformer.Causal.Section5_GapBarrier

/-!
# A genuine forward-invariant region away from the leading axis

Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5` and
`eqn:csa`, at `V=diag(2,1)`, `Q=I`, `K=diag(0,4)`, `β=1`.
The original two-token region has six strictly positive coordinate
margins. Their proved actual derivatives point inward at every
feasible edge. The finite first-crossing theorem therefore keeps
every interior angular solution in this region for all future time.

Global ODE existence and the original circle chart give a genuine
solution of the original masked causal dynamics through each initial
point in the region. Its follower's second coordinate remains above
one quarter, so it cannot converge to either leading eigenvector.
The positive-measure argument for the source's almost-everywhere
quantifier is separate; no conclusion about that measure is assumed here.
-/

@[expose] public section

open Real Set Filter Topology

namespace Transformer.Causal

open Perceptron

/-- The actual open coordinate region in the original two-token sphere.
Source: arXiv:2411.04990v2, §4, the original state space of `eqn:csa`. -/
def anisotropicInitialRegion (X : SphereTuple 2 2) : Prop :=
  |(X 0 : EucSpace 2) 1| < 1 / 100 ∧ 99 / 100 < (X 0 : EucSpace 2) 0 ∧
    (X 1 : EucSpace 2) 1 ∈ Ioo (1 / 4) (1 / 2) ∧ (X 1 : EucSpace 2) 0 < 0

/-- The original coordinate inequalities agree with positivity of all
six actual margins. Source: arXiv:2411.04990v2, §4, the circle chart of `eqn:csa`. -/
theorem anisotropicInitialRegion_iff_margins (φ : Fin 2 → ℝ) :
    anisotropicInitialRegion (fun i => circlePoint (φ i)) ↔
      ∀ j : Fin 6, 0 < anisotropicMargins φ j := by
  unfold anisotropicInitialRegion
  simp only [circlePoint_coord_zero, circlePoint_coord_one]
  constructor
  · rintro ⟨hs₀, hc₀, hs₁, hc₁⟩
    have habs := abs_lt.mp hs₀
    intro j
    fin_cases j
    · change 0 < 1 / 100 - Real.sin (φ 0); linarith [habs.2]
    · change 0 < 1 / 100 + Real.sin (φ 0); linarith [habs.1]
    · change 0 < Real.cos (φ 0) - 99 / 100; linarith
    · change 0 < Real.sin (φ 1) - 1 / 4; linarith [hs₁.1]
    · change 0 < 1 / 2 - Real.sin (φ 1); linarith [hs₁.2]
    · change 0 < -Real.cos (φ 1); linarith
  · intro h
    have h₀ := h 0; have h₁ := h 1; have h₂ := h 2
    have h₃ := h 3; have h₄ := h 4; have h₅ := h 5
    change 0 < 1 / 100 - Real.sin (φ 0) at h₀
    change 0 < 1 / 100 + Real.sin (φ 0) at h₁
    change 0 < Real.cos (φ 0) - 99 / 100 at h₂
    change 0 < Real.sin (φ 1) - 1 / 4 at h₃
    change 0 < 1 / 2 - Real.sin (φ 1) at h₄
    change 0 < -Real.cos (φ 1) at h₅
    exact ⟨abs_lt.mpr ⟨by linarith, by linarith⟩, by linarith, ⟨by linarith, by linarith⟩, by linarith⟩

/-- Every original interior angular solution keeps all six margins
strictly positive for all nonnegative time. Source: arXiv:2411.04990v2,
§4, `eqn:csa`, using the actual derivatives and proved inward signs.
The finite first-crossing argument is the one proved in §5 for `lemma:meta`. -/
theorem anisotropicMargins_pos_for_nonneg_time (φ : ℝ → Fin 2 → ℝ)
    (hφ : ∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t)
    (hinit : ∀ j : Fin 6, 0 < anisotropicMargins (φ 0) j) :
    ∀ t : ℝ, 0 ≤ t → ∀ j : Fin 6, 0 < anisotropicMargins (φ t) j := by
  intro t ht j
  have h := finite_gap_barrier Finset.univ
    (fun i s => anisotropicMargins (φ s) i) (fun i s => anisotropicMarginVelocities (φ s) i) t
    (fun i _ s => hasDerivAt_anisotropicMargins φ hφ s i) (fun i _ => hinit i)
    (fun s _ hall i _ hi => anisotropicMarginVelocities_pos_on_boundary (φ s)
      (fun k => hall k (Finset.mem_univ k)) i hi)
  exact h t ⟨ht, le_rfl⟩ j (Finset.mem_univ j)

/-- An actual unit-circle pair lies strictly inside the original region.
Source: arXiv:2411.04990v2, §4, `eqn:csa`, with leader at its top eigenvector
and follower sine three eighths in the negative-cosine semicircle. -/
theorem anisotropicInitialRegion_witness :
    anisotropicInitialRegion (fun i => circlePoint ((![0, π - Real.arcsin (3 / 8)] : Fin 2 → ℝ) i)) := by
  have hs : Real.sin (π - Real.arcsin (3 / 8 : ℝ)) = 3 / 8 := by
    rw [Real.sin_pi_sub, Real.sin_arcsin (by norm_num) (by norm_num)]
  unfold anisotropicInitialRegion
  simp only [circlePoint_coord_zero, circlePoint_coord_one,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_fin_one]
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  · rw [hs]
    norm_num
  · rw [Real.cos_pi_sub, Real.cos_arcsin]
    exact neg_lt_zero.mpr (Real.sqrt_pos.mpr (by norm_num))

/-- Every original initial point in the region has a true global causal
solution remaining there for all future time. Source: arXiv:2411.04990v2,
§4, the actual `eqn:csa` at the three specified matrices.
The invariant behavior is proved for a genuine original solution, not supplied
as an assumption on the dynamics or built into its definition. -/
theorem exists_anisotropic_CSA_in_region (X₀ : SphereTuple 2 2)
    (hinit : anisotropicInitialRegion X₀) :
    ∃ X : ℝ → SphereTuple 2 2, X 0 = X₀ ∧
      CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) anisotropicKeyMap anisotropicValueMap X ∧
      ∀ t : ℝ, 0 ≤ t → anisotropicInitialRegion (X t) := by
  choose φ₀ hφ₀ using fun i : Fin 2 => circlePoint_surjective (X₀ i)
  have heq : (fun i => circlePoint (φ₀ i)) = X₀ := funext hφ₀
  have hm : ∀ j : Fin 6, 0 < anisotropicMargins φ₀ j :=
    (anisotropicInitialRegion_iff_margins φ₀).mp (heq.symm ▸ hinit)
  obtain ⟨φ, h0, hφ⟩ := exists_anisotropicAngleFlow φ₀
  have hi : ∀ j : Fin 6, 0 < anisotropicMargins (φ 0) j := by simpa only [h0] using hm
  refine ⟨fun t i => circlePoint (φ t i), ?_, CSA_of_anisotropicAngleFlow φ hφ, ?_⟩
  · funext i
    change circlePoint (φ 0 i) = X₀ i
    rw [h0]
    exact hφ₀ i
  · intro t ht
    exact (anisotropicInitialRegion_iff_margins (φ t)).mpr
      (anisotropicMargins_pos_for_nonneg_time φ hφ hi t ht)

/-- A follower staying in the proved region cannot tend to any point
of the leading coordinate axis. Source: arXiv:2411.04990v2, §4,
the proposed limits `±ξ` in Conjecture `thm1.5`.
Its original second coordinate stays above one quarter for all future time. -/
theorem not_tendsto_axis_of_anisotropicRegion (X : ℝ → SphereTuple 2 2)
    (hX : ∀ t : ℝ, 0 ≤ t → anisotropicInitialRegion (X t)) (v : EucSpace 2) (hv : v 1 = 0) :
    ¬ Tendsto (fun t => (X t 1 : EucSpace 2)) atTop (nhds v) := by
  intro hlim
  have hl : Tendsto (fun t => (X t 1 : EucSpace 2) 1) atTop (nhds (v 1)) :=
    ((EuclideanSpace.proj (1 : Fin 2) : EucSpace 2 →L[ℝ] ℝ).continuous.tendsto v).comp hlim
  rw [hv] at hl
  have hbound : ∀ᶠ t : ℝ in atTop, (1 / 4 : ℝ) ≤ (X t 1 : EucSpace 2) 1 := by
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
    exact (hX t ht).2.2.1.1.le
  have h := ge_of_tendsto hl hbound
  norm_num at h

/-- The forward-invariance hypotheses hold jointly for a genuine
global solution from the strictly interior witness, on the original circle.
Source: arXiv:2411.04990v2, §4, `eqn:csa`. -/
example : ∃ φ : ℝ → Fin 2 → ℝ,
    (∀ t, HasDerivAt φ (anisotropicAngleField (φ t)) t) ∧
    (∀ j : Fin 6, 0 < anisotropicMargins (φ 0) j) := by
  obtain ⟨φ, h0, hφ⟩ := exists_anisotropicAngleFlow ![0, π - Real.arcsin (3 / 8)]
  refine ⟨φ, hφ, ?_⟩
  rw [h0]
  exact (anisotropicInitialRegion_iff_margins _).mp anisotropicInitialRegion_witness

/-- The original-region and limiting-axis hypotheses are realized on
the true global trajectory from the interior witness, with the original
leading unit vector. Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`. -/
example : ∃ X : ℝ → SphereTuple 2 2,
    CSA 2 2 1 (ContinuousLinearMap.id ℝ (EucSpace 2)) anisotropicKeyMap anisotropicValueMap X ∧
    (∀ t : ℝ, 0 ≤ t → anisotropicInitialRegion (X t)) ∧ (basePoint 1 : EucSpace 2) 1 = 0 := by
  obtain ⟨X, -, hX, hi⟩ := exists_anisotropic_CSA_in_region _ anisotropicInitialRegion_witness
  exact ⟨X, hX, hi, by simp [basePoint]⟩

end Transformer.Causal
