module

public import Transformer.Interpolation.Section3_TwoRidgeFlow
public import Transformer.Interpolation.Section1_AutonomousLaw

/-!
# Original weak well-posedness for the constant two-neuron construction

ArXiv:2411.04551v3, §1, `eq: vf`, `eq: ips`, and `eq: cauchy.pb`,
and Appendix, the constant-control construction toward `lem: two.balls`.

The actual complete two-neuron flow has the required spatial and temporal
regularity by the original particle equation. Pushing any initial probability
along it therefore gives an actual original global weak solution.
The characteristic identity identifies every weak solution with that law,
so existence and uniqueness hold without any atomicity assumption.

If both activation regions lie in a measurable set, its mass is preserved
by the actual flow and by every original weak solution. This is the mass
accounting needed for the subsequent concentration argument.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every actual original flow slice is Lipschitz in its initial point.
Source: arXiv:2411.04551v3, §1, the Lipschitz characteristic flow. -/
theorem exists_lipschitzWith_twoRidgeFlow (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c t : ℝ) :
    ∃ L : ℝ≥0, LipschitzWith L (fun x : SSphere d => twoRidgeFlow p u b q v c x t) := by
  obtain ⟨K, hK⟩ := exists_lipschitzOnWith_twoRidgeField p u b q v c
  exact exists_lipschitzWith_sphericalFlow_initial _ K hK (twoRidgeFlow p u b q v c)
    (hasDerivAt_twoRidgeFlow p u b q v c) (twoRidgeFlow_zero p u b q v c) t

/-- The actual original two-neuron flow is jointly continuous.
Source: arXiv:2411.04551v3, §1, the characteristic flow. -/
theorem continuous_twoRidgeFlow (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) :
    Continuous (fun r : ℝ × SSphere d => twoRidgeFlow p u b q v c r.2 r.1) := by
  obtain ⟨K, hK⟩ := exists_lipschitzOnWith_twoRidgeField p u b q v c
  exact continuous_sphericalFlow _ (continuous_twoRidgeField p u b q v c) K hK
    (twoRidgeFlow p u b q v c) (hasDerivAt_twoRidgeFlow p u b q v c) (twoRidgeFlow_zero p u b q v c)

/-- Every actual original flow slice is measurable.
Source: arXiv:2411.04551v3, §1, the characteristic pushforward. -/
theorem measurable_twoRidgeFlow_initial (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c t : ℝ) :
    Measurable (fun x : SSphere d => twoRidgeFlow p u b q v c x t) :=
  ((continuous_twoRidgeFlow p u b q v c).comp (continuous_const.prodMk continuous_id)).measurable

/-- The original probability transported by the actual two-neuron flow.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
noncomputable def twoRidgeLaw (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) : Perspective.ProbSphere d :=
  sphericalFlowLaw (twoRidgeFlow p u b q v c) ν t

/-- The constructed law is the exact original characteristic pushforward.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem coe_twoRidgeLaw (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) :
    (twoRidgeLaw p u b q v c ν t : Measure (SSphere d)) =
      Measure.map (fun x => twoRidgeFlow p u b q v c x t) (ν : Measure (SSphere d)) := rfl

/-- The actual transported law has the prescribed initial probability.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem twoRidgeLaw_zero (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (ν : Perspective.ProbSphere d) :
    twoRidgeLaw p u b q v c ν 0 = ν := sphericalFlowLaw_zero _ (twoRidgeFlow_zero p u b q v c) ν

/-- Continuous tests integrate against the actual original trajectories.
Source: arXiv:2411.04551v3, §1, the characteristic formula. -/
theorem integral_twoRidgeLaw (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (f : EucSpace d → ℝ) (hf : Continuous f) (t : ℝ) :
    (∫ x : SSphere d, f x ∂(twoRidgeLaw p u b q v c ν t : Measure (SSphere d))) =
      ∫ x : SSphere d, f (twoRidgeFlow p u b q v c x t) ∂(ν : Measure (SSphere d)) :=
  integral_sphericalFlowLaw _ (continuous_twoRidgeFlow p u b q v c) ν f hf t

/-- The actual two-neuron pushforward solves the original weak equation.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem cauchyPB_twoRidgeLaw (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (ν : Perspective.ProbSphere d) :
    cauchyPB d (fun _ => twoRidgeParams i j p u b q v c) (twoRidgeLaw p u b q v c ν) :=
  cauchyPB_sphericalFlowLaw _ _ (continuous_twoRidgeField p u b q v c)
    (fun ρ s x => fullVF_twoRidgeParams i j hij p u b q v c ρ s x)
    (twoRidgeFlow p u b q v c) (continuous_twoRidgeFlow p u b q v c)
    (hasDerivAt_twoRidgeFlow p u b q v c) ν

/-- Every actual original weak solution is the actual flow pushforward.
Source: arXiv:2411.04551v3, §1, `eq: ips` and `eq: cauchy.pb`. -/
theorem cauchyPB_twoRidge_transport (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => twoRidgeParams i j p u b q v c) μ) (t : ℝ) :
    μ t = twoRidgeLaw p u b q v c (μ 0) t := by
  obtain ⟨K, hK⟩ := exists_lipschitzOnWith_twoRidgeField p u b q v c
  apply Subtype.ext
  change (μ t : Measure (SSphere d)) = (twoRidgeLaw p u b q v c (μ 0) t : Measure (SSphere d))
  rw [coe_twoRidgeLaw]
  exact cauchyPB_autonomous_transport _ _ (continuous_twoRidgeField p u b q v c)
    (fun ρ s x => fullVF_twoRidgeParams i j hij p u b q v c ρ s x) K hK
    (twoRidgeFlow p u b q v c) (hasDerivAt_twoRidgeFlow p u b q v c) (twoRidgeFlow_zero p u b q v c)
    (exists_lipschitzWith_twoRidgeFlow p u b q v c) μ hμ t

/-- The entire original weak curve is fixed by its initial probability.
Source: arXiv:2411.04551v3, §1, well-posedness of `eq: cauchy.pb`. -/
theorem cauchyPB_twoRidge_eq (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (ρ : Perspective.ProbSphere d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (fun _ => twoRidgeParams i j p u b q v c) μ)
    (h0 : μ 0 = ρ) : μ = twoRidgeLaw p u b q v c ρ := by
  funext t
  rw [cauchyPB_twoRidge_transport i j hij p u b q v c μ hμ t, h0]

/-- The actual original two-neuron controls have global weak uniqueness.
Source: arXiv:2411.04551v3, §1, well-posedness of `eq: cauchy.pb`. -/
theorem cauchyPB_twoRidge_unique (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (μ ν : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => twoRidgeParams i j p u b q v c) μ)
    (hν : cauchyPB d (fun _ => twoRidgeParams i j p u b q v c) ν) (h0 : μ 0 = ν 0) : μ = ν := by
  rw [cauchyPB_twoRidge_eq i j hij p u b q v c (μ 0) μ hμ rfl,
    cauchyPB_twoRidge_eq i j hij p u b q v c (μ 0) ν hν h0.symm]

/-- Every initial probability has exactly one actual original weak law.
Source: arXiv:2411.04551v3, §1, well-posedness of `eq: cauchy.pb`. -/
theorem exists_unique_cauchyPB_twoRidge (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (ρ : Perspective.ProbSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧ cauchyPB d (fun _ => twoRidgeParams i j p u b q v c) μ := by
  refine ⟨twoRidgeLaw p u b q v c ρ, ⟨twoRidgeLaw_zero _ _ _ _ _ _ _, cauchyPB_twoRidgeLaw i j hij _ _ _ _ _ _ _⟩,
    fun μ hμ => cauchyPB_twoRidge_eq i j hij p u b q v c ρ μ hμ.2 hμ.1⟩


/-- The actual transported law preserves the mass of the invariant region.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem twoRidgeLaw_mass_invariant (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (S : Set (SSphere d)) (hS : ∀ x : SSphere d, x ∉ S →
      b - inner ℝ (x : EucSpace d) u ≤ 0 ∧ c - inner ℝ (x : EucSpace d) v ≤ 0)
    (hs : MeasurableSet S) (ρ : Perspective.ProbSphere d) (t : ℝ) :
    (twoRidgeLaw p u b q v c ρ t : Measure (SSphere d)) S = (ρ : Measure (SSphere d)) S := by
  rw [coe_twoRidgeLaw, Measure.map_apply (measurable_twoRidgeFlow_initial p u b q v c t) hs]
  congr 1
  ext x
  exact twoRidgeFlow_mem_iff p u b q v c S hS x t

/-- Every original weak solution preserves the invariant-region mass.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem cauchyPB_twoRidge_mass_invariant (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (S : Set (SSphere d)) (hS : ∀ x : SSphere d, x ∉ S →
      b - inner ℝ (x : EucSpace d) u ≤ 0 ∧ c - inner ℝ (x : EucSpace d) v ≤ 0)
    (hs : MeasurableSet S) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => twoRidgeParams i j p u b q v c) μ) (t : ℝ) :
    (μ t : Measure (SSphere d)) S = (μ 0 : Measure (SSphere d)) S := by
  rw [cauchyPB_twoRidge_transport i j hij p u b q v c μ hμ t]
  exact twoRidgeLaw_mass_invariant p u b q v c S hS hs (μ 0) t

/-- Two actual neurons, a transported two-atom probability, a genuine
activation region and a continuous linear test jointly realize all hypotheses. -/
example : ∃ (p q u : EucSpace 2) (ρ : Perspective.ProbSphere 2)
    (μ ν : ℝ → Perspective.ProbSphere 2) (f : EucSpace 2 → ℝ) (S : Set (SSphere 2)),
    (0 : Fin 2) ≠ 1 ∧ cauchyPB 2 (fun _ => twoRidgeParams 0 1 p u 0 q u 0) μ ∧
    cauchyPB 2 (fun _ => twoRidgeParams 0 1 p u 0 q u 0) ν ∧ μ 0 = ρ ∧ μ 0 = ν 0 ∧ Continuous f ∧
    MeasurableSet S ∧ (∀ x : SSphere 2, x ∉ S →
      0 - inner ℝ (x : EucSpace 2) u ≤ 0 ∧ 0 - inner ℝ (x : EucSpace 2) u ≤ 0) := by
  let p : EucSpace 2 := basePoint 1
  let q : EucSpace 2 := EuclideanSpace.single 1 1
  let u := p
  let ρ := halfDirac 2 (basePoint 1) (antipode 2 (basePoint 1))
  let μ := twoRidgeLaw p u 0 q u 0 ρ
  let S := {x : SSphere 2 | 0 < 0 - inner ℝ (x : EucSpace 2) u}
  have hμ := cauchyPB_twoRidgeLaw (0 : Fin 2) 1 (by norm_num) p u 0 q u 0 ρ
  refine ⟨p, q, u, ρ, μ, μ, fun x => inner ℝ p x, S, by norm_num,
    hμ, hμ, twoRidgeLaw_zero _ _ _ _ _ _ _, rfl, continuous_const.inner continuous_id, ?_, ?_⟩
  · exact (isOpen_lt continuous_const (continuous_const.sub
      (continuous_subtype_val.inner continuous_const))).measurableSet
  · intro x hx
    exact ⟨not_lt.mp hx, not_lt.mp hx⟩

end Transformer.Interpolation
