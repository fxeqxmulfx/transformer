/-
# Actual weak transport before the first control switch

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and Appendix,
the successive constant-control phases of `lem: tubular.mass.movement`.

If the original vector field agrees with a two-neuron autonomous field
before a positive switch, every weak solution agrees with its actual
characteristic law up to and including that switch. The controls after
the switch are arbitrary. Append a genuine autonomous weak solution at
the switch; the extended curve has the autonomous field globally, so
the proved global transport identity identifies the entire original prefix.

The same complete particle flow gives composition and inverse identities
for the original probability solution maps, without atomicity assumptions.
-/

import Transformer.Interpolation.Section1_WeakConcatenation
import Transformer.Interpolation.Section3_TwoRidgeLaw

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual two-neuron probability maps compose with their true clocks.
Source: arXiv:2411.04551v3, §1, the solution map of `eq: cauchy.pb`. -/
theorem twoRidgeLaw_add (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (s t : ℝ) :
    twoRidgeLaw p u b q v c (twoRidgeLaw p u b q v c ν s) t = twoRidgeLaw p u b q v c ν (s + t) := by
  apply Subtype.ext
  change Measure.map (fun x => twoRidgeFlow p u b q v c x t)
    (Measure.map (fun x => twoRidgeFlow p u b q v c x s) (ν : Measure (SSphere d))) =
      Measure.map (fun x => twoRidgeFlow p u b q v c x (s + t)) (ν : Measure (SSphere d))
  rw [Measure.map_map (measurable_twoRidgeFlow_initial _ _ _ _ _ _ t)
    (measurable_twoRidgeFlow_initial _ _ _ _ _ _ s)]
  congr 1
  funext x
  exact (twoRidgeFlow_add p u b q v c x s t).symm

/-- Reversing time inverts the actual probability solution map.
Source: arXiv:2411.04551v3, §1, the invertible characteristic map. -/
theorem twoRidgeLaw_inverse (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) :
    twoRidgeLaw p u b q v c (twoRidgeLaw p u b q v c ν t) (-t) = ν := by
  rw [twoRidgeLaw_add, add_neg_cancel, twoRidgeLaw_zero]

/-- Every actual probability solution map is a bijection on probabilities.
Source: arXiv:2411.04551v3, §1, the characteristic solution map. -/
theorem bijective_twoRidgeLaw (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c t : ℝ) :
    Function.Bijective (fun ν : Perspective.ProbSphere d => twoRidgeLaw p u b q v c ν t) := by
  have hl : Function.LeftInverse (fun ν => twoRidgeLaw p u b q v c ν (-t))
      (fun ν => twoRidgeLaw p u b q v c ν t) := fun ν => twoRidgeLaw_inverse _ _ _ _ _ _ ν t
  refine ⟨hl.injective, fun ν => ⟨twoRidgeLaw p u b q v c ν (-t), ?_⟩⟩
  simpa only [neg_neg] using twoRidgeLaw_inverse p u b q v c ν (-t)

/-- An autonomous first phase fixes every original weak law up to and
including its switch, even when the later controls are arbitrary.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_prefix (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (S : ℝ) (hS : 0 < S) (θ : TimeParams d)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ), s < S → ∀ x : SSphere d,
      fullVF d θ ρ s x = twoRidgeField p u b q v c x)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (t : ℝ) (ht : t ≤ S) :
    μ t = twoRidgeLaw p u b q v c (μ 0) t := by
  let θc : TimeParams d := fun _ => twoRidgeParams i j p u b q v c
  let ν := twoRidgeLaw p u b q v c (μ S)
  let M := concatenate S μ ν
  have hmatch : ν 0 = μ S := twoRidgeLaw_zero _ _ _ _ _ _ _
  have hM : cauchyPB d (concatenate S θ θc) M :=
    cauchyPB_concatenate S hS θ θc μ ν hμ
    (cauchyPB_twoRidgeLaw i j hij _ _ _ _ _ _ _) hmatch
  have hglobal (ρ : Perspective.ProbSphere d) (s : ℝ) (x : SSphere d) :
      fullVF d (concatenate S θ θc) ρ s x = twoRidgeField p u b q v c x := by
    have hh := fullVF_concatenate S θ θc (fun _ => ρ) (fun _ => ρ) s x
    by_cases hs : s < S
    · simp only [concatenate, ite_eq_left hs] at hh
      exact hh.trans (hfield ρ s hs x)
    · simp only [concatenate, ite_eq_right hs] at hh
      exact hh.trans (fullVF_twoRidgeParams i j hij p u b q v c ρ (s - S) x)
  obtain ⟨K, hK⟩ := exists_lipschitzOnWith_twoRidgeField p u b q v c
  have he : M t = twoRidgeLaw p u b q v c (M 0) t := by
    apply Subtype.ext
    change (M t : Measure (SSphere d)) = (twoRidgeLaw p u b q v c (M 0) t : Measure (SSphere d))
    rw [coe_twoRidgeLaw]
    exact cauchyPB_autonomous_transport _ _ (continuous_twoRidgeField p u b q v c) hglobal K hK
      (twoRidgeFlow p u b q v c) (hasDerivAt_twoRidgeFlow p u b q v c) (twoRidgeFlow_zero p u b q v c)
      (exists_lipschitzWith_twoRidgeFlow p u b q v c) M hM t
  have hzero : M 0 = μ 0 := concatenate_zero S hS μ ν
  have htime : M t = μ t := concatenate_of_le S μ ν t ht hmatch
  rwa [hzero, htime] at he

/-- The prefix identity retains the specified original initial law.
Auxiliary for arXiv:2411.04551v3, Appendix, the phase-by-phase
transport in `lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_prefix_initial (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (S : ℝ) (hS : 0 < S) (θ : TimeParams d)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ), s < S → ∀ x : SSphere d,
      fullVF d θ ρ s x = twoRidgeField p u b q v c x)
    (ν : Perspective.ProbSphere d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d θ μ) (h0 : μ 0 = ν) (t : ℝ) (ht : t ≤ S) :
    μ t = twoRidgeLaw p u b q v c ν t := by
  rw [cauchyPB_twoRidge_prefix i j hij p u b q v c S hS θ hfield μ hμ t ht, h0]

/-- The original weak solution is unique throughout an autonomous prefix.
Auxiliary for arXiv:2411.04551v3, Appendix, the control switches in
`lem: tubular.mass.movement`. -/
theorem cauchyPB_twoRidge_prefix_unique (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (S : ℝ) (hS : 0 < S) (θ : TimeParams d)
    (hfield : ∀ (ρ : Perspective.ProbSphere d) (s : ℝ), s < S → ∀ x : SSphere d,
      fullVF d θ ρ s x = twoRidgeField p u b q v c x)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (h0 : μ 0 = ν 0) (t : ℝ) (ht : t ≤ S) : μ t = ν t := by
  rw [cauchyPB_twoRidge_prefix i j hij p u b q v c S hS θ hfield μ hμ t ht,
    cauchyPB_twoRidge_prefix i j hij p u b q v c S hS θ hfield ν hν t ht, h0]

/-- Two actual nonzero neurons, two different control speeds and matching
original weak laws jointly realize all prefix and initial-data hypotheses.
The later phase is an actual changed original field. -/
example (ν : Perspective.ProbSphere 2) : ∃ (θ : TimeParams 2)
    (μ η : ℝ → Perspective.ProbSphere 2) (p q : EucSpace 2),
    (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧
    cauchyPB 2 θ μ ∧ cauchyPB 2 θ η ∧ μ 0 = ν ∧ μ 0 = η 0 ∧
    (∀ (ρ : Perspective.ProbSphere 2) (s : ℝ), s < 1 → ∀ x : SSphere 2,
      fullVF 2 θ ρ s x = twoRidgeField p 0 1 q 0 1 x) := by
  let p : EucSpace 2 := basePoint 1
  let q : EucSpace 2 := EuclideanSpace.single 1 1
  let θ₁ : TimeParams 2 := fun _ => twoRidgeParams 0 1 p 0 1 q 0 1
  let θ₂ : TimeParams 2 := fun _ => twoRidgeParams 0 1 ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1
  let μ₁ := twoRidgeLaw p 0 1 q 0 1 ν
  let μ₂ := twoRidgeLaw ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1 (μ₁ 1)
  have hmatch : μ₂ 0 = μ₁ 1 := twoRidgeLaw_zero _ _ _ _ _ _ _
  have hweak := cauchyPB_concatenate 1 zero_lt_one θ₁ θ₂ μ₁ μ₂
    (cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _)
    (cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _) hmatch
  refine ⟨concatenate 1 θ₁ θ₂, concatenate 1 μ₁ μ₂, concatenate 1 μ₁ μ₂,
    p, q, by norm_num, zero_lt_one, le_rfl, hweak, hweak, ?_, rfl, ?_⟩
  · rw [concatenate_zero 1 zero_lt_one]
    exact twoRidgeLaw_zero p 0 1 q 0 1 ν
  · intro ρ s hs x
    have hh := fullVF_concatenate 1 θ₁ θ₂ (fun _ => ρ) (fun _ => ρ) s x
    simp only [concatenate, ite_eq_left hs] at hh
    exact hh.trans (fullVF_twoRidgeParams 0 1 (by norm_num) p 0 1 q 0 1 ρ s x)

end Transformer.Interpolation
