module

public import Transformer.Interpolation.Section1_AutonomousTail

/-!
# Joining an already well-posed original control prefix to one more phase

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and Appendix,
the induction over phases in `lem: tubular.mass.movement`.

An already constructed prefix continues with its last constant field
after its horizon. Every weak solution for the newly joined controls can
be extended at the switch with that old field, producing an original weak
solution for the old controls. The old uniqueness theorem therefore fixes
the entire new solution before the switch. Exact autonomous-tail transport
then identifies every later law.

Existence and uniqueness for the prefix are explicit induction inputs.
The theorem derives these properties for the actual appended two-neuron
controls and every initial probability; it does not assume them for the
joined equation.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Extending with the old constant tail recovers the original prefix
weak equation, so its proved uniqueness fixes every new weak prefix.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem cauchyPB_concatenate_prefix_of_constant_tail (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (hS : 0 < S)
    (θ₁ θ₂ : TimeParams d)
    (htail : ∀ s : ℝ, S ≤ s → θ₁ s = twoRidgeParams i j p u b q v c)
    (hunique : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ₁ α → cauchyPB d θ₁ β →
      α 0 = β 0 → α = β)
    (μ₁ μ : ℝ → Perspective.ProbSphere d) (hμ₁ : cauchyPB d θ₁ μ₁)
    (hμ : cauchyPB d (concatenate S θ₁ θ₂) μ) (h0 : μ 0 = μ₁ 0) (t : ℝ) (ht : t ≤ S) :
    μ t = μ₁ t := by
  let θc : TimeParams d := fun _ => twoRidgeParams i j p u b q v c
  let ν := twoRidgeLaw p u b q v c (μ S)
  let M := concatenate S μ ν
  have hmatch : ν 0 = μ S := twoRidgeLaw_zero _ _ _ _ _ _ _
  have hM := cauchyPB_concatenate S hS (concatenate S θ₁ θ₂) θc μ ν hμ
    (cauchyPB_twoRidgeLaw i j hij _ _ _ _ _ _ _) hmatch
  have hθ : concatenate S (concatenate S θ₁ θ₂) θc = θ₁ := by
    funext s
    by_cases hs : s < S
    · simp only [concatenate, ite_eq_left hs]
    · simp only [concatenate, ite_eq_right hs]
      exact (htail s (le_of_not_gt hs)).symm
  rw [hθ] at hM
  have he : M = μ₁ :=
    hunique M μ₁ hM hμ₁ ((concatenate_zero S hS μ ν).trans h0)
  have htime : M t = μ t := concatenate_of_le S μ ν t ht hmatch
  exact htime.symm.trans (congrFun he t)

/-- Actual prefix uniqueness and exact final-phase transport give global
weak uniqueness for the appended controls. Auxiliary for arXiv:2411.04551v3,
Appendix, the control switches in `lem: tubular.mass.movement`. -/
theorem cauchyPB_concatenate_twoRidge_unique (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (hS : 0 < S)
    (θ : TimeParams d) (htail : ∀ s : ℝ, S ≤ s → θ s = twoRidgeParams i j p u b q v c)
    (hexists : ∀ ρ : Perspective.ProbSphere d,
      ∃ η : ℝ → Perspective.ProbSphere d, η 0 = ρ ∧ cauchyPB d θ η)
    (hunique : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ α → cauchyPB d θ β →
      α 0 = β 0 → α = β)
    (p' u' : EucSpace d) (b' : ℝ) (q' v' : EucSpace d) (c' : ℝ)
    (μ ν : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (concatenate S θ (fun _ => twoRidgeParams i j p' u' b' q' v' c')) μ)
    (hν : cauchyPB d (concatenate S θ (fun _ => twoRidgeParams i j p' u' b' q' v' c')) ν)
    (h0 : μ 0 = ν 0) : μ = ν := by
  obtain ⟨η, hη0, hη⟩ := hexists (μ 0)
  have hpreμ (t : ℝ) (ht : t ≤ S) : μ t = η t :=
    cauchyPB_concatenate_prefix_of_constant_tail i j hij p u b q v c S hS θ _ htail hunique
      η μ hη hμ hη0.symm t ht
  have hpreν (t : ℝ) (ht : t ≤ S) : ν t = η t :=
    cauchyPB_concatenate_prefix_of_constant_tail i j hij p u b q v c S hS θ _ htail hunique
      η ν hη hν (h0.symm.trans hη0.symm) t ht
  have hmatch : μ S = ν S := (hpreμ S le_rfl).trans (hpreν S le_rfl).symm
  funext t
  by_cases ht : t ≤ S
  · exact (hpreμ t ht).trans (hpreν t ht).symm
  · have hs : S ≤ t := (lt_of_not_ge ht).le
    rw [cauchyPB_twoRidge_concatenate_tail i j hij p' u' b' q' v' c' S θ μ hμ t hs,
      cauchyPB_twoRidge_concatenate_tail i j hij p' u' b' q' v' c' S θ ν hν t hs, hmatch]

/-- An already well-posed prefix admits one more actual constant phase,
retaining original global weak existence and uniqueness for every initial
probability. Auxiliary for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`. -/
theorem exists_unique_cauchyPB_concatenate_twoRidge (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (hS : 0 < S)
    (θ : TimeParams d) (htail : ∀ s : ℝ, S ≤ s → θ s = twoRidgeParams i j p u b q v c)
    (hexists : ∀ ρ : Perspective.ProbSphere d,
      ∃ η : ℝ → Perspective.ProbSphere d, η 0 = ρ ∧ cauchyPB d θ η)
    (hunique : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ α → cauchyPB d θ β →
      α 0 = β 0 → α = β)
    (p' u' : EucSpace d) (b' : ℝ) (q' v' : EucSpace d) (c' : ℝ) (ν : Perspective.ProbSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧
      cauchyPB d (concatenate S θ (fun _ => twoRidgeParams i j p' u' b' q' v' c')) μ := by
  have hsecond (ρ : Perspective.ProbSphere d) : ∃ μ : ℝ → Perspective.ProbSphere d,
      μ 0 = ρ ∧ cauchyPB d (fun _ => twoRidgeParams i j p' u' b' q' v' c') μ :=
    ⟨twoRidgeLaw p' u' b' q' v' c' ρ, twoRidgeLaw_zero _ _ _ _ _ _ _,
      cauchyPB_twoRidgeLaw i j hij _ _ _ _ _ _ _⟩
  obtain ⟨μ, h0, hμ⟩ := exists_cauchyPB_concatenate S hS θ
    (fun _ => twoRidgeParams i j p' u' b' q' v' c') ν (hexists ν) hsecond
  refine ⟨μ, ⟨h0, hμ⟩, fun η hη => ?_⟩
  exact cauchyPB_concatenate_twoRidge_unique i j hij p u b q v c S hS θ htail hexists hunique
    p' u' b' q' v' c' η μ hη.2 hμ (hη.1.trans h0.symm)

/-- A genuine constant two-neuron prefix has actual existence for every
law, actual global uniqueness and its true constant tail. Joining a second
phase at twice the speed supplies the new weak curves and matching data. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (θ : TimeParams 2) (p q : EucSpace 2)
    (μ₁ μ ν : ℝ → Perspective.ProbSphere 2),
    (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ s : ℝ, 1 ≤ s → θ s = twoRidgeParams 0 1 p 0 1 q 0 1) ∧
    (∀ σ : Perspective.ProbSphere 2, ∃ η : ℝ → Perspective.ProbSphere 2,
      η 0 = σ ∧ cauchyPB 2 θ η) ∧
    (∀ α β : ℝ → Perspective.ProbSphere 2, cauchyPB 2 θ α → cauchyPB 2 θ β →
      α 0 = β 0 → α = β) ∧ cauchyPB 2 θ μ₁ ∧
    cauchyPB 2 (concatenate 1 θ
      (fun _ => twoRidgeParams 0 1 ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1)) μ ∧
    cauchyPB 2 (concatenate 1 θ
      (fun _ => twoRidgeParams 0 1 ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1)) ν ∧
    μ 0 = μ₁ 0 ∧ μ 0 = ν 0 ∧ μ₁ 0 = ρ := by
  let p : EucSpace 2 := basePoint 1
  let q : EucSpace 2 := EuclideanSpace.single 1 1
  let θ₁ : TimeParams 2 := fun _ => twoRidgeParams 0 1 p 0 1 q 0 1
  let θ₂ : TimeParams 2 :=
    fun _ => twoRidgeParams 0 1 ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1
  let μ₁ := twoRidgeLaw p 0 1 q 0 1 ρ
  let μ₂ :=
    twoRidgeLaw ((2 : ℝ) • p) 0 1 ((2 : ℝ) • q) 0 1 (μ₁ 1)
  have h₁ : cauchyPB 2 θ₁ μ₁ := cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _
  have hmatch : μ₂ 0 = μ₁ 1 := twoRidgeLaw_zero _ _ _ _ _ _ _
  have hweak := cauchyPB_concatenate 1 zero_lt_one θ₁ θ₂ μ₁ μ₂ h₁
    (cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _) hmatch
  refine ⟨θ₁, p, q, μ₁, concatenate 1 μ₁ μ₂, concatenate 1 μ₁ μ₂,
    by norm_num, zero_lt_one, zero_le_one, fun _ _ => rfl, ?_, ?_,
    h₁, hweak, hweak, concatenate_zero 1 zero_lt_one μ₁ μ₂, rfl,
    twoRidgeLaw_zero _ _ _ _ _ _ _⟩
  · intro σ
    exact ⟨twoRidgeLaw p 0 1 q 0 1 σ, twoRidgeLaw_zero _ _ _ _ _ _ _,
      cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _⟩
  · intro α β hα hβ h0
    exact cauchyPB_twoRidge_unique 0 1 (by norm_num) p 0 1 q 0 1 α β hα hβ h0

end Transformer.Interpolation
