/-
# Exact weak laws for appended neural transport phases

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and Appendix,
the successive particle pushforwards in `lem: tubular.mass.movement`.

Every weak solution for the joined controls agrees with the established
prefix through the switch. Its later laws are the exact autonomous
second-phase pushforwards of the switching law. Composition of measurable
pushforwards then gives the joined particle flow at every real time.

The prefix's actual weak existence, uniqueness and representation are
explicit induction inputs. The new representation is proved for every
weak solution of the actual joined controls. The zero original neural
control also gives the identity flow and the constant law, supplying an
initial idle phase without any additional switch in the source bound.
-/

import Transformer.Interpolation.Section1_PhaseConcatenation
import Transformer.Interpolation.Section1_NeuralConcatenation
import Transformer.Interpolation.Section1_ControlPartitions

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- An exact characteristic formula implies uniqueness for the original
weak equation. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem cauchyPB_neural_unique_of_map
    (W U : ℝ → ParamMatrix d) (a : ℝ → EucSpace d) (φ : ℝ → SSphere d → SSphere d)
    (hmap : ∀ η : ℝ → Perspective.ProbSphere d, cauchyPB d (neuralParams d W U a) η →
      ∀ t, (η t : Measure (SSphere d)) = Measure.map (φ t) (η 0 : Measure (SSphere d)))
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (neuralParams d W U a) μ)
    (hν : cauchyPB d (neuralParams d W U a) ν) (h0 : μ 0 = ν 0) : μ = ν := by
  funext t
  apply Subtype.ext
  change (μ t : Measure (SSphere d)) = (ν t : Measure (SSphere d))
  rw [hmap μ hμ t, hmap ν hν t, h0]

/-- Every original weak law is the exact joined-flow pushforward at every
real time. Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, its succession of constant phases. -/
theorem cauchyPB_neural_concatenate_twoRidge_map (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c S : ℝ) (hS : 0 < S)
    (W U : ℝ → ParamMatrix d) (a : ℝ → EucSpace d) (φ : ℝ → SSphere d → SSphere d)
    (htail : ∀ s : ℝ, S ≤ s → neuralParams d W U a s = twoRidgeParams i j p u b q v c)
    (hexists : ∀ ρ : Perspective.ProbSphere d,
      ∃ η : ℝ → Perspective.ProbSphere d, η 0 = ρ ∧ cauchyPB d (neuralParams d W U a) η)
    (hunique : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d (neuralParams d W U a) α →
      cauchyPB d (neuralParams d W U a) β → α 0 = β 0 → α = β)
    (hmap : ∀ η : ℝ → Perspective.ProbSphere d, cauchyPB d (neuralParams d W U a) η →
      ∀ t, (η t : Measure (SSphere d)) = Measure.map (φ t) (η 0 : Measure (SSphere d)))
    (hmeas : Measurable (φ S))
    (p' u' : EucSpace d) (b' : ℝ) (q' v' : EucSpace d) (c' : ℝ)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (concatenate S (neuralParams d W U a)
      (fun _ => twoRidgeParams i j p' u' b' q' v' c')) μ) (t : ℝ) :
    (μ t : Measure (SSphere d)) = Measure.map
      (concatenateFlow S φ (fun s x => twoRidgeFlow p' u' b' q' v' c' x s) t)
      (μ 0 : Measure (SSphere d)) := by
  obtain ⟨η, hη0, hη⟩ := hexists (μ 0)
  have hpre (s : ℝ) (hs : s ≤ S) : μ s = η s :=
    cauchyPB_concatenate_prefix_of_constant_tail i j hij p u b q v c S hS
      (neuralParams d W U a) _ htail hunique η μ hη hμ hη0.symm s hs
  by_cases ht : t < S
  · have hflow : concatenateFlow S φ (fun s x => twoRidgeFlow p' u' b' q' v' c' x s) t = φ t := by
      funext x
      exact concatenateFlow_of_lt S _ _ t ht x
    rw [hpre t ht.le, hmap η hη t, hη0, hflow]
  · have hs : S ≤ t := le_of_not_gt ht
    rw [cauchyPB_twoRidge_concatenate_tail i j hij p' u' b' q' v' c' S
      (neuralParams d W U a) μ hμ t hs, coe_twoRidgeLaw, hpre S le_rfl,
      hmap η hη S, hη0,
      Measure.map_map (measurable_twoRidgeFlow_initial _ _ _ _ _ _ _) hmeas]
    congr 1
    funext x
    exact (concatenateFlow_of_ge S φ (fun s x => twoRidgeFlow p' u' b' q' v' c' x s) t hs x).symm

/-- The actual original flow of six zero neuron data is the identity.
Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the initial idle phase. -/
theorem twoRidgeFlow_zero_data (x : SSphere d) (t : ℝ) :
    twoRidgeFlow (0 : EucSpace d) 0 0 0 0 0 x t = x := by
  apply twoRidgeFlow_inactive
  · simp only [inner_zero_right, sub_self, le_refl]
  · simp only [inner_zero_right, sub_self, le_refl]

/-- Every original weak law for zero neuron data is constant.
Auxiliary for arXiv:2411.04551v3, Appendix,
`lem: tubular.mass.movement`, the induction's idle initial phase. -/
theorem cauchyPB_twoRidge_zero_data (i j : Fin d) (hij : i ≠ j)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => twoRidgeParams i j (0 : EucSpace d) 0 0 0 0 0) μ)
    (t : ℝ) : μ t = μ 0 := by
  rw [cauchyPB_twoRidge_transport i j hij 0 0 0 0 0 0 μ hμ t]
  apply Subtype.ext
  change (twoRidgeLaw 0 0 0 0 0 0 (μ 0) t : Measure (SSphere d)) = (μ 0 : Measure (SSphere d))
  rw [coe_twoRidgeLaw]
  have he : (fun x : SSphere d => twoRidgeFlow (0 : EucSpace d) 0 0 0 0 0 x t) = id := by
    funext x
    exact twoRidgeFlow_zero_data x t
  rw [he, Measure.map_id]

/-- An actual nonzero projection prefix, its complete weak solution, and
a second phase at twice the speed jointly supply every existence,
uniqueness, representation, measurability and matching-data premise. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (p : EucSpace 2)
    (W U : ℝ → ParamMatrix 2) (a : ℝ → EucSpace 2) (φ : ℝ → SSphere 2 → SSphere 2)
    (μ ν : ℝ → Perspective.ProbSphere 2),
    (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧ Measurable (φ 1) ∧
    (∀ s : ℝ, 1 ≤ s → neuralParams 2 W U a s = twoRidgeParams 0 1 p 0 1 0 0 0) ∧
    (∀ σ : Perspective.ProbSphere 2, ∃ η : ℝ → Perspective.ProbSphere 2,
      η 0 = σ ∧ cauchyPB 2 (neuralParams 2 W U a) η) ∧
    (∀ α β : ℝ → Perspective.ProbSphere 2, cauchyPB 2 (neuralParams 2 W U a) α →
      cauchyPB 2 (neuralParams 2 W U a) β → α 0 = β 0 → α = β) ∧
    (∀ η : ℝ → Perspective.ProbSphere 2, cauchyPB 2 (neuralParams 2 W U a) η →
      ∀ t, (η t : Measure (SSphere 2)) = Measure.map (φ t) (η 0 : Measure (SSphere 2))) ∧
    cauchyPB 2 (neuralParams 2 W U a) μ ∧ cauchyPB 2 (neuralParams 2 W U a) ν ∧
    μ 0 = ν 0 ∧ μ 0 = ρ ∧
    (∃ ζ : ℝ → Perspective.ProbSphere 2,
      cauchyPB 2 (concatenate 1 (neuralParams 2 W U a)
        (fun _ => twoRidgeParams 0 1 ((2 : ℝ) • p) 0 1 0 0 0)) ζ) := by
  let p : EucSpace 2 := basePoint 1
  let θ := twoRidgeParams (0 : Fin 2) 1 p 0 1 0 0 0
  let W : ℝ → ParamMatrix 2 := fun _ => θ.W
  let U : ℝ → ParamMatrix 2 := fun _ => θ.U
  let a : ℝ → EucSpace 2 := fun _ => θ.b
  let μ := twoRidgeLaw p 0 1 0 0 0 ρ
  have hweak : cauchyPB 2 (neuralParams 2 W U a) μ :=
    cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _
  refine ⟨p, W, U, a, (fun t x => twoRidgeFlow p 0 1 0 0 0 x t), μ, μ,
    by norm_num, zero_lt_one, measurable_twoRidgeFlow_initial _ _ _ _ _ _ _,
    fun _ _ => rfl, ?_, ?_, ?_, hweak, hweak, rfl, twoRidgeLaw_zero _ _ _ _ _ _ _, ?_⟩
  · intro σ
    exact ⟨twoRidgeLaw p 0 1 0 0 0 σ, twoRidgeLaw_zero _ _ _ _ _ _ _,
      cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _⟩
  · intro α β hα hβ h0
    exact cauchyPB_twoRidge_unique 0 1 (by norm_num) p 0 1 0 0 0 α β hα hβ h0
  · intro η hη t
    rw [cauchyPB_twoRidge_transport 0 1 (by norm_num) p 0 1 0 0 0 η hη t, coe_twoRidgeLaw]
  · exact ⟨concatenate 1 μ (twoRidgeLaw ((2 : ℝ) • p) 0 1 0 0 0 (μ 1)),
      cauchyPB_concatenate 1 zero_lt_one _ _ μ _ hweak
        (cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _)
        (twoRidgeLaw_zero _ _ _ _ _ _ _)⟩

/-- Two actual hidden coordinates and the true zero-control weak solution
realize all zero-data transport hypotheses for any initial law. -/
example (ρ : Perspective.ProbSphere 2) : ∃ μ : ℝ → Perspective.ProbSphere 2,
    (0 : Fin 2) ≠ 1 ∧
    cauchyPB 2 (fun _ => twoRidgeParams 0 1 (0 : EucSpace 2) 0 0 0 0 0) μ ∧ μ 0 = ρ :=
  ⟨twoRidgeLaw 0 0 0 0 0 0 ρ, by norm_num,
    cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _, twoRidgeLaw_zero _ _ _ _ _ _ _⟩

end Transformer.Interpolation
