/-
# Prescribed-time realization by the original constant two-neuron controls

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
and §1, `eq: vf`, `eq: ips` and `eq: cauchy.pb`.

Multiplying both output vectors by a scalar changes the original flow's
clock by that scalar. The activation matrices and biases are unchanged.
This identity follows from the actual particle equation and its uniqueness,
and identifies the transported probability at the rescaled time.

The same explicit controls solve the source's integrated neural-flow
equation, and retain its Lipschitz, bijective and fixed-region clauses.
These statements connect the constructed attractor to the exact controls
and prescribed terminal time required by the original ball-transport lemma.
-/

import Transformer.Interpolation.Section3_TwoRidgeLaw

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Scaling both outputs scales the complete original velocity.
Source: arXiv:2411.04551v3, §1, `eq: vf`, with attention off. -/
theorem twoRidgeField_smul_output (k : ℝ) (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x : EucSpace d) :
    twoRidgeField (k • p) u b (k • q) v c x = k • twoRidgeField p u b q v c x := by
  simp only [twoRidgeField, proj, real_inner_smul_right]
  module

/-- The actual original flow obeys the exact output-scaling clock identity.
Source: arXiv:2411.04551v3, Appendix, time rescaling in `lem: two.balls`. -/
theorem twoRidgeFlow_smul_output (k : ℝ) (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x : SSphere d) (t : ℝ) :
    twoRidgeFlow (k • p) u b (k • q) v c x t = twoRidgeFlow p u b q v c x (k * t) := by
  have hd (s : ℝ) : HasDerivAt (fun r => (twoRidgeFlow p u b q v c x (k * r) : EucSpace d))
      (twoRidgeField (k • p) u b (k • q) v c (twoRidgeFlow p u b q v c x (k * s))) s := by
    rw [twoRidgeField_smul_output]
    simpa only [Function.comp_def, mul_one] using
      (hasDerivAt_twoRidgeFlow p u b q v c x (k * s)).scomp s
      ((hasDerivAt_id s).const_mul k)
  have he := twoRidgeTrajectory_unique (k • p) u b (k • q) v c
    (twoRidgeFlow (k • p) u b (k • q) v c x) (fun s => twoRidgeFlow p u b q v c x (k * s))
    (hasDerivAt_twoRidgeFlow _ _ _ _ _ _ x) hd (by simp only [mul_zero, twoRidgeFlow_zero])
  exact congrFun he t

/-- The actual original probability follows exactly the rescaled clock.
Source: arXiv:2411.04551v3, Appendix, time rescaling in `lem: two.balls`. -/
theorem twoRidgeLaw_smul_output (k : ℝ) (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) :
    twoRidgeLaw (k • p) u b (k • q) v c ν t = twoRidgeLaw p u b q v c ν (k * t) := by
  apply Subtype.ext
  change Measure.map (fun x => twoRidgeFlow (k • p) u b (k • q) v c x t) (ν : Measure (SSphere d)) =
    Measure.map (fun x => twoRidgeFlow p u b q v c x (k * t)) (ν : Measure (SSphere d))
  simp only [twoRidgeFlow_smul_output]

/-- The original neural parameter convention agrees with the constructed
constant controls. Source: arXiv:2411.04551v3, Appendix, `eq: neural.pde.sphere`. -/
theorem constNeural_twoRidgeParams (i j : Fin d) (p u : EucSpace d) (b : ℝ)
    (q v : EucSpace d) (c : ℝ) :
    constNeural d (twoRidgeParams i j p u b q v c).W (twoRidgeParams i j p u b q v c).U
      (twoRidgeParams i j p u b q v c).b = fun _ => twoRidgeParams i j p u b q v c := rfl

/-- The explicit matrices realize the exact original neural vector field.
Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem neuralVF_twoRidgeParams (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c t : ℝ) (x : SSphere d) :
    neuralVF d (fun _ => (twoRidgeParams i j p u b q v c).W)
      (fun _ => (twoRidgeParams i j p u b q v c).U)
      (fun _ => (twoRidgeParams i j p u b q v c).b) t x = twoRidgeField p u b q v c x := by
  have h := fullVF_twoRidgeParams i j hij p u b q v c (Perspective.diracProb d x) t x
  simpa only [fullVF, neuralVF, twoRidgeParams,
    zero_apply, zero_add] using h

/-- The actual flow solves the original integrated neural equation at
every real time. Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem isNeuralFlow_twoRidgeFlow (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) :
    IsNeuralFlow d (fun _ => (twoRidgeParams i j p u b q v c).W) (fun _ => (twoRidgeParams i j p u b q v c).U)
      (fun _ => (twoRidgeParams i j p u b q v c).b) (fun t x => twoRidgeFlow p u b q v c x t) := by
  intro x t
  simp_rw [neuralVF_twoRidgeParams i j hij]
  have hi : IntervalIntegrable
      (fun s => twoRidgeField p u b q v c (twoRidgeFlow p u b q v c x s)) volume 0 t :=
    ((continuous_twoRidgeField p u b q v c).comp
      (continuous_twoRidgeFlow_time p u b q v c x).subtype_val).intervalIntegrable 0 t
  refine ⟨hi, ?_⟩
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_twoRidgeFlow p u b q v c x s) hi
  rw [twoRidgeFlow_zero] at hFTC
  rw [hFTC]
  module

/-- A fixed complement gives precisely the source's Lipschitz, invertible
transport clause. Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem twoRidgeFlow_isTransportOutside (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (S : Set (SSphere d))
    (hfix : ∀ x ∉ S, ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x)
    (T : ℝ) : IsTransportOutside d (fun t x => twoRidgeFlow p u b q v c x t) S T := by
  intro t _
  refine ⟨exists_lipschitzWith_twoRidgeFlow p u b q v c t,
    bijective_twoRidgeFlow p u b q v c t, ?_⟩
  intro x hx
  exact hfix x hx t

/-- Any original flow endpoint can be realized at the specified nonzero
terminal time. Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem twoRidgeFlow_at_prescribed_time (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x : SSphere d) (τ T : ℝ) (hT : T ≠ 0) :
    twoRidgeFlow ((τ / T) • p) u b ((τ / T) • q) v c x T = twoRidgeFlow p u b q v c x τ := by
  rw [twoRidgeFlow_smul_output, div_mul_cancel₀ τ hT]

/-- Every weak solution of the rescaled original controls has exactly the
prescribed flow law at terminal time. Source: arXiv:2411.04551v3,
Appendix, `lem: two.balls` and its time-rescaling step. -/
theorem cauchyPB_twoRidge_prescribed_time (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c τ T : ℝ) (hT : T ≠ 0)
    (ν : Perspective.ProbSphere d) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => twoRidgeParams i j ((τ / T) • p) u b ((τ / T) • q) v c) μ)
    (h0 : μ 0 = ν) : μ T = twoRidgeLaw p u b q v c ν τ := by
  have he := cauchyPB_twoRidge_transport i j hij
    ((τ / T) • p) u b ((τ / T) • q) v c μ hμ T
  rw [h0, twoRidgeLaw_smul_output, div_mul_cancel₀ τ hT] at he
  exact he

/-- The exact source control, flow and arbitrary-law terminal transport
clauses are realized together at a prescribed nonzero time.
Source: arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_twoRidge_neural_realization (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (S : Set (SSphere d))
    (hfix : ∀ x ∉ S, ∀ t : ℝ, twoRidgeFlow p u b q v c x t = x)
    (τ T : ℝ) (hT : T ≠ 0) (ν : Perspective.ProbSphere d) :
    ∃ (W U : ParamMatrix d) (a : EucSpace d) (φ : ℝ → SSphere d → SSphere d),
      IsNeuralFlow d (fun _ => W) (fun _ => U) (fun _ => a) φ ∧
      IsTransportOutside d φ S T ∧
      (∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (constNeural d W U a) μ) ∧
      (∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d (constNeural d W U a) μ →
        μ T = twoRidgeLaw p u b q v c ν τ) ∧
      φ T = (fun x => twoRidgeFlow p u b q v c x τ) := by
  let k := τ / T
  let θ := twoRidgeParams i j (k • p) u b (k • q) v c
  let φ (t : ℝ) (x : SSphere d) := twoRidgeFlow (k • p) u b (k • q) v c x t
  have hconst : constNeural d θ.W θ.U θ.b = (fun _ => θ) := constNeural_twoRidgeParams _ _ _ _ _ _ _ _
  refine ⟨θ.W, θ.U, θ.b, φ, isNeuralFlow_twoRidgeFlow i j hij _ _ _ _ _ _, ?_, ?_, ?_, ?_⟩
  · apply twoRidgeFlow_isTransportOutside
    intro x hx t
    rw [twoRidgeFlow_smul_output]
    exact hfix x hx (k * t)
  · rw [hconst]
    exact ⟨twoRidgeLaw (k • p) u b (k • q) v c ν, twoRidgeLaw_zero _ _ _ _ _ _ _,
      cauchyPB_twoRidgeLaw i j hij _ _ _ _ _ _ _⟩
  · intro μ h0 hμ
    rw [hconst] at hμ
    exact cauchyPB_twoRidge_prescribed_time i j hij p u b q v c τ T hT ν μ hμ h0
  · funext x
    exact twoRidgeFlow_at_prescribed_time p u b q v c x τ T hT

/-- Two distinct actual hidden coordinates, a nonempty activation region,
an actual scaled weak solution and a nonzero terminal time jointly realize
the neural, fixed-region and terminal-transport hypotheses. -/
example : ∃ (p q u : EucSpace 2) (S : Set (SSphere 2)) (ν : Perspective.ProbSphere 2)
    (μ : ℝ → Perspective.ProbSphere 2),
    (0 : Fin 2) ≠ 1 ∧ (1 : ℝ) ≠ 0 ∧
    (∀ x ∉ S, ∀ t : ℝ, twoRidgeFlow p u 0 q u 0 x t = x) ∧
    cauchyPB 2 (fun _ => twoRidgeParams 0 1 ((2 : ℝ) • p) u 0 ((2 : ℝ) • q) u 0) μ ∧
      μ 0 = ν := by
  let p : EucSpace 2 := basePoint 1
  let q : EucSpace 2 := EuclideanSpace.single 1 1
  let S := {x : SSphere 2 | 0 < 0 - inner ℝ (x : EucSpace 2) p}
  let ν := halfDirac 2 (basePoint 1) (antipode 2 (basePoint 1))
  refine ⟨p, q, p, S, ν, twoRidgeLaw ((2 : ℝ) • p) p 0 ((2 : ℝ) • q) p 0 ν,
    by norm_num, by norm_num, ?_, cauchyPB_twoRidgeLaw 0 1 (by norm_num) _ _ _ _ _ _ _,
    twoRidgeLaw_zero _ _ _ _ _ _ _⟩
  intro x hx t
  exact twoRidgeFlow_inactive p p 0 q p 0 x
    (not_lt.mp hx) (not_lt.mp hx) t

end Transformer.Interpolation
