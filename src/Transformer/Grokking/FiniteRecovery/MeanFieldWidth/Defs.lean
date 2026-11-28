module

public import Transformer.Grokking.FiniteRecovery.AdamWTrajectory
public import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-! Mean-field AdamW in width. A particle is one width unit with its own
coordinates P. AdamW acts coordinate by coordinate, so the two moment
buffers of those coordinates travel with the particle. The particles
interact only through a finite vector of empirical averages, the mean
field. The limit replaces those averages by expectations under the
initialization law. The finite system and the limit share the initial
draw of every particle (synchronous coupling) and the minibatch of every
step, which enters through the gradient callback. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.MeanFieldWidth

open MeasureTheory Transformer.AMSGradW.Kernite

variable {P K : Type*}

/-- The ℓ¹ size of a finite real vector. -/
def l1 {ι : Type*} [Fintype ι] (v : ι → ℝ) : ℝ := ∑ i, |v i|

/-- The bias-corrected Adam denominator of `AdamW.scale`, with a general floor. -/
noncomputable def denominator (eps : ℝ) (t : ℕ) (v : ℝ) : ℝ :=
  firstBias t * (Real.sqrt v / Real.sqrt (1 - (95/100 : ℝ)^(t+1)) + eps)

/-- The bias-corrected Adam direction with floor eps. -/
noncomputable def direction (eps : ℝ) (t : ℕ) (m v : ℝ) : ℝ := m / denominator eps t v

/-- The floor 10⁻⁸ gives the direction of the Argand AdamW kernel. -/
theorem direction_literal : direction (1/100000000) = AdamW.direction := rfl

/-- One AdamW step of one particle at rate η after t completed steps,
with the equations of `AdamW.run` coordinate by coordinate. -/
noncomputable def step (eps η : ℝ) (t : ℕ) (g : P → ℝ) (s : AdamW.State P) : AdamW.State P :=
  let m p := (9/10 : ℝ) * s.first p + (1/10 : ℝ) * g p
  let v p := (95/100 : ℝ) * s.second p + (5/100 : ℝ) * (g p)^2
  ⟨fun p => (1 - η/10) * s.weights p - η * direction eps t (m p) (v p), m, v⟩

/-- Floor, schedule, the gradient of one particle on the minibatch of
each step given the mean field, and the features that make the field. -/
structure Training (P K : Type*) where
  eps : ℝ
  rate : ℕ → ℝ
  grad : ℕ → (P → ℝ) → (K → ℝ) → P → ℝ
  field : (P → ℝ) → K → ℝ

/-- The empirical average of the features of N particles. -/
noncomputable def meanField {Θ J : Type*} {N : ℕ} (φ : Θ → J → ℝ) (θ : Fin N → Θ) : J → ℝ :=
  fun j => (∑ i, φ (θ i) j) / N

/-- N interacting particles, each from its own initial weights with zero moments. -/
noncomputable def system (A : Training P K) {N : ℕ} (init : Fin N → P → ℝ) :
    ℕ → Fin N → AdamW.State P
  | 0 => fun i => ⟨init i, 0, 0⟩
  | t+1 => fun i =>
    let s := system A init t
    step A.eps (A.rate t) t (A.grad t (s i).weights (meanField A.field fun j => (s j).weights)) (s i)

/-- The limiting particle from initial weights θ. It sees the field of
the law μ of initial weights transported by the same limiting dynamics. -/
noncomputable def flow (A : Training P K) (μ : Measure (P → ℝ)) : ℕ → (P → ℝ) → AdamW.State P
  | 0 => fun θ => ⟨θ, 0, 0⟩
  | t+1 => fun θ =>
    let X := flow A μ t
    step A.eps (A.rate t) t (A.grad t (X θ).weights fun k => ∫ x, A.field (X x).weights k ∂μ) (X θ)

/-- The deterministic limiting mean field after t steps. -/
noncomputable def limitField (A : Training P K) (μ : Measure (P → ℝ)) (t : ℕ) : K → ℝ :=
  fun k => ∫ x, A.field (flow A μ t x).weights k ∂μ

/-- Distance between two particle states: weights, first moments and the
square roots of the second moments, in ℓ¹ over coordinates. -/
noncomputable def gap [Fintype P] (s s' : AdamW.State P) : ℝ :=
  ∑ p, (|s.weights p - s'.weights p| + |s.first p - s'.first p| +
    |Real.sqrt (s.second p) - Real.sqrt (s'.second p)|)

/-- Mean distance of the N particles to their coupled limiting particles. -/
noncomputable def coupling [Fintype P] (A : Training P K) (μ : Measure (P → ℝ)) {N : ℕ}
    (init : Fin N → P → ℝ) (t : ℕ) : ℝ :=
  (∑ i, gap (system A init t i) (flow A μ t (init i))) / N

/-- Sampling error of the empirical field of N independent limiting particles. -/
noncomputable def fluctuation [Fintype K] (A : Training P K) (μ : Measure (P → ℝ)) {N : ℕ}
    (init : Fin N → P → ℝ) (t : ℕ) : ℝ :=
  l1 (meanField A.field (fun i => (flow A μ t (init i)).weights) - limitField A μ t)

/-- The moment cone of `AdamW.moment_cone`, coordinate by coordinate. -/
def Cone (s : AdamW.State P) : Prop := ∀ p, 14 * s.first p ^ 2 ≤ 19 * s.second p

theorem system_succ (A : Training P K) {N : ℕ} (init : Fin N → P → ℝ) (t : ℕ) (i : Fin N) :
    system A init (t+1) i = step A.eps (A.rate t) t (A.grad t (system A init t i).weights
      (meanField A.field fun j => (system A init t j).weights)) (system A init t i) := rfl

theorem flow_succ (A : Training P K) (μ : Measure (P → ℝ)) (t : ℕ) (θ : P → ℝ) :
    flow A μ (t+1) θ = step A.eps (A.rate t) t
      (A.grad t (flow A μ t θ).weights (limitField A μ t)) (flow A μ t θ) := rfl

end Transformer.Grokking.FiniteRecovery.MeanFieldWidth
