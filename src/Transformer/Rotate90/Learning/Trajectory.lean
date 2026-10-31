module

public import Transformer.Rotate90.Learning.Basic

/-!
# Actual sampled training and a finite-time classification certificate

Source: `review/flat_spectrum.md`, *Learning transfer*, T2--T6. Each update
differentiates the complete network at y on its specified minibatch; the
evaluation loss is at the optimizer's x. Arbitrary finite batches permit the
actual short-tail sampler. Initial history is explicit (the lab starts its
maximum at zero). Source warmup10 equals the lab's inclusive external ramp.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Learning

open Landscape.Neural ScheduleFree
open scoped RealInnerProductSpace BigOperators

section FiniteBounds

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E]

/-- T6, additive form: every step's actual directional effect and curvature
remainder contribute, including steps which make evaluation loss worse. -/
theorem finite_loss_le {F : E → ℝ} (hF : ContDiff ℝ 2 F)
    (x : ℕ → E) (L : ℕ → ℝ) (N : ℕ)
    (hL : ∀ t < N, ∀ w ∈ segment ℝ (x t) (x (t + 1)),
      ‖fderiv ℝ (gradient F) w‖ ≤ L t) :
    F (x N) ≤ F (x 0) + ∑ t ∈ Finset.range N,
      (⟪gradient F (x t), x (t + 1) - x t⟫ + L t * ‖x (t + 1) - x t‖ ^ 2) := by
  induction N with
  | zero => simp
  | succ N ih =>
    have hprevious := ih (fun t ht => hL t (by omega))
    have hstep := segment_loss_le hF (x N) (x (N + 1)) (L N) (hL N (by omega))
    rw [Finset.sum_range_succ]
    linarith

/-- T6, geometric form: a uniform relative first-order advantage over the
curvature remainder gives a finite-time rate. This is an explicit trajectory
condition, not a claim that random-start Schedule-Free necessarily satisfies it. -/
theorem geometric_loss_le {F : E → ℝ} (hF : ContDiff ℝ 2 F)
    (x : ℕ → E) (L : ℕ → ℝ) (N : ℕ) (μ : ℝ) (hμ : μ ≤ 1)
    (hL : ∀ t < N, ∀ w ∈ segment ℝ (x t) (x (t + 1)),
      ‖fderiv ℝ (gradient F) w‖ ≤ L t)
    (hprogress : ∀ t < N,
      ⟪gradient F (x t), x (t + 1) - x t⟫ + L t * ‖x (t + 1) - x t‖ ^ 2 ≤
        -μ * F (x t)) :
    F (x N) ≤ (1 - μ) ^ N * F (x 0) := by
  induction N with
  | zero => simp
  | succ N ih =>
    have hprevious := ih (fun t ht => hL t (by omega)) (fun t ht => hprogress t (by omega))
    have hstep := segment_loss_le hF (x N) (x (N + 1)) (L N) (hL N (by omega))
    have hp := hprogress N (by omega)
    calc
      F (x (N + 1)) ≤ (1 - μ) * F (x N) := by linarith
      _ ≤ (1 - μ) * ((1 - μ) ^ N * F (x 0)) :=
        mul_le_mul_of_nonneg_left hprevious (by linarith)
      _ = _ := by rw [pow_succ]; ring

/- Nonzero strictly improving witnesses at every finite horizon:
F(u)=u, x_t=2^-t, zero curvature and mu=1/2. -/
example (N : ℕ) := finite_loss_le (F := fun u : ℝ => u) (by fun_prop)
  (fun t => (1 / 2 : ℝ) ^ t) (fun _ => 0) N
  (by
    intro t ht w hw
    have hg : gradient (fun u : ℝ => u) = fun _ => 1 := by funext u; simp [gradient_eq_deriv]
    rw [hg]; simp)
example (N : ℕ) := geometric_loss_le (F := fun u : ℝ => u) (by fun_prop)
  (fun t => (1 / 2 : ℝ) ^ t) (fun _ => 0) N (1 / 2) (by norm_num)
  (by
    intro t ht w hw
    have hg : gradient (fun u : ℝ => u) = fun _ => 1 := by funext u; simp [gradient_eq_deriv]
    rw [hg]; simp)
  (by intro t ht; simp [gradient_eq_deriv, pow_succ]; ring_nf; exact le_refl _)

end FiniteBounds

variable {V K H D : Type*} [Fintype V] [Fintype K] [Fintype H] [Fintype D]

/-- Every update takes the gradient of the actual full network at its query.
The sample sequence is fixed independently of the diagnostic evaluation set. -/
def runFrom (eps : ℝ) (order : List D) (samples : ℕ → Finset (V × V))
    (target : V × V → V) (p : AdamW.Parameters)
    (initial : AdamW.TrainingState (Index V K H D)) : ℕ → AdamW.TrainingState (Index V K H D)
  | 0 => initial
  | t + 1 =>
    let s := runFrom eps order samples target p initial t
    AdamW.trainingStep p t s
      (gradient (loss eps order (samples t) target) (WithLp.toLp 2 s.point)).ofLp

/-- Explicit scalar first-order effect: averaging, normalized gradient and decay. -/
def transfer {I : Type*} [Fintype I] (p : AdamW.Parameters) (t : ℕ)
    (s : AdamW.TrainingState I) (g : I → ℝ) (F : EuclideanSpace ℝ I → ℝ) : ℝ :=
  AdamW.coefficient p t s.history *
    ⟪gradient F (evaluated p s), WithLp.toLp 2 s.base - evaluated p s⟫ -
  (AdamW.coefficient p t s.history * AdamW.rate p t) *
    ⟪gradient F (evaluated p s), normalized p t s g⟫ -
  (AdamW.coefficient p t s.history * AdamW.rate p t * p.adam.decay) *
    ⟪gradient F (evaluated p s), WithLp.toLp 2 s.point⟫

/-- T3 on the actual training path, with all training and heldout derivatives
at their correct, generally different, parameter points. -/
theorem run_directional_transfer (eps : ℝ) (order : List D)
    (samples : ℕ → Finset (V × V)) (target : V × V → V)
    (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (initial : AdamW.TrainingState (Index V K H D)) (rows : Finset (V × V)) (t : ℕ) :
    let s := runFrom eps order samples target p initial
    let F := loss eps order rows target
    ⟪gradient F (evaluated p (s t)), evaluated p (s (t + 1)) - evaluated p (s t)⟫ =
      transfer p t (s t)
        (gradient (loss eps order (samples t) target) (WithLp.toLp 2 (s t).point)).ofLp F := by
  exact directional_transfer p hβ t _ _ _

/-- T4--T6 for the actual model and optimizer. The right side includes each
sampled update's complete transfer, even when negative progress occurs. Only
the curvature bound is conditional; no representation or optimizer surrogate
is substituted. A useful small bound still requires favorable transfer. -/
theorem finite_failure_le {eps : ℝ} (heps : 0 < eps) (order : List D)
    (samples : ℕ → Finset (V × V)) (target : V × V → V)
    (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (initial : AdamW.TrainingState (Index V K H D)) (rows : Finset (V × V))
    (L : ℕ → ℝ) (N : ℕ)
    (hL : ∀ t < N, ∀ w ∈ segment ℝ
      (evaluated p (runFrom eps order samples target p initial t))
      (evaluated p (runFrom eps order samples target p initial (t + 1))),
      ‖fderiv ℝ (gradient (loss eps order rows target)) w‖ ≤ L t) :
    let s := runFrom eps order samples target p initial
    let x := fun t => evaluated p (s t)
    let F := loss eps order rows target
    (failures rows (fun r q => score eps order (x N) r.1 r.2 q) target).card /
        (rows.card : ℝ) ≤
      (F (x 0) + ∑ t ∈ Finset.range N,
        (transfer p t (s t)
          (gradient (loss eps order (samples t) target) (WithLp.toLp 2 (s t).point)).ofLp F +
          L t * ‖x (t + 1) - x t‖ ^ 2)) / Real.log 2 := by
  dsimp only
  have h := finite_loss_le (contDiff_loss heps order rows target)
    (fun t => evaluated p (runFrom eps order samples target p initial t)) L N hL
  simp only [run_directional_transfer eps order samples target p hβ initial rows] at h
  exact (failure_fraction_le_loss rows _ target).trans
    (div_le_div_of_nonneg_right h (Real.log_pos (by norm_num : (1 : ℝ) < 2)).le)

/- The actual theorem's hypotheses have a nonempty, positive-depth,
positive-rate, five-update witness. The resulting bound need not be small. -/
section Witness

local notation "p₀" => (⟨⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ : AdamW.Parameters)
local notation "initial₀" => (⟨fun _ => 1, fun _ => 1, ⟨fun _ => 0, 0, 0⟩⟩ :
  AdamW.TrainingState (Index (ZMod 3) (Fin 8) (Fin 43) (Fin 1)))
local notation "samples₀" => (fun _ => {(0, 1), (1, 2)} : ℕ → Finset (ZMod 3 × ZMod 3))
local notation "target₀" => (fun r : ZMod 3 × ZMod 3 => (Prod.fst r / Prod.snd r) ^ 2)
local notation "rows₀" => ({(2, 1)} : Finset (ZMod 3 × ZMod 3))
local notation "s₀" => runFrom (1/8388608) (List.finRange 1) samples₀ target₀ p₀ initial₀
local notation "x₀" => (fun t => evaluated p₀ (s₀ t))
local notation "F₀" => loss (K := Fin 8) (H := Fin 43) (1/8388608) (List.finRange 1) rows₀ target₀

example : ∃ L : ℕ → ℝ,
    (∀ t < 5, ∀ w ∈ segment ℝ (x₀ t) (x₀ (t + 1)), ‖fderiv ℝ (gradient F₀) w‖ ≤ L t) ∧
    (failures rows₀ (fun r q => score (1/8388608) (List.finRange 1) (x₀ 5) r.1 r.2 q)
      target₀).card / ((rows₀).card : ℝ) ≤
      (F₀ (x₀ 0) + ∑ t ∈ Finset.range 5,
        (transfer p₀ t (s₀ t)
          (gradient (loss (1/8388608) (List.finRange 1) (samples₀ t) target₀)
            (WithLp.toLp 2 (s₀ t).point)).ofLp F₀ + L t * ‖x₀ (t + 1) - x₀ t‖ ^ 2)) /
        Real.log 2 := by
  choose L hLpos hL using fun t => exists_segment_curvature
    (contDiff_loss (K := Fin 8) (H := Fin 43) (eps := 1/8388608)
      (by norm_num) (List.finRange 1) rows₀ target₀) (x₀ t) (x₀ (t + 1))
  refine ⟨L, fun t _ => hL t, ?_⟩
  exact finite_failure_le (by norm_num) (List.finRange 1) samples₀ target₀ p₀ (by norm_num)
    initial₀ rows₀ L 5 (fun t _ => hL t)

end Witness

example := run_directional_transfer (K := Fin 8) (H := Fin 43)
  (1/8388608) (List.finRange 1)
  (fun _ => {(0, 1), (1, 2)} : ℕ → Finset (ZMod 3 × ZMod 3))
  (fun r => (r.1 / r.2) ^ 2)
  ⟨⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ (by norm_num)
  ⟨fun _ => 1, fun _ => 1, ⟨fun _ => 0, 0, 0⟩⟩ {(2, 1)} 4

end Transformer.Rotate90.Learning
