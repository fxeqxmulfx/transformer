/-
# Contracting error of the actual AMSGradW state

arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension.
Weights and actual momentum contract toward the frozen-metric
equilibrium, up to a forcing caused by the real denominator history.
-/

import Transformer.AMSGradW.Contraction

noncomputable section

namespace Transformer.AMSGradW

variable {d : ℕ}

/-- Weighted maximum error of actual weights and first moments.
Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
def stateError (ε wd : ℝ) (f : TrainingSpace d → ℝ) (star : Fin d → ℝ)
    (s : TrainingState d) : ℝ :=
  max ‖WithLp.ofLp s.position - star‖ (‖s.momentum - coordinateGradient f star‖ / (wd * ε))

/-- Actual metric perturbation at a frozen equilibrium.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
def metricForcing (η : ℝ) (f : TrainingSpace d → ℝ) (star D nextD : Fin d → ℝ) : ℝ :=
  η * ‖fun i => coordinateGradient f star i * (1 / nextD i - 1 / D i)‖

/-- Exact weight-error estimate using the genuine new momentum and
denominator. Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW extension. -/
theorem position_error_bound (η ε wd β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (star D : Fin d → ℝ)
    (hη : 0 ≤ η) (hε : 0 < ε) (hD : ∀ i, 0 < D i) (hstep : η * wd ≤ 1)
    (hstar : ∀ i, coordinateGradient f star i + wd * D i * star i = 0) :
    let next := trainingStep η ε wd β β₂ f s
    ‖WithLp.ofLp next.position - star‖ ≤
      (1 - η * wd) * ‖WithLp.ofLp s.position - star‖ +
        η / ε * ‖next.momentum - coordinateGradient f star‖ +
          metricForcing η f star D (AMSGrad.trainingDenominator ε next) := by
  let next := trainingStep η ε wd β β₂ f s
  let A := AMSGrad.trainingDenominator ε next
  let g := coordinateGradient f star
  change ‖WithLp.ofLp next.position - star‖ ≤
    (1 - η * wd) * ‖WithLp.ofLp s.position - star‖ +
      η / ε * ‖next.momentum - coordinateGradient f star‖ + metricForcing η f star D A
  have heq : WithLp.ofLp next.position - star =
      ((1 - η * wd) • (WithLp.ofLp s.position - star) -
        η • (fun i => (next.momentum - g) i / A i)) -
          η • (fun i => g i * (1 / A i - 1 / D i)) := by
    ext i
    have hs : g i / D i = -wd * star i := by
      apply (div_eq_iff (hD i).ne').mpr
      have hh := hstar i
      dsimp only [g]
      nlinarith
    have hx := (trainingStep_coordinates η ε wd β β₂ f s i).2
    change next.position i = (1 - η * wd) * s.position i - η * next.momentum i / A i at hx
    simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
    rw [hx]
    have hz : g i / D i + wd * star i = 0 := by rw [hs]; ring
    calc
      (1 - η * wd) * s.position i - η * next.momentum i / A i - star i =
        ((1 - η * wd) * (s.position i - star i) - η * ((next.momentum i - g i) / A i)) -
          η * (g i * (1 / A i - 1 / D i)) - η * (g i / D i + wd * star i) := by ring
      _ = ((1 - η * wd) * (s.position i - star i) - η * ((next.momentum i - g i) / A i)) -
          η * (g i * (1 / A i - 1 / D i)) := by rw [hz]; ring
  rw [heq]
  have h1 := norm_sub_le ((1 - η * wd) • (WithLp.ofLp s.position - star))
    (η • (fun i => (next.momentum - g) i / A i))
  have h2 := norm_sub_le
    ((1 - η * wd) • (WithLp.ofLp s.position - star) - η • (fun i => (next.momentum - g) i / A i))
    (η • (fun i => g i * (1 / A i - 1 / D i)))
  simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr hstep),
    abs_of_nonneg hη] at h1 h2
  have h3 := mul_le_mul_of_nonneg_left
    (division_norm_bound ε (next.momentum - g) A hε (denominator_lower ε next)) hη
  have he : η * (‖next.momentum - g‖ / ε) = η / ε * ‖next.momentum - g‖ := by ring
  rw [he] at h3
  dsimp only [metricForcing, g] at *
  linarith

/-- Nonzero weight/decay parameters and a genuine equilibrium satisfy
the estimate assumptions. Source: arXiv:1904.03590v4, §4, AMSGradW. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) < 1 ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => 1) i) ∧ (1 / 4 : ℝ) * 2 ≤ 1 ∧
    (∀ i : Fin 1, coordinateGradient (Optimization.energy : TrainingSpace 1 → ℝ) 0 i +
      2 * (1 : ℝ) * (0 : Fin 1 → ℝ) i = 0) := by
  refine ⟨by norm_num, by norm_num, fun i => by norm_num, by norm_num, ?_⟩
  intro i
  simp [coordinateGradient, Optimization.energy_gradient]

/-- The real state error has a strict-contraction factor plus only the
metric forcing. The final theorem derives the equilibrium and forcing
limit, rather than assuming them for training.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
theorem stateError_step (η ε wd β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (star D : Fin d → ℝ)
    (hη : 0 ≤ η) (hε : 0 < ε) (hwd : 0 < wd) (hL : 0 ≤ L)
    (hβ : 0 ≤ β) (hβ' : β ≤ 1) (hstep : η * wd ≤ 1)
    (hgrad : LipschitzGradient f L) (hD : ∀ i, 0 < D i)
    (hstar : ∀ i, coordinateGradient f star i + wd * D i * star i = 0) :
    stateError ε wd f star (trainingStep η ε wd β β₂ f s) ≤
      contractionFactor η ε wd β L * stateError ε wd f star s +
        metricForcing η f star D
          (AMSGrad.trainingDenominator ε (trainingStep η ε wd β β₂ f s)) := by
  let next := trainingStep η ε wd β β₂ f s
  let g := coordinateGradient f star
  let E := stateError ε wd f star s
  let K := wd * ε
  let θ := momentumFactor ε wd β L
  let q := contractionFactor η ε wd β L
  have hK : 0 < K := mul_pos hwd hε
  have hE : 0 ≤ E := (norm_nonneg _).trans (le_max_left _ _)
  have hx : ‖WithLp.ofLp s.position - star‖ ≤ E := le_max_left _ _
  have hm : ‖s.momentum - g‖ ≤ K * E := by
    simpa only [mul_comm] using (div_le_iff₀ hK).mp
      (show ‖s.momentum - g‖ / K ≤ E from le_max_right _ _)
  have heq : next.momentum - g = β • (s.momentum - g) +
      (1 - β) • (coordinateGradient f (WithLp.ofLp s.position) - g) := by
    ext i
    simp only [next, trainingStep, AMSGrad.trainingStep, coordinateGradient,
      WithLp.toLp_ofLp, Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have hmn : ‖next.momentum - g‖ ≤ K * θ * E := by
    rw [heq]
    have hn := norm_add_le (β • (s.momentum - g))
      ((1 - β) • (coordinateGradient f (WithLp.ofLp s.position) - g))
    simp only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hβ,
      abs_of_nonneg (sub_nonneg.mpr hβ')] at hn
    have h1 := mul_le_mul_of_nonneg_left hm hβ
    have h2 := mul_le_mul_of_nonneg_left
      ((hgrad (WithLp.ofLp s.position) star).trans
        (mul_le_mul_of_nonneg_left hx hL)) (sub_nonneg.mpr hβ')
    have hid : K * θ = β * K + (1 - β) * L := by
      dsimp only [K, θ, momentumFactor]
      field_simp
    dsimp only [g] at *
    rw [hid]
    nlinarith
  have hf0 : 0 ≤ metricForcing η f star D (AMSGrad.trainingDenominator ε next) :=
    mul_nonneg hη (norm_nonneg _)
  have hq1 : θ ≤ q := le_max_left _ _
  have hq2 : 1 - η * wd + η * wd * θ ≤ q := le_max_right _ _
  apply max_le
  · have hn := position_error_bound η ε wd β β₂ f s star D hη hε hD hstep hstar
    have h1 := mul_le_mul_of_nonneg_left hx (sub_nonneg.mpr hstep)
    have h2 := mul_le_mul_of_nonneg_left hmn (div_nonneg hη hε.le)
    have hid : η / ε * (K * θ * E) = η * wd * θ * E := by
      dsimp only [K]
      field_simp
    have h3 := mul_le_mul_of_nonneg_right hq2 hE
    rw [hid] at h2
    dsimp only [next, E, q] at *
    linarith
  · have hn : ‖next.momentum - g‖ / K ≤ θ * E :=
      (div_le_iff₀ hK).mpr (by nlinarith [hmn])
    have hq := mul_le_mul_of_nonneg_right hq1 hE
    dsimp only [next, E, q, K, g] at *
    linarith

/-- All state-error assumptions hold with nonzero momentum and decay,
arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : (0 : ℝ) ≤ 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) ≤ 1 ∧ (1 / 4 : ℝ) * 2 ≤ 1 ∧
    LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 ∧
    (∀ i : Fin 1, (0 : ℝ) < (fun _ => 1) i) ∧
    (∀ i : Fin 1, coordinateGradient (Optimization.energy : TrainingSpace 1 → ℝ) 0 i +
      2 * (1 : ℝ) * (0 : Fin 1 → ℝ) i = 0) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, energy_lipschitz, fun i => by norm_num, ?_⟩
  intro i
  simp [coordinateGradient, Optimization.energy_gradient]

end Transformer.AMSGradW
