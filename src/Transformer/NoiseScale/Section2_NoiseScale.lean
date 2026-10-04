/-
# The best step of a batch and the noise scale

arXiv:1812.06162, §2.2, eqs. (2.6)–(2.8), in the model (2.4) of `Section2_Quadratic`.
Minimizing eq. (2.5) over `ε` gives the best step `ε_opt(B) = ε_max/(1 + B_noise/B)` of
eq. (2.6) (`stepOpt`, `integral_quadModel_stepOpt_lt`) and the best expected drop
`ΔL_opt(B) = ΔL_max/(1 + B_noise/B)`, `ΔL_max = ½|G|⁴/GᵀHG`, of eq. (2.7)
(`lossDropMax`, `sub_integral_quadModel_stepOpt`), with the noise scale
`B_noise = tr(HΣ)/GᵀHG` of eq. (2.8) (`noiseScale`).  A step beyond `2ε_opt` raises the
expected loss (`lt_integral_quadModel_of_two_mul_stepOpt_lt`): the paper's "may increase".

The paper states (2.6) and (2.7) without hypotheses; they need the curvature
`GᵀHG > 0`, for `ε_max` and `B_noise` to exist and the mean (2.5) to have a least point,
and `tr(HΣ) ≥ 0`, which holds for a positive semidefinite `H`; both are assumed here.
-/

import Transformer.NoiseScale.Section2_Quadratic

open MeasureTheory ProbabilityTheory
open scoped Matrix

namespace Transformer.NoiseScale

variable {ι : Type*} [Fintype ι]

/-- The noise scale `B_noise = tr(HΣ)/GᵀHG`, eq. (2.8). -/
noncomputable def noiseScale (G : ι → ℝ) (H S : Matrix ι ι ℝ) : ℝ :=
  (H * S).trace / (G ⬝ᵥ H *ᵥ G)

/-- The best step `ε_opt(B) = ε_max/(1 + B_noise/B)` of a batch of `B`, eq. (2.6). -/
noncomputable def stepOpt (G : ι → ℝ) (H S : Matrix ι ι ℝ) (B : ℝ) : ℝ :=
  stepMax G H / (1 + noiseScale G H S / B)

/-- The largest drop `ΔL_max = ½|G|⁴/GᵀHG` of a step, eq. (2.7). -/
noncomputable def lossDropMax (G : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ :=
  (G ⬝ᵥ G) ^ 2 / (2 * (G ⬝ᵥ H *ᵥ G))

variable {G : ι → ℝ} {H S : Matrix ι ι ℝ}

/-- `ε_opt(B) = |G|²/(GᵀHG + tr(HΣ)/B)`. -/
theorem stepOpt_eq (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 ≤ (H * S).trace) {B : ℝ} (hB : 0 < B) :
    stepOpt G H S B = G ⬝ᵥ G / (G ⬝ᵥ H *ᵥ G + (H * S).trace / B) := by
  have : 0 < G ⬝ᵥ H *ᵥ G * B + (H * S).trace := by positivity
  unfold stepOpt stepMax noiseScale
  field_simp

/-- The hypotheses of `stepOpt_eq` are satisfiable: `G = H = 1`, `Σ = 0`. -/
example : stepOpt (fun _ : Unit => 1) 1 0 1 = (fun _ : Unit => (1 : ℝ)) ⬝ᵥ (fun _ => 1) /
    ((fun _ : Unit => (1 : ℝ)) ⬝ᵥ (1 : Matrix Unit Unit ℝ) *ᵥ (fun _ => 1) +
      ((1 : Matrix Unit Unit ℝ) * 0).trace / 1) :=
  stepOpt_eq (by simp) (by simp) one_pos

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {B : ℕ} {X : Fin B → Ω → ι → ℝ}

/-- **Eq. (2.6)**: `ε_opt(B) = ε_max/(1 + B_noise/B)` is the least point, and the only one,
of `E[L(θ - εG_est)]` over `ε`, in the model (2.4). -/
theorem integral_quadModel_stepOpt_lt (hB : B ≠ 0) (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 ≤ (H * S).trace)
    (L : ℝ) {ε : ℝ} (hε : ε ≠ stepOpt G H S B) :
    ∫ ω, quadModel L G H (stepOpt G H S B) (batchGrad X ω) ∂P <
      ∫ ω, quadModel L G H ε (batchGrad X ω) ∂P := by
  have hB' : (0 : ℝ) < B := Nat.cast_pos.2 (Nat.pos_of_ne_zero hB)
  have ha : 0 < G ⬝ᵥ H *ᵥ G + (H * S).trace / B := by positivity
  rw [stepOpt_eq hH hHS hB'] at hε ⊢
  simp only [integral_quadModel_batchGrad hB hX hind hG hS, quadratic_eq_sq ha.ne', sub_self]
  have := mul_pos (half_pos ha) (sq_pos_of_ne_zero (sub_ne_zero.2 hε))
  linarith [zero_pow (M₀ := ℝ) two_ne_zero]

/-- The hypotheses of `integral_quadModel_stepOpt_lt` are satisfiable: copies of `G = 1`,
`H = 1`. -/
example (L : ℝ) :
    ∫ ω, quadModel L (fun _ : Unit => 1) 1 (stepOpt (fun _ : Unit => 1) 1 0 ((2 : ℕ) : ℝ))
      (batchGrad (fun (_ : Fin 2) (_ : Unit) (_ : Unit) => (1 : ℝ)) ω) ∂Measure.dirac () <
    ∫ ω, quadModel L (fun _ : Unit => 1) 1 0
      (batchGrad (fun (_ : Fin 2) (_ : Unit) (_ : Unit) => (1 : ℝ)) ω) ∂Measure.dirac () :=
  integral_quadModel_stepOpt_lt (X := fun _ _ _ => 1) (G := fun _ => 1) two_ne_zero
    (fun _ _ => memLp_const _) (fun _ _ _ => indepFun_const_left _ _) (fun _ _ => by simp)
    (fun _ => covMatrix_const _) (by simp) (by simp) L (by simp [stepOpt, stepMax, noiseScale])

/-- **Eq. (2.7)**: the best expected drop `ΔL_opt(B) = L - E[L(θ - ε_opt(B) G_est)]` is
`ΔL_max/(1 + B_noise/B)`, in the model (2.4). -/
theorem sub_integral_quadModel_stepOpt (hB : B ≠ 0) (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 ≤ (H * S).trace)
    (L : ℝ) : L - ∫ ω, quadModel L G H (stepOpt G H S B) (batchGrad X ω) ∂P =
      lossDropMax G H / (1 + noiseScale G H S / B) := by
  have hB' : (0 : ℝ) < B := Nat.cast_pos.2 (Nat.pos_of_ne_zero hB)
  have : 0 < G ⬝ᵥ H *ᵥ G * B + (H * S).trace := by positivity
  rw [integral_quadModel_batchGrad hB hX hind hG hS, stepOpt_eq hH hHS hB']
  unfold lossDropMax noiseScale
  field_simp
  ring

/-- The hypotheses of `sub_integral_quadModel_stepOpt` are satisfiable: copies of `G = 1`,
`H = 1`. -/
example (L : ℝ) :
    L - ∫ ω, quadModel L (fun _ : Unit => 1) 1 (stepOpt (fun _ : Unit => 1) 1 0 ((2 : ℕ) : ℝ))
      (batchGrad (fun (_ : Fin 2) (_ : Unit) (_ : Unit) => (1 : ℝ)) ω) ∂Measure.dirac () =
    lossDropMax (fun _ : Unit => 1) 1 /
      (1 + noiseScale (fun _ : Unit => 1) 1 0 / ((2 : ℕ) : ℝ)) :=
  sub_integral_quadModel_stepOpt (X := fun _ _ _ => 1) (G := fun _ => 1) two_ne_zero
    (fun _ _ => memLp_const _) (fun _ _ _ => indepFun_const_left _ _) (fun _ _ => by simp)
    (fun _ => covMatrix_const _) (by simp) (by simp) L

/-- §2.2: a step beyond twice `ε_opt(B)` raises the expected loss above `L`, in the model
(2.4); the paper says it "may increase". -/
theorem lt_integral_quadModel_of_two_mul_stepOpt_lt (hB : B ≠ 0)
    (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 ≤ (H * S).trace)
    (L : ℝ) {ε : ℝ} (hε : 2 * stepOpt G H S B < ε) :
    L < ∫ ω, quadModel L G H ε (batchGrad X ω) ∂P := by
  have hB' : (0 : ℝ) < B := Nat.cast_pos.2 (Nat.pos_of_ne_zero hB)
  have ha : 0 < G ⬝ᵥ H *ᵥ G + (H * S).trace / B := by positivity
  have hg : 0 ≤ G ⬝ᵥ G := Finset.sum_nonneg fun a _ => mul_self_nonneg (G a)
  rw [stepOpt_eq hH hHS hB', ← mul_div_assoc, div_lt_iff₀ ha] at hε
  have hε0 : 0 < ε := by nlinarith
  rw [integral_quadModel_batchGrad hB hX hind hG hS]
  nlinarith

/-- The hypotheses of `lt_integral_quadModel_of_two_mul_stepOpt_lt` are satisfiable: copies
of `G = 1`, `H = 1`, and the step `3 > 2ε_opt = 2`. -/
example (L : ℝ) : L < ∫ ω, quadModel L (fun _ : Unit => 1) 1 3
    (batchGrad (fun (_ : Fin 2) (_ : Unit) (_ : Unit) => (1 : ℝ)) ω) ∂Measure.dirac () :=
  lt_integral_quadModel_of_two_mul_stepOpt_lt (X := fun _ _ _ => 1) (G := fun _ => 1)
    (S := 0) two_ne_zero (fun _ _ => memLp_const _) (fun _ _ _ => indepFun_const_left _ _)
    (fun _ _ => by simp) (fun _ => covMatrix_const _) (by simp) (by simp) L
    (by norm_num [stepOpt, stepMax, noiseScale])

end Transformer.NoiseScale
