/-
# Implications and simplifications

arXiv:1812.06162, §2.2, "Implications and Simplifications".  The best drop of eq. (2.7),
`ΔL_opt(B) = ΔL_max/(1 + B_noise/B)`, as a function of the batch: up to `B_noise` it grows
linearly in `B` within a factor 2 (`lossDrop_le`, `le_lossDrop_of_le`), beyond it it lies
within a factor 2 of its limit `ΔL_max` (`le_lossDrop_of_ge`, `tendsto_lossDrop`), and at
`B = B_noise` both the drop and the step are half the largest, the turning point of
Fig. 3 (`lossDrop_noiseScale`, `stepOpt_noiseScale`).  The paper's `B ≪ B_noise` and
`B ≫ B_noise` are read as these bounds; they need `B_noise > 0`, that is `tr(HΣ) > 0`.

For a Hessian `hI`, `h ≠ 0`, the noise scale is the simple one `B_simple = tr(Σ)/|G|²` of
eq. (2.9) (`simpleNoiseScale`, `noiseScale_smul_one`), and
`E[|G_est - G|²]/|G|² = tr(Σ)/(B|G|²) = B_simple/B`, eq. (2.10)
(`integral_dotProduct_sub_batchGrad`).
-/

import Transformer.NoiseScale.Section2_NoiseScale

open MeasureTheory ProbabilityTheory Filter Topology
open scoped Matrix

namespace Transformer.NoiseScale

variable {ι : Type*} [Fintype ι] {G : ι → ℝ} {H S : Matrix ι ι ℝ}

theorem lossDropMax_nonneg (hH : 0 < G ⬝ᵥ H *ᵥ G) : 0 ≤ lossDropMax G H := by
  unfold lossDropMax
  positivity

/-- The hypothesis of `lossDropMax_nonneg` is satisfiable: `G = H = 1`. -/
example : 0 ≤ lossDropMax (fun _ : Unit => 1) 1 := lossDropMax_nonneg (by simp)

theorem noiseScale_pos (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 < (H * S).trace) :
    0 < noiseScale G H S :=
  div_pos hHS hH

/-- The hypotheses of `noiseScale_pos` are satisfiable: `G = H = Σ = 1`. -/
example : 0 < noiseScale (fun _ : Unit => 1) 1 1 := noiseScale_pos (by simp) (by simp)

/-- Fig. 3: at `B = B_noise` the best drop is half the largest, `ΔL_max/2`. -/
theorem lossDrop_noiseScale (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 < (H * S).trace) :
    lossDropMax G H / (1 + noiseScale G H S / noiseScale G H S) = lossDropMax G H / 2 := by
  rw [div_self (noiseScale_pos hH hHS).ne']
  norm_num

/-- The hypotheses of `lossDrop_noiseScale` are satisfiable: `G = H = Σ = 1`. -/
example : lossDropMax (fun _ : Unit => 1) 1 / (1 + noiseScale (fun _ : Unit => 1) 1 1 /
    noiseScale (fun _ : Unit => 1) 1 1) = lossDropMax (fun _ : Unit => 1) 1 / 2 :=
  lossDrop_noiseScale (by simp) (by simp)

/-- Fig. 3: at `B = B_noise` the best step is half the largest, `ε_max/2`. -/
theorem stepOpt_noiseScale (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 < (H * S).trace) :
    stepOpt G H S (noiseScale G H S) = stepMax G H / 2 := by
  rw [stepOpt, div_self (noiseScale_pos hH hHS).ne']
  norm_num

/-- The hypotheses of `stepOpt_noiseScale` are satisfiable: `G = H = Σ = 1`. -/
example : stepOpt (fun _ : Unit => 1) 1 1 (noiseScale (fun _ : Unit => 1) 1 1) =
    stepMax (fun _ : Unit => 1) 1 / 2 :=
  stepOpt_noiseScale (by simp) (by simp)

/-- §2.2, small batches: the best drop is at most linear in the batch,
`ΔL_opt(B) ≤ ΔL_max B/B_noise`. -/
theorem lossDrop_le (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 < (H * S).trace) {B : ℝ} (hB : 0 < B) :
    lossDropMax G H / (1 + noiseScale G H S / B) ≤ lossDropMax G H * B / noiseScale G H S := by
  have hN := noiseScale_pos hH hHS
  have hD := lossDropMax_nonneg hH
  rw [div_le_div_iff₀ (by positivity) hN]
  field_simp
  nlinarith [mul_nonneg hD hB.le]

/-- The hypotheses of `lossDrop_le` are satisfiable: `G = H = Σ = 1`, `B = 1`. -/
example : lossDropMax (fun _ : Unit => 1) 1 / (1 + noiseScale (fun _ : Unit => 1) 1 1 / 1) ≤
    lossDropMax (fun _ : Unit => 1) 1 * 1 / noiseScale (fun _ : Unit => 1) 1 1 :=
  lossDrop_le (by simp) (by simp) one_pos

/-- §2.2, small batches: up to `B_noise` the best drop is at least half the linear one,
`ΔL_max B/(2B_noise) ≤ ΔL_opt(B)`. -/
theorem le_lossDrop_of_le (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 < (H * S).trace) {B : ℝ}
    (hB : 0 < B) (hBN : B ≤ noiseScale G H S) :
    lossDropMax G H * B / (2 * noiseScale G H S) ≤
      lossDropMax G H / (1 + noiseScale G H S / B) := by
  have hN := noiseScale_pos hH hHS
  have hD := lossDropMax_nonneg hH
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  field_simp
  nlinarith [mul_nonneg (mul_nonneg hD hB.le) (sub_nonneg.2 hBN)]

/-- The hypotheses of `le_lossDrop_of_le` are satisfiable: `G = H = Σ = 1`, `B = 1`. -/
example : lossDropMax (fun _ : Unit => 1) 1 * 1 / (2 * noiseScale (fun _ : Unit => 1) 1 1) ≤
    lossDropMax (fun _ : Unit => 1) 1 / (1 + noiseScale (fun _ : Unit => 1) 1 1 / 1) :=
  le_lossDrop_of_le (by simp) (by simp) one_pos (by simp [noiseScale])

/-- §2.2, large batches: from `B_noise` on, the best drop is within a factor 2 of
`ΔL_max`. -/
theorem le_lossDrop_of_ge (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 < (H * S).trace) {B : ℝ}
    (hBN : noiseScale G H S ≤ B) :
    lossDropMax G H / 2 ≤ lossDropMax G H / (1 + noiseScale G H S / B) := by
  have hN := noiseScale_pos hH hHS
  have hB : 0 < B := hN.trans_le hBN
  exact div_le_div_of_nonneg_left (lossDropMax_nonneg hH) (by positivity)
    (by rw [← one_add_one_eq_two]; gcongr; exact (div_le_one hB).2 hBN)

/-- The hypotheses of `le_lossDrop_of_ge` are satisfiable: `G = H = Σ = 1`, `B = 1`. -/
example : lossDropMax (fun _ : Unit => 1) 1 / 2 ≤
    lossDropMax (fun _ : Unit => 1) 1 / (1 + noiseScale (fun _ : Unit => 1) 1 1 / 1) :=
  le_lossDrop_of_ge (by simp) (by simp) (by simp [noiseScale])

/-- §2.2, large batches: the best drop tends to `ΔL_max` as `B → ∞`. -/
theorem tendsto_lossDrop (G : ι → ℝ) (H S : Matrix ι ι ℝ) :
    Tendsto (fun B : ℝ => lossDropMax G H / (1 + noiseScale G H S / B)) atTop
      (𝓝 (lossDropMax G H)) := by
  have h : Tendsto (fun B : ℝ => 1 + noiseScale G H S / B) atTop (𝓝 (1 + 0)) :=
    tendsto_const_nhds.add (tendsto_const_nhds.div_atTop tendsto_id)
  have h' := (tendsto_const_nhds (x := lossDropMax G H)).div h (by norm_num)
  rw [add_zero, div_one] at h'
  exact h'

/-- The simple noise scale `B_simple = tr(Σ)/|G|²`, eq. (2.9). -/
noncomputable def simpleNoiseScale (G : ι → ℝ) (S : Matrix ι ι ℝ) : ℝ := S.trace / (G ⬝ᵥ G)

/-- **Eq. (2.9)**: for a Hessian `hI`, `h ≠ 0`, the noise scale (2.8) is `B_simple`. -/
theorem noiseScale_smul_one [DecidableEq ι] {h : ℝ} (hh : h ≠ 0) (G : ι → ℝ)
    (S : Matrix ι ι ℝ) : noiseScale G (h • 1) S = simpleNoiseScale G S := by
  simp only [noiseScale, simpleNoiseScale, Matrix.smul_mul, one_mul, Matrix.trace_smul,
    Matrix.smul_mulVec, Matrix.one_mulVec, dotProduct_smul, smul_eq_mul]
  exact mul_div_mul_left _ _ hh

/-- The hypothesis of `noiseScale_smul_one` is satisfiable: `h = 2`. -/
example (G : Unit → ℝ) (S : Matrix Unit Unit ℝ) :
    noiseScale G ((2 : ℝ) • 1) S = simpleNoiseScale G S :=
  noiseScale_smul_one two_ne_zero G S

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {B : ℕ} {X : Fin B → Ω → ι → ℝ}

/-- **Eq. (2.10)**: `E[|G_est - G|²]/|G|² = tr(Σ)/(B|G|²) = B_simple/B`. -/
theorem integral_dotProduct_sub_batchGrad (hB : B ≠ 0)
    (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) :
    (∫ ω, (batchGrad X ω - G) ⬝ᵥ (batchGrad X ω - G) ∂P) / (G ⬝ᵥ G) =
        S.trace / (B * (G ⬝ᵥ G)) ∧
      S.trace / (B * (G ⬝ᵥ G)) = simpleNoiseScale G S / B := by
  have hm : (fun a => ∫ ω, batchGrad X ω a ∂P) = G :=
    funext (integral_batchGrad hB (fun i a => (hX i a).integrable (q := 2) (by norm_num)) hG)
  have h := integral_dotProduct_sub (memLp_batchGrad hX)
  rw [hm, covMatrix_batchGrad hX hind hS, Matrix.trace_smul, smul_eq_mul] at h
  exact ⟨by rw [h, inv_mul_eq_div, div_div], by rw [simpleNoiseScale, div_div, mul_comm]⟩

/-- The hypotheses of `integral_dotProduct_sub_batchGrad` are satisfiable: copies of a
constant. -/
example (G : ι → ℝ) :
    (∫ ω, (batchGrad (fun (_ : Fin 2) (_ : Unit) => G) ω - G) ⬝ᵥ
      (batchGrad (fun (_ : Fin 2) (_ : Unit) => G) ω - G) ∂Measure.dirac ()) / (G ⬝ᵥ G) =
        (0 : Matrix ι ι ℝ).trace / ((2 : ℕ) * (G ⬝ᵥ G)) ∧
      (0 : Matrix ι ι ℝ).trace / ((2 : ℕ) * (G ⬝ᵥ G)) = simpleNoiseScale G 0 / (2 : ℕ) :=
  integral_dotProduct_sub_batchGrad (X := fun _ _ => G) two_ne_zero
    (fun _ _ => memLp_const _) (fun _ _ _ => indepFun_const_left _ _) (fun _ _ => by simp)
    (fun _ => covMatrix_const G)

end Transformer.NoiseScale
