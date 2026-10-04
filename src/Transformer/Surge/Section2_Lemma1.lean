/-
# The optimal learning rate of an update

arXiv:2405.14578, §2.1, Lemma 1, proved in Appendix B.  Along an update `V`, the loss is the
quadratic model (30) of the true gradient `G` and Hessian `H`, as in arXiv:1812.06162
(`quadModel`): the paper "adopt[s] the equal sign here".  The expected improvement over a
step `εV` is `E[ΔL] = εGᵀE[V] - ½ε²E[VᵀHV]`, eq. (31), with
`E[VᵀHV] = tr(H cov(V)) + E[V]ᵀHE[V]` (`integral_lossDrop`).  The learning rate
`ε_opt = GᵀE[V]/(tr(H cov(V)) + E[V]ᵀHE[V])` of eq. (6) (`lrOpt`) is its only maximizer
(`integral_lossDrop_lt`, `integral_lossDrop_le`), where the improvement is
`ΔL_opt = GᵀE[V] ε_opt/2`, eq. (7) (`integral_lossDrop_lrOpt`), never negative
(`lossDropOpt_nonneg`).  Lemma 1 leaves implicit that `tr(H cov(V)) + E[V]ᵀHE[V] > 0`: without
it `E[ΔL]` has no largest value, or is constant.

For the gradient `G_est` of a batch of `B` independent examples, of mean `G` and covariance
`Σ/B`, eq. (6) is eq. (1), the `ε_opt(B) = ε_max/(1 + B_noise/B)` of arXiv:1812.06162, eq. (2.6)
(`lrOpt_batchGrad`).
-/

import Transformer.NoiseScale.Section2_NoiseScale

open MeasureTheory ProbabilityTheory
open scoped Matrix

namespace Transformer.Surge

open Transformer.NoiseScale

variable {ι : Type*} [Fintype ι]

/-- The optimal learning rate `ε_opt = Gᵀm/(tr(HC) + mᵀHm)` of an update of mean `m` and
covariance `C`, eq. (6). -/
noncomputable def lrOpt (G m : ι → ℝ) (H C : Matrix ι ι ℝ) : ℝ :=
  G ⬝ᵥ m / ((H * C).trace + m ⬝ᵥ H *ᵥ m)

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- The mean `E[V]` of a random vector `V`. -/
noncomputable def meanVec (V : Ω → ι → ℝ) (P : Measure Ω) : ι → ℝ := fun a => ∫ ω, V ω a ∂P

/-- **Eq. (31)**: in the model (30), the expected improvement over a step `εV` is
`E[ΔL] = εGᵀE[V] - ½ε²(tr(H cov(V)) + E[V]ᵀHE[V])`. -/
theorem integral_lossDrop {V : Ω → ι → ℝ} (hV : ∀ a, MemLp (fun ω => V ω a) 2 P) (L : ℝ)
    (G : ι → ℝ) (H : Matrix ι ι ℝ) (ε : ℝ) :
    L - ∫ ω, quadModel L G H ε (V ω) ∂P = ε * (G ⬝ᵥ meanVec V P) -
      ε ^ 2 / 2 * ((H * covMatrix V P).trace + meanVec V P ⬝ᵥ H *ᵥ meanVec V P) := by
  rw [integral_quadModel hV]
  unfold meanVec
  ring

/-- The hypothesis of `integral_lossDrop` is satisfiable: a constant update. -/
example (L ε : ℝ) := integral_lossDrop (P := Measure.dirac ())
  (V := fun (_ : Unit) (_ : Unit) => (1 : ℝ)) (fun _ => memLp_const _) L 1 1 ε

/-- `εg - ½ε²a = g²/(2a) - ½a(ε - g/a)²`, `a ≠ 0`. -/
theorem lossDrop_eq_sq {a : ℝ} (ha : a ≠ 0) (g ε : ℝ) :
    ε * g - ε ^ 2 / 2 * a = g ^ 2 / (2 * a) - a / 2 * (ε - g / a) ^ 2 := by
  field_simp
  ring

/-- The hypothesis of `lossDrop_eq_sq` is satisfiable: `a = 1`. -/
example (g ε : ℝ) := lossDrop_eq_sq one_ne_zero g ε

/-- **Lemma 1**, eq. (6): `ε_opt = GᵀE[V]/(tr(H cov(V)) + E[V]ᵀHE[V])` is the only maximizer
of `E[ΔL]` over the learning rate, in the model (30).  Lemma 1 leaves implicit the hypothesis
`tr(H cov(V)) + E[V]ᵀHE[V] > 0`. -/
theorem integral_lossDrop_lt {V : Ω → ι → ℝ} (hV : ∀ a, MemLp (fun ω => V ω a) 2 P) (L : ℝ)
    {G : ι → ℝ} {H : Matrix ι ι ℝ}
    (hD : 0 < (H * covMatrix V P).trace + meanVec V P ⬝ᵥ H *ᵥ meanVec V P) {ε : ℝ}
    (hε : ε ≠ lrOpt G (meanVec V P) H (covMatrix V P)) :
    L - ∫ ω, quadModel L G H ε (V ω) ∂P <
      L - ∫ ω, quadModel L G H (lrOpt G (meanVec V P) H (covMatrix V P)) (V ω) ∂P := by
  rw [integral_lossDrop hV, integral_lossDrop hV]
  unfold lrOpt at hε ⊢
  generalize G ⬝ᵥ meanVec V P = g at hε ⊢
  generalize (H * covMatrix V P).trace + meanVec V P ⬝ᵥ H *ᵥ meanVec V P = a at hD hε ⊢
  rw [lossDrop_eq_sq hD.ne', lossDrop_eq_sq hD.ne', sub_self]
  have := mul_pos (half_pos hD) (sq_pos_of_ne_zero (sub_ne_zero.2 hε))
  linarith [zero_pow (M₀ := ℝ) two_ne_zero]

/-- The hypotheses of `integral_lossDrop_lt` are satisfiable: the constant update `V = 1`, with
`G = H = 1` and `ε = 0`. -/
example (L : ℝ) := integral_lossDrop_lt (P := Measure.dirac ())
  (V := fun (_ : Unit) (_ : Unit) => (1 : ℝ)) (fun _ => memLp_const _) L (G := fun _ => 1)
  (H := 1) (by simp [covMatrix_const, meanVec]) (ε := 0) (by simp [lrOpt, covMatrix_const, meanVec])

/-- **Lemma 1**, eq. (6): no learning rate improves the loss more than `ε_opt`, in the model
(30), under the hypothesis of `integral_lossDrop_lt`. -/
theorem integral_lossDrop_le {V : Ω → ι → ℝ} (hV : ∀ a, MemLp (fun ω => V ω a) 2 P) (L : ℝ)
    {G : ι → ℝ} {H : Matrix ι ι ℝ}
    (hD : 0 < (H * covMatrix V P).trace + meanVec V P ⬝ᵥ H *ᵥ meanVec V P) (ε : ℝ) :
    L - ∫ ω, quadModel L G H ε (V ω) ∂P ≤
      L - ∫ ω, quadModel L G H (lrOpt G (meanVec V P) H (covMatrix V P)) (V ω) ∂P := by
  rcases eq_or_ne ε (lrOpt G (meanVec V P) H (covMatrix V P)) with rfl | hε
  · exact le_rfl
  · exact (integral_lossDrop_lt hV L hD hε).le

/-- The hypotheses of `integral_lossDrop_le` are satisfiable: the constant update `V = 1`. -/
example (L ε : ℝ) := integral_lossDrop_le (P := Measure.dirac ())
  (V := fun (_ : Unit) (_ : Unit) => (1 : ℝ)) (fun _ => memLp_const _) L (G := fun _ => 1)
  (H := 1) (by simp [covMatrix_const, meanVec]) ε

/-- **Lemma 1**, eq. (7): at `ε_opt` the expected improvement is `ΔL_opt = GᵀE[V] ε_opt/2`, in
the model (30). -/
theorem integral_lossDrop_lrOpt {V : Ω → ι → ℝ} (hV : ∀ a, MemLp (fun ω => V ω a) 2 P)
    (L : ℝ) (G : ι → ℝ) (H : Matrix ι ι ℝ) :
    L - ∫ ω, quadModel L G H (lrOpt G (meanVec V P) H (covMatrix V P)) (V ω) ∂P =
      G ⬝ᵥ meanVec V P / 2 * lrOpt G (meanVec V P) H (covMatrix V P) := by
  rw [integral_lossDrop hV]
  unfold lrOpt
  generalize G ⬝ᵥ meanVec V P = g
  generalize (H * covMatrix V P).trace + meanVec V P ⬝ᵥ H *ᵥ meanVec V P = a
  rcases eq_or_ne a 0 with rfl | ha
  · simp
  · field_simp
    ring

/-- The hypothesis of `integral_lossDrop_lrOpt` is satisfiable: a constant update. -/
example (L : ℝ) := integral_lossDrop_lrOpt (P := Measure.dirac ())
  (V := fun (_ : Unit) (_ : Unit) => (1 : ℝ)) (fun _ => memLp_const _) L 1 1

/-- The best improvement `ΔL_opt = GᵀE[V] ε_opt/2` of eq. (7) is never negative when
`tr(H cov(V)) + E[V]ᵀHE[V] ≥ 0`. -/
theorem lossDropOpt_nonneg {G m : ι → ℝ} {H C : Matrix ι ι ℝ}
    (hD : 0 ≤ (H * C).trace + m ⬝ᵥ H *ᵥ m) : 0 ≤ G ⬝ᵥ m / 2 * lrOpt G m H C := by
  unfold lrOpt
  rw [div_mul_div_comm, ← sq]
  exact div_nonneg (sq_nonneg _) (by linarith)

/-- The hypothesis of `lossDropOpt_nonneg` is satisfiable: `m = H = 1`, `C = 0`. -/
example := lossDropOpt_nonneg (G := fun _ : Unit => 1) (m := fun _ => 1) (H := 1) (C := 0)
  (by simp)

/-- Eq. (6) generalizes eq. (1), arXiv:1812.06162's eq. (2.6): for the gradient `G_est` of a
batch of `B` independent examples of mean `G` and covariance `Σ`,
`ε_opt = ε_max/(1 + B_noise/B)`. -/
theorem lrOpt_batchGrad {B : ℕ} {X : Fin B → Ω → ι → ℝ} {G : ι → ℝ} {H S : Matrix ι ι ℝ}
    (hB : B ≠ 0) (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) (hH : 0 < G ⬝ᵥ H *ᵥ G) (hHS : 0 ≤ (H * S).trace) :
    lrOpt G (meanVec (batchGrad X) P) H (covMatrix (batchGrad X) P) = stepOpt G H S B := by
  have hm : meanVec (batchGrad X) P = G := funext fun a =>
    integral_batchGrad hB (fun i a => (hX i a).integrable (q := 2) (by norm_num)) hG a
  have hB' : (0 : ℝ) < B := Nat.cast_pos.2 (Nat.pos_of_ne_zero hB)
  rw [hm, covMatrix_batchGrad hX hind hS, stepOpt_eq hH hHS hB', lrOpt, Matrix.mul_smul,
    Matrix.trace_smul, smul_eq_mul]
  congr 1
  ring

/-- The hypotheses of `lrOpt_batchGrad` are satisfiable: two copies of `G = 1`, with `H = 1`,
`Σ = 0`. -/
example := lrOpt_batchGrad (P := Measure.dirac ())
  (X := fun (_ : Fin 2) (_ : Unit) (_ : Unit) => (1 : ℝ)) (G := fun _ => 1) (H := 1) (S := 0)
  two_ne_zero (fun _ _ => memLp_const _) (fun _ _ _ => indepFun_const_left _ _)
  (fun _ _ => by simp) (fun _ => covMatrix_const _) (by simp) (by simp)

end Transformer.Surge
