/-
# The quadratic model of a step

arXiv:1812.06162, §2.2.  A step `θ - εV` changes the loss by
`L(θ - εV) ≈ L(θ) - εGᵀV + ½ε²VᵀHV`, eq. (2.4) (`quadModel`), the second-order Taylor
expansion: for a twice continuously differentiable `L` of gradient `G` and Hessian `H` at
`θ`, the error is `o(ε²)` (`isLittleO_sub_quadModel`), the precise sense of the paper's
`≈`.  Along the true gradient, `V = G`, the model is least at
`ε_max = |G|²/GᵀHG` (`stepMax`, `quadModel_stepMax_lt`), which needs the curvature
`GᵀHG > 0` the paper leaves implicit.

The mean of the model over a random direction `V` of mean `m` and covariance `C` is
`L - εGᵀm + ½ε²(mᵀHm + tr(HC))` (`integral_quadModel`); over the batch gradient it is
eq. (2.5), `E[L(θ - εG_est)] = L - ε|G|² + ½ε²(GᵀHG + tr(HΣ)/B)`
(`integral_quadModel_batchGrad`), in the model (2.4), as the paper evaluates it.
-/

import Mathlib.Analysis.Calculus.Taylor
import Transformer.NoiseScale.Section2_Batches

open MeasureTheory ProbabilityTheory Filter Asymptotics Topology
open scoped Matrix

namespace Transformer.NoiseScale

variable {ι : Type*} [Fintype ι]

/-- The quadratic model `L - εGᵀV + ½ε²VᵀHV` of the loss at `θ - εV`, eq. (2.4). -/
noncomputable def quadModel (L : ℝ) (G : ι → ℝ) (H : Matrix ι ι ℝ) (ε : ℝ) (V : ι → ℝ) : ℝ :=
  L - ε * (G ⬝ᵥ V) + ε ^ 2 / 2 * (V ⬝ᵥ H *ᵥ V)

/-- **Eq. (2.4)**: `L(θ - εV) = L(θ) - εGᵀV + ½ε²VᵀHV + o(ε²)` as `ε → 0`, for a twice
continuously differentiable loss `L` of gradient `G` and Hessian `H` at `θ`. -/
theorem isLittleO_sub_quadModel {L : (ι → ℝ) → ℝ} (hL : ContDiff ℝ 2 L) {θ G : ι → ℝ}
    {H : Matrix ι ι ℝ} (hG : ∀ v, fderiv ℝ L θ v = G ⬝ᵥ v)
    (hH : ∀ v, iteratedFDeriv ℝ 2 L θ (fun _ => v) = v ⬝ᵥ H *ᵥ v) (V : ι → ℝ) :
    (fun ε => L (θ - ε • V) - quadModel (L θ) G H ε V) =o[𝓝 0] fun ε => ε ^ 2 := by
  set g : ℝ → ℝ := fun ε => L (θ - ε • V) with hg_def
  have hg : ContDiff ℝ 2 g := hL.comp (contDiff_const.sub (contDiff_id.smul contDiff_const))
  have h1 : deriv g 0 = -(G ⬝ᵥ V) := by
    have hline : HasDerivAt (fun t : ℝ => θ - t • V) (-V) 0 := by
      simpa using ((hasDerivAt_id (0 : ℝ)).smul_const V).const_sub θ
    have hf : HasFDerivAt L (fderiv ℝ L θ) (θ - (0 : ℝ) • V) := by
      simpa using (hL.differentiable (by norm_num) θ).hasFDerivAt
    change deriv (L ∘ fun t => θ - t • V) 0 = _
    rw [(hf.comp_hasDerivAt 0 hline).deriv, map_neg, hG]
  have h2 : iteratedDeriv 2 g 0 = V ⬝ᵥ H *ᵥ V := by
    let M : ℝ →L[ℝ] ι → ℝ := (ContinuousLinearMap.id ℝ ℝ).smulRight (-V)
    have hshift : ContDiff ℝ 2 (fun v => L (θ + v)) := hL.comp (contDiff_const.add contDiff_id)
    have hcomp := M.iteratedFDeriv_comp_right hshift 0 (i := 2) (by norm_num)
    have e : g = (fun v => L (θ + v)) ∘ M := by
      funext t
      simp [hg_def, M, sub_eq_add_neg]
    rw [iteratedDeriv_eq_iteratedFDeriv, e, hcomp,
      ContinuousMultilinearMap.compContinuousLinearMap_apply]
    simp [M, iteratedFDeriv_comp_add_left, hH, Matrix.mulVec_neg]
  have ht : ∀ ε, taylorWithinEval g 2 Set.univ 0 ε = quadModel (L θ) G H ε V := fun ε => by
    rw [taylor_within_apply]
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, iteratedDerivWithin_univ,
      iteratedDeriv_zero, iteratedDeriv_one, h1, h2]
    simp [hg_def, quadModel]
    ring
  have key := taylor_isLittleO_univ (x₀ := 0) hg
  simp only [ht, sub_zero] at key
  exact key

/-- The hypotheses of `isLittleO_sub_quadModel` are satisfiable: a constant loss. -/
example (θ V : ι → ℝ) : (fun ε => (fun _ : ι → ℝ => (1 : ℝ)) (θ - ε • V) -
    quadModel 1 0 0 ε V) =o[𝓝 0] fun ε => ε ^ 2 :=
  isLittleO_sub_quadModel (θ := θ) contDiff_const (fun v => by simp)
    (fun v => by simp [iteratedFDeriv_const_of_ne]) V

/-- The best step `ε_max = |G|²/GᵀHG` along the true gradient, §2.2. -/
noncomputable def stepMax (G : ι → ℝ) (H : Matrix ι ι ℝ) : ℝ := G ⬝ᵥ G / (G ⬝ᵥ H *ᵥ G)

/-- A quadratic `L - εg + ½ε²a`, `a > 0`, is `L - g²/(2a) + ½a(ε - g/a)²`. -/
theorem quadratic_eq_sq {a : ℝ} (ha : a ≠ 0) (L g ε : ℝ) :
    L - ε * g + ε ^ 2 / 2 * a = L - g ^ 2 / (2 * a) + a / 2 * (ε - g / a) ^ 2 := by
  field_simp
  ring

/-- The hypothesis of `quadratic_eq_sq` is satisfiable. -/
example : (0 : ℝ) - 1 * 1 + 1 ^ 2 / 2 * 1 = 0 - 1 ^ 2 / (2 * 1) + 1 / 2 * (1 - 1 / 1) ^ 2 :=
  quadratic_eq_sq one_ne_zero 0 1 1

/-- §2.2: the model (2.4) with `V = G` is least exactly at `ε = ε_max`, if `GᵀHG > 0`. -/
theorem quadModel_stepMax_lt {G : ι → ℝ} {H : Matrix ι ι ℝ} (hH : 0 < G ⬝ᵥ H *ᵥ G) (L : ℝ)
    {ε : ℝ} (hε : ε ≠ stepMax G H) : quadModel L G H (stepMax G H) G < quadModel L G H ε G := by
  have h := mul_pos (half_pos hH) (sq_pos_of_ne_zero (sub_ne_zero.2 hε))
  simp only [stepMax] at h
  simp only [quadModel, quadratic_eq_sq hH.ne', stepMax, sub_self]
  linarith [zero_pow (M₀ := ℝ) two_ne_zero]

/-- The hypothesis of `quadModel_stepMax_lt` is satisfiable: `G = H = 1` in one dimension. -/
example (L : ℝ) : quadModel L (fun _ : Unit => 1) 1 (stepMax (fun _ : Unit => 1) 1)
    (fun _ => 1) < quadModel L (fun _ : Unit => 1) 1 0 (fun _ => 1) :=
  quadModel_stepMax_lt (by simp) L (by simp [stepMax])

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

/-- The mean of the model (2.4) over a random direction `V` of mean `m` and covariance `C`:
`L - εGᵀm + ½ε²(mᵀHm + tr(HC))`. -/
theorem integral_quadModel {V : Ω → ι → ℝ} (hV : ∀ a, MemLp (fun ω => V ω a) 2 P) (L : ℝ)
    (G : ι → ℝ) (H : Matrix ι ι ℝ) (ε : ℝ) :
    ∫ ω, quadModel L G H ε (V ω) ∂P = L - ε * (G ⬝ᵥ fun a => ∫ ω, V ω a ∂P) +
      ε ^ 2 / 2 * ((fun a => ∫ ω, V ω a ∂P) ⬝ᵥ H *ᵥ (fun a => ∫ ω, V ω a ∂P) +
        (H * covMatrix V P).trace) := by
  have h1 := fun a => (hV a).integrable (q := 2) (by norm_num)
  have hl : Integrable (fun ω => ε * (G ⬝ᵥ V ω)) P := (integrable_dotProduct h1 G).const_mul ε
  have hd : Integrable (fun ω => L - ε * (G ⬝ᵥ V ω)) P := (integrable_const L).sub hl
  unfold quadModel
  rw [integral_add hd ((integrable_quadForm hV H).const_mul _),
    integral_sub (integrable_const L) hl, integral_const, integral_const_mul,
    integral_const_mul, integral_dotProduct h1, integral_quadForm hV]
  simp

/-- The hypothesis of `integral_quadModel` is satisfiable: a constant direction. -/
example (L : ℝ) (G V : ι → ℝ) (H : Matrix ι ι ℝ) (ε : ℝ) :
    ∫ ω, quadModel L G H ε ((fun _ : Unit => V) ω) ∂Measure.dirac () =
      L - ε * (G ⬝ᵥ fun a => ∫ ω, (fun _ : Unit => V) ω a ∂Measure.dirac ()) +
        ε ^ 2 / 2 * ((fun a => ∫ ω, (fun _ : Unit => V) ω a ∂Measure.dirac ()) ⬝ᵥ
          H *ᵥ (fun a => ∫ ω, (fun _ : Unit => V) ω a ∂Measure.dirac ()) +
            (H * covMatrix (fun _ : Unit => V) (Measure.dirac ())).trace) :=
  integral_quadModel (V := fun _ => V) (fun a => memLp_const (V a)) L G H ε

/-- **Eq. (2.5)**: `E[L(θ - εG_est)] = L - ε|G|² + ½ε²(GᵀHG + tr(HΣ)/B)`, in the model
(2.4). -/
theorem integral_quadModel_batchGrad {B : ℕ} {X : Fin B → Ω → ι → ℝ} {G : ι → ℝ}
    {S : Matrix ι ι ℝ} (hB : B ≠ 0) (hX : ∀ i a, MemLp (fun ω => X i ω a) 2 P)
    (hind : Pairwise fun i j => IndepFun (X i) (X j) P) (hG : ∀ i a, ∫ ω, X i ω a ∂P = G a)
    (hS : ∀ i, covMatrix (X i) P = S) (L : ℝ) (H : Matrix ι ι ℝ) (ε : ℝ) :
    ∫ ω, quadModel L G H ε (batchGrad X ω) ∂P =
      L - ε * (G ⬝ᵥ G) + ε ^ 2 / 2 * (G ⬝ᵥ H *ᵥ G + (H * S).trace / B) := by
  have hm : (fun a => ∫ ω, batchGrad X ω a ∂P) = G :=
    funext (integral_batchGrad hB (fun i a => (hX i a).integrable (q := 2) (by norm_num)) hG)
  rw [integral_quadModel (memLp_batchGrad hX), hm, covMatrix_batchGrad hX hind hS,
    Matrix.mul_smul, Matrix.trace_smul, smul_eq_mul, inv_mul_eq_div]

/-- The hypotheses of `integral_quadModel_batchGrad` are satisfiable: copies of a
constant. -/
example (L : ℝ) (G : ι → ℝ) (H : Matrix ι ι ℝ) (ε : ℝ) :
    ∫ ω, quadModel L G H ε (batchGrad (fun (_ : Fin 2) (_ : Unit) => G) ω) ∂Measure.dirac () =
      L - ε * (G ⬝ᵥ G) + ε ^ 2 / 2 * (G ⬝ᵥ H *ᵥ G + (H * 0).trace / (2 : ℕ)) :=
  integral_quadModel_batchGrad (X := fun _ _ => G) two_ne_zero (fun _ _ => memLp_const _)
    (fun _ _ _ => indepFun_const_left _ _) (fun _ _ => by simp) (fun _ => covMatrix_const G)
    L H ε

end Transformer.NoiseScale
