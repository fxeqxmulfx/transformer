module

public import Transformer.Homogenized.Section7_GaussianPolynomialInterpolation
public import Mathlib.Analysis.Normed.Lp.PiLp

/-!
# Polynomial Gaussian interpolation in the original finite feature coordinates

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and its application
to the two Gaussian feature blocks in `eq:SIGMA_R`.
The proved interpolation theorem uses `Fin m` coordinates. The actual
head features have a block, token and feature index. An isometric
relabeling transfers the proved theorem to every finite index type.

The actual random vectors retain their original probability space.
Coordinate means, second moments and independence are transported
explicitly. The test's genuine derivatives are composed with the inverse
isometry, preserving their polynomial norm bounds exactly. Relabeling
both actual Hessian axes then identifies the entire original trace.
The trace bound remains an explicit hypothesis of this general analytic
interface; bounds for particular softmax observables require their
actual derivative estimates. Empty types and singular laws are allowed.
-/

@[expose] public section
open Real MeasureTheory ProbabilityTheory Set
open scoped BigOperators
namespace Transformer.Homogenized
/-- Relabeling the actual Hessian axes and covariance entries preserves
exactly the original covariance trace. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp`, expressed in an arbitrary finite coordinate system. -/
theorem gaussian_covariance_trace_relabel {ι κ : Type*} [Fintype ι] [Fintype κ]
    [DecidableEq ι] [DecidableEq κ] (e : κ ≃ ι)
    (φ : EuclideanSpace ℝ ι → ℝ) (hφ : ContDiff ℝ 2 φ)
    (S : Matrix ι ι ℝ) (z : EuclideanSpace ℝ κ) :
    (∑ i : κ, ∑ j : κ, S (e i) (e j) *
      iteratedFDeriv ℝ 2 (φ ∘ LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e) z
        ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]) =
      ∑ i : ι, ∑ j : ι, S i j *
        iteratedFDeriv ℝ 2 φ (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e z)
          ![EuclideanSpace.single i 1, EuclideanSpace.single j 1] := by
  change ι → ι → ℝ at S
  let L := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e
  have hentry (i j : κ) :
      iteratedFDeriv ℝ 2 (φ ∘ L) z ![EuclideanSpace.single i 1, EuclideanSpace.single j 1] =
        iteratedFDeriv ℝ 2 φ (L z) ![EuclideanSpace.single (e i) 1, EuclideanSpace.single (e j) 1] := by
    change iteratedFDeriv ℝ 2 (φ ∘ (L : EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι)) z
      ![EuclideanSpace.single i 1, EuclideanSpace.single j 1] = _
    rw [(L : EuclideanSpace ℝ κ →L[ℝ] EuclideanSpace ℝ ι).iteratedFDeriv_comp_right
      hφ z (i := 2) (by norm_num)]
    rw [ContinuousMultilinearMap.compContinuousLinearMap_apply]
    congr 1
    funext k
    fin_cases k <;> simp [L]
  change (∑ i : κ, ∑ j : κ, S (e i) (e j) *
    iteratedFDeriv ℝ 2 (φ ∘ L) z ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]) = _
  simp_rw [hentry]
  apply Fintype.sum_equiv e
  intro i
  apply Fintype.sum_equiv e
  intro j
  rfl

/-- The proved polynomial Gaussian interpolation bounds original
expectation differences for every finite coordinate type. The actual
Gaussian laws, means, second moments, independence, regularity and trace
bound are unchanged by an isometric relabeling to `Fin`.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp` and the integration
step in `eq:Lispchitz_bound_softmax`. Polynomial growth corrects the
source's unrestricted `C²` wording, as in the original proved theorem. -/
theorem gaussian_interp_indexed_expectation_difference_le {ι : Type*} [Fintype ι] [DecidableEq ι]
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (X Y : Ω → EuclideanSpace ℝ ι) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (SX SY : Matrix ι ι ℝ)
    (hXc : ∀ i, ∫ ω, X ω i ∂P = 0) (hYc : ∀ i, ∫ ω, Y ω i ∂P = 0)
    (hXcov : ∀ i j, ∫ ω, X ω i * X ω j ∂P = SX i j)
    (hYcov : ∀ i j, ∫ ω, Y ω i * Y ω j ∂P = SY i j) (hindep : IndepFun X Y P)
    (φ : EuclideanSpace ℝ ι → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ) (p : ℕ)
    (hbdd : ∀ k ≤ 2, ∀ z, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ p)
    (B : ℝ) (htrace : ∀ t ∈ Icc (0 : ℝ) 1,
      |(1 / 2 : ℝ) * ∫ ω, ∑ i : ι, ∑ j : ι, (SX i j - SY i j) *
        iteratedFDeriv ℝ 2 φ (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω)
          ![EuclideanSpace.single i 1, EuclideanSpace.single j 1] ∂P| ≤ B) :
    |(∫ ω, φ (X ω) ∂P) - ∫ ω, φ (Y ω) ∂P| ≤ B := by
  change ι → ι → ℝ at SX SY
  let e := Fintype.equivFin ι
  let L := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e
  let X' : Ω → EucSpace (Fintype.card ι) := fun ω => L (X ω)
  let Y' : Ω → EucSpace (Fintype.card ι) := fun ω => L (Y ω)
  let SX' : Matrix (Fin (Fintype.card ι)) (Fin (Fintype.card ι)) ℝ := fun i j => SX (e.symm i) (e.symm j)
  let SY' : Matrix (Fin (Fintype.card ι)) (Fin (Fintype.card ι)) ℝ := fun i j => SY (e.symm i) (e.symm j)
  have hc (z : EuclideanSpace ℝ ι) (i : Fin (Fintype.card ι)) : L z i = z (e.symm i) := rfl
  have hgrowth : ∀ k ≤ 2, ∀ z : EucSpace (Fintype.card ι),
      ‖iteratedFDeriv ℝ k (φ ∘ L.symm) z‖ ≤ C * (1 + ‖z‖) ^ p := by
    intro k hk z
    rw [L.symm.norm_iteratedFDeriv_comp_right, ← L.symm.norm_map z]
    exact hbdd k hk (L.symm z)
  have hblend (t : ℝ) (ω : Ω) :
      L.symm (gaussInterp X' Y' t ω) = Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω := by
    simp [gaussInterp, X', Y']
  have htr (t : ℝ) :
      (∫ ω, gaussianCovarianceTrace (φ ∘ L.symm) (SX' - SY') (gaussInterp X' Y' t ω) ∂P) =
        ∫ ω, ∑ i : ι, ∑ j : ι, (SX i j - SY i j) *
          iteratedFDeriv ℝ 2 φ (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω)
            ![EuclideanSpace.single i 1, EuclideanSpace.single j 1] ∂P := by
    apply integral_congr_ae
    apply ae_of_all
    intro ω
    change (∑ i : Fin (Fintype.card ι), ∑ j : Fin (Fintype.card ι),
      (SX (e.symm i) (e.symm j) - SY (e.symm i) (e.symm j)) *
      iteratedFDeriv ℝ 2 (φ ∘ L.symm) (gaussInterp X' Y' t ω)
        ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]) =
      ∑ i : ι, ∑ j : ι, (SX i j - SY i j) *
        iteratedFDeriv ℝ 2 φ (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω)
          ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]
    rw [← hblend t ω]
    simpa only [L, LinearIsometryEquiv.piLpCongrLeft_symm] using
      gaussian_covariance_trace_relabel e.symm φ hφ (fun i j => SX i j - SY i j) (gaussInterp X' Y' t ω)
  have h := gaussian_interp_expectation_difference_le P X' Y'
    (hX.map (L : EuclideanSpace ℝ ι →L[ℝ] EucSpace (Fintype.card ι)))
    (hY.map (L : EuclideanSpace ℝ ι →L[ℝ] EucSpace (Fintype.card ι))) SX' SY'
    (fun i => by simpa only [X', hc] using hXc (e.symm i))
    (fun i => by simpa only [Y', hc] using hYc (e.symm i))
    (fun i j => by simpa only [X', hc] using hXcov (e.symm i) (e.symm j))
    (fun i j => by simpa only [Y', hc] using hYcov (e.symm i) (e.symm j))
    (hindep.comp L.continuous.measurable L.continuous.measurable)
    (φ ∘ L.symm) (hφ.comp L.symm.contDiff) C p hgrowth B
    (fun t ht => by rw [htr]; exact htrace t ht)
  simpa only [Function.comp_def, X', Y', L.symm_apply_apply] using h

/-- Actual independent zero Gaussian vectors, a nonconstant unbounded
quadratic and its nonzero Hessian satisfy every original moment,
regularity, polynomial-growth and actual trace-bound hypothesis.
The finite coordinate types also admit the identity relabeling. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (fun _ : Unit => (0 : EucSpace 2)) (Measure.dirac ()) ∧
    IndepFun (fun _ : Unit => (0 : EucSpace 2)) (fun _ : Unit => (0 : EucSpace 2)) (Measure.dirac ()) ∧
    (∀ i : Fin 2, ∫ _ω : Unit, (0 : EucSpace 2) i ∂Measure.dirac () = 0) ∧
    (∀ i j : Fin 2, ∫ _ω : Unit, (0 : EucSpace 2) i * (0 : EucSpace 2) j ∂Measure.dirac () =
      (0 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧ ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C * (1 + ‖z‖) ^ 2) ∧
    (∀ r : ℝ, φ (r • EuclideanSpace.single (0 : Fin 2) 1) = r ^ 2) ∧
    iteratedFDeriv ℝ 2 φ 0 ![EuclideanSpace.single (0 : Fin 2) 1, EuclideanSpace.single (0 : Fin 2) 1] = 2 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, |(1 / 2 : ℝ) * ∫ _ω : Unit,
      ∑ i : Fin 2, ∑ j : Fin 2,
        ((0 : Matrix (Fin 2) (Fin 2) ℝ) i j - (0 : Matrix (Fin 2) (Fin 2) ℝ) i j) *
        iteratedFDeriv ℝ 2 φ (Real.sqrt t • (0 : EucSpace 2) + Real.sqrt (1 - t) • 0)
          ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]
        ∂Measure.dirac ()| ≤ 0) := by
  obtain ⟨φ, C, hφ, hbdd, hline, hhess⟩ := gaussian_interp_polynomial_test_witness
  refine ⟨φ, C, ?_, indepFun_const_left _ _, fun i => by simp, fun i j => by simp,
    hφ, hbdd, hline, hhess, fun t ht => ?_⟩
  · refine ⟨measurable_const.aemeasurable, ?_⟩
    rw [Measure.map_const]
    simp
    infer_instance
  · simp

end Transformer.Homogenized
