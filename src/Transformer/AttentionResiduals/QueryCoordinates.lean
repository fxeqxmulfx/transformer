module

public import Transformer.AttentionResiduals.Normalized
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.Calculus.FDeriv.Add

/-!
# Exact query coordinates for normalized depth attention

Source: `review/residual_connections.md`, *Query coordinate conditioning*,
QC1--QC5. This is a correction to the proposed Q/K variant, not an equation
attributed to the key-only AttnRes paper. Physical query `q = s • u` uses
positive coordinate scale `s`. Query RMS epsilon must become `eps / s²`;
simply renormalizing the stored query changes the finite-epsilon function.
The coordinate scale is metadata, not a learned normalization gain or bias.
Keys keep their original epsilon. The chain rule gives ordinary coordinate
derivatives, without replacing or detaching the normalization backward.
These are real identities; floating-point preservation is tested separately.
-/

@[expose] public section
noncomputable section

namespace Transformer.AttentionResiduals

open GPTMini

/-- QC1: scaling both the input and its RMS stabilizer preserves normalization.
The scale is strictly positive; zero vectors and zero-dimensional spaces remain
included. No approximation discards the stabilizer. -/
theorem rmsNormEps_scaled {d : ℕ} (eps scale : ℝ) (hs : 0 < scale)
    (value : EucSpace d) :
    rmsNormEps (scale ^ 2 * eps) (scale • value) = rmsNormEps eps value := by
  have hrad : ‖scale • value‖ ^ 2 + (d : ℝ) * (scale ^ 2 * eps) =
      scale ^ 2 * (‖value‖ ^ 2 + (d : ℝ) * eps) := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos hs]
    ring
  unfold rmsNormEps
  rw [hrad, Real.sqrt_mul (sq_nonneg scale), Real.sqrt_sq_eq_abs, abs_of_pos hs,
    smul_smul]
  congr 1
  rw [mul_comm scale, div_mul_eq_div_div, div_mul_cancel₀ _ hs.ne']

/-- QC2: coordinate query `u` represents `scale • u` with a correspondingly
smaller RMS epsilon. Scaling only the query, with fixed epsilon, is different. -/
theorem rmsNormEps_coordinates {d : ℕ} (eps scale : ℝ) (hs : 0 < scale)
    (coordinate : EucSpace d) :
    rmsNormEps (eps / scale ^ 2) coordinate =
      rmsNormEps eps (scale • coordinate) := by
  have h := rmsNormEps_scaled (eps / scale ^ 2) scale hs coordinate
  have hcancel : scale ^ 2 * (eps / scale ^ 2) = eps := by
    rw [mul_comm, div_mul_cancel₀ _ (pow_ne_zero 2 hs.ne')]
  rw [hcancel] at h
  exact h.symm

/-- QC3: source Q/K scoring in a positive query coordinate frame. The key
normalization and score divisor retain their original meanings. -/
def coordinateDepthScore {d : ℕ} (eps scale : ℝ)
    (coordinate key : EucSpace d) : ℝ :=
  inner ℝ (rmsNormEps (eps / scale ^ 2) coordinate) (rmsNormEps eps key) /
    Real.sqrt (d : ℝ)

/-- QC3: the coordinate scorer evaluates the same physical query as the
original finite-epsilon Q/K depth-attention definition. -/
theorem coordinateDepthScore_eq {d : ℕ} (eps scale : ℝ) (hs : 0 < scale)
    (coordinate key : EucSpace d) :
    coordinateDepthScore eps scale coordinate key =
      qkDepthScore eps (scale • coordinate) key := by
  unfold coordinateDepthScore qkDepthScore
  rw [rmsNormEps_coordinates eps scale hs coordinate]

/-- QC3: changing coordinates divides the query and multiplies its scale.
Every score is preserved; hence the same softmax uses the same raw values. -/
theorem coordinateDepthScore_rechart {d : ℕ} (eps scale factor : ℝ)
    (hs : 0 < scale) (hc : 0 < factor) (coordinate key : EucSpace d) :
    coordinateDepthScore eps (scale * factor) (factor⁻¹ • coordinate) key =
      coordinateDepthScore eps scale coordinate key := by
  rw [coordinateDepthScore_eq eps _ (mul_pos hs hc),
    coordinateDepthScore_eq eps scale hs]
  congr 1
  rw [smul_smul]
  have h : scale * factor * factor⁻¹ = scale := by
    field_simp
  rw [h]

/-- QC4: the original normalized score bound survives any positive frame.
It is a bound on logits, not a positive lower bound on a loss gradient. -/
theorem coordinateDepthScore_bound {d : ℕ} (hd : 0 < d) (eps scale : ℝ)
    (heps : 0 < eps) (hs : 0 < scale) (coordinate key : EucSpace d) :
    |coordinateDepthScore eps scale coordinate key| ≤ Real.sqrt (d : ℝ) := by
  rw [coordinateDepthScore_eq eps scale hs]
  exact qkDepthScore_bound hd eps heps _ key

/-- QC3: coordinate metadata affects scoring, while aggregation retains the
original raw source vectors. -/
def coordinateAttentionResidual {ι : Type*} [Fintype ι] {d : ℕ}
    (eps scale : ℝ) (coordinate : EucSpace d) (values : ι → EucSpace d) : EucSpace d :=
  attend (fun i => coordinateDepthScore eps scale coordinate (values i)) values

/-- QC3: the full softmax aggregation is invariant under a positive rechart,
including its unnormalized values and arbitrary finite source count. -/
theorem coordinateAttentionResidual_rechart {ι : Type*} [Fintype ι] {d : ℕ}
    (eps scale factor : ℝ) (hs : 0 < scale) (hc : 0 < factor)
    (coordinate : EucSpace d) (values : ι → EucSpace d) :
    coordinateAttentionResidual eps (scale * factor) (factor⁻¹ • coordinate) values =
      coordinateAttentionResidual eps scale coordinate values := by
  unfold coordinateAttentionResidual
  congr 1
  funext i
  exact coordinateDepthScore_rechart eps scale factor hs hc coordinate (values i)

/-- QC5: the derivative of a scalar loss in a fixed linear coordinate frame
is the physical derivative multiplied by scale. The derivative hypothesis is
explicit; this does not assume an optimizer or replace the loss gradient. -/
theorem query_coordinate_hasFDerivAt {d : ℕ} (loss : EucSpace d → ℝ)
    (derivative : EucSpace d →L[ℝ] ℝ) (scale : ℝ) (coordinate : EucSpace d)
    (h : HasFDerivAt loss derivative (scale • coordinate)) :
    HasFDerivAt (fun u => loss (scale • u)) (scale • derivative) coordinate := by
  have hscale := (hasFDerivAt_id (𝕜 := ℝ) coordinate).const_smul scale
  have hcomp := h.comp coordinate hscale
  simpa only [Function.comp_def, ContinuousLinearMap.comp_smul,
    ContinuousLinearMap.comp_id] using hcomp

example := rmsNormEps_scaled (d := 2) (1 / 1000000) 1000000 (by norm_num)
  (WithLp.toLp 2 ![1, 2])

example := rmsNormEps_coordinates (d := 2) (1 / 1000000) 1000000 (by norm_num)
  (WithLp.toLp 2 ![1, 2])

example := coordinateDepthScore_eq (d := 2) (1 / 1000000) 1000000 (by norm_num)
  (WithLp.toLp 2 ![1, 2]) (WithLp.toLp 2 ![3, 4])

example := coordinateDepthScore_rechart (d := 2) (1 / 1000000) 1 1000000
  (by norm_num) (by norm_num) (WithLp.toLp 2 ![1000000, 2000000])
  (WithLp.toLp 2 ![3, 4])

example := coordinateDepthScore_bound (d := 2) (by decide) (1 / 1000000)
  1000000 (by norm_num) (by norm_num) (WithLp.toLp 2 ![1, 2])
  (WithLp.toLp 2 ![3, 4])

example := query_coordinate_hasFDerivAt (d := 2)
  (innerSL ℝ (WithLp.toLp 2 ![3, 4])) (innerSL ℝ (WithLp.toLp 2 ![3, 4]))
  1000000 (WithLp.toLp 2 ![1, 2]) (ContinuousLinearMap.hasFDerivAt _)

example := coordinateAttentionResidual_rechart (d := 2) (1 / 1000000) 1 1000000
  (by norm_num) (by norm_num) (WithLp.toLp 2 ![1000000, 2000000])
  (fun i : Fin 3 => WithLp.toLp 2 ![(i.val : ℝ), 1])

end Transformer.AttentionResiduals
