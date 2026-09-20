/-
# Homogenized Transformers — the softmax barycenter at low temperature

Formalization of `lemma:Laplace_method` and `lem:delta_method` of
arXiv:2604.01978v1, *Homogenized Transformers*: the two estimates that make the
drift of the overlap collapse to the logistic one, and so carry the proof of
`thm:large_beta_meta`.

Both are statements about the softmax barycenter

  `m_{β,A}[μ](x) = 𝒵_A[μ](x)⁻¹ ∫ e^{β⟨A x, y⟩} y μ(dy)`,

which is the attention field `EQ:VELOCITY_FIELD_SELF_ATTENTION` with the value
matrix removed; `lem:lemma_app` is what puts it back, `ξ_θ[μ] = V m_{β,A}[μ]`.

The first says that at large `β` the barycenter collapses onto the unit vector
`A x / ‖A x‖` — the Laplace method on the sphere — and the second expands the
resulting `E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩` in `1/d` by the delta method.  Together they
are `eq:kernel_expansion`, which is a line of the proof of
`thm:large_beta_meta` and is not restated here.
-/

import Transformer.Homogenized.Metastability

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The softmax barycenter -/

/-- The attention weight of `eq: mbetaA`, `e^{β⟨A x, y⟩}`, as a function of the
query-key matrix alone. -/
noncomputable def softWeight {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x y : EucSpace d) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (Matrix.toEuclideanLin A x) y)

theorem softWeight_pos {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x y : EucSpace d) : 0 < softWeight β A x y :=
  Real.exp_pos _

/-- The weight of `eq: mbetaA` is the attention weight of the head: the value
matrix never enters it. -/
theorem softWeight_eq_attnWeight {d : ℕ} (β : ℝ) (θ : HeadParam d) (x y : EucSpace d) :
    softWeight β θ.2 x y = attnWeight β θ x y := rfl

/-- The **softmax barycenter** `m_{β,A}[μ](x)` of `eq: mbetaA`,

  `m_{β,A}[μ](x) = 𝒵_A[μ](x)⁻¹ ∫ e^{β⟨A x, y⟩} y μ(dy)`.

Source: arXiv:2604.01978v1, `eq: mbetaA`. -/
noncomputable def softBary {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (μ : Measure (EucSpace d)) (x : EucSpace d) : EucSpace d :=
  (∫ y, softWeight β A x y ∂μ)⁻¹ • ∫ y, softWeight β A x y • y ∂μ

/-- At a Dirac mass the barycenter is the atom, at every temperature: the
normalizer cancels the weight. -/
@[simp] theorem softBary_dirac {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (z x : EucSpace d) : softBary β A (Measure.dirac z) x = z := by
  rw [softBary, integral_dirac, integral_dirac, smul_smul,
    inv_mul_cancel₀ (softWeight_pos β A x z).ne', one_smul]

/-- **`m_{β,A}[μ](x)` is a convex combination of unit vectors**, so it lies in
the closed unit ball.  This is the source's parenthetical remark in the proof
of `lem:Ito_formula`, and it is what makes `s_μ(x,y) = E⟨m(x),m(y)⟩` an
integral of a bounded function rather than a junk value.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem norm_softBary_le_one {d : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (μ : Measure (EucSpace d)) [IsProbabilityMeasure μ] (x : EucSpace d)
    (hsph : ∀ᵐ y ∂μ, ‖y‖ = 1)
    (hint : Integrable (fun y => softWeight β A x y) μ) :
    ‖softBary β A μ x‖ ≤ 1 := by
  have hZpos : 0 < ∫ y, softWeight β A x y ∂μ := by
    have hsupp : Function.support (fun y => softWeight β A x y) = Set.univ := by
      ext y
      simp [Function.mem_support, (softWeight_pos β A x y).ne']
    rw [integral_pos_iff_support_of_nonneg (fun y => (softWeight_pos β A x y).le) hint,
      hsupp, measure_univ]
    exact zero_lt_one
  have hbound : ∀ᵐ y ∂μ, ‖softWeight β A x y • y‖ ≤ softWeight β A x y := by
    filter_upwards [hsph] with y hy
    rw [norm_smul, hy, mul_one, Real.norm_eq_abs, abs_of_pos (softWeight_pos β A x y)]
  have hnum : ‖∫ y, softWeight β A x y • y ∂μ‖ ≤ ∫ y, softWeight β A x y ∂μ := by
    refine le_trans (norm_integral_le_integral_norm _) ?_
    exact integral_mono_of_nonneg (Filter.Eventually.of_forall fun y => norm_nonneg _)
      hint hbound
  rw [softBary, norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hZpos, inv_mul_eq_div,
    div_le_one hZpos]
  exact hnum

/-- **`s_μ(x,x')` of `eq:s_mu_defs_clean`**, the correlation of two softmax
barycenters averaged over the head law:

  `s_μ(x,x') = E_A⟨m_{β,A}[μ](x), m_{β,A}[μ](x')⟩`.

The source's other statistic `s_μ(x) = E‖m_{β,A}[μ](x)‖²` is this one at
`x' = x`, by `baryCorr_self`; there is one object, not two.

Source: arXiv:2604.01978v1, `eq:s_mu_defs_clean`. -/
noncomputable def baryCorr {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x y : EucSpace d) : ℝ :=
  ∫ θ, inner (𝕜 := ℝ) (softBary β θ.2 μ x) (softBary β θ.2 μ y) ∂ρ

/-- `s_μ(x) = s_μ(x,x)`. -/
theorem baryCorr_self {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x : EucSpace d) :
    baryCorr β ρ μ x x = ∫ θ, ‖softBary β θ.2 μ x‖ ^ 2 ∂ρ := by
  simp only [baryCorr, real_inner_self_eq_norm_sq]

/-- `s_μ` is symmetric. -/
theorem baryCorr_comm {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (x y : EucSpace d) :
    baryCorr β ρ μ x y = baryCorr β ρ μ y x := by
  simp only [baryCorr]
  exact integral_congr_ae (Filter.Eventually.of_forall fun θ => real_inner_comm _ _)

/-- The normalization `v ↦ v/‖v‖` is invariant under a positive rescaling.
This is why `lemma:Laplace_method` and `lem:delta_method` compose although the
source writes the first at `A = W W'ᵀ` and the second at
`Â = (dσ_A²)⁻¹ Wᵀ W'`: the quantity `E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩` they share does not
see the constant. -/
theorem normalizeLayer_smul {d : ℕ} {c : ℝ} (hc : 0 < c) (v : EucSpace d) :
    normalizeLayer (c • v) = normalizeLayer v := by
  rw [normalizeLayer, normalizeLayer, norm_smul, Real.norm_eq_abs, abs_of_pos hc,
    mul_inv, smul_smul]
  congr 1
  field_simp

/-! ### The two estimates -/

/-- **Lemma (lemma:Laplace_method).**  For `μ` as in `ass:low-temperature`,

  `E⟨m_{β,A}[μ](x), m_{β,A}[μ](y)⟩ = E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩ + O(d^{-3/2} + β^{-1/2})`

for all `x, y ∈ 𝕊^{d-1}`.

**What the source says and what is written here.**

* The `β` of the error is again the temperature of the normalized query-key
  matrix, as in `large_beta_metastability`: the proof's Laplace expansion is in
  `λ_x = β‖A x‖`, and at `A = W W'ᵀ` of `eq: tformers.at.initialization` the
  norm `‖A x‖` is of order `d σ_A²`, so `λ_x ≍ β d σ_A² = effBeta`.  Writing
  `β^{-1/2}` here would be the statement at `‖A x‖ ≍ 1`, which is not the
  ensemble the lemma is applied to.
* The lemma is stated at a Gaussian head law and not at an arbitrary one: the
  proof discards the event `{‖A x‖ < ε}` by a Gaussian concentration
  inequality, and that is where `e^{-c(ε)d} = O(d^{-3/2})` comes from.  At an
  arbitrary law the claim is false — `ρ* = δ_0` makes the right-hand side a
  junk value while the left-hand side is `⟨m_{β,0}[μ](x), m_{β,0}[μ](y)⟩`.
* `ass:low-temperature` is an assumption on a *path* `t ↦ μ(t)`; the lemma uses
  it at one time, so it is instantiated at the constant path.
* `ρ_min`, `ρ_max`, `L` come before `C`, which is how the source's "all
  constants here depend only on bounds on `ρ` and `∇ρ`" is said.

Not proved here.

Source: arXiv:2604.01978v1, `lemma:Laplace_method`. -/
theorem laplace_method (ρmin ρmax L : ℝ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d : ℕ), 0 < d → ∀ β : ℝ, 0 < β → ∀ σV σA : ℝ≥0, 0 < σA →
      ∀ ρ : Measure (HeadParam d), IsGaussianHeadLaw d σV σA ρ →
      ∀ (σ μ : Measure (EucSpace d)) (dens : EucSpace d → ℝ),
        IsLowTemperature d σ (fun _ => μ) (fun _ => dens) ρmin ρmax L →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        |baryCorr β ρ μ x y
            - ∫ θ, inner (𝕜 := ℝ)
                (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂ρ|
          ≤ C * ((d : ℝ) ^ (-(3 : ℝ) / 2) + effBeta d β σA ^ (-(1 : ℝ) / 2)) := by
  sorry

/-- **Lemma (lem:delta_method).**  For `A = (dσ_A²)⁻¹ Wᵀ W'`,

  `E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩ = ⟨x,y⟩ + (⟨x,y⟩³ - ⟨x,y⟩)/(2d) + O(d^{-3/2})`.

**What the source says and what is written here.**  The statement is written
at the head law's own `A = W W'ᵀ` of `eq: tformers.at.initialization` rather
than at the source's `Â`.  The two differ by the positive constant `(dσ_A²)⁻¹`,
which `normalizeLayer_smul` shows the left-hand side does not see, and by
transposing both factors, which is a relabeling of independent identically
distributed entries and so does not change the law of `A`.  Stating it at `Â`
would therefore be the same lemma written in a matrix the rest of the
development never forms.

Not proved here.

Source: arXiv:2604.01978v1, `lem:delta_method`. -/
theorem delta_method :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d : ℕ), 0 < d → ∀ σV σA : ℝ≥0, 0 < σA →
      ∀ ρ : Measure (HeadParam d), IsGaussianHeadLaw d σV σA ρ →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        |(∫ θ, inner (𝕜 := ℝ)
              (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂ρ)
            - (inner (𝕜 := ℝ) x y
                + ((inner (𝕜 := ℝ) x y : ℝ) ^ 3 - inner (𝕜 := ℝ) x y) / (2 * (d : ℝ)))|
          ≤ C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  sorry

/-- The hypotheses of `laplace_method` are satisfiable, at the constants
`ρ_min = ρ_max = 1`, `L = 0` its statement is quantified after: dimension `1`,
temperature `1`, the Gaussian head law at `σ_V² = 1/d`, and the uniform measure
with density `1`, tested at the unit vector against itself. -/
example :
    (0 : ℕ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ≥0) < 1 ∧
      IsGaussianHeadLaw 1 (stdSigmaV 1) 1 (gaussHeadLaw 1 (stdSigmaV 1) 1) ∧
      IsLowTemperature 1 (uniformAmbient 1) (fun _ => uniformAmbient 1)
        (fun _ => (fun _ => 1)) 1 1 0 ∧
      ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 :=
  ⟨one_pos, one_pos, one_pos, isGaussianHeadLaw_gaussHeadLaw 1 _ _,
    isLowTemperature_uniformAmbient one_pos, by simp [PiLp.norm_single]⟩

/-- The hypotheses of `delta_method` are satisfiable: the same Gaussian head
law, with no assumption on the token measure. -/
example :
    (0 : ℕ) < 1 ∧ (0 : ℝ≥0) < 1 ∧
      IsGaussianHeadLaw 1 (stdSigmaV 1) 1 (gaussHeadLaw 1 (stdSigmaV 1) 1) ∧
      ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 :=
  ⟨one_pos, one_pos, isGaussianHeadLaw_gaussHeadLaw 1 _ _, by simp [PiLp.norm_single]⟩

end Homogenized
end Transformer
