/-
# Homogenized Transformers — the barycenter at low temperature

The dimension-uniform version of `lemma:Laplace_method` of
arXiv:2604.01978v1, *Homogenized Transformers*, is false. The actual
uniform spherical law and actual original Gaussian product head at entry
variance `1/d` violate its claimed rate beyond arbitrary dimension and
temperature thresholds. `laplace_method_false` proves this with all the
original density, initialization and unit-query conditions retained.

The estimate `lem:delta_method`, which expands normalized-query
correlation in `1/d`, remains separately stated and unproved below.
The paper combines these two estimates in `eq:kernel_expansion`, in the
proof of `thm:large_beta_meta`; no such combination is asserted here.
The counterexample concerns the uniform simultaneous-dimension estimate
required by that argument. Exact dimension-dependent barycenter bounds
and the actual Gaussian query moments are proved in the imported modules.
-/

import Transformer.Homogenized.Section7_UniformLaplaceCounterexample
import Transformer.Homogenized.Section7_DeltaNormCovariance
import Transformer.Homogenized.Section7_GaussianFactorDelta
import Transformer.Homogenized.Section7_CorrelationDeviation
import Transformer.Homogenized.Metastability

open scoped BigOperators ENNReal NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The two estimates -/

/-- **Counterexample to the dimension-uniform `lemma:Laplace_method`.**
The source asserts, for `μ` as in `ass:low-temperature`,

  `E⟨m_{β,A}[μ](x), m_{β,A}[μ](y)⟩ = E⟨Ax/‖Ax‖, Ay/‖Ay‖⟩ + O(d^{-3/2} + β^{-1/2})`

for all original unit queries, with constants depending only on the
low-temperature density and gradient bounds.

Fix those bounds to `(ρ_min, ρ_max, L) = (1,1,0)`. The exact uniform
token law satisfies them jointly with the Gaussian initialization (G),
standard value variance `1/d`, and common query/key entry variance `1/d`.
At that scaling `effBeta = β`, so the counterexample also refutes the
former effective-temperature version of this formal statement.

For every prefactor `C`, choose `β=s²`, cutoff `R=s⁴` and dimension
larger than `16 exp(2s⁴)`. The proved actual-query second moment and
uniform barycenter bound give `E‖m‖² < 1/4`, while
`E‖Ax/‖Ax‖‖² = 1`. The original self-correlation error is therefore
greater than `3/4`, and the proposed bound can be made less than `1/4`.
Both temperature and dimension exceed arbitrary prescribed thresholds.

The theorem below explicitly permits such thresholds, as a uniform
Big-O assertion does. Their availability does not restore the claim.
No zero head, zero query or failing density hypothesis is used.
The original matrix remains the product of two independent Gaussian
factors; each coordinate is treated by its actual bilinear row law.
The uniform law is a probability measure carried by the original sphere,
and the low-temperature assumption is instantiated at its constant path.

The source's Step 1 includes `O(d ‖w‖²/λ_x)` before dropping dimension
from the final error. The refutation addresses this simultaneous-limit
claim; the fixed-dimension Laplace asymptotic has a different uniformity.

Source: arXiv:2604.01978v1, §2.3, initialization (G), §2.3.3,
`ass:low-temperature`, and §7.3, `lemma:Laplace_method`, Step 1. -/
theorem laplace_method_false :
    ¬ ∃ (C B : ℝ) (D : ℕ), 0 < C ∧ 0 ≤ B ∧
      ∀ (d : ℕ), 0 < d → D ≤ d → ∀ β : ℝ, 0 < β → B ≤ β →
      ∀ σV σA : ℝ≥0, 0 < σA →
      ∀ ρ : Measure (HeadParam d), IsGaussianHeadLaw d σV σA ρ →
      ∀ (σ μ : Measure (EucSpace d)) (dens : EucSpace d → ℝ),
        IsLowTemperature d σ (fun _ => μ) (fun _ => dens) 1 1 0 →
      ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
        |baryCorr β ρ μ x y
            - ∫ θ, inner (𝕜 := ℝ)
                (normalizeLayer (qkMap θ x)) (normalizeLayer (qkMap θ y)) ∂ρ|
          ≤ C * ((d : ℝ) ^ (-(3 : ℝ) / 2) + effBeta d β σA ^ (-(1 : ℝ) / 2)) := by
  rintro ⟨C, B, D, hC, hB, h⟩
  obtain ⟨d, hd, hDd, β, ν, x, hβ, hBβ, hx, -, hlow, hG, hbad⟩ :=
    uniform_laplace_self_error_counterexample C B D hC hB
  have hdpos : 0 < d := by omega
  have hb := h d hdpos hDd β hβ hBβ (stdSigmaV d) (stdSigmaV d)
    (stdSigmaV_pos hdpos)
    (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)) hG
    (ν.map (Subtype.val : SSphere d → EucSpace d)) (ν.map Subtype.val) (fun _ => 1)
    hlow x x hx hx
  rw [effBeta_stdSigmaV hdpos] at hb
  exact not_lt_of_ge hb hbad

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

The source's displayed covariance used in its proof is false:
`not_gaussian_delta_norm_covariance` refutes its `2/d + O(d⁻²)`
diagonal under the actual original Gaussian product law. The proved
exact normalized squared-query-norm variance is `4/d + 4/d²`.
The normalized-correlation expansion below remains unproved; this
covariance discrepancy alone does not establish its corrected remainder.

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

/-- The original hypotheses used by the refutation hold jointly at the
common Gaussian scale in dimension two, actual uniform density one and
an actual coordinate unit query. The full counterexample theorem supplies
such data beyond arbitrary dimension and temperature thresholds too.
The entry variance is exactly `1/d`, effective temperature is the actual
temperature, and the original normalized-query self-correlation is one. -/
example :
    (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ≥0) < stdSigmaV 2 ∧
      IsGaussianHeadLaw 2 (stdSigmaV 2) (stdSigmaV 2)
        (gaussHeadLaw 2 (stdSigmaV 2) (stdSigmaV 2)) ∧
      IsLowTemperature 2 (uniformAmbient 2) (fun _ => uniformAmbient 2)
        (fun _ => (fun _ => 1)) 1 1 0 ∧
      ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
      (stdSigmaV 2 : ℝ) ^ 2 = 1 / 2 ∧ effBeta 2 1 (stdSigmaV 2) = 1 ∧
      (∫ θ : HeadParam 2, inner ℝ
        (normalizeLayer (qkMap θ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))))
        (normalizeLayer (qkMap θ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ))))
        ∂gaussHeadLaw 2 (stdSigmaV 2) (stdSigmaV 2)) = 1 :=
  ⟨by decide, one_pos, stdSigmaV_pos (by decide), isGaussianHeadLaw_gaussHeadLaw 2 _ _,
    isLowTemperature_uniformAmbient (by decide), by simp [PiLp.norm_single],
    by simp, effBeta_stdSigmaV (by decide) 1,
    integral_scaled_gaussianHead_normalized_query_self _ (stdSigmaV_pos (by decide)) 0⟩

/-- The hypotheses of `delta_method` are satisfiable: the same Gaussian head
law, with no assumption on the token measure. -/
example :
    (0 : ℕ) < 1 ∧ (0 : ℝ≥0) < 1 ∧
      IsGaussianHeadLaw 1 (stdSigmaV 1) 1 (gaussHeadLaw 1 (stdSigmaV 1) 1) ∧
      ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 :=
  ⟨one_pos, one_pos, isGaussianHeadLaw_gaussHeadLaw 1 _ _, by simp [PiLp.norm_single]⟩

end Homogenized
end Transformer
