import Transformer.Modes.Section3_ScaledSumMoments
import Transformer.Modes.Section3_CumulantMoment

/-
# The number of modes of a Gaussian KDE — the moment identity for the density
of the normalized sum

The second equality of the moment identity of arXiv:2412.09080v3, §3.1, is

  `𝔼_{Z ~ N(0,I₂)}[(q_t/φ)(Z) H^α(Z)] = ∫ q_t H^α = n^{-1/2} κ_t^α`,

`q_t` being the density of `S_n = n^{-1/2} Σ Yᵢ(t)`.  Three steps, each proved
elsewhere in this directory, meet here:

* `∫ q H^α dx = 𝔼 H^α(S_n)`, since `q` is a density — the only place
  `IsDensityOf` is used, and the reason it asks for measurability;
* `𝔼 H^α(S_n) = 𝔼 (S_n)₁^k (S_n)₂^{3-k}` — `Section3_CumulantMoment.lean`,
  which needs only that the law of `S_n` is centred, proved here;
* `𝔼 (S_n)₁^k (S_n)₂^{3-k} = n^{-1/2} 𝔼 Y₁^k Y₂^{3-k}` —
  `Section3_ScaledSumMoments.lean`, the scaling law.

**The source omits the factor `n^{-1/2}`.**  It writes `κ_t^α = 𝔼[H^α(Y(t))] =
𝔼[(q_t/φ)(Z) H^α(Z)]`, but the last expectation is the third cumulant of the
*normalized sum*, and cumulants are additive over independent summands and
homogeneous of degree three, so it is `n^{-1/2} κ_t^α`.

The slip is in that one line, and the source's own Edgeworth display contradicts
it: from `q_t/φ = 1 + n^{-1/2} Σ_{|α|=3} κ_t^α/α! H^α + r`, multiplying by
`φ H^β` and integrating with `∫ φ H^α H^β = α! δ_{αβ}` and `∫ φ H^β = 0` gives
`∫ q_t H^β = n^{-1/2} κ_t^β`, the factor and not its absence.  The theorem below
carries it; `cumulant_three_eq_integral_hermite` carries the first equality as
the source writes it.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`.
-/

open Real MeasureTheory

namespace Transformer
namespace Modes

variable {μ : Measure (ℝ × ℝ)} {ε : ℝ} {n : ℕ}

/-- **The law of a normalized sum of centred summands is centred**, first
coordinate.  arXiv:2412.09080v3, §3 ("by construction `q_t` has mean `0`"). -/
theorem integral_map_scaledSum_fst [IsProbabilityMeasure μ] (hE : HasExpMomentsOn μ ε)
    (hε : 0 < ε) (hfst : ∫ x, x.1 ∂μ = 0) (n : ℕ) :
    ∫ x, x.1 ∂((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) = 0 := by
  have hfi : Integrable (fun x : ℝ × ℝ => x.1) μ := by
    simpa using integrable_pow_mul_pow hE hε 1 0
  rw [integral_map (measurable_scaledSum n).aemeasurable measurable_fst.aestronglyMeasurable]
  have hL : ∀ X : Fin n → ℝ × ℝ, (scaledSum n X).1 = ∑ i, (Real.sqrt n)⁻¹ * (X i).1 := fun X => by
    simpa using scaledSum_dot n 1 0 X
  have hterm : ∀ i : Fin n, ∫ X : Fin n → ℝ × ℝ, (Real.sqrt n)⁻¹ * (X i).1
      ∂(Measure.pi fun _ : Fin n => μ) = 0 := by
    intro i
    have h : ∫ X : Fin n → ℝ × ℝ, (X i).1 ∂(Measure.pi fun _ : Fin n => μ) = ∫ x, x.1 ∂μ :=
      integral_eval_pi hfi n i
    rw [integral_const_mul, h, hfst, mul_zero]
  simp only [hL]
  rw [integral_finsetSum _ fun i _ => (integrable_eval_pi hfi n i).const_mul _]
  simp [hterm]

/-- **The law of a normalized sum of centred summands is centred**, second
coordinate.  arXiv:2412.09080v3, §3 ("by construction `q_t` has mean `0`"). -/
theorem integral_map_scaledSum_snd [IsProbabilityMeasure μ] (hE : HasExpMomentsOn μ ε)
    (hε : 0 < ε) (hsnd : ∫ x, x.2 ∂μ = 0) (n : ℕ) :
    ∫ x, x.2 ∂((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) = 0 := by
  have hsi : Integrable (fun x : ℝ × ℝ => x.2) μ := by
    simpa using integrable_pow_mul_pow hE hε 0 1
  rw [integral_map (measurable_scaledSum n).aemeasurable measurable_snd.aestronglyMeasurable]
  have hL : ∀ X : Fin n → ℝ × ℝ, (scaledSum n X).2 = ∑ i, (Real.sqrt n)⁻¹ * (X i).2 := fun X => by
    simpa using scaledSum_dot n 0 1 X
  have hterm : ∀ i : Fin n, ∫ X : Fin n → ℝ × ℝ, (Real.sqrt n)⁻¹ * (X i).2
      ∂(Measure.pi fun _ : Fin n => μ) = 0 := by
    intro i
    have h : ∫ X : Fin n → ℝ × ℝ, (X i).2 ∂(Measure.pi fun _ : Fin n => μ) = ∫ x, x.2 ∂μ :=
      integral_eval_pi hsi n i
    rw [integral_const_mul, h, hsnd, mul_zero]
  simp only [hL]
  rw [integral_finsetSum _ fun i _ => (integrable_eval_pi hsi n i).const_mul _]
  simp [hterm]

/-- The hypotheses of `integral_map_scaledSum_fst` and `integral_map_scaledSum_snd`
are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ (0 : ℝ) < 1 ∧ ∫ x, x.1 ∂stdGauss2 = 0 ∧
    ∫ x, x.2 ∂stdGauss2 = 0 :=
  ⟨hasExpMomentsOn_stdGauss2, one_pos, isStandardized_stdGauss2.mean_fst,
    isStandardized_stdGauss2.mean_snd⟩

/-- **Integrating against a density is taking an expectation**:
`∫ q G dx = 𝔼 G(S)` when `q` is a density of the law of `S`.

Source: arXiv:2412.09080v3, §3, the display after `eq:psi`. -/
theorem integral_density_mul {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {S : Ω → ℝ × ℝ}
    {q : ℝ × ℝ → ℝ} (hq : IsDensityOf P S q) (G : ℝ × ℝ → ℝ) :
    ∫ x, q x * G x = ∫ x, G x ∂(P.map S) := by
  rw [hq.map_eq, integral_withDensity_eq_integral_toReal_smul hq.measurable.ennreal_ofReal
    (Filter.Eventually.of_forall fun x => ENNReal.ofReal_lt_top)]
  refine integral_congr_ae (Filter.Eventually.of_forall fun x => ?_)
  show q x * G x = (ENNReal.ofReal (q x)).toReal • G x
  rw [smul_eq_mul, ENNReal.toReal_ofReal (hq.nonneg x)]

/-- The hypothesis of `integral_density_mul` is satisfiable. -/
example : IsDensityOf (Measure.pi fun _ : Fin 1 => stdGauss2) (scaledSum 1) phi2 :=
  isDensityOf_stdGauss2

/-- **The second equality of the moment identity, corrected**: if `q` is the
density of `n^{-1/2} Σ Xᵢ` for `X₁, …, Xₙ` i.i.d. of law `μ`, then
`𝔼_{Z ~ N(0,I₂)}[(q/φ)(Z) H^α(Z)] = ∫ q H^α = n^{-1/2} κ^α` for `|α| = 3`.

arXiv:2412.09080v3, §3.1, the display after `eq:psi`, states it without the
factor `n^{-1/2}`; see the module docstring. -/
theorem integral_density_mul_hermite (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hμ : IsStandardized μ) (hexp : HasExpMoments μ) (n : ℕ) (hn : 1 ≤ n)
    (q : ℝ × ℝ → ℝ) (hq : IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q)
    (k : ℕ) (hk : k ≤ 3) :
    ∫ x, q x * hermite3 k x = (Real.sqrt n)⁻¹ * cumulantOf μ k (3 - k) := by
  obtain ⟨ε, hε, hE⟩ := hexp
  rw [integral_density_mul hq,
    integral_hermite3_eq_of_centred (hasExpMomentsOn_map_scaledSum hE hn) hε
      (integral_map_scaledSum_fst hE hε hμ.mean_fst n)
      (integral_map_scaledSum_snd hE hε hμ.mean_snd n) k hk,
    integral_map_scaledSum_monomial hE hε hμ.mean_fst hμ.mean_snd hn k hk,
    ← integral_hermite3_eq hE hε hμ k hk,
    ← cumulant_three_eq_integral_hermite μ hμ ⟨ε, hε, hE⟩ k hk]

/-- The hypotheses of `integral_density_mul_hermite` are satisfiable. -/
example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ 1 ≤ 1 ∧
    IsDensityOf (Measure.pi fun _ : Fin 1 => stdGauss2) (scaledSum 1) phi2 ∧ (0 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2, le_rfl,
    isDensityOf_stdGauss2, by norm_num⟩

end Modes
end Transformer
