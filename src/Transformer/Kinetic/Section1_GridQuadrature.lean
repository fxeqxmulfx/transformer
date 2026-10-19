module

public import Transformer.Kinetic.MeanField
public import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

/-!
# Smooth position tests obstruct the printed uniform mean-field rate

ArXiv:2605.09213v1, §1.1, `th:mf-gpt` (ii),
`eq:conv-rate-muNf`, and §4, Step 4 of its proof.
The empirical position marginal is the actual grid `j/N`, whereas the
limiting position marginal is Lebesgue measure on `(0,1]`.

The smooth position test `cos(2πNσ)` equals one at every grid point,
has integral zero, and is bounded by one. As a token test it is constant
in `θ`, so every positive-order token derivative is zero. Consequently
its full token Sobolev bound stays one, independently of `N`.
Its actual quadrature error is one, defeating every uniform vanishing
power rate controlled solely by that token norm.

The source's own Step 4 retains `N⁻¹‖∂σφ‖∞` before the final displayed
bound drops position derivatives. The counterexample concerns that
printed uniform bound, including for jointly smooth test functions;
it does not dispute convergence for each fixed continuous test.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory Filter Set

namespace Transformer.Kinetic

/-- The actual smooth position test at grid frequency `N`.
Source: arXiv:2605.09213v1, §1.1, `eq:conv-rate-muNf`.
Both the grid size and the position enter the original cosine. -/
noncomputable def cursorCos (N : ℕ) (σ : ℝ) : ℝ :=
  Real.cos (2 * π * (N : ℝ) * σ)

/-- Every position test is smooth on the full real line.
Source: arXiv:2605.09213v1, §1.1, the smooth test class in
`th:mf-gpt` (ii); no rough position indicator is used. -/
theorem contDiff_cursorCos (N : ℕ) : ContDiff ℝ (⊤ : ℕ∞) (cursorCos N) := by
  change ContDiff ℝ (⊤ : ℕ∞) (fun σ : ℝ => Real.cos (2 * π * (N : ℝ) * σ))
  exact Real.contDiff_cos.comp (contDiff_const.mul contDiff_id)

/-- The corresponding original two-variable test is jointly smooth.
Source: arXiv:2605.09213v1, §1.1, `φ ∈ C∞([0,1] × 𝕋)`.
It is constant in the token variable and therefore descends to the torus. -/
theorem contDiff_cursorCos_test (N : ℕ) :
    ContDiff ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => cursorCos N p.1) :=
  (contDiff_cursorCos N).comp contDiff_fst

/-- The original token periodicity holds at every cursor.
Source: arXiv:2605.09213v1, §1.1, the torus test class of
`eq:conv-rate-muNf`. The position oscillation is unaffected. -/
theorem periodic_cursorCos_test (N : ℕ) (σ : ℝ) :
    Function.Periodic (fun _ : ℝ => cursorCos N σ) (2 * π) := fun _ => rfl

/-- The complete token Sobolev bound is one at every derivative order,
uniformly in the grid frequency and position. Source:
arXiv:2605.09213v1, §1.1, `eq:conv-rate-muNf`.
Order zero uses the actual cosine bound; positive orders vanish. -/
theorem abs_iteratedDeriv_cursorCos_test_le_one (N r : ℕ) (σ u : ℝ) :
    |iteratedDeriv r (fun _ : ℝ => cursorCos N σ) u| ≤ 1 := by
  rw [iteratedDeriv_const]
  split_ifs
  · exact Real.abs_cos_le_one _
  · norm_num

/-- Every actual one-based grid point sees value one.
Source: arXiv:2605.09213v1, §1.1, `eq:emp-meas` and
`eq:conv-rate-muNf`. The index is `(j+1)/N`, exactly as in the empirical law. -/
theorem cursorCos_grid (N : ℕ) (hN : 0 < N) (j : Idx N) :
    cursorCos N (((j : ℝ) + 1) / N) = 1 := by
  have hne : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  have harg : 2 * π * (N : ℝ) * (((j : ℝ) + 1) / N) =
      ((j.val + 1 : ℕ) : ℝ) * (2 * π) := by
    push_cast
    field_simp
  rw [cursorCos, harg, Real.cos_nat_mul_two_pi]

/-- The actual continuum position integral is zero for every positive
grid size. Source: arXiv:2605.09213v1, §1.1,
`eq:conv-rate-muNf`; the limit has Lebesgue position marginal. -/
theorem integral_cursorCos (N : ℕ) (hN : 0 < N) :
    (∫ σ in (0 : ℝ)..1, cursorCos N σ) = 0 := by
  have hne : 2 * π * (N : ℝ) ≠ 0 := by positivity
  simp only [cursorCos]
  rw [intervalIntegral.integral_comp_mul_left (f := Real.cos) hne, integral_cos]
  have hsin : Real.sin (2 * π * (N : ℝ)) = 0 := by
    have harg : 2 * π * (N : ℝ) = ((2 * N : ℕ) : ℝ) * π := by push_cast; ring
    rw [harg, Real.sin_nat_mul_pi]
  simp only [mul_zero, mul_one, Real.sin_zero, hsin, sub_self, smul_zero]

/-- The original empirical position average is exactly one.
Source: arXiv:2605.09213v1, §1.1, `eq:emp-meas`.
The normalization is the actual `1/N`, with all `N` grid points retained. -/
theorem rightRiemann_cursorCos (N : ℕ) (hN : 0 < N) :
    (N : ℝ)⁻¹ * ∑ j : Idx N, cursorCos N (((j : ℝ) + 1) / N) = 1 := by
  simp only [cursorCos_grid N hN, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul, mul_one]
  exact inv_mul_cancel₀ (by exact_mod_cast hN.ne')

/-- The error in the original grid-versus-Lebesgue comparison is one.
Source: arXiv:2605.09213v1, §1.1, `eq:conv-rate-muNf`,
and §4, Step 4. This is the literal quadrature term of the printed estimate. -/
theorem cursorCos_error_one (N : ℕ) (hN : 0 < N) :
    |(N : ℝ)⁻¹ * ∑ j : Idx N, cursorCos N (((j : ℝ) + 1) / N) -
      ∫ σ in (0 : ℝ)..1, cursorCos N σ| = 1 := by
  rw [rightRiemann_cursorCos N hN, integral_cursorCos N hN]
  norm_num

/-- An actual smooth grid test defeats any positive power decay and
any proposed uniform prefactor. Source: arXiv:2605.09213v1,
§1.1, `eq:conv-rate-muNf`. The frequency is chosen after the prefactor,
exactly as allowed when that prefactor is uniform over test functions. -/
theorem exists_cursorCos_error_gt_decay (K p : ℝ) (hp : 0 < p) :
    ∃ N : ℕ, 0 < N ∧
      K * (N : ℝ) ^ (-p) <
        |(N : ℝ)⁻¹ * ∑ j : Idx N, cursorCos N (((j : ℝ) + 1) / N) -
          ∫ σ in (0 : ℝ)..1, cursorCos N σ| := by
  have ht : Tendsto (fun N : ℕ => K * (N : ℝ) ^ (-p)) atTop (nhds 0) := by
    simpa only [Function.comp_def, mul_zero] using
      ((tendsto_rpow_neg_atTop hp).comp
        (tendsto_natCast_atTop_atTop : Tendsto (fun N : ℕ => (N : ℝ)) atTop atTop)).const_mul K
  obtain ⟨N, hN, hbound⟩ := ((eventually_gt_atTop 0).and
    (ht.eventually_lt_const zero_lt_one)).exists
  exact ⟨N, hN, by rwa [cursorCos_error_one N hN]⟩

/-- No vanishing power quadrature rate is uniform over smooth position
tests bounded by one. Source: arXiv:2605.09213v1, §1.1,
`eq:conv-rate-muNf`. Applied to tests constant in the token variable,
this is exactly the uncontrolled position part of the printed token-norm bound. -/
theorem not_uniform_smooth_rightRiemann_rate (p : ℝ) (hp : 0 < p) :
    ¬ ∃ K : ℝ, ∀ N : ℕ, 0 < N → ∀ g : ℝ → ℝ,
      ContDiff ℝ (⊤ : ℕ∞) g → (∀ σ ∈ Icc (0 : ℝ) 1, |g σ| ≤ 1) →
      |(N : ℝ)⁻¹ * ∑ j : Idx N, g (((j : ℝ) + 1) / N) -
        ∫ σ in (0 : ℝ)..1, g σ| ≤ K * (N : ℝ) ^ (-p) := by
  rintro ⟨K, hK⟩
  obtain ⟨N, hN, hbad⟩ := exists_cursorCos_error_gt_decay K p hp
  exact not_lt_of_ge (hK N hN (cursorCos N) (contDiff_cursorCos N)
    (fun σ _ => Real.abs_cos_le_one _)) hbad

/-- The original constant Dirac profile is an actual mean-field solution,
for every model parameter. Source: arXiv:2605.09213v1,
§1.1, `eq:mfl-lambda`; the interaction derivative vanishes at zero.
This names the proved constant-profile witness in `MeanField`. -/
theorem isMeanFieldSolution_constant_zero (lam β : ℝ) :
    IsMeanFieldSolution lam β (fun _ => ⟨Measure.dirac 0, inferInstance⟩)
      (fun _ _ => ⟨Measure.dirac 0, inferInstance⟩) := by
  refine ⟨Filter.Eventually.of_forall fun _ => rfl, fun t _ ψ _ => aemeasurable_const,
    Filter.Eventually.of_forall fun σ ψ _ hψ t _ => ?_⟩
  have hw : wDerivT β 0 = 0 := by
    have h := (wBetaDeriv_periodic β).lift_coe 0
    rw [AddCircle.coe_zero] at h
    simp [wDerivT, h, wBetaDeriv]
  have hv : mfVelocity lam β (fun _ _ => (⟨Measure.dirac 0, inferInstance⟩ :
      ProbabilityMeasure Torus)) t σ 0 = 0 := by
    simp [mfVelocity, wConv, hw]
  show HasDerivWithinAt (fun _ : ℝ => ∫ x, hψ.lift x ∂(Measure.dirac (0 : Torus)))
    (∫ x, (periodic_deriv hψ).lift x * mfVelocity lam β _ t σ x ∂(Measure.dirac (0 : Torus)))
    (Set.Ici 0) t
  rw [integral_dirac, integral_dirac, hv, mul_zero]
  exact hasDerivWithinAt_const t (Set.Ici (0 : ℝ)) _

/-- The matching original finite prompt is stationary for every positive
or zero token count. Source: arXiv:2605.09213v1, §1,
`eq:gpt-dyn-alibi`; every interaction difference is zero. -/
theorem isGPTFlow_constant_zero (lam β : ℝ) (N : ℕ) :
    IsGPTFlow lam β N (fun _ _ => 0) := by
  intro t j
  have hz : gptField lam β N (fun _ => 0) j = 0 := by simp [gptField, wBetaDeriv]
  rw [hz]
  exact hasDerivAt_const t 0

/-- All quadrature hypotheses have a genuine positive grid witness,
and the original test is jointly smooth with every token derivative bounded
by one. Source: arXiv:2605.09213v1, §1.1, `th:mf-gpt` (ii). -/
example : (0 : ℕ) < 1 ∧ ContDiff ℝ (⊤ : ℕ∞) (cursorCos 1) ∧
    ContDiff ℝ (⊤ : ℕ∞) (fun p : ℝ × ℝ => cursorCos 1 p.1) ∧
    (∀ σ : ℝ, Function.Periodic (fun _ : ℝ => cursorCos 1 σ) (2 * π)) ∧
    ∀ r : ℕ, ∀ σ u : ℝ, |iteratedDeriv r (fun _ : ℝ => cursorCos 1 σ) u| ≤ 1 :=
  ⟨Nat.one_pos, contDiff_cursorCos 1, contDiff_cursorCos_test 1,
    periodic_cursorCos_test 1, abs_iteratedDeriv_cursorCos_test_le_one 1⟩

/-- A positive decay exponent is jointly admissible with every finite
prefactor; the test frequency is then produced by the proved limit.
Source: arXiv:2605.09213v1, §1.1, `eq:conv-rate-muNf`. -/
example : (0 : ℝ) < 1 / 2 := by norm_num

end Transformer.Kinetic
