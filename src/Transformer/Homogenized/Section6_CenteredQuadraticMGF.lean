module

public import Transformer.Homogenized.Section6_OffDiagonalQuadraticMGF

/-!
# MGF of the original mean-centered quadratic form

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The actual mean-centered quadratic form splits into its original
centered diagonal and off-diagonal terms. Their proved doubled-coefficient
exponentials dominate the original exponential by the elementary
arithmetic-geometric mean bound. This establishes integrability and
combines the two estimates on the original probability space.
The fixed universal constant and coefficient interval retain the exact
original matrix Frobenius square and Euclidean operator norm.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual sum exponential is bounded by the average of its
doubled-component exponentials. Source: arXiv:2604.01978v1,
§6.3, the two-part MGF combination for `lem:Hanson_wright`. -/
theorem exp_sum_le_exp_double_average (x y : ℝ) :
    Real.exp (x + y) ≤ (Real.exp (2 * x) + Real.exp (2 * y)) / 2 := by
  have hx : Real.exp (2 * x) = Real.exp x ^ 2 := by simpa only [Nat.cast_ofNat] using Real.exp_nat_mul x 2
  have hy : Real.exp (2 * y) = Real.exp y ^ 2 := by simpa only [Nat.cast_ofNat] using Real.exp_nat_mul y 2
  rw [Real.exp_add, hx, hy]
  apply (le_div_iff₀ (by norm_num : (0 : ℝ) < 2)).mpr
  nlinarith [sq_nonneg (Real.exp x - Real.exp y)]

/-- The original quadratic form centered by its actual integral has
an integrable exponential and universal quadratic MGF bound on a fixed
operator-norm interval. Source: arXiv:2604.01978v1, §6.3,
the full centered exponential estimate for `lem:Hanson_wright`. -/
theorem subgaussian_centered_quadratic_exp_integral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hsmall : |t| * (c : ℝ) * matOpNorm B ≤ 1 / (64 * Real.exp 1)) :
    Integrable (fun ω => Real.exp (t * (hansonWrightQuadraticForm B (Z ω) -
      ∫ η, hansonWrightQuadraticForm B (Z η) ∂P))) P ∧
    (∫ ω, Real.exp (t * (hansonWrightQuadraticForm B (Z ω) -
      ∫ η, hansonWrightQuadraticForm B (Z η) ∂P)) ∂P) ≤
      Real.exp (8192 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by
  let D : Ω → ℝ := fun ω => ∑ i : Fin n, B i i * ((Z ω i) ^ 2 - ∫ η, (Z η i) ^ 2 ∂P)
  let O : Ω → ℝ := fun ω => hansonWrightQuadraticForm (B - Matrix.diagonal (fun i => B i i)) (Z ω)
  let F : Ω → ℝ := fun ω => Real.exp (t * (hansonWrightQuadraticForm B (Z ω) -
    ∫ η, hansonWrightQuadraticForm B (Z η) ∂P))
  let G : Ω → ℝ := fun ω => (Real.exp ((2 * t) * D ω) + Real.exp ((2 * t) * O ω)) / 2
  let V : ℝ := 8192 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2
  have hn : 0 ≤ |t| * (c : ℝ) * matOpNorm B :=
    mul_nonneg (mul_nonneg (abs_nonneg t) c.coe_nonneg) (matOpNorm_nonneg B)
  have hs := (le_div_iff₀ (show 0 < 64 * Real.exp 1 by positivity)).mp hsmall
  have he : 1 ≤ Real.exp 1 := Real.one_le_exp (by norm_num : (0 : ℝ) ≤ 1)
  have hr : |t| * (c : ℝ) * matOpNorm B ≤ 1 / 64 := by
    have hh := mul_le_mul_of_nonneg_left he (show 0 ≤ 64 * (|t| * (c : ℝ) * matOpNorm B) by positivity)
    nlinarith [hh]
  have hdsmall : ∀ i : Fin n, |2 * t| * (c : ℝ) * |B i i| ≤ 1 / (32 * Real.exp 1) := by
    intro i
    rw [abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
    apply (le_div_iff₀ (show 0 < 32 * Real.exp 1 by positivity)).mpr
    have hb := mul_le_mul_of_nonneg_left (abs_matrix_entry_le_matOpNorm B i i)
      (mul_nonneg (abs_nonneg t) c.coe_nonneg)
    have hh := mul_le_mul_of_nonneg_right hb (show 0 ≤ 64 * Real.exp 1 by positivity)
    nlinarith [hh]
  have hosmall : |2 * t| * (c : ℝ) * matOpNorm B ≤ 1 / 8 := by
    rw [abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
    nlinarith [hr]
  obtain ⟨hdi, hdb⟩ := subgaussian_centered_diagonal_exp_integral_le P Z c hindep hZ (fun i => B i i) (2 * t) hdsmall
  obtain ⟨hoi, hob⟩ := subgaussian_offDiagonal_exp_integral_of_abs_le P Z c hindep hZ B (2 * t) hosmall
  have hd : (∫ ω, Real.exp ((2 * t) * D ω) ∂P) ≤ Real.exp V := by
    apply hdb.trans
    apply Real.exp_le_exp.mpr
    have hc : 2048 * (Real.exp 1) ^ 4 * (2 * t) ^ 2 * (c : ℝ) ^ 2 * (∑ i : Fin n, B i i ^ 2) =
        8192 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 * ∑ i : Fin n, B i i ^ 2 := by ring
    rw [hc]
    exact mul_le_mul_of_nonneg_left (diagonal_sq_sum_le_frobenius_sq B) (by positivity)
  have ho : (∫ ω, Real.exp ((2 * t) * O ω) ∂P) ≤ Real.exp V := by
    apply hob.trans
    apply Real.exp_le_exp.mpr
    have he4 : 1 ≤ (Real.exp 1) ^ 4 := by
      have h := Real.one_le_exp (by norm_num : (0 : ℝ) ≤ 4)
      simpa only [show (4 : ℝ) = (4 : ℕ) * (1 : ℝ) by norm_num, Real.exp_nat_mul] using h
    have hp : 0 ≤ t ^ 2 * (c : ℝ) ^ 2 * (∑ i : Fin n, ∑ j : Fin n, B i j ^ 2) := by positivity
    have hh := mul_le_mul_of_nonneg_right (show 256 ≤ 8192 * (Real.exp 1) ^ 4 by nlinarith [he4]) hp
    dsimp only [V]
    nlinarith [hh]
  have hgi : Integrable G P := (hdi.fun_add hoi).div_const 2
  have hp : ∀ ω, F ω ≤ G ω := by
    intro ω
    have hs : hansonWrightQuadraticForm B (Z ω) - (∫ η, hansonWrightQuadraticForm B (Z η) ∂P) = D ω + O ω :=
      centered_hansonWrightQuadraticForm_split P Z c hindep hZ B ω
    have h := exp_sum_le_exp_double_average (t * D ω) (t * O ω)
    dsimp only [F, G]
    rw [hs, mul_add]
    simpa only [mul_assoc] using h
  have hz := aemeasurable_eucSpace_of_coordinates P Z (fun i => (hZ i).aemeasurable)
  have hfm : AEMeasurable F P :=
    (((aemeasurable_hansonWrightQuadraticForm P Z hz B).fun_sub aemeasurable_const).const_mul t).exp
  have hfi : Integrable F P := by
    apply hgi.mono' hfm.aestronglyMeasurable
    apply ae_of_all
    intro ω
    rw [Real.norm_eq_abs, abs_of_nonneg (show 0 ≤ F ω from Real.exp_nonneg _)]
    exact hp ω
  refine ⟨hfi, ?_⟩
  have hh := integral_mono hfi hgi hp
  have hg : (∫ ω, G ω ∂P) ≤ Real.exp V := by
    dsimp only [G]
    rw [integral_div, integral_add hdi hoi]
    linarith [hd, ho]
  exact hh.trans hg

/-- The actual mean-centered quadratic exponential has the same finite
extended-real integral bound, ready for the Chernoff tail estimate.
Source: arXiv:2604.01978v1, §6.3, the MGF input to `lem:Hanson_wright`. -/
theorem subgaussian_centered_quadratic_lintegral_le {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (t : ℝ)
    (hsmall : |t| * (c : ℝ) * matOpNorm B ≤ 1 / (64 * Real.exp 1)) :
    (∫⁻ ω, ENNReal.ofReal (Real.exp (t * (hansonWrightQuadraticForm B (Z ω) -
      ∫ η, hansonWrightQuadraticForm B (Z η) ∂P))) ∂P) ≤
      ENNReal.ofReal (Real.exp (8192 * (Real.exp 1) ^ 4 * t ^ 2 * (c : ℝ) ^ 2 *
        ∑ i : Fin n, ∑ j : Fin n, B i j ^ 2)) := by
  obtain ⟨hi, hb⟩ := subgaussian_centered_quadratic_exp_integral_le P Z c hindep hZ B t hsmall
  have h := ENNReal.ofReal_le_ofReal hb
  rw [ofReal_integral_eq_lintegral_ofReal hi (ae_of_all _ (fun ω => Real.exp_nonneg _))] at h
  exact h

/-- Genuine standard coordinates and a nonzero coefficient witness
the original small-operator-norm interval on a nonempty probability space. -/
example : ∃ Z : (Fin 2 → ℝ) → EucSpace 2,
    iIndepFun (fun i : Fin 2 => fun z => Z z i) (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z => Z z i) 1 (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    |(1 / (128 * Real.exp 1) : ℝ)| * (1 : ℝ≥0) * matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) ≤
      1 / (64 * Real.exp 1) := by
  refine ⟨WithLp.toLp 2, ?_, hasSubgaussianMGF_standardProduct_coordinate, inferInstance, ?_⟩
  · simpa only [WithLp.ofLp_toLp, id_eq] using iIndepFun_pi
      (μ := fun _ : Fin 2 => gaussianReal 0 1) (X := fun _ => id) (fun i => measurable_id.aemeasurable)
  · have hn : matOpNorm (1 : Matrix (Fin 2) (Fin 2) ℝ) = 1 := by rw [← Matrix.diagonal_one, matOpNorm_diagonal]; simp
    rw [hn, abs_of_nonneg (by positivity), NNReal.coe_one, mul_one, mul_one]
    apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
    nlinarith [Real.exp_pos (1 : ℝ)]

end Transformer.Homogenized
