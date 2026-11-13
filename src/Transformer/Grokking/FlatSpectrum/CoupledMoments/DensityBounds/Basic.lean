module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
public import Mathlib.Analysis.Matrix.PosDef

/-!
# Convex tolerances on actual density readings

Source: `review/flat_spectrum.md`, CM22. Relax the affine unit diagonal by a
declared norm tolerance. Positivity and trace one remain exact mathematical
constraints; a numerical residual is not an exact certificate of them.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds

open Finset Matrix DensityExtraction
open scoped ComplexOrder

variable {ι X : Type*} [Fintype ι]

/-- PSD trace-one densities with bounded actual diagonal error, CM22. -/
def Relaxed (B : Matrix X ι ℂ) (ε : ℝ) (M : Matrix ι ι ℂ) : Prop :=
  Density M ∧ ∀ x, ‖reading B M x x - 1‖ ≤ ε

/-- PSD kernel diagonals are real, CM22; this justifies real interval
constraints without dropping an unmeasured imaginary component. -/
theorem reading_diagonal_real [Finite X] (B : Matrix X ι ℂ) {M : Matrix ι ι ℂ}
    (hM : Density M) (x : X) : (reading B M x x).im = 0 := by
  have hp := hM.1.mul_mul_conjTranspose_same B
  exact (Complex.nonneg_iff.mp (hp.diag_nonneg (i := x))).2.symm

/-- On actual PSD densities the complex norm tolerance is precisely a real
box constraint, CM22. This is the domain a native interval solver may use. -/
theorem relaxed_interval [Finite X] (B : Matrix X ι ℂ) (ε : ℝ) (M : Matrix ι ι ℂ) :
    Relaxed B ε M ↔ Density M ∧
      ∀ x, 1 - ε ≤ (reading B M x x).re ∧ (reading B M x x).re ≤ 1 + ε := by
  have hn (hM : Density M) (x : X) : ‖reading B M x x - 1‖ =
      |(reading B M x x).re - 1| := by
    have hr : reading B M x x - 1 = (((reading B M x x).re - 1 : ℝ) : ℂ) := by
      apply Complex.ext
      · simp
      · simp [reading_diagonal_real B hM x]
    rw [hr, Complex.norm_real, Real.norm_eq_abs]
  constructor
  · rintro ⟨hM, he⟩
    refine ⟨hM, fun x => ?_⟩
    have h := he x
    rw [hn hM x, abs_le] at h
    constructor <;> linarith [h.1, h.2]
  · rintro ⟨hM, he⟩
    refine ⟨hM, fun x => ?_⟩
    rw [hn hM x, abs_le]
    constructor <;> linarith [(he x).1, (he x).2]

/-- The tolerance domain is convex without imposing purity, CM22. -/
theorem relaxed_convex (B : Matrix X ι ℂ) (ε : ℝ) :
    Convex ℝ {M | Relaxed B ε M} := by
  intro M hM N hN a b ha hb hab
  refine ⟨density_convex hM.1 hN.1 ha hb hab, fun x => ?_⟩
  have he : reading B (a • M + b • N) x x - 1 =
      a • (reading B M x x - 1) + b • (reading B N x x - 1) := by
    rw [reading_mix, smul_sub, smul_sub]
    have hc : a • (1 : ℂ) + b • (1 : ℂ) = 1 := by rw [← add_smul, hab, one_smul]
    linear_combination hc
  rw [he]
  calc
    _ ≤ ‖a • (reading B M x x - 1)‖ + ‖b • (reading B N x x - 1)‖ := norm_add_le _ _
    _ = a * ‖reading B M x x - 1‖ + b * ‖reading B N x x - 1‖ := by
      rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
        abs_of_nonneg ha, abs_of_nonneg hb]
    _ ≤ a * ε + b * ε := add_le_add
      (mul_le_mul_of_nonneg_left (hM.2 x) ha) (mul_le_mul_of_nonneg_left (hN.2 x) hb)
    _ = ε := by rw [← add_mul, hab, one_mul]

/-- Exact feasibility is admitted at every nonnegative tolerance, CM22. -/
theorem exact_relaxed {B : Matrix X ι ℂ} {M : Matrix ι ι ℂ}
    (hM : Feasible B M) {ε : ℝ} (hε : 0 ≤ ε) : Relaxed B ε M := by
  refine ⟨hM.1, fun x => ?_⟩
  simpa only [hM.2 x, sub_self, norm_zero] using hε

/-- Zero tolerance recovers the exact domain, CM22. -/
theorem relaxed_zero (B : Matrix X ι ℂ) (M : Matrix ι ι ℂ) :
    Relaxed B 0 M ↔ Feasible B M := by
  constructor
  · rintro ⟨hM, hd⟩
    refine ⟨hM, fun x => ?_⟩
    exact sub_eq_zero.mp (norm_eq_zero.mp (le_antisymm (hd x) (norm_nonneg _)))
  · intro hM
    exact exact_relaxed hM le_rfl

/-- The diagonal of every density is a probability vector, even when the
matrix has off-diagonal terms, CM22. -/
theorem density_weights {M : Matrix ι ι ℂ} (hM : Density M) :
    Weights (fun i => (M i i).re) := by
  refine ⟨fun i => (Complex.nonneg_iff.mp (hM.1.diag_nonneg (i := i))).1, ?_⟩
  simpa [Matrix.trace, Matrix.diag] using congrArg Complex.re hM.2

/-- A density's diagonal entries lie in the unit interval, CM22. -/
theorem density_diagonal_le_one {M : Matrix ι ι ℂ} (hM : Density M) (i : ι) :
    (M i i).re ≤ 1 := by
  have hw := density_weights hM
  calc
    _ ≤ ∑ j, (M j j).re := single_le_sum (fun j hj => hw.1 j) (mem_univ i)
    _ = 1 := hw.2

/-- Two-by-two principal minors bound every entry of a density by one,
CM22. This bound is derived from PSD and trace, not added as a hypothesis. -/
theorem density_entry_le_one {M : Matrix ι ι ℂ} (hM : Density M) (i j : ι) :
    ‖M i j‖ ≤ 1 := by
  have hd := (hM.1.submatrix (![i, j] : Fin 2 → ι)).det_nonneg
  have hi := Complex.nonneg_iff.mp (hM.1.diag_nonneg (i := i))
  have hj := Complex.nonneg_iff.mp (hM.1.diag_nonneg (i := j))
  have hh := hM.1.isHermitian.apply j i
  have hr := (Complex.nonneg_iff.mp hd).1
  simp only [Matrix.det_fin_two, Matrix.submatrix_apply, Matrix.cons_val_zero,
    Matrix.cons_val_one] at hr
  rw [← hh] at hr
  simp only [Complex.sub_re, Complex.mul_re, Complex.star_def, Complex.conj_re,
    Complex.conj_im, ← hi.2] at hr
  have hu := mul_le_mul (density_diagonal_le_one hM i) (density_diagonal_le_one hM j)
    hj.1 (by norm_num : (0 : ℝ) ≤ 1)
  have hn := Complex.sq_norm (M i j)
  rw [Complex.normSq_apply] at hn
  nlinarith [norm_nonneg (M i j)]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds
