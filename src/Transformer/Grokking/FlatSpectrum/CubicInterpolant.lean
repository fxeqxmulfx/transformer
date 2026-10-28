module

public import Transformer.Grokking.FlatSpectrum.Interpolation
public import Transformer.Grokking.FlatSpectrum.Kernels
public import Transformer.Grokking.FlatSpectrum.Unimodular

/-!
# A flat kernel is never its own minimum-norm interpolant

Source: `review/flat_spectrum.md`, Theorem A, last paragraph. When every singular value of `K`
equals `c`, the polar factor is `W = K / c`. For the `a³ + a b` kernel `c = √p`, so every entry
of `W` has modulus `p^{-1/2}` and none vanishes. By Theorem A, `K` minimizes `‖X‖_S1` over
`{X = K on Ω}` for no proper training set `Ω`.

The same holds for every square `K` with equal singular values and no zero entry (added,
`not_isMinOn_of_singularValues_eq`): `∑ σ_i² = ‖K‖_S2² > 0` gives `c ≠ 0`, so `K` is invertible
and `W = K / c` vanishes nowhere. So `K` does not have the least nuclear norm on any proper `Ω`,
and M recovers it from none (`not_forall_isMinOn_iff_of_singularValues_eq`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- With every singular value equal to `c`, the polar factor is `K / c`. -/
lemma polarFactor_of_singularValues_eq {K : Matrix m n ℂ} {c : ℝ}
    (h : ∀ i, singularValues K i = c) : polarFactor K = ((c⁻¹ : ℝ) : ℂ) • K := by
  rw [polarFactor_eq]
  simp only [h]
  rw [← smul_one_eq_diagonal, Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul,
    Matrix.mul_assoc, eigenvectorUnitary_mul_conjTranspose, Matrix.mul_one]

/-- A square matrix whose singular values all equal `c ≠ 0` is invertible. -/
lemma det_ne_zero_of_singularValues_eq {K : Matrix n n ℂ} {c : ℝ} (hc : c ≠ 0)
    (h : ∀ i, singularValues K i = c) : K.det ≠ 0 := by
  intro hK
  have hd := (isHermitian_conjTranspose_mul_self K).det_eq_prod_eigenvalues
  rw [det_mul, hK, mul_zero] at hd
  obtain ⟨i, -, hi⟩ := Finset.prod_eq_zero_iff.mp hd.symm
  have h0 : (isHermitian_conjTranspose_mul_self K).eigenvalues i = 0 := RCLike.ofReal_eq_zero.mp hi
  have hci := h i
  rw [singularValues, h0, Real.sqrt_zero] at hci
  exact hc hci.symm

/-- **Theorem A for a flat kernel** (`review/flat_spectrum.md`, Theorem A, last paragraph,
stated there for `a³ + a b`): if all singular values of `K` are equal and no entry of `K`
vanishes, then for every proper `Ω` some matrix equal to `K` on `Ω` has a smaller nuclear norm
than `K`. -/
theorem not_isMinOn_of_singularValues_eq {K : Matrix n n ℂ} {c : ℝ}
    (h : ∀ i, singularValues K i = c) (hK : ∀ i j, K i j ≠ 0) {Ω : Set (n × n)}
    (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} K := by
  obtain ⟨⟨a, b⟩, hab⟩ := (Set.ne_univ_iff_exists_notMem Ω).mp hΩ
  have hc : c ≠ 0 := by
    rintro rfl
    have h2 : frobeniusSq K = 0 := by
      rw [← sum_singularValues_sq]
      simp [h]
    exact hK a b (by rw [eq_zero_of_frobeniusSq_eq_zero h2, Matrix.zero_apply])
  intro hmin
  rw [isMinOn_nuclearNorm_iff (det_ne_zero_of_singularValues_eq hc h)] at hmin
  have hW := hmin a b hab
  rw [polarFactor_of_singularValues_eq h, Matrix.smul_apply, smul_eq_mul] at hW
  exact mul_ne_zero (Complex.ofReal_ne_zero.mpr (inv_ne_zero hc)) (hK a b) hW

/-- **Theorem A for a flat kernel**, as recovery: M recovers such a `K` from no proper `Ω`. -/
theorem not_forall_isMinOn_iff_of_singularValues_eq {K : Matrix n n ℂ} {c : ℝ}
    (h : ∀ i, singularValues K i = c) (hK : ∀ i j, K i j ≠ 0) {Ω : Set (n × n)}
    (hΩ : Ω ≠ Set.univ) :
    ¬ ∀ X ∈ {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix n n ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2} X ↔
        X = K) :=
  fun hrec => not_isMinOn_of_singularValues_eq h hK hΩ ((hrec K fun _ _ => rfl).mpr rfl)

variable (p : ℕ) [NeZero p]

/-- The polar factor of the `a³ + a b` kernel is `K / √p`, Theorem A. -/
theorem polarFactor_cubicKernel :
    polarFactor (cubicKernel p) = (((√p)⁻¹ : ℝ) : ℂ) • cubicKernel p :=
  polarFactor_of_singularValues_eq (singularValues_cubicKernel p)

/-- Every entry of the polar factor of the `a³ + a b` kernel has modulus `p^{-1/2}`,
Theorem A. -/
theorem norm_polarFactor_cubicKernel (a b : ZMod p) :
    ‖polarFactor (cubicKernel p) a b‖ = (√p)⁻¹ := by
  rw [polarFactor_cubicKernel, Matrix.smul_apply, smul_eq_mul, norm_mul, Complex.norm_real,
    Real.norm_of_nonneg (inv_nonneg.mpr (Real.sqrt_nonneg _))]
  simp [cubicKernel, taskKernel]

/-- The `a³ + a b` kernel is invertible: `K* K = p I`. -/
theorem det_cubicKernel_ne_zero : (cubicKernel p).det ≠ 0 := by
  intro h
  have h1 := congrArg det (conjTranspose_mul_cubicKernel p)
  rw [det_mul, h, mul_zero, det_smul, det_one, mul_one] at h1
  exact pow_ne_zero _ (Nat.cast_ne_zero.mpr (NeZero.ne p)) h1.symm

example (Ω : Set (ZMod 3 × ZMod 3)) :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = cubicKernel 3 ij.1 ij.2} (cubicKernel 3) ↔
      ∀ i j, (i, j) ∉ Ω → polarFactor (cubicKernel 3) i j = 0 :=
  isMinOn_nuclearNorm_iff (det_cubicKernel_ne_zero 3) Ω

/-- **Theorem A for `a³ + a b`**, `review/flat_spectrum.md`: the kernel `K` is not the
minimum-nuclear-norm matrix agreeing with `K` on `Ω`, for any proper `Ω`. -/
theorem not_isMinOn_cubicKernel {Ω : Set (ZMod p × ZMod p)} (hΩ : Ω ≠ Set.univ) :
    ¬IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = cubicKernel p ij.1 ij.2}
      (cubicKernel p) := by
  rw [isMinOn_nuclearNorm_iff (det_cubicKernel_ne_zero p)]
  obtain ⟨⟨a, b⟩, hab⟩ := (Set.ne_univ_iff_exists_notMem Ω).mp hΩ
  intro h
  have h1 := norm_polarFactor_cubicKernel p a b
  rw [h a b hab, norm_zero] at h1
  have hp : (0 : ℝ) < p := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne p))
  exact (inv_pos.mpr (Real.sqrt_pos.mpr hp)).ne h1

example : ¬IsMinOn nuclearNorm
    {X | ∀ ij ∈ (∅ : Set (ZMod 3 × ZMod 3)), X ij.1 ij.2 = cubicKernel 3 ij.1 ij.2}
      (cubicKernel 3) :=
  not_isMinOn_cubicKernel 3 Set.empty_ne_univ

example : polarFactor (cubicKernel 2) = ((((√(2 : ℕ) : ℝ))⁻¹ : ℝ) : ℂ) • cubicKernel 2 :=
  polarFactor_of_singularValues_eq (singularValues_cubicKernel 2)

example := det_ne_zero_of_singularValues_eq (K := cubicKernel 2)
  (Real.sqrt_ne_zero'.mpr (by norm_num)) (singularValues_cubicKernel 2)

/-- With no cell seen, the `a³ + a b` kernel for `p = 3` does not have the least nuclear norm. -/
example := not_isMinOn_of_singularValues_eq (singularValues_cubicKernel 3)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3))

/-- M does not recover the `a³ + a b` kernel for `p = 3` from the empty set of cells. -/
example := not_forall_isMinOn_iff_of_singularValues_eq (singularValues_cubicKernel 3)
  (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
  (Set.empty_ne_univ (α := ZMod 3 × ZMod 3))

end Transformer.Grokking.FlatSpectrum
