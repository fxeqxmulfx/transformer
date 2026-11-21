module

public import Transformer.Grokking.FlatSpectrum.ProductCost

/-!
# The product cost is attained at the SVD and scales with exponent 2/3

Source: `review/flat_spectrum.md`, NV10. With `A` an eigenbasis of `Xᴴ X` and `σ` the singular
values, the factors `G = X A diag(σ^{-2/3})`, `o = σ^{1/3}`, `W = A diag(σ^{1/3})` write `X` and
each has squared norm `∑ σ^{2/3}`. So `productCost X` is the least weight of a one-layer
product factorization of `X`, over every inner width. Rescaling the three factors then gives
`productCost (c X) = c^{2/3} productCost X`, where the factorization model pays `c ‖X‖_S1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- Scaling the columns by reals scales their squared lengths. -/
lemma frobeniusSq_mul_diagonal_ofReal (H : Matrix m n ℂ) (d : n → ℝ) :
    frobeniusSq (H * diagonal fun i => (d i : ℂ)) = ∑ i, d i ^ 2 * ∑ k, ‖H k i‖ ^ 2 := by
  rw [frobeniusSq, Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [mul_diagonal, norm_mul, Complex.norm_real, Real.norm_eq_abs, mul_pow, sq_abs]
  ring

omit [DecidableEq n] in
/-- `‖v A‖_S2² = |v|² ‖A‖_S2²`. -/
lemma frobeniusSq_smul_eq (v : ℂ) (A : Matrix m n ℂ) :
    frobeniusSq (v • A) = ‖v‖ ^ 2 * frobeniusSq A := by
  simp only [frobeniusSq, Matrix.smul_apply, smul_eq_mul, norm_mul, mul_pow, Finset.mul_sum]

/-- **The product cost is attained** at the SVD, through inner width `n`. -/
theorem exists_productFactorization (X : Matrix m n ℂ) :
    ∃ (G : Matrix m n ℂ) (o : n → ℂ) (W : Matrix n n ℂ), X = G * diagonal o * Wᴴ ∧
      (frobeniusSq G + ∑ u, ‖o u‖ ^ 2 + frobeniusSq W) / 2 = productCost X := by
  obtain ⟨A, hA, hXA⟩ : ∃ A : Matrix n n ℂ, Aᴴ * A = 1 ∧ Aᴴ * (Xᴴ * X) * A =
      diagonal fun i => ((singularValues X i ^ 2 : ℝ) : ℂ) :=
    ⟨_, eigenvectorUnitary_conjTranspose_mul (isHermitian_conjTranspose_mul_self _),
      gram_diagonalization _⟩
  have hA' : A * Aᴴ = 1 := mul_eq_one_comm.mp hA
  obtain ⟨s, hsdef⟩ : ∃ s : n → ℝ, s = singularValues X := ⟨_, rfl⟩
  rw [← hsdef] at hXA
  have hs (i : n) : 0 ≤ s i := by
    rw [hsdef]
    exact Real.sqrt_nonneg _
  obtain ⟨a, hadef⟩ : ∃ a : n → ℝ, a = fun i => s i ^ (1 / 3 : ℝ) := ⟨_, rfl⟩
  have ha (i : n) : 0 ≤ a i := by
    rw [hadef]
    exact Real.rpow_nonneg (hs i) _
  have ha3 (i : n) : s i = a i ^ 3 := by
    simp only [hadef]
    rw [← Real.rpow_natCast, ← Real.rpow_mul (hs i)]
    norm_num
  have ha2 (i : n) : s i ^ (2 / 3 : ℝ) = a i ^ 2 := by
    simp only [hadef]
    rw [← Real.rpow_natCast, ← Real.rpow_mul (hs i)]
    norm_num
  have hH : (X * A)ᴴ * (X * A) = diagonal fun i => ((s i ^ 2 : ℝ) : ℂ) := by
    rw [← hXA]
    simp only [conjTranspose_mul, Matrix.mul_assoc]
  have hcol (i : n) : ∑ k, ‖(X * A) k i‖ ^ 2 = s i ^ 2 := by
    have h := congrFun (congrFun hH i) i
    rw [conjTranspose_mul_self_apply_self, diagonal_apply_eq] at h
    exact_mod_cast h
  refine ⟨X * A * diagonal fun i => (((a i ^ 2)⁻¹ : ℝ) : ℂ), fun i => (a i : ℂ),
    A * diagonal fun i => (a i : ℂ), ?_, ?_⟩
  · have hfix : X * A * (diagonal fun i => (((a i ^ 2)⁻¹ : ℝ) : ℂ)) *
        (diagonal fun i => (a i : ℂ)) * (diagonal fun i => (a i : ℂ)) = X * A := by
      rw [Matrix.mul_assoc (X * A), diagonal_mul_diagonal, Matrix.mul_assoc (X * A),
        diagonal_mul_diagonal]
      ext k i
      rw [mul_diagonal]
      rcases (ha i).eq_or_lt with h | h
      · have h0 : (X * A) k i = 0 :=
          apply_eq_zero_of_gram hH (by simp [ha3 i, ← h]) k
        rw [h0, zero_mul]
      · have h1 : (a i ^ 2)⁻¹ * a i * a i = 1 := by field_simp
        rw [← Complex.ofReal_mul, ← Complex.ofReal_mul, h1, Complex.ofReal_one, mul_one]
    rw [conjTranspose_mul, conjTranspose_diagonal_ofReal]
    calc X = X * A * Aᴴ := by rw [Matrix.mul_assoc, hA', Matrix.mul_one]
      _ = X * A * (diagonal fun i => (((a i ^ 2)⁻¹ : ℝ) : ℂ)) *
          (diagonal fun i => (a i : ℂ)) * (diagonal fun i => (a i : ℂ)) * Aᴴ := by rw [hfix]
      _ = _ := by simp only [Matrix.mul_assoc]
  · rw [frobeniusSq_mul_diagonal_ofReal, frobeniusSq_mul_diagonal_ofReal]
    simp only [hcol, sum_norm_sq_col_eq_one hA, mul_one]
    have h1 : ∑ i, ((a i ^ 2)⁻¹) ^ 2 * s i ^ 2 = ∑ i, a i ^ 2 := by
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [ha3 i]
      rcases (ha i).eq_or_lt with h | h
      · rw [← h]
        simp
      · field_simp
    have h2 : ∑ u, ‖(a u : ℂ)‖ ^ 2 = ∑ i, a i ^ 2 :=
      Finset.sum_congr rfl fun i _ => by rw [Complex.norm_real, Real.norm_eq_abs, sq_abs]
    rw [h1, h2, productCost, ← hsdef]
    simp only [ha2]
    ring

/-- **The product-layer analogue of Lemma 1** (NV10): `productCost X` is the least of
`½ (‖G‖² + ‖o‖² + ‖W‖²)` over factorizations `X = G diag(o) Wᴴ` of every inner width. -/
theorem productCost_isLeast (X : Matrix m n ℂ) :
    IsLeast {c | ∃ (r : ℕ) (G : Matrix m (Fin r) ℂ) (o : Fin r → ℂ) (W : Matrix n (Fin r) ℂ),
      X = G * diagonal o * Wᴴ ∧ c = (frobeniusSq G + ∑ u, ‖o u‖ ^ 2 + frobeniusSq W) / 2}
      (productCost X) := by
  refine ⟨?_, ?_⟩
  · obtain ⟨G, o, W, hX, hc⟩ := exists_productFactorization X
    set e := (Fintype.equivFin n).symm
    refine ⟨Fintype.card n, G.submatrix id e, o ∘ e, W.submatrix id e, ?_, ?_⟩
    · rw [← submatrix_diagonal_equiv o e, submatrix_mul_equiv, conjTranspose_submatrix,
        submatrix_mul_equiv, submatrix_id_id]
      exact hX
    · have hsum : ∑ u, ‖(o ∘ e) u‖ ^ 2 = ∑ u, ‖o u‖ ^ 2 := e.sum_comp fun u => ‖o u‖ ^ 2
      rw [frobeniusSq_submatrix_equiv, frobeniusSq_submatrix_equiv, hsum, hc]
  · rintro c ⟨r, G, o, W, h, rfl⟩
    rw [h]
    exact productCost_le G o W

/-- Scaling by `c` costs at most the factor `c^{2/3}`. -/
lemma productCost_smul_le {c : ℝ} (hc : 0 < c) (X : Matrix m n ℂ) :
    productCost ((c : ℂ) • X) ≤ c ^ (2 / 3 : ℝ) * productCost X := by
  obtain ⟨G, o, W, hX, hcost⟩ := exists_productFactorization X
  obtain ⟨v, hvdef⟩ : ∃ v : ℝ, v = c ^ (1 / 3 : ℝ) := ⟨_, rfl⟩
  have hv : 0 < v := hvdef ▸ Real.rpow_pos_of_pos hc _
  have hv3 : c = v ^ 3 := by
    rw [hvdef, ← Real.rpow_natCast, ← Real.rpow_mul hc.le]
    norm_num
  have hv2 : c ^ (2 / 3 : ℝ) = v ^ 2 := by
    rw [hvdef, ← Real.rpow_natCast, ← Real.rpow_mul hc.le]
    norm_num
  have hfac : (c : ℂ) • X = ((v : ℂ) • G) * diagonal ((v : ℂ) • o) * ((v : ℂ) • W)ᴴ := by
    rw [diagonal_smul, conjTranspose_smul, Complex.star_def, Complex.conj_ofReal]
    simp only [Matrix.smul_mul, Matrix.mul_smul, smul_smul]
    rw [← hX, hv3]
    push_cast
    congr 1
    ring
  have hb := productCost_le ((v : ℂ) • G) ((v : ℂ) • o) ((v : ℂ) • W)
  rw [← hfac, frobeniusSq_smul_eq, frobeniusSq_smul_eq, Complex.norm_real,
    Real.norm_of_nonneg hv.le] at hb
  have ho : ∑ u, ‖((v : ℂ) • o) u‖ ^ 2 = v ^ 2 * ∑ u, ‖o u‖ ^ 2 := by
    simp only [Pi.smul_apply, smul_eq_mul, norm_mul, Complex.norm_real,
      Real.norm_of_nonneg hv.le, mul_pow, Finset.mul_sum]
  rw [ho] at hb
  rw [hv2, ← hcost]
  linarith

example := productCost_smul_le (c := 1) one_pos (1 : Matrix (Fin 2) (Fin 2) ℂ)

/-- **The degree of homogeneity** (NV10): `productCost (c X) = c^{2/3} productCost X`. -/
theorem productCost_smul {c : ℝ} (hc : 0 < c) (X : Matrix m n ℂ) :
    productCost ((c : ℂ) • X) = c ^ (2 / 3 : ℝ) * productCost X := by
  refine le_antisymm (productCost_smul_le hc X) ?_
  have h := productCost_smul_le (inv_pos.mpr hc) ((c : ℂ) • X)
  rw [smul_smul, Complex.ofReal_inv, inv_mul_cancel₀ (by exact_mod_cast hc.ne'), one_smul,
    Real.inv_rpow hc.le] at h
  have hcp : 0 < c ^ (2 / 3 : ℝ) := Real.rpow_pos_of_pos hc _
  calc c ^ (2 / 3 : ℝ) * productCost X
      ≤ c ^ (2 / 3 : ℝ) * ((c ^ (2 / 3 : ℝ))⁻¹ * productCost ((c : ℂ) • X)) := by gcongr
    _ = productCost ((c : ℂ) • X) := by field_simp

example := productCost_smul (c := 1) one_pos (1 : Matrix (Fin 2) (Fin 2) ℂ)

end Transformer.Grokking.FlatSpectrum
