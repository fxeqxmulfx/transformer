module

public import Transformer.Grokking.FlatSpectrum.CycleSpectrum
public import Transformer.Grokking.FlatSpectrum.BlockBound
public import Mathlib.Analysis.CStarAlgebra.Matrix

/-!
# The sharp norm of the two-permutation certificate

Source: `review/flat_spectrum.md`, Theorem D, items 1 and 3, and the *Numerical check*: "The
certificate of item 3 for the diagonals `{0, 1}` has `‖Y − Q‖_op = cos(π/p)`". Item 3 bounds
`‖(Y − Q) v‖² ≤ (1 − 1/(4N²)) ‖v‖²` for `Y = (P_σ + P_τ)/2`, `τ σ⁻¹` one `N`-cycle. Read along
the cycle, `l ↦ σ⁻¹ (e l)`, `Y − Q` is the certificate of the diagonals `{0, 1}` of `ℤ/N`
(`permAvg_sub_meanProj_mulVec_symm_apply`), so by `CycleSpectrum`
`‖(Y − Q) v‖² ≤ cos²(π/N) ‖v‖²` (`sum_norm_sq_mulVec_permAvg_sub_le_cos`), with equality at
`v = ψ ∘ e⁻¹` for `N ≥ 2` (`sum_norm_sq_mulVec_permAvg_sub_stdAddChar`). So the L2 operator norm
is `cos(π/N)` (`l2_opNorm_permAvg_sub_meanProj`; for two diagonals whose offsets differ by a
unit, `l2_opNorm_permAvg_addRight_sub`), and `c (Y − Q)` is a contraction exactly when
`|c| cos(π/N) ≤ 1` (`isContraction_smul_permAvg_sub_iff`). For `N ≥ 3`, item 1 holds with
`c = 1/cos(π/N)` (`le_nuclearNorm_onesMatrix_add_of_cycle`), and no larger `c` makes
`c (Y − Q)` a contraction.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset Real

variable {n : Type*} [Fintype n] [DecidableEq n] {N : ℕ} [NeZero N]

/-- Read along the cycle, `Y − Q` is the certificate of the diagonals `{0, 1}` of `ℤ/N`:
`((Y − Q) v)(σ⁻¹ (e l)) = ((Y₀ − Q₀)(v ∘ e))(l)`. -/
lemma permAvg_sub_meanProj_mulVec_symm_apply {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) (v : n → ℂ) (l : ZMod N) :
    ((permAvg σ τ - meanProj n) *ᵥ v) (σ.symm (e l)) =
      ((permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) - meanProj (ZMod N)) *ᵥ (v ∘ e)) l := by
  have hcard : Fintype.card n = N := by rw [← Fintype.card_congr e, ZMod.card]
  simp only [sub_mulVec, Pi.sub_apply, permAvg_mulVec, Equiv.apply_symm_apply, he,
    Equiv.refl_apply, Equiv.coe_addRight, Function.comp_apply]
  congr 1
  simp only [meanProj, mulVec, dotProduct, of_apply, ← mul_sum, hcard, ZMod.card,
    Function.comp_apply, Equiv.sum_comp e v]

example := permAvg_sub_meanProj_mulVec_symm_apply (N := 3) (σ := Equiv.refl _)
  (τ := Equiv.addRight 1) (Equiv.refl _) (fun _ => rfl) ![1, 0, 0] 0

/-- **The sharp bound**: `‖(Y − Q) v‖² ≤ cos²(π/N) ‖v‖²` for `Y = (P_σ + P_τ)/2` when `τ σ⁻¹`
is one `N`-cycle. -/
theorem sum_norm_sq_mulVec_permAvg_sub_le_cos {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) (v : n → ℂ) :
    ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ v) i‖ ^ 2 ≤ cos (π / N) ^ 2 * ∑ i, ‖v i‖ ^ 2 := by
  have h1 : ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ v) i‖ ^ 2 = ∑ l, ‖((permAvg (Equiv.refl (ZMod N))
      (Equiv.addRight 1) - meanProj (ZMod N)) *ᵥ (v ∘ e)) l‖ ^ 2 := by
    rw [← Equiv.sum_comp (e.trans σ.symm)]
    simp only [Equiv.trans_apply, permAvg_sub_meanProj_mulVec_symm_apply e he]
  have h2 : ∑ i, ‖v i‖ ^ 2 = ∑ l, ‖(v ∘ e) l‖ ^ 2 := (Equiv.sum_comp e fun i => ‖v i‖ ^ 2).symm
  rw [h1, h2]
  exact sum_norm_sq_mulVec_cycleAvg_le (v ∘ e)

example := sum_norm_sq_mulVec_permAvg_sub_le_cos (N := 3) (σ := Equiv.refl _)
  (τ := Equiv.addRight 1) (Equiv.refl _) (fun _ => rfl) ![1, 0, 0]

/-- **Equality at `ψ ∘ e⁻¹`**: for `N ≥ 2` and `v i = ψ(e⁻¹ i)`, `‖v‖² = N` and
`‖(Y − Q) v‖² = cos²(π/N) N`. -/
theorem sum_norm_sq_mulVec_permAvg_sub_stdAddChar {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) (hN : 2 ≤ N) :
    ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ fun i => (ZMod.stdAddChar (e.symm i) : ℂ)) i‖ ^ 2 =
      cos (π / N) ^ 2 * N ∧ ∑ i, ‖(ZMod.stdAddChar (e.symm i) : ℂ)‖ ^ 2 = N := by
  have hv : (fun i => (ZMod.stdAddChar (e.symm i) : ℂ)) ∘ e = fun l => ZMod.stdAddChar l := by
    ext l
    simp
  have hnorm : ∑ l : ZMod N, ‖(ZMod.stdAddChar l : ℂ)‖ ^ 2 = N := by
    simp [ZMod.stdAddChar_apply, ZMod.card]
  have h1 : ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ fun i => (ZMod.stdAddChar (e.symm i) : ℂ)) i‖ ^ 2 =
      ∑ l, ‖((permAvg (Equiv.refl (ZMod N)) (Equiv.addRight 1) - meanProj (ZMod N)) *ᵥ
        ((fun i => (ZMod.stdAddChar (e.symm i) : ℂ)) ∘ e)) l‖ ^ 2 := by
    rw [← Equiv.sum_comp (e.trans σ.symm)]
    simp only [Equiv.trans_apply, permAvg_sub_meanProj_mulVec_symm_apply e he]
  refine ⟨?_, ?_⟩
  · rw [h1, hv, sum_norm_sq_mulVec_cycleAvg_stdAddChar hN, hnorm]
  · rw [← Equiv.sum_comp e]
    simp

example := sum_norm_sq_mulVec_permAvg_sub_stdAddChar (N := 3) (σ := Equiv.refl _)
  (τ := Equiv.addRight 1) (Equiv.refl _) (fun _ => rfl) (by norm_num)

open scoped Matrix.Norms.L2Operator in
/-- **The sharp norm**: `‖(P_σ + P_τ)/2 − J/N‖_op = cos(π/N)` when `τ σ⁻¹` is one `N`-cycle and
`N ≥ 2`. -/
theorem l2_opNorm_permAvg_sub_meanProj {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) (hN : 2 ≤ N) :
    ‖permAvg σ τ - meanProj n‖ = cos (π / N) := by
  have hc := cos_pi_div_nonneg hN
  have hN0 : (0 : ℝ) < N := by exact_mod_cast (show 0 < N by omega)
  rw [← l2_opNorm_toEuclideanCLM]
  set T := toEuclideanCLM (n := n) (𝕜 := ℂ) (permAvg σ τ - meanProj n)
  refine le_antisymm (T.opNorm_le_bound hc fun x => ?_) ?_
  · refine (sq_le_sq₀ (norm_nonneg _) (by positivity)).mp ?_
    rw [mul_pow, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq]
    simpa only [T, ofLp_toEuclideanCLM] using
      sum_norm_sq_mulVec_permAvg_sub_le_cos e he (WithLp.ofLp x)
  · obtain ⟨h1, h2⟩ := sum_norm_sq_mulVec_permAvg_sub_stdAddChar e he hN
    set x : EuclideanSpace ℂ n := WithLp.toLp 2 fun i => (ZMod.stdAddChar (e.symm i) : ℂ)
    have hx2 : ‖x‖ ^ 2 = N := by
      rw [EuclideanSpace.norm_sq_eq]
      simpa [x] using h2
    have hTx2 : ‖T x‖ ^ 2 = cos (π / N) ^ 2 * N := by
      rw [EuclideanSpace.norm_sq_eq]
      simpa only [T, x, ofLp_toEuclideanCLM, WithLp.ofLp_toLp] using h1
    have hxpos : 0 < ‖x‖ := by
      by_contra h
      have h0 : ‖x‖ = 0 := le_antisymm (not_lt.mp h) (norm_nonneg _)
      rw [h0] at hx2
      linarith
    have hTx : ‖T x‖ = cos (π / N) * ‖x‖ := by
      rw [← sq_eq_sq₀ (norm_nonneg _) (by positivity), mul_pow, hTx2, hx2]
    have hle := T.le_opNorm x
    rw [hTx] at hle
    exact le_of_mul_le_mul_right hle hxpos

open scoped Matrix.Norms.L2Operator in
example : ‖permAvg (Equiv.refl (ZMod 3)) (Equiv.addRight 1) - meanProj (ZMod 3)‖ =
    cos (π / (3 : ℕ)) :=
  l2_opNorm_permAvg_sub_meanProj (Equiv.refl _) (fun _ => rfl) (by norm_num)

/-- **The best certificate constant**: for `N ≥ 2`, `c (Y − Q)` is a contraction exactly when
`|c| cos(π/N) ≤ 1`. -/
theorem isContraction_smul_permAvg_sub_iff {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) (hN : 2 ≤ N) (c : ℂ) :
    IsContraction (c • (permAvg σ τ - meanProj n)) ↔ ‖c‖ * cos (π / N) ≤ 1 := by
  have hc := cos_pi_div_nonneg hN
  have hN0 : (0 : ℝ) < N := by exact_mod_cast (show 0 < N by omega)
  have hsmul (v : n → ℂ) : ∑ i, ‖((c • (permAvg σ τ - meanProj n)) *ᵥ v) i‖ ^ 2 =
      ‖c‖ ^ 2 * ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ v) i‖ ^ 2 := by
    simp only [smul_mulVec, Pi.smul_apply, smul_eq_mul, norm_mul, mul_pow, ← mul_sum]
  have hpos : 0 ≤ ‖c‖ * cos (π / N) := mul_nonneg (norm_nonneg c) hc
  constructor
  · intro h
    obtain ⟨h1, h2⟩ := sum_norm_sq_mulVec_permAvg_sub_stdAddChar e he hN
    have hle := h.sum_norm_sq_mulVec_le fun i => (ZMod.stdAddChar (e.symm i) : ℂ)
    rw [hsmul, h1, h2] at hle
    have hsq : (‖c‖ * cos (π / N)) ^ 2 ≤ 1 ^ 2 := by nlinarith
    exact (sq_le_sq₀ hpos zero_le_one).mp hsq
  · intro h
    refine isContraction_of_forall_sum_norm_sq_le fun v => ?_
    have hb := sum_norm_sq_mulVec_permAvg_sub_le_cos e he v
    have hV : 0 ≤ ∑ i, ‖v i‖ ^ 2 := sum_nonneg fun _ _ => sq_nonneg _
    have h2 : (‖c‖ * cos (π / N)) ^ 2 ≤ 1 := by nlinarith
    rw [hsmul]
    calc ‖c‖ ^ 2 * ∑ i, ‖((permAvg σ τ - meanProj n) *ᵥ v) i‖ ^ 2
        ≤ ‖c‖ ^ 2 * (cos (π / N) ^ 2 * ∑ i, ‖v i‖ ^ 2) :=
          mul_le_mul_of_nonneg_left hb (sq_nonneg _)
      _ = (‖c‖ * cos (π / N)) ^ 2 * ∑ i, ‖v i‖ ^ 2 := by ring
      _ ≤ ∑ i, ‖v i‖ ^ 2 := by nlinarith

example : IsContraction ((2 : ℂ) • (permAvg (Equiv.refl (ZMod 3)) (Equiv.addRight 1) -
    meanProj (ZMod 3))) := by
  rw [isContraction_smul_permAvg_sub_iff (Equiv.refl _) (fun _ => rfl) (by norm_num)]
  norm_num [cos_pi_div_three]

open scoped Matrix.Norms.L2Operator in
/-- **Two diagonals**: for `N ≥ 2` and `s₂ − s₁` a unit of `ℤ/N`, the certificate of the
diagonals `b = a + s₁` and `b = a + s₂` has `‖Y − J/N‖_op = cos(π/N)`. -/
theorem l2_opNorm_permAvg_addRight_sub {s₁ s₂ : ZMod N} (hs : IsUnit (s₂ - s₁)) (hN : 2 ≤ N) :
    ‖permAvg (Equiv.addRight s₁) (Equiv.addRight s₂) - meanProj (ZMod N)‖ = cos (π / N) :=
  l2_opNorm_permAvg_sub_meanProj hs.unit.mulRight (fun l => by
    simp only [Equiv.addRight_symm, Equiv.coe_addRight, Units.mulRight_apply, IsUnit.unit_spec]
    ring) hN

open scoped Matrix.Norms.L2Operator in
example : ‖permAvg (Equiv.addRight (0 : ZMod 7)) (Equiv.addRight 1) - meanProj (ZMod 7)‖ =
    cos (π / (7 : ℕ)) :=
  l2_opNorm_permAvg_addRight_sub (by simp) (by norm_num)

/-- **Theorem D, item 1, with the sharp constant** `c = 1/cos(π/N)`: if `Ω` holds the graphs of
`σ` and `τ`, `τ σ⁻¹` is one `N`-cycle and `N ≥ 3`, every `H` vanishing on `Ω` has
`N + (1 − cos(π/N)) ‖Π H Π‖_S1 ≤ ‖J + H‖_S1`, with `Π = 1 − J/N`. -/
theorem le_nuclearNorm_onesMatrix_add_of_cycle {σ τ : Equiv.Perm n} (e : ZMod N ≃ n)
    (he : ∀ l, τ (σ.symm (e l)) = e (l + 1)) (hN : 3 ≤ N) {Ω : Set (n × n)}
    (hσ : ∀ i, (i, σ i) ∈ Ω) (hτ : ∀ i, (i, τ i) ∈ Ω) {H : Matrix n n ℂ}
    (hH : ∀ ij ∈ Ω, H ij.1 ij.2 = 0) :
    N + (1 - cos (π / N)) * nuclearNorm ((1 - meanProj n) * H * (1 - meanProj n)) ≤
      nuclearNorm (onesMatrix n + H) := by
  have : Nonempty n := ⟨e 0⟩
  have hN' : (3 : ℝ) ≤ N := by exact_mod_cast hN
  have hN0 : (0 : ℝ) < N := by linarith
  have hcos : 0 < cos (π / N) := cos_pos_of_mem_Ioo
    ⟨by linarith [pi_pos, div_pos pi_pos hN0], div_lt_div_of_pos_left pi_pos two_pos (by linarith)⟩
  have hW := (isContraction_smul_permAvg_sub_iff e he (by omega) ((cos (π / N))⁻¹ : ℝ)).mpr
    (by rw [Complex.norm_real, norm_inv, Real.norm_of_nonneg hcos.le, inv_mul_cancel₀ hcos.ne'])
  have h := le_nuclearNorm_onesMatrix_add (permAvg_eq_zero hσ hτ) (sum_permAvg_row σ τ)
    (sum_permAvg_col σ τ) (inv_pos.mpr hcos) hW hH
  rwa [inv_inv, ← Fintype.card_congr e, ZMod.card] at h

example := le_nuclearNorm_onesMatrix_add_of_cycle (N := 3) (σ := Equiv.refl _)
  (τ := Equiv.addRight 1) (Equiv.refl _) (fun _ => rfl) le_rfl (Ω := Set.univ)
  (fun _ => Set.mem_univ _) (fun _ => Set.mem_univ _) (H := 0) fun _ _ => rfl

end Transformer.Grokking.FlatSpectrum
