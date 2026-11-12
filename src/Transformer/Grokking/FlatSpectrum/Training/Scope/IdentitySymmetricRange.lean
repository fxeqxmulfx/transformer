module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentitySymmetricLimit

/-!
# The identity kernel: which projection the run from equal factors reaches

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statement 7; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size, `0 < λ < 1` and the bounds of
WD8, the limit `(U, U)` of the run from a pair `(U₀, U₀)`, of any width, has a product
`X = U Uᴴ` with `X U₀ = (1 − λ) U₀` (`exists_tendsto_decayRun_identity_self`). Here:

* the stack `[U₀; U₀]` has at most the rank of `U₀` (`rank_stack_self_le`), so by WD14
  `rank X ≤ rank U₀`, and `X` has the column space of `U₀`
  (`range_mul_conjTranspose_of_tendsto_decayRun_identity_self`): with `X² = (1 − λ) X` and
  `X` Hermitian, the product of the limit is `1 − λ` times the orthogonal projection onto the
  column space of the start;
* a run from equal factors, for any gradient field, whose limit has a product `c · 1` with
  `c ≠ 0` starts at a factor with a right inverse (`exists_mul_eq_one_of_tendsto_decayRun_self`);
* so the limit is a decayed minimum exactly when `U₀` has a right inverse
  (`isDecayedMin_iff_of_tendsto_decayRun_identity_self`): among the starts `V₀ = U₀` the
  exceptional ones are those without full row rank.

The witnesses are `shearFactor`, which is invertible, and `diag(1, 0)`, which is not.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {η lam κ ρ : ℝ}
  {U₀ U V : Matrix m k ℂ}

local notation "G₁" => fun X : Matrix m m ℂ => mask Set.univ (X - 1)

omit [DecidableEq m] in
/-- The stack `[U; U]` has at most the rank of `U`. -/
lemma rank_stack_self_le (U : Matrix m k ℂ) : (stack U U).rank ≤ U.rank :=
  rank_le_of_forall_mulVec_eq_zero fun z hz => by
    rw [stack, fromRows_mulVec, hz]
    ext (i | i) <;> simp

/-- **A run from equal factors that reaches a product of full rank starts at a factor with a
right inverse** (`review/flat_spectrum.md`, item WD16, statement 7; WD14, statement 2): for
every gradient field, step and weight decay, if the limit has the product `c · 1` with
`c ≠ 0`. -/
theorem exists_mul_eq_one_of_tendsto_decayRun_self {G : Matrix m m ℂ → Matrix m m ℂ} {c : ℝ}
    (hc : c ≠ 0) (hlim : Tendsto (decayRun G η lam U₀ U₀) atTop (𝓝 (U, V)))
    (hX : U * Vᴴ = c • 1) : ∃ R : Matrix k m ℂ, U₀ * R = 1 := by
  have h1 : (1 : Matrix m m ℂ) = U * Vᴴ * (c⁻¹ • 1) := by
    rw [hX, Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul, smul_smul, mul_inv_cancel₀ hc,
      one_smul]
  have hr1 : Fintype.card m ≤ (U * Vᴴ).rank := by
    have h := Matrix.rank_mul_le_left (U * Vᴴ) (c⁻¹ • (1 : Matrix m m ℂ))
    rwa [← h1, Matrix.rank_one] at h
  have hr : U₀.rank = Fintype.card m :=
    le_antisymm (Matrix.rank_le_card_height U₀)
      (hr1.trans ((rank_mul_conjTranspose_le_of_tendsto hlim).trans (rank_stack_self_le U₀)))
  exact Matrix.mulVec_surjective_iff_exists_right_inverse.1
    (Matrix.mulVec_surjective_iff_rank_eq_card.2 hr)

/-- **The product of the limit of a run from equal factors has the column space of the start**
(`review/flat_spectrum.md`, item WD16, statement 7): for the identity, `0 < λ < 1` and the
bounds of WD8. With `X² = (1 − λ) X` and `X` Hermitian, the product is `1 − λ` times the
orthogonal projection onto the column space of `U₀`. -/
theorem range_mul_conjTranspose_of_tendsto_decayRun_identity_self (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2)
    (hρ : κ ^ 2 ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : 2 * frobeniusSq U₀ ≤ ρ)
    (hlim : Tendsto (decayRun G₁ η lam U₀ U₀) atTop (𝓝 (U, V))) :
    LinearMap.range (U * Vᴴ).mulVecLin = LinearMap.range U₀.mulVecLin := by
  obtain ⟨U', hlim', -, h⟩ := exists_tendsto_decayRun_identity_self hη hlam hlam1 hκ0 hκ hρ hstep
    hdesc hw
  have hrank := (rank_mul_conjTranspose_le_of_tendsto hlim).trans (rank_stack_self_le U₀)
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (tendsto_nhds_unique hlim hlim')
  have hδ : (1 - lam) ≠ 0 := (sub_pos.2 hlam1).ne'
  have hle : LinearMap.range U₀.mulVecLin ≤ LinearMap.range (V * Vᴴ).mulVecLin := by
    rintro _ ⟨y, rfl⟩
    refine ⟨(1 - lam)⁻¹ • U₀ *ᵥ y, ?_⟩
    rw [Matrix.mulVecLin_apply, Matrix.mulVecLin_apply, Matrix.mulVec_smul,
      Matrix.mulVec_mulVec, h, Matrix.smul_mulVec, smul_smul, inv_mul_cancel₀ hδ, one_smul]
  exact (Submodule.eq_of_le_of_finrank_le hle hrank).symm

/-- **Among the starts with equal factors the exceptional ones are those without full row
rank** (`review/flat_spectrum.md`, item WD16, statement 7): for the identity, `0 < λ < 1` and
the bounds of WD8, the limit of the run from `(U₀, U₀)` is a decayed minimum exactly when `U₀`
has a right inverse. -/
theorem isDecayedMin_iff_of_tendsto_decayRun_identity_self (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2)
    (hρ : κ ^ 2 ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : 2 * frobeniusSq U₀ ≤ ρ)
    (hlim : Tendsto (decayRun G₁ η lam U₀ U₀) atTop (𝓝 (U, V))) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam U V ↔
      ∃ R : Matrix k m ℂ, U₀ * R = 1 := by
  refine ⟨fun h => ?_, fun ⟨R, hR⟩ => ?_⟩
  · exact exists_mul_eq_one_of_tendsto_decayRun_self (sub_pos.2 hlam1).ne' hlim
      ((isDecayedMin_identity_iff_of_tendsto hη hlam hlam1.le hlim).1 h)
  · obtain ⟨U', hlim', hmin⟩ := exists_tendsto_decayRun_identity_self_isDecayedMin hη hlam hlam1
      hκ0 hκ hρ hstep hdesc hw hR
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (tendsto_nhds_unique hlim hlim')
    exact hmin

/-! The witnesses, for the `2 × 2` identity with `λ = ½`, `κ = 2`, `ρ = 8` and `η = 1/200`:
the start `U₀ = V₀ = [[1, 1], [0, 1]]`, and the start `U₀ = V₀ = diag(1, 0)` of width two,
whose limit is not a decayed minimum. -/

example := rank_stack_self_le shearFactor

example : ∃ U : Matrix (Fin 2) (Fin 2) ℂ,
    Tendsto (decayRun (fun X : Matrix (Fin 2) (Fin 2) ℂ => mask Set.univ (X - 1)) (1 / 200)
      (1 / 2) shearFactor shearFactor) atTop (𝓝 (U, U)) ∧
    LinearMap.range (U * Uᴴ).mulVecLin = LinearMap.range shearFactor.mulVecLin ∧
    (IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (1 / 2) U U ↔
      ∃ R : Matrix (Fin 2) (Fin 2) ℂ, shearFactor * R = 1) := by
  have hκ : frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) ≤ (2 : ℝ) ^ 2 := by
    rw [frobeniusSq_one_fin_two]
    norm_num
  have hw : 2 * frobeniusSq shearFactor ≤ (8 : ℝ) := by
    rw [frobeniusSq_shearFactor]
    norm_num
  obtain ⟨U, hlim, -⟩ := exists_tendsto_decayRun_identity_self (η := 1 / 200) (lam := 1 / 2)
    (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hκ (by norm_num)
    (by norm_num) (by norm_num) hw
  exact ⟨U, hlim,
    range_mul_conjTranspose_of_tendsto_decayRun_identity_self (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) hκ (by norm_num) (by norm_num) (by norm_num) hw hlim,
    isDecayedMin_iff_of_tendsto_decayRun_identity_self (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) hκ (by norm_num) (by norm_num) (by norm_num) hw hlim⟩

example : ∃ U : Matrix (Fin 2) (Fin 2) ℂ,
    Tendsto (decayRun (fun X : Matrix (Fin 2) (Fin 2) ℂ => mask Set.univ (X - 1)) (1 / 200)
      (1 / 2) (Matrix.diagonal ![1, 0]) (Matrix.diagonal ![1, 0])) atTop (𝓝 (U, U)) ∧
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (1 / 2) U U := by
  have hκ : frobeniusSq (1 : Matrix (Fin 2) (Fin 2) ℂ) ≤ (2 : ℝ) ^ 2 := by
    rw [frobeniusSq_one_fin_two]
    norm_num
  have hw : 2 * frobeniusSq (Matrix.diagonal ![1, 0] : Matrix (Fin 2) (Fin 2) ℂ) ≤ (8 : ℝ) := by
    simp [frobeniusSq, Fin.sum_univ_two, Matrix.diagonal]
    norm_num
  obtain ⟨U, hlim, -⟩ := exists_tendsto_decayRun_identity_self (η := 1 / 200) (lam := 1 / 2)
    (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hκ (by norm_num)
    (by norm_num) (by norm_num) hw
  refine ⟨U, hlim, fun h => ?_⟩
  obtain ⟨R, hR⟩ := (isDecayedMin_iff_of_tendsto_decayRun_identity_self (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) hκ (by norm_num) (by norm_num) (by norm_num) hw
    hlim).1 h
  have h11 := congrFun (congrFun hR 1) 1
  simp [Matrix.mul_apply, Fin.sum_univ_two] at h11

example : ∃ R : Matrix (Fin 2) (Fin 2) ℂ, shearFactor * R = 1 := by
  obtain ⟨U, hlim, hmin⟩ := exists_tendsto_decayRun_identity_self_isDecayedMin (η := 1 / 200)
    (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
    (U₀ := shearFactor) (by rw [frobeniusSq_shearFactor]; norm_num) shearFactor_mul
  exact exists_mul_eq_one_of_tendsto_decayRun_self (by norm_num) hlim
    ((isDecayedMin_identity_iff_of_tendsto (by norm_num) (by norm_num) (by norm_num) hlim).1 hmin)

end Transformer.Grokking.FlatSpectrum.Training.Scope
