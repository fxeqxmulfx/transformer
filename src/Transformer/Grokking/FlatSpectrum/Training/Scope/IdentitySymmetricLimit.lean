module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentitySymmetric

/-!
# The identity kernel: the limit of the run from equal factors

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statement 7; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size, `0 < λ < 1` and the bounds of
WD8, the run from a pair `(U₀, U₀)`, of any width, converges to a pair `(U, U)` whose product
`X = U Uᴴ` has `X U₀ = (1 − λ) U₀` (`exists_tendsto_decayRun_identity_self`):

* `X` commutes with `U₀ U₀ᴴ`, as every `U_t U_tᴴ` does;
* `E = 1 − X / (1 − λ)` is a Hermitian idempotent with `E U = 0`, by `X² = (1 − λ) X` and
  `X U = (1 − λ) U`, so `E U_t` is the run from `(E U₀, E U₀)` and converges to the origin;
* a run converges to the origin only from a start whose factors sum to zero
  (`tendsto_decayRun_identity_zero_iff`), so `E U₀ = 0`.

So the product of the limit is `1 − λ` on the columns of `U₀`, and when `U₀` has a right
inverse the limit is a decayed minimum (`exists_tendsto_decayRun_identity_self_isDecayedMin`):
these are starts of the width of the identity, whose stack has half of the full row rank.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {η lam κ ρ : ℝ}
  {U₀ U : Matrix m k ℂ}

local notation "G₁" => fun X : Matrix m m ℂ => mask Set.univ (X - 1)

/-- **The limit of a run from equal factors fixes the columns of the start**
(`review/flat_spectrum.md`, item WD16, statement 7): for the
identity, `0 < λ < 1` and the bounds of WD8, the run from `(U₀, U₀)` converges to a pair
`(U, U)` with `U Uᴴ U = (1 − λ) U` and `U Uᴴ U₀ = (1 − λ) U₀`. -/
theorem exists_tendsto_decayRun_identity_self (hη : 0 < η) (hlam : 0 < lam) (hlam1 : lam < 1)
    (hκ0 : 0 ≤ κ) (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : 2 * frobeniusSq U₀ ≤ ρ) :
    ∃ U : Matrix m k ℂ,
      Tendsto (decayRun G₁ η lam U₀ U₀) atTop (𝓝 (U, U)) ∧ U * Uᴴ * U = (1 - lam) • U ∧
      U * Uᴴ * U₀ = (1 - lam) • U₀ := by
  obtain ⟨U, hlim, h⟩ := exists_tendsto_decayRun_identity (U₀ := U₀) (V₀ := U₀) hη hlam hκ0 hκ
    hρ hstep hdesc (by linarith)
  refine ⟨U, hlim, h, ?_⟩
  have hδ : (1 - lam) ≠ 0 := (sub_pos.2 hlam1).ne'
  obtain ⟨E, hEdef⟩ : ∃ E : Matrix m m ℂ, E = 1 - (1 - lam)⁻¹ • (U * Uᴴ) := ⟨_, rfl⟩
  have hXX : U * Uᴴ * (U * Uᴴ) = (1 - lam) • (U * Uᴴ) := by
    rw [← Matrix.mul_assoc, h, Matrix.smul_mul]
  have hXH : (U * Uᴴ)ᴴ = U * Uᴴ := by rw [conjTranspose_mul, conjTranspose_conjTranspose]
  have hEH : Eᴴ = E := by
    rw [hEdef, conjTranspose_sub, conjTranspose_one, conjTranspose_smul, hXH, star_trivial]
  have hEE : E * E = E := by
    rw [hEdef, Matrix.sub_mul, Matrix.one_mul, Matrix.mul_sub, Matrix.mul_one, Matrix.smul_mul,
      Matrix.mul_smul, hXX, smul_smul, smul_smul,
      show (1 - lam)⁻¹ * (1 - lam)⁻¹ * (1 - lam) = (1 - lam)⁻¹ by field_simp]
    module
  have hEU : E * U = 0 := by
    rw [hEdef, Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, h, smul_smul,
      inv_mul_cancel₀ hδ, one_smul, sub_self]
  -- the run in terms of the step of equal factors
  have hrun : ∀ W₀ : Matrix m k ℂ, decayRun G₁ η lam W₀ W₀ =
      fun t => ((symStep η lam)^[t] W₀, (symStep η lam)^[t] W₀) := fun W₀ =>
    funext fun t => decayRun_identity_self η lam W₀ t
  rw [hrun] at hlim
  have hlimU : Tendsto (fun t => (symStep η lam)^[t] U₀) atTop (𝓝 U) :=
    (continuous_fst.tendsto (U, U)).comp hlim
  have hlimX : Tendsto (fun t => (symStep η lam)^[t] U₀ * ((symStep η lam)^[t] U₀)ᴴ) atTop
      (𝓝 (U * Uᴴ)) := (continuous_mul_conjTranspose.tendsto (U, U)).comp hlim
  -- the limit commutes with the Gram matrix of the start
  have hcX : Commute (U₀ * U₀ᴴ) (U * Uᴴ) := by
    have h1 : Tendsto (fun t => U₀ * U₀ᴴ * ((symStep η lam)^[t] U₀ *
        ((symStep η lam)^[t] U₀)ᴴ)) atTop (𝓝 (U₀ * U₀ᴴ * (U * Uᴴ))) :=
      ((continuous_const.matrix_mul continuous_id).tendsto (U * Uᴴ)).comp hlimX
    have h2 : Tendsto (fun t => (symStep η lam)^[t] U₀ * ((symStep η lam)^[t] U₀)ᴴ *
        (U₀ * U₀ᴴ)) atTop (𝓝 (U * Uᴴ * (U₀ * U₀ᴴ))) :=
      ((continuous_id.matrix_mul continuous_const).tendsto (U * Uᴴ)).comp hlimX
    have h3 : (fun t => U₀ * U₀ᴴ * ((symStep η lam)^[t] U₀ * ((symStep η lam)^[t] U₀)ᴴ)) =
        fun t => (symStep η lam)^[t] U₀ * ((symStep η lam)^[t] U₀)ᴴ * (U₀ * U₀ᴴ) :=
      funext fun t => commute_iterate_symStep (Commute.refl (U₀ * U₀ᴴ)) η lam t
    rw [h3] at h1
    exact tendsto_nhds_unique h1 h2
  have hcE : Commute E (U₀ * U₀ᴴ) := by
    rw [hEdef]
    exact (Commute.one_left _).sub_left (hcX.symm.smul_left _)
  -- the projected run is a run, and it converges to the origin
  have hlimE : Tendsto (fun t => (symStep η lam)^[t] (E * U₀)) atTop (𝓝 0) := by
    have h1 : Tendsto (fun t => E * (symStep η lam)^[t] U₀) atTop (𝓝 (E * U)) :=
      ((continuous_const.matrix_mul continuous_id).tendsto U).comp hlimU
    rw [hEU] at h1
    exact h1.congr fun t => mul_iterate_symStep hEE hEH hcE η lam t
  have hball : frobeniusSq (E * U₀) + frobeniusSq (E * U₀) ≤ ρ := by
    have := frobeniusSq_mul_le_of_idempotent hEE hEH U₀
    linarith
  have hzero : E * U₀ + E * U₀ = 0 :=
    (tendsto_decayRun_identity_zero_iff hη hlam hlam1 hκ0 hκ hρ hstep hdesc hball).1 (by
      rw [hrun]
      exact hlimE.prodMk_nhds hlimE)
  have hEU₀ : E * U₀ = 0 := by
    ext i j
    have := congrFun (congrFun hzero i) j
    simpa using this
  rw [hEdef, Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul, sub_eq_zero] at hEU₀
  conv_rhs => rw [hEU₀]
  rw [smul_smul, mul_inv_cancel₀ hδ, one_smul]

/-- **A run from equal factors of full row rank converges to a decayed minimum**
(`review/flat_spectrum.md`, item WD16, statement 7): for the
identity, `0 < λ < 1` and the bounds of WD8, when `U₀` has a right inverse. -/
theorem exists_tendsto_decayRun_identity_self_isDecayedMin (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2)
    (hρ : κ ^ 2 ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : 2 * frobeniusSq U₀ ≤ ρ)
    {R : Matrix k m ℂ} (hR : U₀ * R = 1) :
    ∃ U : Matrix m k ℂ,
      Tendsto (decayRun G₁ η lam U₀ U₀) atTop (𝓝 (U, U)) ∧
      IsDecayedMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam U U := by
  obtain ⟨U, hlim, -, h⟩ := exists_tendsto_decayRun_identity_self hη hlam hlam1 hκ0 hκ hρ hstep
    hdesc hw
  refine ⟨U, hlim, (isDecayedMin_identity_iff_of_tendsto hη hlam hlam1.le hlim).2 ?_⟩
  calc U * Uᴴ = U * Uᴴ * (U₀ * R) := by rw [hR, Matrix.mul_one]
    _ = (1 - lam) • 1 := by rw [← Matrix.mul_assoc, h, Matrix.smul_mul, hR]

/-! The witness: the start `U₀ = V₀ = [[1, 1], [0, 1]]` of the `2 × 2` identity, with `λ = ½`,
`κ = 2`, `ρ = 8` and `η = 1/200`. -/

example := exists_tendsto_decayRun_identity_self (η := 1 / 200) (lam := 1 / 2) (κ := 2) (ρ := 8)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
  (U₀ := shearFactor) (by rw [frobeniusSq_shearFactor]; norm_num)

example := exists_tendsto_decayRun_identity_self_isDecayedMin (η := 1 / 200) (lam := 1 / 2)
  (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
  (U₀ := shearFactor) (by rw [frobeniusSq_shearFactor]; norm_num) shearFactor_mul

end Transformer.Grokking.FlatSpectrum.Training.Scope
