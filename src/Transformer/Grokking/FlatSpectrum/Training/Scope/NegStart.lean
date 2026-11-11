module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.PaddedRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.RankStart

/-!
# The starts `V₀ = −U₀`: exceptional starts of full rank

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD14,
statement 7; the item "Width" of WD11.

`StackRank` shows that a start whose stack has a rank below that of every trained product is
in the exceptional set of WD10. The converse fails.

* Let the gradient field map Hermitian matrices to Hermitian matrices. A step from a pair
  `(U, −U)` gives a pair `(U', −U')` (`decayStep_neg`), so the run from `V₀ = −U₀` has
  `V_t = −U_t` at every step (`decayRun_neg`) and at a limit (`eq_neg_of_tendsto_decayRun_neg`).
  The stack `[U; −U]` has the rank of `U` (`rank_stack_neg`).
* For the squared loss on all cells of the `2 × 2` identity and `0 ≤ λ < 1`, a product
  `U (−U)ᴴ = −U Uᴴ` is not a trained product (`not_isNuclearMin_one_mul_neg`):
  `Re tr (−U Uᴴ) ≤ 0` gives `½ ‖−U Uᴴ − 1‖² ≥ 1`, above the value `2 λ − λ²` of `(1 − λ) · 1`.
* So the run from `V₀ = −U₀` does not converge to a decayed minimum, at any width and with any
  step (`not_isDecayedMin_one_of_neg`), and under the bounds of WD8 it converges to a
  stationary point that is not one (`exists_tendsto_decayRun_one_neg_not_isDecayedMin`).

With `U₀ = 1` of width two the stack `[1; −1]` has rank two, the rank of the trained products
(`RankStart`): the exceptional set of WD10 is not described by the rank of the stack.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] {η lam κ ρ : ℝ}
  {G : Matrix m m ℂ → Matrix m m ℂ}

/-- **A step keeps `V = −U`** (`review/flat_spectrum.md`, item WD14, statement 7), when the
gradient field maps Hermitian matrices to Hermitian matrices: the product `−U Uᴴ` is
Hermitian. -/
lemma decayStep_neg (hG : ∀ X : Matrix m m ℂ, X.IsHermitian → (G X).IsHermitian) (η lam : ℝ)
    (U : Matrix m k ℂ) :
    (decayStep G η lam (U, -U)).2 = -(decayStep G η lam (U, -U)).1 := by
  have hX : (U * (-U)ᴴ).IsHermitian := by
    rw [Matrix.IsHermitian, conjTranspose_mul, conjTranspose_conjTranspose, conjTranspose_neg,
      Matrix.neg_mul, Matrix.mul_neg]
  have hH : (G (U * (-U)ᴴ))ᴴ = G (U * (-U)ᴴ) := hG _ hX
  simp only [decayStep, hH, Matrix.mul_neg]
  module

/-- **The run from `V₀ = −U₀` has `V_t = −U_t` at every step** (`review/flat_spectrum.md`,
item WD14, statement 7). -/
lemma decayRun_neg (hG : ∀ X : Matrix m m ℂ, X.IsHermitian → (G X).IsHermitian) (η lam : ℝ)
    (U₀ : Matrix m k ℂ) (t : ℕ) :
    (decayRun G η lam U₀ (-U₀) t).2 = -(decayRun G η lam U₀ (-U₀) t).1 := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [decayRun_succ]
    have h := decayStep_neg hG η lam (decayRun G η lam U₀ (-U₀) t).1
    rw [← ih] at h
    exact h

/-- A limit of the run from `V₀ = −U₀` is a pair `(U, −U)`. -/
lemma eq_neg_of_tendsto_decayRun_neg
    (hG : ∀ X : Matrix m m ℂ, X.IsHermitian → (G X).IsHermitian) {U₀ U V : Matrix m k ℂ}
    (hlim : Tendsto (decayRun G η lam U₀ (-U₀)) atTop (𝓝 (U, V))) : V = -U := by
  have hc : Continuous fun W : Matrix m k ℂ × Matrix m k ℂ => W.1 + W.2 :=
    continuous_fst.add continuous_snd
  have h := tendsto_nhds_unique ((hc.tendsto (U, V)).comp hlim)
    (tendsto_const_nhds.congr fun t => by
      simp only [Function.comp_apply]
      rw [decayRun_neg hG, add_neg_cancel])
  exact eq_neg_of_add_eq_zero_right h

/-- The stack `[U; −U]` has the rank of `U`. -/
lemma rank_stack_neg (U : Matrix m k ℂ) : (stack U (-U)).rank = U.rank := by
  refine le_antisymm (rank_le_of_forall_mulVec_eq_zero fun z hz => ?_)
    (rank_le_of_forall_mulVec_eq_zero fun z hz => ?_)
  · rw [stack, fromRows_mulVec, hz, Matrix.neg_mulVec, hz]
    ext (i | i) <;> simp
  · rw [stack, fromRows_mulVec] at hz
    funext i
    exact congrFun hz (Sum.inl i)

/-- **A product `−U Uᴴ` is not a trained product of the `2 × 2` identity**
(`review/flat_spectrum.md`, item WD14, statement 7), for the squared loss on all cells and
`0 ≤ λ < 1`: its value is at least `1`, above the value `2 λ − λ²` of `(1 − λ) · 1`. -/
theorem not_isNuclearMin_one_mul_neg (h0 : 0 ≤ lam) (h1 : lam < 1) (U : Matrix (Fin 2) k ℂ) :
    ¬ IsNuclearMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam (U * (-U)ᴴ) := by
  intro hX
  have hup : fitLoss Set.univ 1 (U * (-U)ᴴ) + lam * nuclearNorm (U * (-U)ᴴ) ≤
      2 * lam - lam ^ 2 := by
    have h := hX ((√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) * (√(1 - lam) • 1)ᴴ)
    have h2 := mul_le_mul_of_nonneg_left (nuclearNorm_le_of_eq_mul
      (√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ))
      (√(1 - lam) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) rfl) h0
    have h3 := decayedLoss_sqrt_smul_one h1.le
    rw [decayedLoss] at h3
    linarith
  have hmask : mask Set.univ (U * (-U)ᴴ - 1) = U * (-U)ᴴ - 1 := by
    ext i j
    simp [mask]
  have hp : pairing (U * (-U)ᴴ) 1 = -frobeniusSq U := by
    rw [pairing_mul_right, conjTranspose_conjTranspose, Matrix.one_mul,
      show -U = (-1 : ℝ) • U by simp, pairing_smul_right, pairing_self]
    ring
  have hL : fitLoss Set.univ 1 (U * (-U)ᴴ) =
      (frobeniusSq (U * (-U)ᴴ) - 2 * pairing (U * (-U)ᴴ) 1 + 2) / 2 := by
    rw [fitLoss, hmask, frobeniusSq_sub, frobeniusSq_one_fin_two]
  have hn := mul_nonneg h0 (nuclearNorm_nonneg (U * (-U)ᴴ))
  have hf := frobeniusSq_nonneg (U * (-U)ᴴ)
  have hu := frobeniusSq_nonneg U
  rw [hL, hp] at hup
  nlinarith [sq_pos_of_pos (sub_pos.2 h1)]

/-- The gradient field of the squared loss on all cells of the `2 × 2` identity maps Hermitian
matrices to Hermitian matrices. -/
lemma isHermitian_mask_sub_one (X : Matrix (Fin 2) (Fin 2) ℂ) (hX : X.IsHermitian) :
    (mask Set.univ (X - 1)).IsHermitian := by
  have hmask : mask Set.univ (X - 1) = X - 1 := by
    ext i j
    simp [mask]
  rw [hmask]
  exact hX.sub isHermitian_one

/-- **The run from `V₀ = −U₀` does not reach a decayed minimum of the `2 × 2` identity**
(`review/flat_spectrum.md`, item WD14, statement 7): for the squared loss on all cells and
`0 ≤ λ < 1`, at every width and with every step, whatever the rank of the stack. -/
theorem not_isDecayedMin_one_of_neg (h0 : 0 ≤ lam) (h1 : lam < 1)
    {U₀ U V : Matrix (Fin 2) k ℂ}
    (hlim : Tendsto (decayRun (fun X => mask Set.univ (X - 1)) η lam U₀ (-U₀)) atTop
      (𝓝 (U, V))) :
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam U V := by
  rw [eq_neg_of_tendsto_decayRun_neg isHermitian_mask_sub_one hlim]
  exact fun h => not_isNuclearMin_one_mul_neg h0 h1 U (h.isNuclearMin h0)

/-- **Such a run converges, to a stationary point that is not a decayed minimum**
(`review/flat_spectrum.md`, item WD14, statement 7): for the squared loss on all cells of the
`2 × 2` identity, `0 < λ < 1` and the bounds of WD8 with `‖1‖ ≤ κ`, the run from a start
`(U₀, −U₀)` in the ball, of any width. -/
theorem exists_tendsto_decayRun_one_neg_not_isDecayedMin (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : 2 ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) {U₀ : Matrix (Fin 2) k ℂ}
    (hw : 2 * frobeniusSq U₀ ≤ ρ) :
    ∃ U V : Matrix (Fin 2) k ℂ,
      Tendsto (decayRun (fun X => mask Set.univ (X - 1)) η lam U₀ (-U₀)) atTop (𝓝 (U, V)) ∧
      lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0 ∧
      ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) lam U V := by
  have hw' : frobeniusSq U₀ + frobeniusSq (-U₀) ≤ ρ := by
    rw [show -U₀ = (-1 : ℝ) • U₀ by simp, frobeniusSq_smul]
    linarith
  obtain ⟨U, V, hlim, hstat⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) hη hlam hκ0
    (frobeniusSq_one_fin_two.trans_le hκ) hρ hstep hdesc hw'
  exact ⟨U, V, hlim, hstat, not_isDecayedMin_one_of_neg hlam.le hlam1 hlim⟩

/-! The witness: the start `U₀ = 1`, `V₀ = −1` of width two, whose stack has rank two, with
`λ = ½`, `κ = 2`, `ρ = 8` and `η = 1/200`. -/

example := isHermitian_mask_sub_one 1 isHermitian_one

example := decayStep_neg isHermitian_mask_sub_one (1 / 200) (1 / 2)
  (1 : Matrix (Fin 2) (Fin 2) ℂ)

example := decayRun_neg isHermitian_mask_sub_one (1 / 200) (1 / 2)
  (1 : Matrix (Fin 2) (Fin 2) ℂ) 3

example : (stack (1 : Matrix (Fin 2) (Fin 2) ℂ) (-1 : Matrix (Fin 2) (Fin 2) ℂ)).rank = 2 := by
  rw [rank_stack_neg, Matrix.rank_one, Fintype.card_fin]

example := not_isNuclearMin_one_mul_neg (lam := 1 / 2) (by norm_num) (by norm_num)
  (1 : Matrix (Fin 2) (Fin 2) ℂ)

example := exists_tendsto_decayRun_one_neg_not_isDecayedMin (k := Fin 2) (η := 1 / 200)
  (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (U₀ := 1)
  (by rw [frobeniusSq_one_fin_two]; norm_num)

example : ∃ U V : Matrix (Fin 2) (Fin 2) ℂ, V = -U ∧
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (1 / 2) U V := by
  obtain ⟨U, V, hlim, -, -⟩ := exists_tendsto_decayRun_one_neg_not_isDecayedMin (k := Fin 2)
    (η := 1 / 200) (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (U₀ := 1)
    (by rw [frobeniusSq_one_fin_two]; norm_num)
  exact ⟨U, V, eq_neg_of_tendsto_decayRun_neg isHermitian_mask_sub_one hlim,
    not_isDecayedMin_one_of_neg (by norm_num) (by norm_num) hlim⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope
