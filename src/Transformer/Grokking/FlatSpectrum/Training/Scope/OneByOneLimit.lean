module

public import Transformer.Grokking.FlatSpectrum.Training.Certificate
public import Transformer.Grokking.FlatSpectrum.Training.Scope.NegStart
public import Transformer.Grokking.FlatSpectrum.Training.Scope.OneByOneRun

/-!
# The `1 × 1` identity: the exceptional starts are `U₀ + V₀ = 0`

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD15,
statements 4 to 6 and the theorem; the item "Width" of WD11.

For the squared loss on the `1 × 1` identity:

* for `λ > 0` a stationary point other than the origin is a decayed minimum
  (`isDecayedMin_one_of_stationary`): `G V = −λ U` and `Ḡ U = −λ V` give `|G| = λ`, so the
  lifted matrix is positive semidefinite (WD1, WD2);
* for `0 ≤ λ < 1` a pair `(U, −U)` is not a decayed minimum
  (`not_isDecayedMin_one_fin_one_neg`): its weight-decayed loss is at least `½`, above the
  value `λ − ½ λ²` of the factorization `√(1 − λ) · √(1 − λ)`;
* so for `0 < λ < 1` and the bounds of WD8 with `κ ≥ 1`, at every width, the run from a start
  in the ball converges, and its limit is a decayed minimum exactly when `U₀ + V₀ ≠ 0`
  (`exists_tendsto_decayRun_one_isDecayedMin_iff`);
* for `λ ≥ 1` the origin is a decayed minimum too (`isDecayedMin_one_of_stationary_of_one_le`),
  so every run from the ball ends at one (`exists_tendsto_decayRun_one_isDecayedMin_of_one_le`).

So the exceptional starts are those with `U₀ + V₀ = 0` when `λ < 1`, and there are none when
`λ ≥ 1`; the step is the explicit one of WD8, and width one, which is `min(m, n)` and below
`m + n`, is included.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {k : Type*} [Fintype k] {η lam κ ρ : ℝ}

local notation "R₁" => decayRun (fun X : Matrix (Fin 1) (Fin 1) ℂ => mask Set.univ (X - 1))

variable {U₀ V₀ U V : Matrix (Fin 1) k ℂ}

/-- **A stationary point of the `1 × 1` identity other than the origin is a decayed minimum**
(`review/flat_spectrum.md`, item WD15, statement 4), for the squared loss and `λ > 0`. -/
theorem isDecayedMin_one_of_stationary (hlam : 0 < lam)
    (hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0)
    (hne : 0 < frobeniusSq U + frobeniusSq V) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) lam U V := by
  refine isDecayedMin_of_posSemidef (G := fun X => mask Set.univ (X - 1))
    (fitLoss_first_order Set.univ 1) hstat (lift_posSemidef_of_frobeniusSq_le hlam.le ?_)
  rw [lift_mul_stack] at hstat
  have h1 : mask Set.univ (U * Vᴴ - 1) * V + lam • U = 0 := by
    ext i j
    have := congrFun (congrFun hstat (Sum.inl i)) j
    simpa [stack] using this
  have h2 : (mask Set.univ (U * Vᴴ - 1))ᴴ * U + lam • V = 0 := by
    ext i j
    have := congrFun (congrFun hstat (Sum.inr i)) j
    simpa [stack] using this
  rw [fin_one_mul] at h1 h2
  generalize mask Set.univ (U * Vᴴ - 1) = G at h1 h2 ⊢
  have e1 : ‖G 0 0‖ ^ 2 * frobeniusSq V = lam ^ 2 * frobeniusSq U := by
    rw [← frobeniusSq_complex_smul, ← frobeniusSq_smul, eq_neg_of_add_eq_zero_left h1,
      show -(lam • U) = (-1 : ℝ) • (lam • U) by simp, frobeniusSq_smul]
    ring
  have e2 : ‖G 0 0‖ ^ 2 * frobeniusSq U = lam ^ 2 * frobeniusSq V := by
    have h3 : ‖Gᴴ 0 0‖ = ‖G 0 0‖ := by rw [conjTranspose_apply, norm_star]
    rw [← h3, ← frobeniusSq_complex_smul, ← frobeniusSq_smul, eq_neg_of_add_eq_zero_left h2,
      show -(lam • V) = (-1 : ℝ) • (lam • V) by simp, frobeniusSq_smul]
    ring
  have hG : frobeniusSq G = ‖G 0 0‖ ^ 2 := by simp [frobeniusSq]
  rw [hG]
  nlinarith

/-- **A pair `(U, −U)` is not a decayed minimum of the `1 × 1` identity**
(`review/flat_spectrum.md`, item WD15, statement 5), for the squared loss and `0 ≤ λ < 1`. -/
lemma not_isDecayedMin_one_fin_one_neg (h0 : 0 ≤ lam) (h1 : lam < 1) (U : Matrix (Fin 1) k ℂ) :
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) lam U (-U) := by
  intro h
  have hcomp := h 1 (√(1 - lam) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) (√(1 - lam) • 1)
  have hs : √(1 - lam) * √(1 - lam) = 1 - lam := Real.mul_self_sqrt (by linarith)
  have hmul : (√(1 - lam) • (1 : Matrix (Fin 1) (Fin 1) ℂ)) * (√(1 - lam) • 1)ᴴ - 1 =
      (-lam) • (1 : Matrix (Fin 1) (Fin 1) ℂ) := by
    rw [conjTranspose_smul, conjTranspose_one, star_trivial, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.one_mul, smul_smul, hs]
    module
  have hmask : ∀ X : Matrix (Fin 1) (Fin 1) ℂ, mask Set.univ X = X := fun X => by
    ext i j
    simp [mask]
  have hp : pairing (U * (-U)ᴴ) 1 = -frobeniusSq U := by
    rw [pairing_mul_right, conjTranspose_conjTranspose, Matrix.one_mul,
      show -U = (-1 : ℝ) • U by simp, pairing_smul_right, pairing_self]
    ring
  rw [decayedLoss, decayedLoss, fitLoss, fitLoss, hmul, hmask, hmask, frobeniusSq_smul,
    frobeniusSq_smul, frobeniusSq_sub, hp, frobeniusSq_one_fin_one, sq √(1 - lam), hs] at hcomp
  have hf := frobeniusSq_nonneg (U * (-U)ᴴ)
  have hu := frobeniusSq_nonneg U
  have hu' := frobeniusSq_nonneg (-U)
  nlinarith [sq_pos_of_pos (sub_pos.2 h1), mul_nonneg h0 hu, mul_nonneg h0 hu']

/-- **The `1 × 1` identity: the run reaches a decayed minimum exactly from the starts with
`U₀ + V₀ ≠ 0`** (`review/flat_spectrum.md`, item WD15, theorem): for the squared loss,
`0 < λ < 1` and the bounds of WD8 with `κ ≥ 1`, at every width, the run from a start in the
ball converges, and its limit is a decayed minimum if and only if `U₀ + V₀ ≠ 0`. -/
theorem exists_tendsto_decayRun_one_isDecayedMin_iff (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : 1 ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    ∃ U V : Matrix (Fin 1) k ℂ,
      Tendsto (R₁ η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
      (IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) lam U V ↔
        U₀ + V₀ ≠ 0) := by
  obtain ⟨U, V, hlim, hstat⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) hη hlam hκ0
    (frobeniusSq_one_fin_one.trans_le hκ) hρ hstep hdesc hw
  refine ⟨U, V, hlim, fun hmin h0 => ?_, fun h0 => isDecayedMin_one_of_stationary hlam hstat
    (frobeniusSq_pos_of_tendsto_decayRun_one hη hlam hlam1 hκ0 hκ hρ hstep hdesc hw h0 hlim)⟩
  have hV : V₀ = -U₀ := eq_neg_of_add_eq_zero_right h0
  subst hV
  have hG : ∀ X : Matrix (Fin 1) (Fin 1) ℂ, X.IsHermitian →
      (mask Set.univ (X - 1)).IsHermitian := fun X hX => by
    rw [show mask Set.univ (X - 1) = X - 1 by ext i j; simp [mask]]
    exact hX.sub isHermitian_one
  rw [eq_neg_of_tendsto_decayRun_neg hG hlim] at hmin
  exact not_isDecayedMin_one_fin_one_neg hlam.le hlam1 U hmin

/-- **For `λ ≥ 1` every stationary point of the `1 × 1` identity is a decayed minimum**
(`review/flat_spectrum.md`, item WD15, statement 6): at the origin `|G| = 1 ≤ λ`. -/
theorem isDecayedMin_one_of_stationary_of_one_le (hlam : 1 ≤ lam)
    (hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) lam U V := by
  have hU := frobeniusSq_nonneg U
  have hV := frobeniusSq_nonneg V
  rcases (add_nonneg hU hV).eq_or_lt with h | h
  · obtain rfl : U = 0 := eq_zero_of_frobeniusSq_eq_zero (by linarith)
    obtain rfl : V = 0 := eq_zero_of_frobeniusSq_eq_zero (by linarith)
    refine isDecayedMin_of_posSemidef (G := fun X => mask Set.univ (X - 1))
      (fitLoss_first_order Set.univ 1) hstat
      (lift_posSemidef_of_frobeniusSq_le (by linarith) ?_)
    rw [show mask Set.univ ((0 : Matrix (Fin 1) k ℂ) * (0 : Matrix (Fin 1) k ℂ)ᴴ - 1) =
      (-1 : ℝ) • (1 : Matrix (Fin 1) (Fin 1) ℂ) by ext i j; simp [mask], frobeniusSq_smul,
      frobeniusSq_one_fin_one]
    nlinarith
  · exact isDecayedMin_one_of_stationary (by linarith) hstat h

/-- **The `1 × 1` identity with `λ ≥ 1`: no start is exceptional** (`review/flat_spectrum.md`,
item WD15, statement 6): under the bounds of WD8 with `κ ≥ 1`, at every width, the run from a
start in the ball converges to a decayed minimum. -/
theorem exists_tendsto_decayRun_one_isDecayedMin_of_one_le (hη : 0 < η) (hlam : 1 ≤ lam)
    (hκ0 : 0 ≤ κ) (hκ : 1 ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    ∃ U V : Matrix (Fin 1) k ℂ,
      Tendsto (R₁ η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
      IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) lam U V := by
  obtain ⟨U, V, hlim, hstat⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) hη
    (by linarith) hκ0 (frobeniusSq_one_fin_one.trans_le hκ) hρ hstep hdesc hw
  exact ⟨U, V, hlim, isDecayedMin_one_of_stationary_of_one_le hlam hstat⟩

/-! The witnesses, of width one with `κ = 1`, `ρ = 4` and `η = 1/100`: for `λ = ½` the start
`u₀ = v₀ = 1`, whose run ends at a decayed minimum, and the start `(1, −1)`, whose run does
not; for `λ = 1` the origin and the run from `(1, −1)`. -/

example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ,
    IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 1) (Fin 1) ℂ)) (1 / 2) U V := by
  have hw : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) +
      frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) ≤ 4 := by rw [frobeniusSq_one_fin_one]; norm_num
  obtain ⟨U, V, hlim, hstat⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) (K := 1)
    (η := 1 / 100) (lam := 1 / 2) (κ := 1) (ρ := 4) (by norm_num) (by norm_num) (by norm_num)
    (by rw [frobeniusSq_one_fin_one]; norm_num) (by norm_num) (by norm_num) (by norm_num) hw
  exact ⟨U, V, isDecayedMin_one_of_stationary (by norm_num) hstat
    (frobeniusSq_pos_of_tendsto_decayRun_one (κ := 1) (ρ := 4) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hw
      one_add_one_fin_one_ne_zero hlim)⟩

example := not_isDecayedMin_one_fin_one_neg (lam := 1 / 2) (by norm_num) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℂ)

example := exists_tendsto_decayRun_one_isDecayedMin_iff (k := Fin 1) (η := 1 / 100)
  (lam := 1 / 2) (κ := 1) (ρ := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (U₀ := 1) (V₀ := 1)
  (by rw [frobeniusSq_one_fin_one]; norm_num)

example := exists_tendsto_decayRun_one_isDecayedMin_iff (k := Fin 1) (η := 1 / 100)
  (lam := 1 / 2) (κ := 1) (ρ := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (U₀ := 1) (V₀ := -1)
  (by simp [frobeniusSq]; norm_num)

example := isDecayedMin_one_of_stationary_of_one_le (lam := 1) le_rfl
  (U := (0 : Matrix (Fin 1) (Fin 1) ℂ)) (V := 0) (by simp [stack])

example := exists_tendsto_decayRun_one_isDecayedMin_of_one_le (k := Fin 1) (η := 1 / 100)
  (lam := 1) (κ := 1) (ρ := 4) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (U₀ := 1) (V₀ := -1) (by simp [frobeniusSq]; norm_num)

end Transformer.Grokking.FlatSpectrum.Training.Scope
