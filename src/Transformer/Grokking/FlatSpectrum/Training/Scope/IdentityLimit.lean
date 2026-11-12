module

public import Transformer.Grokking.FlatSpectrum.Training.SquaredRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityMin
public import Transformer.Grokking.FlatSpectrum.Training.Scope.NegStart

/-!
# The identity kernel: the limits of the runs

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statement 5 and the theorem; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size, with factors of any width and
`λ > 0`:

* a limit of a run, with any step `η > 0`, is a pair `(U, U)` with `U Uᴴ U = (1 − λ) U`
  (`eq_and_mul_of_tendsto_decayRun_identity`), and for `λ ≤ 1` it is a decayed minimum exactly
  when `U Uᴴ = (1 − λ) · 1` (`isDecayedMin_identity_iff_of_tendsto`);
* under the bounds of WD8 the run from a start in the ball converges to such a pair
  (`exists_tendsto_decayRun_identity`); at a width below the size of the identity and
  `λ < 1` the limit is not a decayed minimum (`not_isDecayedMin_identity_of_card_lt`);
* for `λ ≥ 1` the run converges to the origin (`tendsto_decayRun_identity_zero`), a decayed
  minimum;
* the run from `V₀ = −U₀` converges to the origin (`tendsto_decayRun_identity_neg`), for every
  `λ > 0`: its limit is a pair `(U, −U)` (WD14) and a pair `(U, U)`.

So the product of the limit of a run is `1 − λ` times an orthogonal projection, of a rank at
most that of the stack of the start (WD14). Which projection a run reaches is not determined.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {η lam κ ρ : ℝ}
  {U₀ V₀ U V : Matrix m k ℂ}

local notation "G₁" => fun X : Matrix m m ℂ => mask Set.univ (X - 1)

/-- **A limit of a run on the identity has equal factors** (`review/flat_spectrum.md`, item
WD16, statement 5), for every step `η > 0` and `λ > 0`: `V = U` and `U Uᴴ U = (1 − λ) U`. -/
theorem eq_and_mul_of_tendsto_decayRun_identity (hη : 0 < η) (hlam : 0 < lam)
    (hlim : Tendsto (decayRun G₁ η lam U₀ V₀) atTop (𝓝 (U, V))) :
    V = U ∧ U * Uᴴ * U = (1 - lam) • U := by
  have hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0 :=
    lift_mul_stack_eq_zero_of_tendsto_decayRun (G := G₁) (continuous_mask_sub Set.univ 1) hη hlim
  exact eq_and_mul_of_stationary_identity hlam hstat

/-- **A limit of a run on the identity is a decayed minimum exactly when its product is
`(1 − λ) · 1`** (`review/flat_spectrum.md`, item WD16, statement 5), for `0 < λ ≤ 1`. -/
theorem isDecayedMin_identity_iff_of_tendsto (hη : 0 < η) (hlam : 0 < lam) (hlam1 : lam ≤ 1)
    (hlim : Tendsto (decayRun G₁ η lam U₀ V₀) atTop (𝓝 (U, V))) :
    IsDecayedMin (fitLoss Set.univ (1 : Matrix m m ℂ)) lam U V ↔ U * Vᴴ = (1 - lam) • 1 := by
  have hstat : lift lam (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0 :=
    lift_mul_stack_eq_zero_of_tendsto_decayRun (G := G₁) (continuous_mask_sub Set.univ 1) hη hlim
  exact isDecayedMin_identity_iff_of_stationary hlam.le hlam1 hstat

/-- **The runs on the identity converge to pairs of equal factors** (`review/flat_spectrum.md`,
item WD16, theorem): for `λ > 0` and the bounds of WD8 with `‖1‖ ≤ κ`, at every width, the run
from a start in the ball converges to a pair `(U, U)` with `U Uᴴ U = (1 − λ) U`. -/
theorem exists_tendsto_decayRun_identity (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    ∃ U : Matrix m k ℂ,
      Tendsto (decayRun G₁ η lam U₀ V₀) atTop (𝓝 (U, U)) ∧ U * Uᴴ * U = (1 - lam) • U := by
  obtain ⟨U, V, hlim, hstat⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) hη hlam hκ0 hκ
    hρ hstep hdesc hw
  obtain ⟨rfl, h⟩ := eq_and_mul_of_stationary_identity hlam hstat
  exact ⟨V, hlim, h⟩

/-- **For `λ ≥ 1` every run on the identity converges to the origin**
(`review/flat_spectrum.md`, item WD16, theorem), under the bounds of WD8. -/
theorem tendsto_decayRun_identity_zero (hη : 0 < η) (hlam : 1 ≤ lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    Tendsto (decayRun G₁ η lam U₀ V₀) atTop (𝓝 (0, 0)) := by
  obtain ⟨U, V, hlim, hstat⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) hη
    (by linarith) hκ0 hκ hρ hstep hdesc hw
  obtain ⟨rfl, rfl⟩ := eq_zero_of_stationary_identity hlam hstat
  exact hlim

/-- **The run from `V₀ = −U₀` on the identity converges to the origin**
(`review/flat_spectrum.md`, item WD16, theorem; WD14, statement 7), for `λ > 0` and the bounds
of WD8. -/
theorem tendsto_decayRun_identity_neg (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : 2 * frobeniusSq U₀ ≤ ρ) :
    Tendsto (decayRun G₁ η lam U₀ (-U₀)) atTop (𝓝 (0, 0)) := by
  have hw' : frobeniusSq U₀ + frobeniusSq (-U₀) ≤ ρ := by
    rw [show -U₀ = (-1 : ℝ) • U₀ by simp, frobeniusSq_smul]
    linarith
  obtain ⟨U, hlim, -⟩ := exists_tendsto_decayRun_identity hη hlam hκ0 hκ hρ hstep hdesc hw'
  have hG : ∀ X : Matrix m m ℂ, X.IsHermitian → (mask Set.univ (X - 1)).IsHermitian :=
    fun X hX => by
      rw [mask_univ]
      exact hX.sub isHermitian_one
  have hU : U = 0 := by
    have h := eq_neg_of_tendsto_decayRun_neg hG hlim
    ext i j
    exact CharZero.eq_neg_self_iff.mp (congrFun (congrFun h i) j)
  rwa [hU] at hlim

/-! The witnesses, for the `2 × 2` identity with `κ = 2`, `ρ = 8` and `η = 1/200`: the run of
width one from `u₀ = v₀ = (1, 1)` with `λ = ½`, whose limit is not a decayed minimum; the
same run with `λ = 1`; and the run of width two from `U₀ = 1`, `V₀ = −1` with `λ = ½`. -/

example : ∃ u : Matrix (Fin 2) (Fin 1) ℂ,
    Tendsto (decayRun (fun X : Matrix (Fin 2) (Fin 2) ℂ => mask Set.univ (X - 1)) (1 / 200)
      (1 / 2) (of fun _ _ => 1) (of fun _ _ => 1)) atTop (𝓝 (u, u)) ∧
    u * uᴴ * u = (1 - 1 / 2 : ℝ) • u ∧
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (1 / 2) u u := by
  obtain ⟨u, hlim, h⟩ := exists_tendsto_decayRun_identity (m := Fin 2) (k := Fin 1)
    (η := 1 / 200) (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num)
    (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
    (U₀ := of fun _ _ => 1) (V₀ := of fun _ _ => 1) (by simp [frobeniusSq]; norm_num)
  exact ⟨u, hlim, h, not_isDecayedMin_identity_of_card_lt (by norm_num) (by norm_num)
    (by simp) u u⟩

example (U V : Matrix (Fin 2) (Fin 1) ℂ)
    (hlim : Tendsto (decayRun (fun X : Matrix (Fin 2) (Fin 2) ℂ => mask Set.univ (X - 1))
      (1 / 200) (1 / 2) (of fun _ _ => 1) (of fun _ _ => 1)) atTop (𝓝 (U, V))) :
    V = U ∧ U * Uᴴ * U = (1 - 1 / 2 : ℝ) • U ∧
      (IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (1 / 2) U V ↔
        U * Vᴴ = (1 - 1 / 2 : ℝ) • 1) :=
  ⟨(eq_and_mul_of_tendsto_decayRun_identity (by norm_num) (by norm_num) hlim).1,
    (eq_and_mul_of_tendsto_decayRun_identity (by norm_num) (by norm_num) hlim).2,
    isDecayedMin_identity_iff_of_tendsto (by norm_num) (by norm_num) (by norm_num) hlim⟩

example := tendsto_decayRun_identity_zero (m := Fin 2) (k := Fin 1) (η := 1 / 200)
  (lam := 1) (κ := 2) (ρ := 8) (by norm_num) le_rfl (by norm_num)
  (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
  (U₀ := of fun _ _ => 1) (V₀ := of fun _ _ => 1) (by simp [frobeniusSq]; norm_num)

example := tendsto_decayRun_identity_neg (m := Fin 2) (k := Fin 2) (η := 1 / 200)
  (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num)
  (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
  (U₀ := 1) (by rw [frobeniusSq_one_fin_two]; norm_num)

end Transformer.Grokking.FlatSpectrum.Training.Scope
