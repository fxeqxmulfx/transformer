module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityOrigin

/-!
# The identity kernel: the runs that converge to the origin

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD16,
statement 6 and the theorem; the item "Width" of WD11.

For the squared loss on all cells of the identity of any size, `0 < λ`, and the bounds of WD8,
the run from a start in the ball `‖U₀‖² + ‖V₀‖² ≤ ρ`, of any width:

* stays in the ball (`frobeniusSq_decayRun_identity_le`, WD6);
* has `U_t + V_t ≠ 0` at every step when `U₀ + V₀ ≠ 0`
  (`frobeniusSq_add_decayRun_identity_pos`), by `‖P'‖² ≥ η² ‖P‖²` in the ball;
* for `λ < 1` does not converge to the origin when `U₀ + V₀ ≠ 0`
  (`frobeniusSq_pos_of_tendsto_decayRun_identity`): otherwise the norms tend to `0`, from some
  step on `‖U_t + V_t‖` does not decrease, and it is positive;
* so for `0 < λ < 1` it converges to the origin exactly when `U₀ + V₀ = 0`
  (`tendsto_decayRun_identity_zero_iff`): its limit is a pair `(U, U)` with
  `U Uᴴ U = (1 − λ) U`, and `U = 0` exactly when `U₀ + V₀ = 0`
  (`exists_tendsto_decayRun_identity_eq_zero_iff`).

So the starts attracted to the origin are those of the subspace `U₀ + V₀ = 0`, at every size
and width. For the `1 × 1` identity a limit other than the origin is a decayed minimum (WD15).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m] {η lam κ ρ : ℝ}
  {U₀ V₀ U V : Matrix m k ℂ}

local notation "G₁" => fun X : Matrix m m ℂ => mask Set.univ (X - 1)

/-- **The run on the identity stays in the ball** (`review/flat_spectrum.md`, item WD16,
statement 6; WD6 with the constants of the squared loss). -/
lemma frobeniusSq_decayRun_identity_le (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (t : ℕ) :
    frobeniusSq (decayRun G₁ η lam U₀ V₀ t).1 + frobeniusSq (decayRun G₁ η lam U₀ V₀ t).2 ≤
      ρ := by
  have hρ0 : 0 ≤ ρ := (add_nonneg (frobeniusSq_nonneg U₀) (frobeniusSq_nonneg V₀)).trans hw
  exact frobeniusSq_decayRun_le hη.le hlam (frobeniusSq_mask_sub_le hρ0 hκ0 hκ)
    (fun X _ => neg_le_pairing_mask_sub_of_le hκ X) (by linarith) hstep hw t

/-- **The sum of the factors stays nonzero along a run on the identity**
(`review/flat_spectrum.md`, item WD16, statement 6): under the bounds of WD8, from a start in
the ball with `U₀ + V₀ ≠ 0`. -/
lemma frobeniusSq_add_decayRun_identity_pos (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (h0 : U₀ + V₀ ≠ 0) (t : ℕ) :
    0 < frobeniusSq ((decayRun G₁ η lam U₀ V₀ t).1 + (decayRun G₁ η lam U₀ V₀ t).2) := by
  induction t with
  | zero =>
    exact lt_of_le_of_ne (frobeniusSq_nonneg _) fun h =>
      h0 (eq_zero_of_frobeniusSq_eq_zero h.symm)
  | succ t ih =>
    rw [decayRun_succ]
    exact lt_of_lt_of_le (mul_pos (pow_pos hη 2) ih) (frobeniusSq_add_decayStep_identity hη hκ0
      hdesc (frobeniusSq_decayRun_identity_le hη hlam hκ0 hκ hρ hstep hw t)).1

/-- **A run on the identity from `U₀ + V₀ ≠ 0` does not converge to the origin**
(`review/flat_spectrum.md`, item WD16, statement 6), for `0 < λ < 1` and the bounds of WD8. -/
lemma frobeniusSq_pos_of_tendsto_decayRun_identity (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2)
    (hρ : κ ^ 2 ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (h0 : U₀ + V₀ ≠ 0) (hlim : Tendsto (decayRun G₁ η lam U₀ V₀) atTop (𝓝 (U, V))) :
    0 < frobeniusSq U + frobeniusSq V := by
  by_contra hcon
  have hz : frobeniusSq U + frobeniusSq V = 0 :=
    le_antisymm (not_lt.1 hcon) (add_nonneg (frobeniusSq_nonneg U) (frobeniusSq_nonneg V))
  have hc : Continuous fun W : Matrix m k ℂ × Matrix m k ℂ =>
      frobeniusSq W.1 + frobeniusSq W.2 :=
    (continuous_frobeniusSq.comp continuous_fst).add (continuous_frobeniusSq.comp continuous_snd)
  have hn : Tendsto (fun t => frobeniusSq (decayRun G₁ η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun G₁ η lam U₀ V₀ t).2) atTop (𝓝 0) := by
    have h := (hc.tendsto (U, V)).comp hlim
    rwa [show frobeniusSq (U, V).1 + frobeniusSq (U, V).2 = 0 from hz] at h
  obtain ⟨T, hT⟩ := eventually_atTop.1
    (hn.eventually_lt_const (by linarith : (0 : ℝ) < 1 - lam))
  have hmono : ∀ t, T ≤ t →
      frobeniusSq ((decayRun G₁ η lam U₀ V₀ T).1 + (decayRun G₁ η lam U₀ V₀ T).2) ≤
        frobeniusSq ((decayRun G₁ η lam U₀ V₀ t).1 + (decayRun G₁ η lam U₀ V₀ t).2) :=
    fun t ht => by
      induction t, ht using Nat.le_induction with
      | base => exact le_rfl
      | succ t ht ih =>
        refine ih.trans ?_
        rw [decayRun_succ]
        exact (frobeniusSq_add_decayStep_identity hη hκ0 hdesc
          (frobeniusSq_decayRun_identity_le hη hlam hκ0 hκ hρ hstep hw t)).2 (hT t ht).le
  have hp : Tendsto (fun t => frobeniusSq ((decayRun G₁ η lam U₀ V₀ t).1 +
      (decayRun G₁ η lam U₀ V₀ t).2)) atTop (𝓝 0) :=
    squeeze_zero (g := fun t => 2 * (frobeniusSq (decayRun G₁ η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun G₁ η lam U₀ V₀ t).2)) (fun t => frobeniusSq_nonneg _)
      (fun t => by
        have := frobeniusSq_add_le (decayRun G₁ η lam U₀ V₀ t).1 (decayRun G₁ η lam U₀ V₀ t).2
        linarith) (by simpa using hn.const_mul 2)
  have hpos := frobeniusSq_add_decayRun_identity_pos hη hlam hκ0 hκ hρ hstep hdesc hw h0 T
  have := ge_of_tendsto hp (eventually_atTop.2 ⟨T, hmono⟩)
  linarith

/-- **A run on the identity converges to the origin exactly from the starts with
`U₀ + V₀ = 0`** (`review/flat_spectrum.md`, item WD16, theorem), for `0 < λ < 1` and the
bounds of WD8, at every size and width. -/
theorem tendsto_decayRun_identity_zero_iff (hη : 0 < η) (hlam : 0 < lam) (hlam1 : lam < 1)
    (hκ0 : 0 ≤ κ) (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    Tendsto (decayRun G₁ η lam U₀ V₀) atTop (𝓝 (0, 0)) ↔ U₀ + V₀ = 0 := by
  refine ⟨fun hlim => ?_, fun h0 => ?_⟩
  · by_contra h0
    have h := frobeniusSq_pos_of_tendsto_decayRun_identity hη hlam hlam1 hκ0 hκ hρ hstep hdesc
      hw h0 hlim
    simp [frobeniusSq] at h
  · obtain rfl : V₀ = -U₀ := eq_neg_of_add_eq_zero_right h0
    refine tendsto_decayRun_identity_neg hη hlam hκ0 hκ hρ hstep hdesc ?_
    rw [show -U₀ = (-1 : ℝ) • U₀ by simp, frobeniusSq_smul] at hw
    linarith

/-- **The limit of a run on the identity is the origin exactly when `U₀ + V₀ = 0`**
(`review/flat_spectrum.md`, item WD16, theorem): under
the bounds of WD8 with `0 < λ < 1` the run converges to a pair `(U, U)` with
`U Uᴴ U = (1 − λ) U`, and `U = 0` if and only if `U₀ + V₀ = 0`. -/
theorem exists_tendsto_decayRun_identity_eq_zero_iff (hη : 0 < η) (hlam : 0 < lam)
    (hlam1 : lam < 1) (hκ0 : 0 ≤ κ) (hκ : frobeniusSq (1 : Matrix m m ℂ) ≤ κ ^ 2)
    (hρ : κ ^ 2 ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) :
    ∃ U : Matrix m k ℂ,
      Tendsto (decayRun G₁ η lam U₀ V₀) atTop (𝓝 (U, U)) ∧ U * Uᴴ * U = (1 - lam) • U ∧
      (U = 0 ↔ U₀ + V₀ = 0) := by
  obtain ⟨U, hlim, h⟩ := exists_tendsto_decayRun_identity hη hlam hκ0 hκ hρ hstep hdesc hw
  have hiff := tendsto_decayRun_identity_zero_iff (U₀ := U₀) (V₀ := V₀) hη hlam hlam1 hκ0 hκ hρ
    hstep hdesc hw
  refine ⟨U, hlim, h, fun hU => hiff.1 (by rwa [hU] at hlim), fun h0 => ?_⟩
  exact (Prod.ext_iff.1 (tendsto_nhds_unique hlim (hiff.2 h0))).1

/-! The witnesses, for the `2 × 2` identity with `λ = ½`, `κ = 2`, `ρ = 8` and `η = 1/200`: the
run of width one from `u₀ = v₀ = (1, 1)`, whose limit is a pair `(u, u)` with `u ≠ 0`. -/

/-- The start `u₀ = v₀ = (1, 1)` has `u₀ + v₀ ≠ 0`. -/
lemma ones_add_ones_ne_zero :
    (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) + (of fun _ _ => 1) ≠ 0 := fun h => by
  have := congrFun (congrFun h 0) 0
  simp at this

/-- The start `u₀ = v₀ = (1, 1)` is in the ball of radius `8`. -/
lemma frobeniusSq_ones_add_le :
    frobeniusSq (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) +
      frobeniusSq (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) ≤ 8 := by
  simp [frobeniusSq]
  norm_num

example := frobeniusSq_decayRun_identity_le (m := Fin 2) (η := 1 / 200) (lam := 1 / 2) (κ := 2)
  (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by rw [frobeniusSq_one_fin_two]; norm_num)
  (by norm_num) (by norm_num) frobeniusSq_ones_add_le 3

example := frobeniusSq_add_decayRun_identity_pos (m := Fin 2) (η := 1 / 200) (lam := 1 / 2)
  (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num)
  (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
  frobeniusSq_ones_add_le ones_add_ones_ne_zero 3

example : ∃ u : Matrix (Fin 2) (Fin 1) ℂ,
    Tendsto (decayRun (fun X : Matrix (Fin 2) (Fin 2) ℂ => mask Set.univ (X - 1)) (1 / 200)
      (1 / 2) (of fun _ _ => 1) (of fun _ _ => 1)) atTop (𝓝 (u, u)) ∧
    u * uᴴ * u = (1 - 1 / 2 : ℝ) • u ∧ u ≠ 0 := by
  obtain ⟨u, hlim, h, hiff⟩ := exists_tendsto_decayRun_identity_eq_zero_iff (m := Fin 2)
    (k := Fin 1) (η := 1 / 200) (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num)
    (by norm_num) (by norm_num) frobeniusSq_ones_add_le
  exact ⟨u, hlim, h, fun hu => ones_add_ones_ne_zero (hiff.1 hu)⟩

example := (tendsto_decayRun_identity_zero_iff (m := Fin 2) (η := 1 / 200) (lam := 1 / 2)
  (κ := 2) (ρ := 8) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num) (by norm_num)
  frobeniusSq_ones_add_le).not.2 ones_add_ones_ne_zero

example (U V : Matrix (Fin 2) (Fin 1) ℂ)
    (hlim : Tendsto (decayRun (fun X : Matrix (Fin 2) (Fin 2) ℂ => mask Set.univ (X - 1))
      (1 / 200) (1 / 2) (of fun _ _ => 1) (of fun _ _ => 1)) atTop (𝓝 (U, V))) :
    0 < frobeniusSq U + frobeniusSq V :=
  frobeniusSq_pos_of_tendsto_decayRun_identity (κ := 2) (ρ := 8) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num)
    (by norm_num) (by norm_num) frobeniusSq_ones_add_le ones_add_ones_ne_zero hlim

end Transformer.Grokking.FlatSpectrum.Training.Scope
