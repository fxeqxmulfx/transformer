module

public import Transformer.Grokking.FlatSpectrum.Training.SquaredRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.OneByOne

/-!
# The `1 × 1` identity: a run from `U₀ + V₀ ≠ 0` does not end at the origin

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD15,
statement 3; the item "Width" of WD11.

For the squared loss on the `1 × 1` identity, `0 < λ`, and the bounds of WD8 with `κ ≥ 1`, the
run from a start in the ball `‖U₀‖² + ‖V₀‖² ≤ ρ`, of any width:

* stays in the ball (`frobeniusSq_decayRun_one_le`, WD6);
* has `U_t + V_t ≠ 0` at every step when `U₀ + V₀ ≠ 0`
  (`frobeniusSq_add_decayRun_one_pos`), by `‖P'‖² ≥ η² ‖P‖²` in the ball;
* for `λ < 1` does not converge to the origin when `U₀ + V₀ ≠ 0`
  (`frobeniusSq_pos_of_tendsto_decayRun_one`): otherwise the norms tend to `0`, from some step
  on `‖U_t + V_t‖` does not decrease, and it is positive.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {k : Type*} [Fintype k] {η lam κ ρ : ℝ}

local notation "R₁" => decayRun (fun X : Matrix (Fin 1) (Fin 1) ℂ => mask Set.univ (X - 1))

variable {U₀ V₀ U V : Matrix (Fin 1) k ℂ}

/-- **The run on the `1 × 1` identity stays in the ball** (`review/flat_spectrum.md`, item
WD15, statement 3; WD6 with the constants of the squared loss). -/
lemma frobeniusSq_decayRun_one_le (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ) (hκ : 1 ≤ κ ^ 2)
    (hρ : κ ^ 2 ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ) (t : ℕ) :
    frobeniusSq (R₁ η lam U₀ V₀ t).1 + frobeniusSq (R₁ η lam U₀ V₀ t).2 ≤ ρ := by
  have hρ0 : 0 ≤ ρ := (add_nonneg (frobeniusSq_nonneg U₀) (frobeniusSq_nonneg V₀)).trans hw
  have hκ' : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) ≤ κ ^ 2 :=
    frobeniusSq_one_fin_one.trans_le hκ
  exact frobeniusSq_decayRun_le hη.le hlam (frobeniusSq_mask_sub_le hρ0 hκ0 hκ')
    (fun X _ => neg_le_pairing_mask_sub_of_le hκ' X) (by linarith) hstep hw t

/-- **The sum of the factors stays nonzero** (`review/flat_spectrum.md`, item WD15, statement
3): under the bounds of WD8, from a start in the ball with `U₀ + V₀ ≠ 0`. -/
lemma frobeniusSq_add_decayRun_one_pos (hη : 0 < η) (hlam : 0 < lam) (hκ0 : 0 ≤ κ)
    (hκ : 1 ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ) (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (h0 : U₀ + V₀ ≠ 0) (t : ℕ) :
    0 < frobeniusSq ((R₁ η lam U₀ V₀ t).1 + (R₁ η lam U₀ V₀ t).2) := by
  induction t with
  | zero =>
    exact lt_of_le_of_ne (frobeniusSq_nonneg _) fun h =>
      h0 (eq_zero_of_frobeniusSq_eq_zero h.symm)
  | succ t ih =>
    rw [decayRun_succ]
    exact lt_of_lt_of_le (mul_pos (pow_pos hη 2) ih) (frobeniusSq_add_decayStep_one hη hκ0 hdesc
      (frobeniusSq_decayRun_one_le hη hlam hκ0 hκ hρ hstep hw t)).1

/-- **A run from `U₀ + V₀ ≠ 0` does not converge to the origin** (`review/flat_spectrum.md`,
item WD15, statement 3), for `0 < λ < 1` and the bounds of WD8. -/
lemma frobeniusSq_pos_of_tendsto_decayRun_one (hη : 0 < η) (hlam : 0 < lam) (hlam1 : lam < 1)
    (hκ0 : 0 ≤ κ) (hκ : 1 ≤ κ ^ 2) (hρ : κ ^ 2 ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam)
    (hdesc : η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1) (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (h0 : U₀ + V₀ ≠ 0) (hlim : Tendsto (R₁ η lam U₀ V₀) atTop (𝓝 (U, V))) :
    0 < frobeniusSq U + frobeniusSq V := by
  by_contra hcon
  have hz : frobeniusSq U + frobeniusSq V = 0 :=
    le_antisymm (not_lt.1 hcon) (add_nonneg (frobeniusSq_nonneg U) (frobeniusSq_nonneg V))
  have hc : Continuous fun W : Matrix (Fin 1) k ℂ × Matrix (Fin 1) k ℂ =>
      frobeniusSq W.1 + frobeniusSq W.2 :=
    (continuous_frobeniusSq.comp continuous_fst).add (continuous_frobeniusSq.comp continuous_snd)
  have hn : Tendsto (fun t => frobeniusSq (R₁ η lam U₀ V₀ t).1 +
      frobeniusSq (R₁ η lam U₀ V₀ t).2) atTop (𝓝 0) := by
    have h := (hc.tendsto (U, V)).comp hlim
    rwa [show frobeniusSq (U, V).1 + frobeniusSq (U, V).2 = 0 from hz] at h
  obtain ⟨T, hT⟩ := eventually_atTop.1
    (hn.eventually_lt_const (by linarith : (0 : ℝ) < 2 * (1 - lam) / 3))
  have hmono : ∀ t, T ≤ t → frobeniusSq ((R₁ η lam U₀ V₀ T).1 + (R₁ η lam U₀ V₀ T).2) ≤
      frobeniusSq ((R₁ η lam U₀ V₀ t).1 + (R₁ η lam U₀ V₀ t).2) := fun t ht => by
    induction t, ht using Nat.le_induction with
    | base => exact le_rfl
    | succ t ht ih =>
      refine ih.trans ?_
      rw [decayRun_succ]
      exact (frobeniusSq_add_decayStep_one hη hκ0 hdesc
        (frobeniusSq_decayRun_one_le hη hlam hκ0 hκ hρ hstep hw t)).2 (hT t ht).le
  have hp : Tendsto (fun t => frobeniusSq ((R₁ η lam U₀ V₀ t).1 + (R₁ η lam U₀ V₀ t).2)) atTop
      (𝓝 0) :=
    squeeze_zero (g := fun t => 2 * (frobeniusSq (R₁ η lam U₀ V₀ t).1 +
      frobeniusSq (R₁ η lam U₀ V₀ t).2)) (fun t => frobeniusSq_nonneg _)
      (fun t => by
        have := frobeniusSq_add_le (R₁ η lam U₀ V₀ t).1 (R₁ η lam U₀ V₀ t).2
        linarith) (by simpa using hn.const_mul 2)
  have hpos := frobeniusSq_add_decayRun_one_pos hη hlam hκ0 hκ hρ hstep hdesc hw h0 T
  have := ge_of_tendsto hp (eventually_atTop.2 ⟨T, hmono⟩)
  linarith

/-! The witness: the start `u₀ = v₀ = 1` of width one, with `λ = ½`, `κ = 1`, `ρ = 4` and
`η = 1/100`. -/

/-- The start `u₀ = v₀ = 1` has `u₀ + v₀ ≠ 0`. -/
lemma one_add_one_fin_one_ne_zero : (1 : Matrix (Fin 1) (Fin 1) ℂ) + 1 ≠ 0 := fun h => by
  have := congrFun (congrFun h 0) 0
  simp at this

example := frobeniusSq_decayRun_one_le (k := Fin 1) (η := 1 / 100) (lam := 1 / 2) (κ := 1)
  (ρ := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (U₀ := 1) (V₀ := 1) (by rw [frobeniusSq_one_fin_one]; norm_num) 3

example := frobeniusSq_add_decayRun_one_pos (k := Fin 1) (η := 1 / 100) (lam := 1 / 2)
  (κ := 1) (ρ := 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (U₀ := 1) (V₀ := 1) (by rw [frobeniusSq_one_fin_one]; norm_num)
  one_add_one_fin_one_ne_zero 3

example : ∃ U V : Matrix (Fin 1) (Fin 1) ℂ, 0 < frobeniusSq U + frobeniusSq V := by
  have hw : frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) +
      frobeniusSq (1 : Matrix (Fin 1) (Fin 1) ℂ) ≤ 4 := by
    rw [frobeniusSq_one_fin_one]
    norm_num
  obtain ⟨U, V, hlim, -⟩ := exists_tendsto_decayRun_fitLoss (Ω := Set.univ) (K := 1)
    (η := 1 / 100) (lam := 1 / 2) (κ := 1) (ρ := 4) (by norm_num) (by norm_num) (by norm_num)
    (by rw [frobeniusSq_one_fin_one]; norm_num) (by norm_num) (by norm_num) (by norm_num) hw
  exact ⟨U, V, frobeniusSq_pos_of_tendsto_decayRun_one (κ := 1) (ρ := 4) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hw one_add_one_fin_one_ne_zero hlim⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope
