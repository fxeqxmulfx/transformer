module

public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies

/-!
# Phasor v2: LeRoPE

The second element of Phasor v2 is LeRoPE (Karypis et al., arXiv:2607.10134v1, §3.1). RoPE gives
the band `m` of the `p = d/2` bands of a head the frequency `θ_m = b^{−2m/d}` (Eq. (1)); LeRoPE
scales it by a learned `e^{α_m}`, Eq. (5) (`leropeFrequency`). A head vector here is `p` complex
numbers, one per band. The rotation `ρ(θ_m)^t` of Eq. (5) acts on the band `ℂ = ℝ²` as
multiplication by `e^{i t θ_m}`, so LeRoPE at the position `t` multiplies the band `m` by it
(`turn`). The score is the real dot product of the bands, `Re ∑_m conj(q_m) k_m` (`score`,
`score_eq_sum`).

* Eq. (3): the query turned to `s` against the key turned to `t` scores as the query against the
  key turned by `t − s` (`score_turn`), so the score depends only on the offset
  (`score_turn_add`);
* the turn keeps the length of every band (`norm_turn`), so the score is at most
  `∑_m |q_m| |k_m|` at all positions and frequencies (`abs_score_turn_le`);
* §3.1: the frequencies are positive (`leropeFrequency_pos`), `α = 0` is RoPE
  (`leropeFrequency_zero`), a step in `α_m` multiplies `θ_m` (`leropeFrequency_add`), and any
  positive frequencies are reached (`exists_leropeFrequency_eq`).

`Transformer.PhasorV2.LeRoPEBands` puts Partial RoPE and NoPE inside LeRoPE and shows that LeRoPE
sees what NoPE cannot.
-/

@[expose] public section

open scoped ComplexConjugate

namespace Transformer.PhasorV2

variable {p : ℕ}

/-- **LeRoPE's frequencies** (arXiv:2607.10134v1, §3.1, Eq. (5)): RoPE's `θ_m = b^{−2m/d}` of
Eq. (1) for the band `m` of the `p = d/2` bands, scaled by `e^{α_m}`. -/
noncomputable def leropeFrequency (b : ℝ) (α : Fin p → ℝ) (m : Fin p) : ℝ :=
  RoPE.LeRoPE.learnedFrequency (RoPE.frequency b (2 * p) m) (α m)

/-- **LeRoPE at the position `t`** (Eq. (5)): the band `m` of the head vector `q`, a complex
number, turned by the angle `t θ_m`. -/
noncomputable def turn (θ : Fin p → ℝ) (t : ℝ) (q : Fin p → ℂ) : Fin p → ℂ :=
  fun m => Complex.exp ((t * θ m : ℝ) * Complex.I) * q m

/-- **The score of a query and a key** (Eq. (3)): `Re ∑_m conj(q_m) k_m`. -/
noncomputable def score (q k : Fin p → ℂ) : ℝ :=
  (∑ m, conj (q m) * k m).re

/-- The score is the real dot product `qᵀk` of the bands as vectors of `ℝ²`. -/
theorem score_eq_sum (q k : Fin p → ℂ) :
    score q k = ∑ m, ((q m).re * (k m).re + (q m).im * (k m).im) := by
  simp only [score, Complex.re_sum, Complex.mul_re, Complex.conj_re, Complex.conj_im]
  refine Finset.sum_congr rfl fun m _ => ?_
  ring

/-- **Eq. (3)**: the query turned to `s` against the key turned to `t` scores as the query
against the key turned by `t − s`. -/
theorem score_turn (θ : Fin p → ℝ) (s t : ℝ) (q k : Fin p → ℂ) :
    score (turn θ s q) (turn θ t k) = score q (turn θ (t - s) k) := by
  unfold score turn
  congr 1
  refine Finset.sum_congr rfl fun m _ => ?_
  have h : Complex.exp (((t - s) * θ m : ℝ) * Complex.I) =
      conj (Complex.exp ((s * θ m : ℝ) * Complex.I)) *
        Complex.exp ((t * θ m : ℝ) * Complex.I) := by
    rw [← Complex.exp_conj, map_mul, Complex.conj_ofReal, Complex.conj_I, ← Complex.exp_add]
    congr 1
    push_cast
    ring
  rw [map_mul, h]
  ring

/-- **The score depends only on the offset**: moving both positions by `c` keeps it. -/
theorem score_turn_add (θ : Fin p → ℝ) (s t c : ℝ) (q k : Fin p → ℂ) :
    score (turn θ (s + c) q) (turn θ (t + c) k) = score (turn θ s q) (turn θ t k) := by
  rw [score_turn, score_turn, add_sub_add_right_eq_sub]

/-- **The turn keeps the length of every band.** -/
theorem norm_turn (θ : Fin p → ℝ) (t : ℝ) (q : Fin p → ℂ) (m : Fin p) :
    ‖turn θ t q m‖ = ‖q m‖ := by
  simp only [turn, norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]

/-- The score is at most `∑_m |q_m| |k_m|`. -/
theorem abs_score_le (q k : Fin p → ℂ) : |score q k| ≤ ∑ m, ‖q m‖ * ‖k m‖ := by
  refine (Complex.abs_re_le_norm _).trans ((norm_sum_le _ _).trans (le_of_eq ?_))
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [norm_mul, Complex.norm_conj]

/-- **LeRoPE moves no score beyond the bound of the unturned vectors**: at all positions and
frequencies the score is at most `∑_m |q_m| |k_m|`. -/
theorem abs_score_turn_le (θ : Fin p → ℝ) (s t : ℝ) (q k : Fin p → ℂ) :
    |score (turn θ s q) (turn θ t k)| ≤ ∑ m, ‖q m‖ * ‖k m‖ := by
  refine (abs_score_le _ _).trans (le_of_eq ?_)
  simp only [norm_turn]

/-- §3.1: for `b > 0` the learned frequencies are positive, whatever the scales `α`. -/
theorem leropeFrequency_pos {b : ℝ} (hb : 0 < b) (α : Fin p → ℝ) (m : Fin p) :
    0 < leropeFrequency b α m :=
  RoPE.LeRoPE.learned_frequency_pos (Real.rpow_pos_of_pos hb _) (α m)

/-- The main model's base `10000`, sixteen bands, at the initial scales. -/
example := leropeFrequency_pos (p := 16) (b := 10000) (by norm_num) 0 3

/-- §3.1: "We initialize `α_m = 0`, making our method equivalent to RoPE on initialization." -/
theorem leropeFrequency_zero (b : ℝ) (m : Fin p) :
    leropeFrequency b 0 m = RoPE.frequency b (2 * p) m :=
  RoPE.LeRoPE.learned_frequency_init _

/-- §3.1: "steps in `α_m` make multiplicative changes to `θ̂^{(m)}`". -/
theorem leropeFrequency_add (b : ℝ) (α δ : Fin p → ℝ) (m : Fin p) :
    leropeFrequency b (α + δ) m = Real.exp (δ m) * leropeFrequency b α m := by
  rw [leropeFrequency, leropeFrequency, Pi.add_apply,
    RoPE.LeRoPE.learned_frequency_increment]

/-- **Any positive frequencies are reached** (Eq. (5)): for `b > 0` some scales give every band
its prescribed positive frequency. -/
theorem exists_leropeFrequency_eq {b : ℝ} (hb : 0 < b) {θ : Fin p → ℝ} (hθ : ∀ m, 0 < θ m) :
    ∃ α : Fin p → ℝ, leropeFrequency b α = θ := by
  have hω (m : Fin p) : 0 < RoPE.frequency b (2 * p) m := Real.rpow_pos_of_pos hb _
  choose α hα using fun m => RoPE.LeRoPE.learned_frequency_surjective (hω m) (hθ m)
  exact ⟨α, funext hα⟩

/-- Sixteen bands, all at the frequency `π`. -/
example := exists_leropeFrequency_eq (p := 16) (b := 10000) (by norm_num)
  (θ := fun _ => Real.pi) fun _ => Real.pi_pos

end Transformer.PhasorV2
