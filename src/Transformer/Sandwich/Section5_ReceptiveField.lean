module

public import Mathlib.Order.Interval.Set.Card
public import Transformer.Sandwich.Section5_Averaging
public import Transformer.Sandwich.Section5_WindowLayer

/-!
# §5 The theoretical receptive field

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §1: "Windowed attention allows the easy
derivation of a theoretical (maximum) receptive field: `wR` for an `R` layer transformer model
with windowed attention size `w`"; §5: "an `R` layer transformer has a theoretical receptive
field (TRF) of `wR`, which is the maximum number of tokens that contribute to the prediction of
the next token."

**The TRF is `R(w − 1) + 1`, not `wR`.** The windows of consecutive layers share a token: the
query `m` of the second layer reads the first layer's outputs at `m − w + 1, …, m`, each of which
reads the `w` tokens ending at it, so together they reach back to `m − 2(w − 1)`: `2w − 1`
tokens. `wR` is an upper bound, attained only by `R = 1`.

* `Model.ncard_contributes_le`: at most `R(w − 1) + 1` tokens contribute to the prediction of a
  model whose heads all mask the distances `≥ w`, `w ≥ 1`.
* `avgModel_contributes`: in the averaging model of `Section5_Averaging` (zero queries and keys,
  the identity as values, one head) under windowed attention, all of them contribute.
* `isGreatest_trf`: the maximum over all models with `R ≥ 1` layers of windowed attention of
  size `w ≥ 1` is `R(w − 1) + 1`; `not_isGreatest_trf_mul`: for `R ≥ 2` it is not `wR`.
* The numbers of §5, `R = 12` and `L_tr = 512`: the TRF is covered by `L_tr` exactly when
  `w ≤ 43` (`trf_twelve_le_iff`); the paper's `w < 42.6 = 512/12`, derived from `wR`, is
  sufficient, and `512/12 = 42.66…` (`trf_twelve_of_lt`). With `w = 100` the TRF is `1189`, not
  `100 * 12 = 1200`, still `≫ 512` (`trf_twelve_hundred`).
-/

@[expose] public section

namespace Transformer.Sandwich

open Matrix Transformer.BAM

variable {d k H v w R : ℕ}

/-- **At most `R(w − 1) + 1` tokens contribute** to the prediction of an `R`-layer model whose
heads all mask the distances `≥ w ≥ 1`. -/
theorem Model.ncard_contributes_le (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) (m : ℕ) :
    {n | Contributes (M.predict b) m n}.ncard ≤ M.depth * (w - 1) + 1 :=
  (M.hasWindow_predict hb hw).1.ncard_contributes_le m

/-- The hypotheses of `Model.ncard_contributes_le` hold for windowed attention of size `1`. -/
example (M : Model d k H v) (m : ℕ) :
    {n | Contributes (M.predict fun _ => windowBias 1) m n}.ncard ≤ M.depth * (1 - 1) + 1 :=
  M.ncard_contributes_le (fun _ => isWindowed_windowBias 1) le_rfl m

/-- **Every token of the window contributes** to the averaging model: with `R ≥ 1` layers whose
bias is finite at the distances `< w`, `w ≥ 1`, changing the token `n` alone can change the
prediction at `m` whenever `n ≤ m < n + R(w − 1) + 1`. Beyond the distance `w` the bias is
arbitrary, `−∞` included. -/
theorem avgModel_contributes_of_finite {b : ℕ → EReal} (hb : ∀ t, t < w → b t ≠ ⊥ ∧ b t ≠ ⊤)
    (hR : 1 ≤ R) (hw : 1 ≤ w) {m n : ℕ} (hn : n ≤ m) (hmn : m < n + (R * (w - 1) + 1)) :
    Contributes ((avgModel R).predict fun _ => b) m n := by
  set y : Embs 2 := fun j => if j = n then ![1, -1] else 0 with hy
  have hy0 : Spread n 0 y := spread_single n
  have hout := (hy0.hidden_avgLayer hb hw (R - 1)).avgAttn_out hb hw
  rw [show 0 + (R - 1) * (w - 1) + (w - 1) = R * (w - 1) by
    rw [zero_add, ← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel hR]] at hout
  have hpred : ∀ e j, (avgModel R).predict (fun _ => b) e m j =
      softmax (fun i => ((avgAttn.out (fun _ => b)
        (hidden 1 (fun _ => b) (List.replicate (R - 1) avgLayer) e) m i : ℝ) : EReal)) j :=
    fun e j => by simp only [Model.predict, Model.out, avgModel, one_mulVec]
  refine ⟨0, y, fun j hj => by simp [y, hj], fun heq => ?_⟩
  have h0 : (avgModel R).predict (fun _ => b) 0 m 0 = (avgModel R).predict (fun _ => b) 0 m 1 := by
    simp [hpred, hidden_avgLayer_zero, avgAttn_out_apply, softmax]
  have h1 : (avgModel R).predict (fun _ => b) y m 1 < (avgModel R).predict (fun _ => b) y m 0 := by
    have hc := hout.2 m hn (by omega)
    simp only [hpred, softmax, weight_coe]
    rw [(hout.1 m).1]
    refine div_lt_div_of_pos_right (Real.exp_lt_exp.2 (by linarith)) ?_
    simp only [partition, weight_coe]
    exact Finset.sum_pos (fun i _ => Real.exp_pos _) Finset.univ_nonempty
  rw [heq] at h0
  exact h1.ne h0.symm

/-- The hypotheses of `avgModel_contributes_of_finite` hold for the bias `0`. -/
example : Contributes ((avgModel 1).predict fun _ _ => 0) 0 0 :=
  avgModel_contributes_of_finite (w := 1) (fun _ _ => ⟨by simp, by simp⟩) le_rfl le_rfl le_rfl
    (by norm_num)

/-- **Every token of the window contributes** to the averaging model under windowed attention:
with `R ≥ 1` layers of size `w ≥ 1`, changing the token `n` alone can change the prediction at
`m` whenever `n ≤ m < n + R(w − 1) + 1`. -/
theorem avgModel_contributes (hR : 1 ≤ R) (hw : 1 ≤ w) {m n : ℕ} (hn : n ≤ m)
    (hmn : m < n + (R * (w - 1) + 1)) :
    Contributes ((avgModel R).predict fun _ => windowBias w) m n :=
  avgModel_contributes_of_finite (fun t ht => by simp [windowBias, ht]) hR hw hn hmn

/-- The hypotheses of `avgModel_contributes` hold for the token `0` and the position `0`. -/
example : Contributes ((avgModel 1).predict fun _ => windowBias 1) 0 0 :=
  avgModel_contributes le_rfl le_rfl le_rfl (by norm_num)

/-- The numbers of tokens contributing to the prediction of an `R`-layer model with windowed
attention of size `w`, over all models and positions. -/
def trf (R w : ℕ) : Set ℕ :=
  {N | ∃ (d k H v : ℕ) (M : Model d k H v) (m : ℕ), M.depth = R ∧
    {n | Contributes (M.predict fun _ => windowBias w) m n}.ncard = N}

/-- **The theoretical receptive field is `R(w − 1) + 1`** (§5, corrected): the maximum number of
tokens that contribute to the prediction of the next token in an `R ≥ 1` layer model with
windowed attention of size `w ≥ 1`. -/
theorem isGreatest_trf (hR : 1 ≤ R) (hw : 1 ≤ w) : IsGreatest (trf R w) (R * (w - 1) + 1) := by
  refine ⟨⟨2, 2, 1, 2, avgModel R, R * (w - 1), avgModel_depth hR, ?_⟩, ?_⟩
  · have hset : {n | Contributes ((avgModel R).predict fun _ => windowBias w) (R * (w - 1)) n} =
        Set.Icc 0 (R * (w - 1)) := by
      ext n
      simp only [Set.mem_ofPred_eq, Set.mem_Icc, zero_le, true_and]
      refine ⟨fun h => ?_, fun h => avgModel_contributes hR hw h (by omega)⟩
      exact (((avgModel R).hasWindow_predict (fun _ => isWindowed_windowBias w) hw).1.contributes
        h).1
    rw [hset, Set.ncard_Icc, Nat.card_Icc, Nat.sub_zero]
  · rintro N ⟨d, k, H, v, M, m, hM, rfl⟩
    rw [← hM]
    exact M.ncard_contributes_le (fun _ => isWindowed_windowBias w) hw m

/-- The hypotheses of `isGreatest_trf` hold for one layer of size `1`. -/
example : IsGreatest (trf 1 1) (1 * (1 - 1) + 1) := isGreatest_trf le_rfl le_rfl

/-- **The TRF is not `wR`** for `R ≥ 2` layers (§1, §5): `wR` exceeds `R(w − 1) + 1`, the
maximum. -/
theorem not_isGreatest_trf_mul (hR : 2 ≤ R) (hw : 1 ≤ w) : ¬IsGreatest (trf R w) (w * R) := by
  intro h
  have := (isGreatest_trf (by omega) hw).unique h
  rw [Nat.mul_sub, Nat.mul_one, Nat.mul_comm w R] at this
  have : R ≤ R * w := Nat.le_mul_of_pos_right R hw
  omega

/-- The hypotheses of `not_isGreatest_trf_mul` hold for two layers of size `2`. -/
example : ¬IsGreatest (trf 2 2) (2 * 2) := not_isGreatest_trf_mul le_rfl (by norm_num)

/-- **When `R = 12` layers' TRF fits in `L_tr = 512`** (§5): exactly when `w ≤ 43`. -/
theorem trf_twelve_le_iff (w : ℕ) : 12 * (w - 1) + 1 ≤ 512 ↔ w ≤ 43 := by
  omega

/-- **The paper's `w < 42.6 = 512/12` is sufficient** (§5): it gives `wR ≤ 512`, hence
`R(w − 1) + 1 ≤ 512`; `512/12` is `42.66…`. -/
theorem trf_twelve_of_lt {w : ℕ} (hw : (w : ℝ) < 42.6) :
    w * 12 ≤ 512 ∧ 12 * (w - 1) + 1 ≤ 512 ∧ (42.6 : ℝ) < 512 / 12 ∧ (512 : ℝ) / 12 < 42.7 := by
  have : w ≤ 42 := by
    by_contra h
    have : (43 : ℝ) ≤ w := by exact_mod_cast Nat.lt_of_not_le h
    linarith
  refine ⟨by omega, by omega, by norm_num, by norm_num⟩

/-- The hypothesis of `trf_twelve_of_lt` holds for `w = 42`. -/
example : 42 * 12 ≤ 512 ∧ 12 * (42 - 1) + 1 ≤ 512 ∧ (42.6 : ℝ) < 512 / 12 ∧
    (512 : ℝ) / 12 < 42.7 :=
  trf_twelve_of_lt (by norm_num)

/-- **The TRF of `w = 100`** (§5): `12 · 99 + 1 = 1189`, not `100 * 12 = 1200`; both exceed
`L_tr = 512`. -/
theorem trf_twelve_hundred : 12 * (100 - 1) + 1 = 1189 ∧ 100 * 12 = 1200 ∧ 512 < 1189 := by
  norm_num

end Transformer.Sandwich
