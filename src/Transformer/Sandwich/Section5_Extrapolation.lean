module

public import Transformer.Sandwich.Section5_WindowLayer

/-!
# §1 and §5: windowed attention extrapolates when its TRF fits in `L_tr`

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §1: "A windowed attention model can extrapolate
if `L_tr > wR` because 1) `wR` is fully covered by `L_tr` during the training stage, and 2) it
simply ignores the additional `L_ex − wR` tokens during the testing stage." §5: "Consider the case
where TRF `≤ L_tr`: This model can extrapolate easily because its TRF is fully covered and
trained."

With the TRF `R(w − 1) + 1` of `Section5_ReceptiveField`:

* `Model.not_contributes`: the tokens `n` with `n + R(w − 1) + 1 ≤ m` do not contribute to the
  prediction at `m`, (2).
* `Model.predict_shift`: the prediction at `s + m` is the prediction at `m` of the sequence
  without its first `s` tokens, once `m ≥ R(w − 1)`.
* `Model.extrapolates`: if `R(w − 1) + 1 ≤ L_tr`, every prediction, at any position of any input,
  is the prediction at a position `< L_tr` of some input, a computation the training stage
  covers, (1). `Model.extrapolates_of_mul_le` assumes the paper's `wR ≤ L_tr` instead.
-/

@[expose] public section

namespace Transformer.Sandwich

variable {d k H v w : ℕ}

/-- **A windowed model ignores the tokens beyond its TRF** (§1): the token `n` does not
contribute to the prediction at `m ≥ n + R(w − 1) + 1`. -/
theorem Model.not_contributes (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) {m n : ℕ}
    (hmn : n + (M.depth * (w - 1) + 1) ≤ m) : ¬Contributes (M.predict b) m n := fun h => by
  have := ((M.hasWindow_predict hb hw).1.contributes h).2
  omega

/-- The hypotheses of `Model.not_contributes` hold for the token `0` and a position past the
TRF. -/
example (M : Model d k H v) :
    ¬Contributes (M.predict fun _ => windowBias 1) (M.depth * (1 - 1) + 1) 0 :=
  M.not_contributes (fun _ => isWindowed_windowBias 1) le_rfl (by omega)

/-- **The prediction does not see absolute positions**: the prediction at `s + m` is the
prediction at `m` of the sequence without its first `s` tokens, once `m ≥ R(w − 1)`. -/
theorem Model.predict_shift (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) (x : Embs d) (s : ℕ) {m : ℕ}
    (hm : M.depth * (w - 1) ≤ m) : M.predict b x (s + m) = M.predict b (fun j => x (s + j)) m :=
  (M.hasWindow_predict hb hw).2 x s m (by omega)

/-- The hypotheses of `Model.predict_shift` hold at the end of the TRF. -/
example (M : Model d k H v) (x : Embs d) (s : ℕ) :
    M.predict (fun _ => windowBias 1) x (s + M.depth * (1 - 1)) =
      M.predict (fun _ => windowBias 1) (fun j => x (s + j)) (M.depth * (1 - 1)) :=
  M.predict_shift (fun _ => isWindowed_windowBias 1) le_rfl x s le_rfl

/-- **A windowed model whose TRF fits in `L_tr` extrapolates** (§1, §5): every prediction, at any
position `m` of any input `x`, is the prediction at a position `j < L_tr` of some input `y`. -/
theorem Model.extrapolates (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) {Ltr : ℕ} (hL : M.depth * (w - 1) + 1 ≤ Ltr)
    (x : Embs d) (m : ℕ) : ∃ (y : Embs d) (j : ℕ), j < Ltr ∧ M.predict b x m = M.predict b y j := by
  by_cases hm : m < Ltr
  · exact ⟨x, m, hm, rfl⟩
  · refine ⟨fun j => x (m - M.depth * (w - 1) + j), M.depth * (w - 1), by omega, ?_⟩
    rw [← M.predict_shift hb hw x _ le_rfl, Nat.sub_add_cancel (by omega)]

/-- The hypotheses of `Model.extrapolates` hold for windowed attention of size `1` and
`L_tr = 1`. -/
example (M : Model d k H v) (x : Embs d) (m : ℕ) : ∃ (y : Embs d) (j : ℕ), j < 1 ∧
    M.predict (fun _ => windowBias 1) x m = M.predict (fun _ => windowBias 1) y j :=
  M.extrapolates (fun _ => isWindowed_windowBias 1) le_rfl (by omega) x m

/-- **The paper's condition `wR ≤ L_tr`** (§1, §5) suffices: it implies `R(w − 1) + 1 ≤ L_tr`. -/
theorem Model.extrapolates_of_mul_le (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) {Ltr : ℕ} (hL : w * M.depth ≤ Ltr)
    (x : Embs d) (m : ℕ) : ∃ (y : Embs d) (j : ℕ), j < Ltr ∧ M.predict b x m = M.predict b y j := by
  refine M.extrapolates hb hw ?_ x m
  have h1 : 1 ≤ M.depth := Nat.succ_pos _
  have h2 : M.depth * (w - 1) + M.depth = w * M.depth := by
    zify [hw]
    ring
  omega

/-- The hypotheses of `Model.extrapolates_of_mul_le` hold for windowed attention of size `1` and
`L_tr = R`. -/
example (M : Model d k H v) (x : Embs d) (m : ℕ) : ∃ (y : Embs d) (j : ℕ), j < 1 * M.depth ∧
    M.predict (fun _ => windowBias 1) x m = M.predict (fun _ => windowBias 1) y j :=
  M.extrapolates_of_mul_le (fun _ => isWindowed_windowBias 1) le_rfl le_rfl x m

end Transformer.Sandwich
