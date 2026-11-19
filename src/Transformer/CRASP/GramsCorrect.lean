module

public import Transformer.CRASP.GramsRun

/-!
# Positional logits read the last tokens: the transformer

In no paper; the construction is `Transformer.CRASP.Grams`, for the rounded transformers of
arXiv:2506.16055v3 (Appendix B.1, `def:transformer`), with a logit `L` of the positions and the
dot product (`Transformer.CRASP.LTfr`).

At precision `p ≥ 3`, with `b < 2^(p-1)`, `ReadsPrev L b`, and `M` scoring every pair `0`: the
residual of a reading layer adds the gram at `i` to the attention of
`Transformer.CRASP.GramsRead`, so the coordinate of `c i` holds `2` or `3` and that of
`c (i - 1)` holds `1` or `3` (`val_input_read`); the feed-forward map reads both, and the layer
writes at `i` the gram `push (c i) (c (i - 1))` (`layer_read`). So after `ℓ ≤ n` layers the
position `j` of `⊲ · w` holds its gram `window w ℓ j` (`act_read`); after
`n + m` layers its first `d` coordinates hold the activation of `M` after `m` layers at `j` in
`⊲ · grams n κ w` (`lo_act_run`). So the transformer accepts `⊲ · w` iff `M` accepts
`⊲ · grams n κ w` (`accepts_model`), and recognizes the words whose grams `M` recognizes
(`recognizes_model`).
-/

@[expose] public section

namespace Transformer.CRASP.Grams

universe u v
variable {σ : Type u} {Γ : Type v} {n p d k : ℕ} [DecidableEq σ] [Fintype σ]
  (L : ℕ → ℕ → ℝ → ℝ) (b : ℕ) (M : RTfr (Option Γ) p 0 d k) (κ : Gram σ n → Γ)

omit [DecidableEq σ] [Fintype σ] in
/-- The token at `j` of `⊲ · w` is `tok w j`. -/
theorem getElem_bos (w : List σ) (j : ℕ) (hj : j < (bos w).length) : (bos w)[j] = tok w j := by
  simp [tok, List.getElem?_eq_getElem hj]

/-- The hypothesis of `getElem_bos` holds at the letter of a one-letter word. -/
example : (bos [true])[1] = tok [true] 1 := getElem_bos [true] 1 (by simp)

/-- After the residual, the coordinate of `c'` holds `2` if `c'` is the gram at `i`, plus `1` if
it is the gram at `i - 1`. -/
theorem val_input_read (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) (hL : ReadsPrev L b) {ℓ : ℕ}
    (hℓ : ℓ < n) (c : ℕ → Gram σ n) {N : ℕ} (i : Fin N) (c' : Gram σ n) :
    (Fx.add ((model L b M κ).attention ℓ (fun j => enc M κ (c j.val)) i
      (Fin.natAdd d (idx c'))) (enc M κ (c i.val) (Fin.natAdd d (idx c')))).val =
        2 * (if c' = c i.val then 1 else 0) + (if c' = c (i.val - 1) then 1 else 0) := by
  rw [Fx.add, val_attention_read L b M κ hp hb hL hℓ, enc, mk_natAdd, val_onehot hp]
  exact val_round_small hp (by split_ifs <;> norm_num) (by split_ifs <;> norm_num)

/-- The hypotheses of `val_input_read` hold for ALiBi of slope `1` at precision `3`. -/
example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c : ℕ → Gram Bool 1) (i : Fin 2)
    (c' : Gram Bool 1) :=
  val_input_read (alibiLogit (1 : ℕ)) 1 M id le_rfl (by norm_num) (readsPrev_alibi le_rfl)
    (ℓ := 0) (by norm_num) c i c'

theorem ff_read {ℓ : ℕ} (hℓ : ℓ < n) (x : Fin (width σ n d) → Fx p 0) :
    (model L b M κ).ff ℓ x = enc M κ (push (pick fun c => 2 ≤ (hi x c).val)
      (pick fun c => (hi x c).val = 1 ∨ (hi x c).val = 3)) := by
  simp [model, hℓ]

example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (x : Fin (width Bool 1 0) → Fx 3 0) :=
  ff_read (alibiLogit (1 : ℕ)) 1 M id (ℓ := 0) (by norm_num) x

/-- **A reading layer brings the gram of the previous position**: on positions that hold the
grams `c j`, it writes at `i` the gram `push (c i) (c (i - 1))`. -/
theorem layer_read (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) (hL : ReadsPrev L b) {ℓ : ℕ}
    (hℓ : ℓ < n) (c : ℕ → Gram σ n) {N : ℕ} (i : Fin N) :
    (model L b M κ).layer ℓ (fun j => enc M κ (c j.val)) i =
      enc M κ (push (c i.val) (c (i.val - 1))) := by
  simp only [LTfr.layer]
  rw [ff_read L b M κ hℓ]
  congr 2
  · refine pick_eq fun c' => ?_
    simp only [hi]
    rw [val_input_read L b M κ hp hb hL hℓ]
    split_ifs with h1 h2 <;> norm_num [h1]
  · refine pick_eq fun c' => ?_
    simp only [hi]
    rw [val_input_read L b M κ hp hb hL hℓ]
    split_ifs with h1 h2 h2 <;> norm_num [h2]

/-- The hypotheses of `layer_read` hold for ALiBi of slope `1` at precision `3`. -/
example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c : ℕ → Gram Bool 1) (i : Fin 2) :
    (model (alibiLogit (1 : ℕ)) 1 M id).layer 0 (fun j => enc M id (c j.val)) i =
      enc M id (push (c i.val) (c (i.val - 1))) :=
  layer_read _ 1 M id le_rfl (by norm_num) (readsPrev_alibi le_rfl) (by norm_num) c i

/-- **After `ℓ ≤ n` layers the position `j` holds its gram.** -/
theorem act_read (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) (hL : ReadsPrev L b) (w : List σ) :
    ∀ ℓ, ℓ ≤ n → ∀ j : Fin (bos w).length,
      (model L b M κ).act (bos w) ℓ j = enc M κ (window w ℓ j.val)
  | 0, _, j => by
      have hw : (fun m : Fin (n + 1) => if m.val = 0 then (bos w)[j] else none) =
          window w 0 j.val := by
        funext m
        simp only [window, Nat.le_zero]
        split_ifs with hm
        · rw [hm, Nat.sub_zero]
          exact getElem_bos w j.val j.isLt
        · rfl
      calc (model L b M κ).act (bos w) 0 j =
          enc M κ (fun m : Fin (n + 1) => if m.val = 0 then (bos w)[j] else none) := rfl
        _ = enc M κ (window w 0 j.val) := by rw [hw]
  | ℓ + 1, hℓ, j => by
      have ih : (model L b M κ).act (bos w) ℓ = fun j => enc M κ (window w ℓ j.val) :=
        funext (act_read hp hb hL w ℓ (by omega))
      show (model L b M κ).layer ℓ ((model L b M κ).act (bos w) ℓ) j = _
      rw [ih, layer_read L b M κ hp hb hL (by omega) (window w ℓ), window_succ]

/-- The hypotheses of `act_read` hold for ALiBi of slope `1` at precision `3`, after its one
reading layer. -/
example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (w : List Bool) :=
  act_read (alibiLogit (1 : ℕ)) 1 M id le_rfl (by norm_num) (readsPrev_alibi le_rfl) w 1 le_rfl

omit [DecidableEq σ] [Fintype σ] in
theorem length_bos_grams (w : List σ) : (bos (grams n κ w)).length = (bos w).length := by simp

omit [DecidableEq σ] [Fintype σ] in
/-- Renaming the positions commutes with a layer of `M`. -/
theorem layer_cast {N N' : ℕ} (e : N = N') (ℓ : ℕ) (G : Fin N' → Fin d → Fx p 0) (i : Fin N) :
    M.layer ℓ (fun j => G (Fin.cast e j)) i = M.layer ℓ G (Fin.cast e i) := by
  subst e
  rfl

example (M : RTfr (Option Bool) 3 0 0 0) (G : Fin 2 → Fin 0 → Fx 3 0) (i : Fin 2) :=
  layer_cast M rfl 0 G i

/-- **After `n + m` layers the first `d` coordinates run `M`** on `⊲ · grams n κ w`. -/
theorem lo_act_run (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) (hL : ReadsPrev L b)
    (hM : ∀ ℓ x y, M.score ℓ x y = 0) (w : List σ) :
    ∀ m, ∀ j : Fin (bos w).length, lo ((model L b M κ).act (bos w) (n + m) j) =
      M.act (bos (grams n κ w)) m (Fin.cast (length_bos_grams κ w).symm j)
  | 0, j => by
      rw [Nat.add_zero, act_read L b M κ hp hb hL w n le_rfl j, enc, lo_mk]
      exact congrArg M.E (getElem_bos_grams κ w j.val _).symm
  | m + 1, j => by
      have ih := lo_act_run hp hb hL hM w m
      show lo ((model L b M κ).layer (n + m) ((model L b M κ).act (bos w) (n + m)) j) =
        M.layer m (M.act (bos (grams n κ w)) m) (Fin.cast (length_bos_grams κ w).symm j)
      rw [lo_layer_run L b M κ hp hL hM (Nat.le_add_right n m), Nat.add_sub_cancel_left]
      simp only [ih]
      exact layer_cast M _ m _ j

/-- The hypotheses of `lo_act_run` hold for ALiBi of slope `1` and the transformer of
`thm:TLCl_to_rtfr` of a formula of capacity `1`. -/
example (w : List Bool) := lo_act_run (n := 1) (alibiLogit (1 : ℕ)) 1
  (TemporalProgram.model (Form.lt .one (.add .one .one) : Form Bool) 0)
  (fun c : Gram Bool 1 => decide (c 1 = some true)) (by decide) (by norm_num)
  (readsPrev_alibi le_rfl) (fun ℓ x y => by simp [RTfr.score, TemporalProgram.model]) w

/-- **The transformer accepts `⊲ · w` iff `M` accepts `⊲ · grams n κ w`.** -/
theorem accepts_model (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) (hL : ReadsPrev L b)
    (hM : ∀ ℓ x y, M.score ℓ x y = 0) (w : List σ) :
    (model L b M κ).Accepts (bos w) ↔ M.Accepts (bos (grams n κ w)) := by
  have hw : 0 < (bos w).length := by simp
  have hg : 0 < (bos (grams n κ w)).length := by simp
  have hout : (model L b M κ).out (bos w) = M.out (bos (grams n κ w)) := by
    rw [LTfr.out, RTfr.out, dite_eq_left hw, dite_eq_left hg]
    show M.Wout (lo ((model L b M κ).act (bos w) (n + k) _)) = _
    rw [lo_act_run L b M κ hp hb hL hM w k]
    congr 2
    ext
    simp
  rw [LTfr.Accepts, RTfr.Accepts, hout]

example (w : List Bool) := accepts_model (n := 1) (alibiLogit (1 : ℕ)) 1
  (TemporalProgram.model (Form.lt .one (.add .one .one) : Form Bool) 0)
  (fun c : Gram Bool 1 => decide (c 1 = some true)) (by decide) (by norm_num)
  (readsPrev_alibi le_rfl) (fun ℓ x y => by simp [RTfr.score, TemporalProgram.model]) w

/-- **The transformer recognizes the words whose grams `M` recognizes.** -/
theorem recognizes_model (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) (hL : ReadsPrev L b)
    (hM : ∀ ℓ x y, M.score ℓ x y = 0) {Λ : Set (List Γ)} (hR : M.Recognizes Λ) :
    (model L b M κ).Recognizes {w | grams n κ w ∈ Λ} := fun w =>
  (accepts_model L b M κ hp hb hL hM w).trans (hR (grams n κ w))

/-- The hypotheses of `recognizes_model` hold for ALiBi of slope `1` and the transformer of
`thm:TLCl_to_rtfr` of a formula of capacity `1`. -/
example := recognizes_model (n := 1) (alibiLogit (1 : ℕ)) 1
  (TemporalProgram.model (Form.lt .one (.add .one .one) : Form Bool) 0)
  (fun c : Gram Bool 1 => decide (c 1 = some true)) (by decide) (by norm_num)
  (readsPrev_alibi le_rfl) (fun ℓ x y => by simp [RTfr.score, TemporalProgram.model])
  (TemporalProgram.model_recognizes _ 0 ⟨rfl, rfl, le_rfl⟩)

end Transformer.CRASP.Grams
