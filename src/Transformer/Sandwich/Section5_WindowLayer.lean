module

public import Transformer.Sandwich.Section3_Biases
public import Transformer.Sandwich.Section5_Locality

/-!
# §5 The window of a windowed transformer

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §3.3 and §5: windowed attention of size `w`
lets the query `m` see the keys `n` with `n + w > m ≥ n`, and §5 counts the tokens an `R`-layer
model can see.

* `weight_score_eq_zero`: a masked key gets the weight `0`.
* `hasWindow_headOut`: a head whose bias masks every distance `≥ w` reads the `w` most recent
  tokens and is shift-invariant: its output at `s + m` is the output at `m` of the sequence
  without its first `s` tokens, once `m + 1 ≥ w`. The bias depends on the distance only, and the
  masked keys add nothing to the Softmax's sum.
* The layer, which applies LayerNorm position by position, has the same window
  (`Layer.hasWindow_apply`); the first `R − 1` layers have the window `(R − 1)(w − 1) + 1`
  (`hasWindow_hidden`), and the model's `o_m` and its prediction `R(w − 1) + 1`
  (`Model.hasWindow_out`, `Model.hasWindow_predict`).
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix Transformer.BAM

variable {d k H v w : ℕ}

/-- A key masked by the bias gets the weight `0`. -/
theorem weight_score_eq_zero {b : ℕ → EReal} (hb : IsWindowed w b) (A : Head d k) (x : Embs d)
    {m n : ℕ} (h : w ≤ m - n) : weight (score A b x m n) = 0 := by
  rw [score, hb _ h, EReal.add_bot, weight_bot]

/-- The hypotheses of `weight_score_eq_zero` hold for the key `0` of the query `1` under windowed
attention of size `1`. -/
example (A : Head d k) (x : Embs d) : weight (score A (windowBias 1) x 1 0) = 0 :=
  weight_score_eq_zero (isWindowed_windowBias 1) A x le_rfl

/-- **A windowed head reads the `w` most recent tokens.** -/
theorem isLocal_headOut {b : ℕ → EReal} (hb : IsWindowed w b) (A : Head d k) :
    IsLocal w (headOut A b) := by
  intro x y m h
  have hw : ∀ n ∈ range (m + 1), weight (score A b x m n) = weight (score A b y m n) := by
    intro n hn
    rw [mem_range] at hn
    by_cases hmn : m < n + w
    · rw [score, score, h n (by omega) hmn, h m le_rfl (by omega)]
    · rw [weight_score_eq_zero hb A x (by omega), weight_score_eq_zero hb A y (by omega)]
  have hZ : ∑ i ∈ range (m + 1), weight (score A b x m i) =
      ∑ i ∈ range (m + 1), weight (score A b y m i) := sum_congr rfl hw
  unfold headOut attnWeight
  refine sum_congr rfl fun n hn => ?_
  rw [hZ, hw n hn]
  by_cases hmn : m < n + w
  · rw [h n (by rw [mem_range] at hn; omega) hmn]
  · rw [weight_score_eq_zero hb A y (by omega), zero_div, zero_smul, zero_smul]

/-- The hypothesis of `isLocal_headOut` holds for windowed attention. -/
example (A : Head d k) : IsLocal w (headOut A (windowBias w)) :=
  isLocal_headOut (isWindowed_windowBias w) A

/-- **A windowed head is shift-invariant**: dropping the first `s` tokens does not change its
output at `s + m` once `m + 1 ≥ w`. -/
theorem isShiftInvariant_headOut {b : ℕ → EReal} (hb : IsWindowed w b) (A : Head d k) :
    IsShiftInvariant w (headOut A b) := by
  intro x s m hm
  have hscore : ∀ i, score A b x (s + m) (s + i) = score A b (fun j => x (s + j)) m i :=
    fun i => by simp only [score, Nat.add_sub_add_left]
  have hzero : ∀ n, n < s → weight (score A b x (s + m) n) = 0 := fun n hn =>
    weight_score_eq_zero hb A x (by omega)
  have hsplit : ∀ {M : Type} [AddCommMonoid M] (f : ℕ → M), (∀ n, n < s → f n = 0) →
      ∑ n ∈ range (s + m + 1), f n = ∑ i ∈ range (m + 1), f (s + i) := by
    intro M _ f hf
    rw [show s + m + 1 = s + (m + 1) by omega, sum_range_add,
      sum_eq_zero fun n hn => hf n (mem_range.1 hn), zero_add]
  have hZ : ∑ i ∈ range (s + m + 1), weight (score A b x (s + m) i) =
      ∑ i ∈ range (m + 1), weight (score A b (fun j => x (s + j)) m i) := by
    rw [hsplit _ hzero]
    simp only [hscore]
  simp only [headOut, attnWeight]
  rw [hZ, hsplit]
  · simp only [hscore]
  · intro n hn
    rw [hzero n hn, zero_div, zero_smul]

/-- The hypothesis of `isShiftInvariant_headOut` holds for windowed attention. -/
example (A : Head d k) : IsShiftInvariant w (headOut A (windowBias w)) :=
  isShiftInvariant_headOut (isWindowed_windowBias w) A

/-- A windowed head has the window `w`. -/
theorem hasWindow_headOut {b : ℕ → EReal} (hb : IsWindowed w b) (A : Head d k) :
    HasWindow w (headOut A b) :=
  ⟨isLocal_headOut hb A, isShiftInvariant_headOut hb A⟩

/-- The hypothesis of `hasWindow_headOut` holds for windowed attention. -/
example (A : Head d k) : HasWindow w (headOut A (windowBias w)) :=
  hasWindow_headOut (isWindowed_windowBias w) A

/-- The attention of a layer whose heads are all windowed has the window `w`. -/
theorem Attn.hasWindow_out (A : Attn d k H) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) : HasWindow w (A.out b) := by
  refine ⟨fun x y m hxy => ?_, fun x s m hm => ?_⟩
  · simp only [Attn.out]
    congr 1
    funext j
    rw [isLocal_headOut (hb j.1) _ x y m hxy]
  · simp only [Attn.out]
    congr 1
    funext j
    rw [isShiftInvariant_headOut (hb j.1) _ x s m hm]

/-- The hypothesis of `Attn.hasWindow_out` holds for windowed attention. -/
example (A : Attn d k H) : HasWindow w (A.out fun _ => windowBias w) :=
  A.hasWindow_out fun _ => isWindowed_windowBias w

/-- A layer whose heads are all windowed has the window `w`: LayerNorm acts position by
position. -/
theorem Layer.hasWindow_apply (ε : ℝ) (P : Layer d k H) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) : HasWindow w (P.apply ε b) :=
  (P.toAttn.hasWindow_out hb).map (layerNorm ε P.gain P.bias)

/-- The hypothesis of `Layer.hasWindow_apply` holds for windowed attention. -/
example (ε : ℝ) (P : Layer d k H) : HasWindow w (P.apply ε fun _ => windowBias w) :=
  P.hasWindow_apply ε fun _ => isWindowed_windowBias w

/-- **The first layers' window**: layers whose heads are all windowed with `w ≥ 1` have the
window `(number of layers)(w − 1) + 1`. -/
theorem hasWindow_hidden (ε : ℝ) {b : Fin H → ℕ → EReal} (hb : ∀ h, IsWindowed w (b h))
    (hw : 1 ≤ w) (Ps : List (Layer d k H)) :
    HasWindow (Ps.length * (w - 1) + 1) (hidden ε b Ps) := by
  induction Ps with
  | nil =>
    rw [List.length_nil, Nat.zero_mul, Nat.zero_add]
    exact hasWindow_id
  | cons P Ps ih =>
    have h := ih.comp (P.hasWindow_apply ε hb) hw (by omega)
    rw [List.length_cons, Nat.add_mul, Nat.one_mul]
    rwa [show w + (Ps.length * (w - 1) + 1) - 1 = Ps.length * (w - 1) + (w - 1) + 1 by
      omega] at h

/-- The hypotheses of `hasWindow_hidden` hold for windowed attention of size `1`. -/
example (ε : ℝ) (Ps : List (Layer d k H)) :
    HasWindow (Ps.length * (1 - 1) + 1) (hidden ε (fun _ => windowBias 1) Ps) :=
  hasWindow_hidden ε (fun _ => isWindowed_windowBias 1) le_rfl Ps

/-- The window of the model's last `o_m`: `R(w − 1) + 1` for `R` layers. -/
theorem Model.hasWindow_out (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) :
    HasWindow (M.depth * (w - 1) + 1) (M.out b) := by
  have h := (M.last.hasWindow_out hb).comp (hasWindow_hidden M.eps hb hw M.init) (by omega) hw
  rw [Model.depth, Nat.add_mul, Nat.one_mul]
  rwa [show M.init.length * (w - 1) + 1 + w - 1 = M.init.length * (w - 1) + (w - 1) + 1 by
    omega] at h

/-- The hypotheses of `Model.hasWindow_out` hold for windowed attention of size `1`. -/
example (M : Model d k H v) :
    HasWindow (M.depth * (1 - 1) + 1) (M.out fun _ => windowBias 1) :=
  M.hasWindow_out (fun _ => isWindowed_windowBias 1) le_rfl

/-- **The window of the prediction**: an `R`-layer model whose heads are all windowed with
`w ≥ 1` predicts the token after `m` from the tokens `n` with `n + R(w − 1) + 1 > m ≥ n`, and
in the same way at every position. -/
theorem Model.hasWindow_predict (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) :
    HasWindow (M.depth * (w - 1) + 1) (M.predict b) :=
  (M.hasWindow_out hb hw).map fun o => softmax fun j => ((M.V *ᵥ o) j : EReal)

/-- The hypotheses of `Model.hasWindow_predict` hold for windowed attention of size `1`. -/
example (M : Model d k H v) :
    HasWindow (M.depth * (1 - 1) + 1) (M.predict fun _ => windowBias 1) :=
  M.hasWindow_predict (fun _ => isWindowed_windowBias 1) le_rfl

end Transformer.Sandwich
