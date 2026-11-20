module

public import Transformer.RoPE.RoFormer.Section3_Properties
public import Transformer.Sandwich.Section5_Locality

/-!
# HWFA meets ReRoPE, 温故 (Review): HWFA's window layers

Su Jianlin, *Transformer升级之路：14、当HWFA遇见ReRoPE* ("The road to upgrading the Transformer,
14: when HWFA meets ReRoPE"), kexue.fm/archives/9731, 2023-08-24. The post's sections 温故
(Review), 知新 (New), 实验 (Experiments) and 小结 (Summary) are unnumbered; the files count them
1–4. HWFA is part 9, kexue.fm/archives/9603, where the window is `w`; here it is `w̃`, and `w` is
ReRoPE's.

Review: "HWFA is '`L − 1` layers of Window RoPE Attention + `1` layer of Full NoPE Attention':
the first `L − 1` attention layers add RoPE and limit the receptive field by a window, so that
the inference cost becomes constant."

* `windowHead θ Q K V w`: a RoPE attention head with the window `w`. The query at `m` reads the
  keys `n ≤ m` with `m < n + w` (`windowKeys`), weighs them by the Softmax of the RoPE scores
  `(R^m Q(x_m))ᵀ(R^n K(x_n))` and sums the values `V(x_n)`. The maps `Q`, `K`, `V` of a token are
  arbitrary, which covers GAU's queries and keys from one shared projection;
* `card_windowKeys`: the query at `m` reads `min(m + 1, w) ≤ w` keys, a bound independent of `m`:
  "the inference cost becomes constant";
* `hasWindow_windowHead`: the head reads the `w` most recent tokens and is shift-invariant, since
  RoPE's score depends on `n − m` only (`RoFormer.rotary_dotProduct`); `hasWindow_windowLayer`: a
  layer combining the head's output with its token position by position, as a residual
  connection, GAU's gate or a feed-forward network does, keeps the window;
* `hasWindow_foldr`: `L − 1` such layers have the window `(w − 1)(L − 1) + 1`, the receptive
  field of HWFA's constraint `(w̃ − 1)(L − 1) + 1 = αN` (`Section2_Combination`);
  `ncard_contributes_foldr_le`: at most that many tokens contribute to an output. The windows are
  those of `Transformer.Sandwich.HasWindow` (Chi et al., arXiv:2212.10356v2, §5).

Not formalized: the comparison with FOT (Focused Transformer), a paper not formalized here, and
the unreported crossbatch-training runs on HWFA.
-/

@[expose] public section

open Finset Matrix

namespace Transformer.RoPE.HWFAReRoPE

open RoFormer Sandwich

variable {p : ℕ} {α β γ : Type*}

/-- Review: the keys of the query `m` under the window `w`, the positions `n ≤ m` with
`m < n + w`. -/
def windowKeys (w m : ℕ) : Finset ℕ :=
  (range (m + 1)).filter fun n => m < n + w

/-- Review: the query at `m` reads `min(m + 1, w)` keys, at most `w` at every position: the
inference cost of a window layer per token is bounded independently of the position. -/
theorem card_windowKeys (w m : ℕ) : #(windowKeys w m) = min (m + 1) w := by
  have h : windowKeys w m = Ico (m + 1 - w) (m + 1) := by
    ext n
    simp only [windowKeys, mem_filter, mem_range, mem_Ico]
    omega
  rw [h, Nat.card_Ico]
  omega

/-- Review: the RoPE score of the query at `m` and the key at `n`, `(R^m Q(x_m))ᵀ(R^n K(x_n))`. -/
noncomputable def ropeScore (θ : Fin p → ℝ) (Q K : α → Fin 2 × Fin p → ℝ) (x : ℕ → α)
    (m n : ℕ) : ℝ :=
  (rotary θ (m : ℝ) *ᵥ Q (x m)) ⬝ᵥ (rotary θ (n : ℝ) *ᵥ K (x n))

/-- Review: Window RoPE Attention, a head whose query at `m` weighs the values `V(x_n)` of the
keys `windowKeys w m` by the Softmax of the RoPE scores. -/
noncomputable def windowHead [AddCommMonoid β] [Module ℝ β] (θ : Fin p → ℝ)
    (Q K : α → Fin 2 × Fin p → ℝ) (V : α → β) (w : ℕ) (x : ℕ → α) (m : ℕ) : β :=
  ∑ n ∈ windowKeys w m, (Real.exp (ropeScore θ Q K x m n) /
    ∑ n' ∈ windowKeys w m, Real.exp (ropeScore θ Q K x m n')) • V (x n)

/-- RoPE's score is unchanged when both positions move by `s` (Eq. (19) of RoFormer). -/
theorem ropeScore_add (θ : Fin p → ℝ) (Q K : α → Fin 2 × Fin p → ℝ) (x : ℕ → α) (s m n : ℕ) :
    ropeScore θ Q K x (s + m) (s + n) = ropeScore θ Q K (fun j => x (s + j)) m n := by
  simp only [ropeScore, rotary_dotProduct, Nat.cast_add, add_sub_add_left_eq_sub]

/-- Review: the window head reads the `w` most recent tokens. -/
theorem isLocal_windowHead [AddCommMonoid β] [Module ℝ β] (θ : Fin p → ℝ)
    (Q K : α → Fin 2 × Fin p → ℝ) (V : α → β) (w : ℕ) : IsLocal w (windowHead θ Q K V w) := by
  intro x y m h
  have hs : ∀ n ∈ windowKeys w m, ropeScore θ Q K x m n = ropeScore θ Q K y m n ∧ x n = y n := by
    intro n hn
    obtain ⟨hn1, hn2⟩ := mem_filter.mp hn
    rw [mem_range] at hn1
    have hxn := h n (by omega) hn2
    have hxm := h m le_rfl (by omega)
    exact ⟨by simp only [ropeScore, hxm, hxn], hxn⟩
  have hZ : ∑ n' ∈ windowKeys w m, Real.exp (ropeScore θ Q K x m n') =
      ∑ n' ∈ windowKeys w m, Real.exp (ropeScore θ Q K y m n') :=
    sum_congr rfl fun n hn => by rw [(hs n hn).1]
  unfold windowHead
  rw [hZ]
  exact sum_congr rfl fun n hn => by rw [(hs n hn).1, (hs n hn).2]

/-- The keys of the query `s + m` are those of `m` moved by `s`, once `m + 1 ≥ w`. -/
theorem windowKeys_add {w m : ℕ} (hm : w ≤ m + 1) (s : ℕ) :
    windowKeys w (s + m) = (windowKeys w m).map (addLeftEmbedding s) := by
  ext n
  simp only [windowKeys, mem_filter, mem_range, mem_map, addLeftEmbedding_apply]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨n - s, ⟨by omega, by omega⟩, by omega⟩
  · rintro ⟨i, ⟨h1, h2⟩, rfl⟩
    omega

/-- HWFA's window `16`: the query `3 + 15` sees the keys of `15` moved by `3`. -/
example : windowKeys 16 (3 + 15) = (windowKeys 16 15).map (addLeftEmbedding 3) :=
  windowKeys_add (by norm_num) 3

/-- Review: the window head is shift-invariant: dropping the first `s` tokens does not change its
output at `s + m` once `m + 1 ≥ w`. -/
theorem isShiftInvariant_windowHead [AddCommMonoid β] [Module ℝ β] (θ : Fin p → ℝ)
    (Q K : α → Fin 2 × Fin p → ℝ) (V : α → β) (w : ℕ) :
    IsShiftInvariant w (windowHead θ Q K V w) := by
  intro x s m hm
  unfold windowHead
  simp only [windowKeys_add hm s, sum_map, addLeftEmbedding_apply, ropeScore_add]

/-- Review: the window head has the window `w`. -/
theorem hasWindow_windowHead [AddCommMonoid β] [Module ℝ β] (θ : Fin p → ℝ)
    (Q K : α → Fin 2 × Fin p → ℝ) (V : α → β) (w : ℕ) : HasWindow w (windowHead θ Q K V w) :=
  ⟨isLocal_windowHead θ Q K V w, isShiftInvariant_windowHead θ Q K V w⟩

/-- Review: a layer combining, position by position, its token with the window head's output, as
a residual connection, GAU's gate or a feed-forward network does, keeps the window `w ≥ 1`. -/
theorem hasWindow_windowLayer [AddCommMonoid β] [Module ℝ β] (θ : Fin p → ℝ)
    (Q K : α → Fin 2 × Fin p → ℝ) (V : α → β) {w : ℕ} (hw : 1 ≤ w) (φ : α → β → γ) :
    HasWindow w fun x m => φ (x m) (windowHead θ Q K V w x m) := by
  refine ⟨fun x y m h => ?_, fun x s m hm => ?_⟩
  · simp only
    rw [h m le_rfl (by omega), isLocal_windowHead θ Q K V w x y m h]
  · simp only
    rw [isShiftInvariant_windowHead θ Q K V w x s m hm]

/-- A residual layer of Window RoPE Attention with the window `16` of part 9. -/
example (θ : Fin 64 → ℝ) (Q K V : (Fin 2 × Fin 64 → ℝ) → Fin 2 × Fin 64 → ℝ) :
    HasWindow 16 fun x m => x m + windowHead θ Q K V 16 x m :=
  hasWindow_windowLayer θ Q K V (by norm_num) (· + ·)

/-- Review: layers that each have the window `w ≥ 1`, applied one after the other, have the
window `(w − 1)·(number of layers) + 1`. -/
theorem hasWindow_foldr {w : ℕ} (hw : 1 ≤ w) :
    ∀ Fs : List ((ℕ → α) → ℕ → α), (∀ F ∈ Fs, HasWindow w F) →
      HasWindow ((w - 1) * Fs.length + 1) (Fs.foldr (fun F G x => F (G x)) fun x => x)
  | [], _ => by simpa using hasWindow_id
  | F :: Fs, h => by
    have ih := hasWindow_foldr hw Fs fun G hG => h G (List.mem_cons_of_mem F hG)
    have hc := (h F List.mem_cons_self).comp ih (by omega) hw
    rw [List.foldr_cons, List.length_cons]
    rwa [show (w - 1) * Fs.length + 1 + w - 1 = (w - 1) * (Fs.length + 1) + 1 by
      rw [Nat.mul_succ]; omega] at hc

/-- HWFA's `23` window layers of part 9, residual layers of window `16`, have the window
`15 · 23 + 1 = 346`. -/
example (θ : Fin 64 → ℝ) (Q K V : (Fin 2 × Fin 64 → ℝ) → Fin 2 × Fin 64 → ℝ) :
    HasWindow ((16 - 1) * (List.replicate 23
        fun (x : ℕ → Fin 2 × Fin 64 → ℝ) m => x m + windowHead θ Q K V 16 x m).length + 1)
      ((List.replicate 23 fun (x : ℕ → Fin 2 × Fin 64 → ℝ) m => x m + windowHead θ Q K V 16 x m
        ).foldr (fun F G x => F (G x)) fun x => x) :=
  hasWindow_foldr (by norm_num) _ fun F hF => by
    rw [List.eq_of_mem_replicate hF]
    exact hasWindow_windowLayer θ Q K V (by norm_num) (· + ·)

/-- Review: at most `(w − 1)·(number of layers) + 1` tokens contribute to each output of layers
that each have the window `w ≥ 1`: the receptive field. -/
theorem ncard_contributes_foldr_le {w : ℕ} (hw : 1 ≤ w) {Fs : List ((ℕ → α) → ℕ → α)}
    (h : ∀ F ∈ Fs, HasWindow w F) (m : ℕ) :
    {n | Contributes (Fs.foldr (fun F G x => F (G x)) fun x => x) m n}.ncard ≤
      (w - 1) * Fs.length + 1 :=
  (hasWindow_foldr hw Fs h).1.ncard_contributes_le m

example (θ : Fin 64 → ℝ) (Q K V : (Fin 2 × Fin 64 → ℝ) → Fin 2 × Fin 64 → ℝ) (m : ℕ) :
    {n | Contributes ([fun (x : ℕ → Fin 2 × Fin 64 → ℝ) m => x m + windowHead θ Q K V 16 x m
        ].foldr (fun F G x => F (G x)) fun x => x) m n}.ncard ≤ (16 - 1) * 1 + 1 :=
  ncard_contributes_foldr_le (by norm_num)
    (fun F hF => by
      rw [List.mem_singleton.mp hF]
      exact hasWindow_windowLayer θ Q K V (by norm_num) (· + ·)) m

end Transformer.RoPE.HWFAReRoPE
