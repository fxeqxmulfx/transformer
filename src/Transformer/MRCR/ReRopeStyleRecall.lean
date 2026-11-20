module

public import Transformer.MRCR.ReRopeStyleKeys
public import Transformer.MRCR.CountLookup

/-!
# Three real-softmax layers with ReRoPE and SSMax answer asks with a style

In no paper. Source of the task: MRCR-V2 (8-needle) of arXiv:2507.06261v6, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, keys of depth three spelled by words
(`Transformer.MRCR.PhraseKeyStyle`), and the query for "the entry with the desired index" of
arXiv:2409.12640v2, `eval_framework.tex`, lines 35–37. Every head is the real softmax at the
scale `s log n` of SSMax, arXiv:2501.19399v1, §2.2.

* Layer 1 (`Transformer.MRCR.ReRopeStyleKeys`): four ReRoPE slot heads at the scale `s log n`,
  rounded, and one ReLU unit per key give the position `x` the one-hot key `rowKey w x` of the
  ask that ends just before it, exactly at every length once `s A (1 − cos θ_m) > 4`.
* Layer 2 (`styleCount`): the counting head of `Transformer.MRCR.CountHead` at the scale
  `s₁ log n`, its raw score the dot product of these one-hot keys (`styleCount_eq`).
* Layer 3 (`styleLookup`): the lookup head of `Transformer.MRCR.CountLookup` at the scale
  `s₂ log n` at the query "`f` about `t` in the style `σ`, number `r`", reading the token of the
  position it finds (`styleLookup_eq`).

The positions of a key are those after its asks, so the position after the ask with `r` asks of
its key before it has `r + 1` positions of that key up to it (`occ_rowKey`); when `v` answers
the query, the token there is `val v`. So for every `s₂ > 1` some `b` makes the three layers
return the answer to every accuracy from some length on, for every number of needles, eight
included, whenever layer 2 runs at scales `s₁ x ≥ b` (`exists_forall_styleLookup`). The scales of
layers 1 and 2 may depend on the position, as Su's scale does (`Transformer.MRCR.ReRopeStyleSu`).

Under the plain softmax the slot heads fade instead (`Transformer.MRCR.ReRopeFading`). The rounded
transformers of `Transformer.MRCR.PhraseKeyStyleLogits` take six layers, as each of their reading
layers reads the token before, where one ReRoPE head reads any distance inside its window. As in
`Transformer.MRCR.CountLookup` the lookup head leaves out the query token; layer 1 attends over
the tokens only, without the BOS of layers 2 and 3.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall

variable {K F N S V R c : ℕ}

/-- **The count at the answer**: after an ask "`f` about `t` in the style `σ`" at the index `i`,
the positions of its key up to `i + 4` are those after its asks at the indices `≤ i`, whatever
injective code `e` the keys carry. -/
theorem occ_rowKey {e : Option (Fin F × Fin N × Fin S) → Fin c} (he : Function.Injective e)
    {w : List (StyleTok F N S V R)} {f : Fin F} {t : Fin N} {σ : Fin S} {i : ℕ}
    (hA : StyleAskAt w f t σ i) :
    occ (fun x => e (rowKey w x)) (i + 4) = Nat.count (StyleAskAt w f t σ) i + 1 := by
  have hk : rowKey w (i + 4) = some (f, t, σ) :=
    rowKey_eq_some.2 ⟨by omega, by rwa [Nat.add_sub_cancel]⟩
  rw [show Nat.count (StyleAskAt w f t σ) i + 1 = Nat.count (StyleAskAt w f t σ) (i + 1) by
      rw [Nat.count_succ, ite_eq_left hA], Nat.count_eq_card_filter_range, occ]
  refine card_nbij' (· - 4) (· + 4) (fun u hu => ?_) (fun j hj => ?_) (fun u hu => ?_)
    (fun j _ => ?_)
  · simp only [mem_coe, mem_filter, mem_range, hk, he.eq_iff, rowKey_eq_some] at hu ⊢
    exact ⟨by omega, hu.2.2⟩
  · simp only [mem_coe, mem_filter, mem_range, hk, he.eq_iff, rowKey_eq_some] at hj ⊢
    exact ⟨by omega, by omega, by rw [Nat.add_sub_cancel]; exact hj.2⟩
  · simp only [mem_coe, mem_filter, mem_range, hk, he.eq_iff, rowKey_eq_some] at hu
    show u - 4 + 4 = u
    omega
  · show j + 4 - 4 = j
    omega

/-- The hypothesis of `occ_rowKey` holds at the ask of the conversation "`0` about `0` in the
style `0`", answer `0`, query. -/
example := occ_rowKey (Fintype.equivFin _).injective (w := [StyleTok.fmt (0 : Fin 1), .about,
  .topic (0 : Fin 1), .style (0 : Fin 1), .val (0 : Fin 1), .query (R := 1) 0 0 0 0]) (f := 0)
  (t := 0) (σ := 0) (i := 0) ⟨rfl, rfl, rfl, rfl⟩

/-- **Layer 2 on layer 1**: the counting head at the scale `s₁ log n` over the BOS and the
positions `u ≤ x`, with the raw score `1` at the BOS and the dot product of the units of `u` and
`x` (`Transformer.MRCR.keyVec`) at `u`, reading `1` at the BOS and `0` at the positions. Layer 1
at the position `u` runs at the scale `s u`, which may depend on the position, as Su's does. -/
noncomputable def styleCount (s : ℕ → ℝ) (A : ℝ) (θ : Fin K → ℝ) (m : Fin K) (W : ℕ) (s₁ : ℝ)
    (w : List (StyleTok F N S V R)) (x : ℕ) : ℝ :=
  logScaleAttention s₁ (fun u : Option (Fin (x + 1)) => u.elim 1 fun u =>
      ∑ o, keyVec (s u) A θ m W w u o * keyVec (s x) A θ m W w x o)
    fun u => u.elim 1 fun _ => 0

/-- **Layer 3 on layers 1 and 2**: at the query "`f` about `t` in the style `σ`, number `r`"
closing `w`, attention at the scale `s₂ log n` over the BOS and the positions `x` before the query,
with the raw score `unit_{(f, t, σ)}(x) − K_r (count(x) − 1/(r + 2))²` and `0` at the BOS, reading
`φ` of the token at `x` and `0` at the BOS; layer 2 at the position `x` runs at the scale `s₁ x`. -/
noncomputable def styleLookup (s : ℕ → ℝ) (A : ℝ) (θ : Fin K → ℝ) (m : Fin K) (W : ℕ)
    (s₁ : ℕ → ℝ) (s₂ : ℝ) (φ : Option (StyleTok F N S V R) → ℝ) (w : List (StyleTok F N S V R))
    (f : Fin F) (t : Fin N) (σ : Fin S) (r : ℕ) : ℝ :=
  logScaleAttention s₂ (fun u : Option (Fin (w.length - 1)) => u.elim 0 fun x =>
      keyVec (s x) A θ m W w x (some (f, t, σ)) -
        lookupWeight r * (styleCount s A θ m W (s₁ x) w x - 1 / ((r : ℝ) + 2)) ^ 2)
    fun u => u.elim 0 fun x => φ w[(x : ℕ)]?

section Layers

variable {s : ℕ → ℝ} {A : ℝ} (hA : 0 ≤ A) (hs : ∀ x, 0 ≤ s x) {θ : Fin K → ℝ} {m : Fin K}
  (hθ : 0 < θ m) {W : ℕ} (hW : 4 < W) (hθW : θ m * W ≤ Real.pi)
  (hsg : ∀ x, 4 < s x * (A * (1 - Real.cos (θ m))))
include hA hs hθ hW hθW hsg

/-- Layer 2 on the units of layer 1 is the counting head on the keys `rowKey w`. -/
theorem styleCount_eq {e : Option (Fin F × Fin N × Fin S) → Fin c} (he : Function.Injective e)
    (s₁ : ℝ) (w : List (StyleTok F N S V R)) (x : ℕ) :
    styleCount s A θ m W s₁ w x = countHead s₁ (fun y => e (rowKey w y)) x := by
  unfold styleCount countHead
  congr 1
  funext u
  cases u with
  | none => rfl
  | some u => simp only [Option.elim,
      sum_keyVec_mul hA (hs u) hθ hW hθW (hsg u) w (hs x) (hsg x), he.eq_iff]

/-- Layer 3 on layers 1 and 2 is the lookup head on the keys `rowKey w`, reading the tokens. -/
theorem styleLookup_eq {e : Option (Fin F × Fin N × Fin S) → Fin c} (he : Function.Injective e)
    (s₁ : ℕ → ℝ) (s₂ : ℝ) (φ : Option (StyleTok F N S V R) → ℝ) (w : List (StyleTok F N S V R))
    (f : Fin F) (t : Fin N) (σ : Fin S) (r : ℕ) :
    styleLookup s A θ m W s₁ s₂ φ w f t σ r = lookupHead s₁ s₂ φ (fun y => e (rowKey w y))
      (fun y => w[y]?) (e (some (f, t, σ))) r (w.length - 1) := by
  unfold styleLookup lookupHead
  congr 1
  funext u
  cases u with
  | none => rfl
  | some x => simp only [Option.elim, keyVec_eq hA (hs x) hθ hW hθW (hsg x),
      styleCount_eq hA hs hθ hW hθW hsg he, he.eq_iff]

/-- **Three real-softmax layers answer asks with a style at every length** (MRCR-V2,
arXiv:2507.06261v6; MRCR, arXiv:2409.12640v2, `eval_framework.tex`): with layer 1 exact, for every
`s₂ > 1` some `b ≥ 1` makes layer 3 at the query that closes `w`, over counts at the scales
`s₁ x ≥ b`, return `φ (val v)`, `v` the answer, to every accuracy `ε > 0` for all conversations of
more than `M` tokens, for every value read `φ` bounded by `C`. -/
theorem exists_forall_styleLookup {s₂ C : ℝ} (hs₂ : 1 < s₂)
    {φ : Option (StyleTok F N S V R) → ℝ} (hφ : ∀ o, |φ o| ≤ C) :
    ∃ b : ℝ, 1 ≤ b ∧ ∀ ε : ℝ, 0 < ε → ∃ M : ℕ, ∀ s₁ : ℕ → ℝ, (∀ x, b ≤ s₁ x) →
      ∀ (w : List (StyleTok F N S V R)) (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R)
      (v : Fin V), M < w.length → w.getLast? = some (.query f t σ r) → StyleAnswer w v →
      |styleLookup s A θ m W s₁ s₂ φ w f t σ r - φ (some (.val v))| ≤ ε := by
  obtain ⟨b, hb, h⟩ := exists_forall_lookupHead_recall
    (c := Fintype.card (Option (Fin F × Fin N × Fin S))) R hs₂ hφ
  refine ⟨b, hb, fun ε hε => ?_⟩
  obtain ⟨M, hM⟩ := h ε hε
  refine ⟨M, fun s₁ hs₁ w f t σ r v hMw hq ⟨f', t', σ', r', hq', i, hA', hc, hv⟩ => ?_⟩
  rw [hq] at hq'
  simp only [Option.some.injEq, StyleTok.query.injEq] at hq'
  obtain ⟨rfl, rfl, rfl, rfl⟩ := hq'
  have hlt : i + 4 < w.length := (List.getElem?_eq_some_iff.1 hv).1
  have hne : i + 4 ≠ w.length - 1 := fun h' => by
    rw [List.getLast?_eq_getElem?, ← h', hv] at hq
    exact absurd hq (by simp)
  have hk : rowKey w (i + 4) = some (f, t, σ) :=
    rowKey_eq_some.2 ⟨by omega, by rwa [Nat.add_sub_cancel]⟩
  have he := (Fintype.equivFin (Option (Fin F × Fin N × Fin S))).injective
  rw [styleLookup_eq hA hs hθ hW hθW hsg he]
  have hR := hM (w.length - 1) (by omega) s₁ hs₁ (fun y => Fintype.equivFin _ (rowKey w y))
    (fun y => w[y]?) (Fintype.equivFin _ (some (f, t, σ))) r (i + 4) r.isLt (by omega)
    (by simp only [hk]) (by rw [occ_rowKey he hA', hc])
  simpa only [hv] using hR

end Layers

/-- The hypotheses of `styleCount_eq` and `styleLookup_eq` hold at the scale `40` at every
position, `A = 1`, `θ = π/6` and the window `6`. -/
example (w : List (StyleTok 1 1 1 1 8)) (s₁ : ℝ) (x : ℕ) :=
  styleCount_eq (s := fun _ => 40) (A := 1) zero_le_one (fun _ => by norm_num)
    (θ := fun _ : Fin 1 => Real.pi / 6) (m := 0) (by positivity) (W := 6) (by norm_num)
    (by push_cast; linarith) (fun _ => four_lt_forty_mul_one_sub_cos)
    (Fintype.equivFin _).injective s₁ w x

example (w : List (StyleTok 1 1 1 1 8)) (s₁ : ℕ → ℝ) (s₂ : ℝ)
    (φ : Option (StyleTok 1 1 1 1 8) → ℝ) :=
  styleLookup_eq (s := fun _ => 40) (A := 1) zero_le_one (fun _ => by norm_num)
    (θ := fun _ : Fin 1 => Real.pi / 6) (m := 0) (by positivity) (W := 6) (by norm_num)
    (by push_cast; linarith) (fun _ => four_lt_forty_mul_one_sub_cos)
    (Fintype.equivFin _).injective s₁ s₂ φ w 0 0 0 3

/-- Eight needles at the scale `40` at every position, `A = 1`, `θ = π/6` and the window `6`,
with `s₂ = 2` and the value read `1` at `val 0` and `0` elsewhere. -/
example := exists_forall_styleLookup (F := 1) (N := 1) (S := 1) (V := 2) (R := 8)
  (s := fun _ => 40) (A := 1) zero_le_one (fun _ => by norm_num)
  (θ := fun _ : Fin 1 => Real.pi / 6) (m := 0) (by positivity) (W := 6) (by norm_num)
  (by push_cast; linarith) (fun _ => four_lt_forty_mul_one_sub_cos) (s₂ := 2) (C := 1)
  (by norm_num) (φ := fun o => if o = some (.val 0) then 1 else 0) fun o => by split_ifs <;> simp

end Transformer.MRCR
