module

public import Transformer.RoPE.HWFAReRoPE.Section1_HWFA
public import Transformer.RoPE.ReRoPE.Section2_Fusion

/-!
# HWFA meets ReRoPE, 知新 (New): full RoPE layers read with ReRoPE

Su, *Transformer升级之路：14、当HWFA遇见ReRoPE*, kexue.fm/archives/9731, section 知新 (New):
"in training replace HWFA's Full NoPE Attention by Full RoPE Attention, and at inference change it
to Full ReRoPE Attention", ReRoPE being part 12's relative position `min(d, w)`
(`ReRoPE.rectifiedPosition`).

* `exists_alpha_iff`: HWFA's constraint "`(w̃ − 1)(L − 1) + 1 = αN`, `L` the number of layers,
  `N` the training length, `0 < α ≤ 1`" holds for some `α` exactly when the receptive field
  `(w̃ − 1)(L − 1) + 1` of the window layers (`hasWindow_foldr`) is at most `N`;
* `inferencePosition_lt`: "the window can be taken somewhat larger, the full attention can use
  RoPE and sit in middle layers, there can be more than one, … and thanks to ReRoPE the
  extrapolation does not drop". For any placement of the full layers and any input length, each
  relative position an unmasked query–key pair uses at inference is below `N` when `w̃ ≤ N` and
  ReRoPE's `w < N`: a window layer's distance is below `w̃`, a full layer's `min(d, w)` at most
  `w`. The positions stay in the trained range; the extrapolation itself is measured
  (`Section3_Table`);
* cost, counted in scores on an input of `n` tokens: a full layer computes
  `∑_{m<n}(m + 1) = n(n + 1)/2` (`sum_range_succ_mul_two`), a window layer
  `∑_{m<n} min(m + 1, w̃) ≤ nw̃` (`sum_card_windowKeys_le`), which also bounds its training cost
  with block-parallel kernels, and a full ReRoPE layer both of part 12's attention matrices, twice
  the full count. "The extra cost of switching to ReRoPE at inference becomes very small":
  `cost_lt_rerope`, the `f` full layers of HWFA+ReRoPE against the `L` of ReRoPE in a full model;
* the author's reply of 2023-08-31 on the post: "with 24 layers, all 24 attentions had to be
  computed twice, now only 2 do, and the other 22 cut the computation a lot; in the end it is even
  faster": `cost_lt` against `24` full RoPE layers when `11w̃ < 5(n + 1)`; `cost_gt`: on inputs no
  longer than the window it is slower.
-/

@[expose] public section

open Finset

namespace Transformer.RoPE.HWFAReRoPE

open ReRoPE

/-- New, HWFA's constraint: some `0 < α ≤ 1` has `(w̃ − 1)(L − 1) + 1 = αN` exactly when the
receptive field `(w̃ − 1)(L − 1) + 1` of the `L − 1` window layers is at most `N`. -/
theorem exists_alpha_iff (W L : ℕ) {N : ℕ} (hN : 0 < N) :
    (∃ α : ℝ, 0 < α ∧ α ≤ 1 ∧ (((W - 1) * (L - 1) + 1 : ℕ) : ℝ) = α * N) ↔
      (W - 1) * (L - 1) + 1 ≤ N := by
  have hN' : (0 : ℝ) < N := Nat.cast_pos.mpr hN
  constructor
  · rintro ⟨α, -, hα1, h⟩
    have : (((W - 1) * (L - 1) + 1 : ℕ) : ℝ) ≤ N := by
      rw [h]
      nlinarith
    exact_mod_cast this
  · intro h
    refine ⟨(((W - 1) * (L - 1) + 1 : ℕ) : ℝ) / N, div_pos (by positivity) hN',
      (div_le_one hN').mpr (by exact_mod_cast h), (div_mul_cancel₀ _ hN'.ne').symm⟩

example : ∃ α : ℝ, 0 < α ∧ α ≤ 1 ∧ (((16 - 1) * (24 - 1) + 1 : ℕ) : ℝ) = α * (512 : ℕ) :=
  (exists_alpha_iff 16 24 (by norm_num)).mpr (by norm_num)

/-- New: the relative position a layer of HWFA+ReRoPE uses at inference for the distance `d`: a
window layer RoPE's `d`, a full layer ReRoPE's `min(d, w)`. -/
noncomputable def inferencePosition (w : ℝ) (full : Bool) (d : ℕ) : ℝ :=
  if full then rectifiedPosition w d else d

/-- New: at inference every relative position HWFA+ReRoPE uses stays below the training length
`N`, in every layer, full or window, and for every causal pair the layer's mask keeps, when the
window `w̃` is at most `N` and ReRoPE's `w` below `N`. -/
theorem inferencePosition_lt {N W : ℕ} {w : ℝ} (hW : W ≤ N) (hw : w < N) (full : Bool)
    {m n : ℕ} (hnm : n ≤ m) (hmask : full = true ∨ m < n + W) :
    inferencePosition w full (m - n) < N := by
  cases full with
  | true =>
    simp only [inferencePosition, ite_true, rectifiedPosition]
    exact (min_le_right _ _).trans_lt hw
  | false =>
    simp only [inferencePosition, Bool.false_eq_true, ite_false]
    have : m - n < N := by
      rcases hmask with h | h
      · exact absurd h Bool.false_ne_true
      · omega
    exact_mod_cast this

/-- The runs of `Section3_Table`: `w̃ = 256` and ReRoPE's `w = 256` below `N = 512`; a full layer
and a window layer. -/
example (m n : ℕ) (hnm : n ≤ m) (hmask : m < n + 256) :
    inferencePosition 256 true (m - n) < (512 : ℕ) ∧
      inferencePosition 256 false (m - n) < (512 : ℕ) :=
  ⟨inferencePosition_lt (W := 256) (by norm_num) (by norm_num) true hnm (Or.inl rfl),
    inferencePosition_lt (by norm_num) (by norm_num) false hnm (Or.inr hmask)⟩

/-- New: a full causal layer computes `n(n + 1)/2` scores on `n` tokens. -/
theorem sum_range_succ_mul_two (n : ℕ) : (∑ m ∈ range n, (m + 1)) * 2 = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ, add_mul, ih]
    ring

/-- New: a window layer computes at most `nw̃` scores on `n` tokens. -/
theorem sum_card_windowKeys_le (w n : ℕ) : ∑ m ∈ range n, #(windowKeys w m) ≤ n * w := by
  calc ∑ m ∈ range n, #(windowKeys w m) ≤ ∑ _m ∈ range n, w :=
        sum_le_sum fun m _ => by rw [card_windowKeys]; exact min_le_right _ _
    _ = n * w := by rw [sum_const, card_range, smul_eq_mul]

/-- New: a window layer computes at most as many scores as a full layer, and as many on inputs
no longer than the window. -/
theorem sum_card_windowKeys_le_full (w n : ℕ) :
    ∑ m ∈ range n, #(windowKeys w m) ≤ ∑ m ∈ range n, (m + 1) ∧
      (n ≤ w → ∑ m ∈ range n, #(windowKeys w m) = ∑ m ∈ range n, (m + 1)) := by
  refine ⟨sum_le_sum fun m _ => by rw [card_windowKeys]; exact min_le_left _ _,
    fun hn => sum_congr rfl fun m hm => ?_⟩
  rw [card_windowKeys]
  rw [mem_range] at hm
  omega

/-- Summary, "the extra inference cost of ReRoPE is clearly lowered": with `f < L` full layers
on `n ≥ 1` tokens, HWFA+ReRoPE, its full layers computing both attention matrices, computes fewer
scores than ReRoPE in all `L` layers of a full model. -/
theorem cost_lt_rerope {L f w n : ℕ} (hf : f < L) (hn : 0 < n) :
    2 * f * ∑ m ∈ range n, (m + 1) + (L - f) * ∑ m ∈ range n, #(windowKeys w m) <
      2 * L * ∑ m ∈ range n, (m + 1) := by
  have hWF := (sum_card_windowKeys_le_full w n).1
  have hF : 0 < ∑ m ∈ range n, (m + 1) :=
    sum_pos (fun _ _ => Nat.succ_pos _) (nonempty_range_iff.mpr hn.ne')
  obtain ⟨g, rfl⟩ : ∃ g, L = f + (g + 1) := ⟨L - f - 1, by omega⟩
  rw [show f + (g + 1) - f = g + 1 by omega]
  have h1 := Nat.mul_le_mul_left (g + 1) hWF
  nlinarith

example : 2 * 2 * ∑ m ∈ range 512, (m + 1) + (24 - 2) * ∑ m ∈ range 512, #(windowKeys 64 m) <
    2 * 24 * ∑ m ∈ range 512, (m + 1) :=
  cost_lt_rerope (by norm_num) (by norm_num)

/-- The author's reply of 2023-08-31: with `24` layers, `2` full ReRoPE layers computing two
attention matrices and `22` window layers compute fewer scores than `24` full RoPE layers once
`11w̃ < 5(n + 1)`. -/
theorem cost_lt {w n : ℕ} (hn : 0 < n) (h : 11 * w < 5 * (n + 1)) :
    2 * (2 * ∑ m ∈ range n, (m + 1)) + 22 * ∑ m ∈ range n, #(windowKeys w m) <
      24 * ∑ m ∈ range n, (m + 1) := by
  have hF := sum_range_succ_mul_two n
  have hW := sum_card_windowKeys_le w n
  have h1 : n * (11 * w) < n * (5 * (n + 1)) := Nat.mul_lt_mul_of_pos_left h hn
  rw [show n * (11 * w) = 11 * (n * w) by ring,
    show n * (5 * (n + 1)) = 5 * (n * (n + 1)) by ring] at h1
  omega

/-- The runs' window `64` at the training length `512` and at the test length `4096`. -/
example : 2 * (2 * ∑ m ∈ range 512, (m + 1)) + 22 * ∑ m ∈ range 512, #(windowKeys 64 m) <
      24 * ∑ m ∈ range 512, (m + 1) ∧
    2 * (2 * ∑ m ∈ range 4096, (m + 1)) + 22 * ∑ m ∈ range 4096, #(windowKeys 64 m) <
      24 * ∑ m ∈ range 4096, (m + 1) :=
  ⟨cost_lt (by norm_num) (by norm_num), cost_lt (by norm_num) (by norm_num)⟩

/-- The reply's speed-up needs long inputs: on `1 ≤ n ≤ w̃` tokens the window layers compute as
much as full ones, and the `24`-layer HWFA+ReRoPE computes more scores than `24` full RoPE
layers. -/
theorem cost_gt {w n : ℕ} (hn : 0 < n) (hnw : n ≤ w) :
    24 * ∑ m ∈ range n, (m + 1) <
      2 * (2 * ∑ m ∈ range n, (m + 1)) + 22 * ∑ m ∈ range n, #(windowKeys w m) := by
  rw [(sum_card_windowKeys_le_full w n).2 hnw]
  have : 0 < ∑ m ∈ range n, (m + 1) :=
    sum_pos (fun _ _ => Nat.succ_pos _) (nonempty_range_iff.mpr hn.ne')
  omega

example : 24 * ∑ m ∈ range 64, (m + 1) <
    2 * (2 * ∑ m ∈ range 64, (m + 1)) + 22 * ∑ m ∈ range 64, #(windowKeys 64 m) :=
  cost_gt (by norm_num) le_rfl

end Transformer.RoPE.HWFAReRoPE
