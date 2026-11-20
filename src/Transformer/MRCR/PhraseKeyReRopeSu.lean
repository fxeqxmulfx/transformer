module

public import Transformer.MRCR.PhraseKeyReRope
public import Transformer.RoPE.ReRoPE.Section4_Experiments

/-!
# ReRoPE's own formulas behind the clamped logit

Source: Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708
(`papers/kexue-9708-rerope`), formalized in `Transformer.RoPE.ReRoPE`.
`Transformer.MRCR.reropeLogit` adds to the dot product the RoPE score at the distance
`min (i − j) W`. This file derives that logit from the source's formulas.

* `reropeLogit_eq_rectifiedPosition`: the clamped distance is the position of Su's matrix
  (rerope), `rectifiedPosition W (i − j) = min(i − j, W)`.
* `dotProduct_rotary_pair`: the sine coefficient `B` of one frequency is the score of the
  constant pair `q = e_{(0,m)}`, `k = B e_{(1,m)}`: `qᵀ R^{−t} k = B sin(θ_m t)`.
* `reropeLogit_eq_merge`: the logit is the dot product plus Su's merged score of that pair,
  the first matrix `(R^i q)ᵀ(R^j k)` inside the window and the second `(R^W q)ᵀ k` outside it
  (`Transformer.RoPE.ReRoPE.merge_rectified`).
* `exists_logScale_rerope_phraseAnswer`: Su's `log n` runs (实验, Experiments) multiply each
  query `q_n` by `max(1, log_T n)` (`Transformer.RoPE.ReRoPE.logScale`). The rotation keeps the
  scale, so the whole logit is multiplied by it, at every training length `T`. A factor of at
  least `1` keeps `ReadsPrev` (`readsPrev_mul`), and five layers answer spelled asks with that
  logit too. It is not SSMax, whose `s log n` vanishes at `n = 1` and grows from the start.

One deviation from the source remains. The dot product `x` is not rotated: the query and key
carry their content in coordinates that no rotation touches, as in partial RoPE. In Su's
ReRoPE every plane rotates.
-/

@[expose] public section

namespace Transformer.MRCR

open Matrix Transformer.RoPE.RoFormer
open Transformer.CRASP Transformer.CRASP.Grams
open Transformer.RoPE.ReRoPE (rectifiedPosition merge_rectified logScale)

variable {K F N V R : ℕ}

/-- **The clamped distance is Su's**: `reropeLogit` takes the RoPE score at the position of the
matrix (rerope) of kexue.fm/archives/9708, `rectifiedPosition W (i − j) = min(i − j, W)`. -/
theorem reropeLogit_eq_rectifiedPosition (c : ℝ) (θ a b : Fin K → ℝ) (W i j : ℕ) (x : ℝ) :
    reropeLogit c θ a b W i j x =
      x / c + ropeScore θ a b (rectifiedPosition W ((i - j : ℕ) : ℝ)) := by
  rw [reropeLogit, rectifiedPosition, Nat.cast_min]

/-- **One sine is the score of a constant pair**: `e_{(0,m)}ᵀ R^{−t} (B e_{(1,m)}) = B sin(θ_m t)`,
with RoFormer's rotation `R^t` (arXiv:2104.09864v5, Eq. (15)). -/
theorem dotProduct_rotary_pair (θ : Fin K → ℝ) (m : Fin K) (B t : ℝ) :
    Pi.single (0, m) 1 ⬝ᵥ (rotary θ (-t) *ᵥ Pi.single (1, m) B) = B * Real.sin (θ m * t) := by
  simp [rotary, blockDiagonal_apply, Transformer.RoPE.RoFormer.rotation, mul_comm]

/-- **The logit is the dot product plus Su's merged score**: for `j ≤ i`, `reropeLogit` with the
sine coefficient `B` at the frequency `θ m` is `x / c` plus the merged attention score of
kexue.fm/archives/9708, 计算 (Computation), of the pair `q = e_{(0,m)}`, `k = B e_{(1,m)}`:
`(R^i q)ᵀ(R^j k)` for `i − j < W` and `(R^W q)ᵀ k` otherwise. -/
theorem reropeLogit_eq_merge (c : ℝ) (θ : Fin K → ℝ) (m : Fin K) (B : ℝ) (W : ℕ) {i j : ℕ}
    (hji : j ≤ i) (x : ℝ) :
    reropeLogit c θ 0 (Pi.single m B) W i j x = x / c +
      (if (i : ℝ) - j < W then
          (rotary θ i *ᵥ Pi.single (0, m) 1) ⬝ᵥ (rotary θ j *ᵥ Pi.single (1, m) B)
        else (rotary θ W *ᵥ Pi.single (0, m) 1) ⬝ᵥ Pi.single (1, m) B) := by
  rw [reropeLogit_eq_rectifiedPosition, merge_rectified,
    dotProduct_rotary_pair, ropeScore_single, Nat.cast_sub hji]

/-- The hypothesis of `reropeLogit_eq_merge` holds at the positions `7 ≥ 2`, the window `512`. -/
example := reropeLogit_eq_merge 1 (fun _ : Fin 1 => Real.pi / 1024) 0 (-3) 512
  (i := 7) (j := 2) (by norm_num) 0

/-- **A factor of at least `1` keeps `ReadsPrev`**: the logit at the query position `i` multiplied
by `g i ≥ 0`, with `g i ≥ 1` for `i > 0`. -/
theorem readsPrev_mul {L : ℕ → ℕ → ℝ → ℝ} {b : ℕ} (hL : ReadsPrev L b) {g : ℕ → ℝ}
    (hg0 : ∀ i, 0 ≤ g i) (hg : ∀ i, 0 < i → 1 ≤ g i) :
    ReadsPrev (fun i j x => g i * L i j x) b := by
  have hle : ∀ i, 0 < i → ∀ z : ℝ, z ≤ -1 → g i * z ≤ -1 := fun i hi z hz => by
    nlinarith [hg i hi]
  refine ⟨fun i => by simp [hL.self i], fun i hi => by simp [hL.prev i hi],
    fun i hi => hle i hi _ (hL.prevSame i hi),
    fun i j hij => ⟨hle i (by omega) _ (hL.far i j hij).1, hle i (by omega) _ (hL.far i j hij).2⟩,
    fun i => ?_⟩
  rcases (hg0 i).eq_or_lt with h0 | hpos
  · exact Or.inr fun j _ => by simp [← h0]
  · rcases hL.run i with h | h
    · exact Or.inl fun j hj => mul_neg_of_pos_of_neg hpos (h j hj)
    · exact Or.inr fun j hj => by simp [h j hj]

/-- The hypotheses of `readsPrev_mul` hold for ALiBi of slope `1` and the factor `2`. -/
example := readsPrev_mul (readsPrev_alibi le_rfl) (g := fun _ => 2) (fun _ => zero_le_two)
  (fun _ _ => one_le_two)

/-- Su's scale `max(1, log_T n)` is at least `1`. -/
theorem one_le_logScale (T n : ℝ) : 1 ≤ logScale T n :=
  le_max_left _ _

/-- **ReRoPE with Su's `log n` scale answers spelled asks**: five layers, the logit of
`exists_rerope_phraseAnswer` multiplied at the query position `i` by `max(1, log_T (i + 1))`, the
scale of the `log n` runs of kexue.fm/archives/9708 and of `rerope_patch.py`, at every training
length `T`. -/
theorem exists_logScale_rerope_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d) (T : ℝ)
    {θ : Fin K → ℝ} {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 2 ≤ W)
    (hθW : θ m * (W + 2) ≤ Real.pi) (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (Tr : LTfr (Option (PhraseTok F N V R)) p 0 d 5),
      Tr.logit = (fun (i j : ℕ) (x : ℝ) => logScale T (i + 1) * reropeLogit (c d) θ 0 b W i j x) ∧
      Tr.Recognizes {w | PhraseAnswer w v} := by
  obtain ⟨p, d, Tr, -, hT, hR⟩ := exists_phraseAnswer (F := F) (N := N) (R := R)
    (fun d i j x => logScale T ((i : ℝ) + 1) * reropeLogit (c d) θ 0
      (Pi.single m (-(((3 * ⌈c d⌉₊ : ℕ) : ℝ) / c d) / Real.sin (θ m))) W i j x)
    (fun d => 3 * ⌈c d⌉₊) (fun d hd => readsPrev_mul (readsPrev_rerope (hc d hd)
      (by push_cast; linarith [Nat.le_ceil (c d)]) hθ hW hθW)
      (fun i => zero_le_one.trans (one_le_logScale _ _)) (fun i _ => one_le_logScale _ _)) v
  exact ⟨p, d, _, Tr, hT, hR⟩

/-- The hypotheses of `exists_logScale_rerope_phraseAnswer` hold for Su's training length `4096`
and window `512`, the frequency `π/1024` and the divisor `√d`. -/
example := exists_logScale_rerope_phraseAnswer (F := 1) (N := 1) (R := 8) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) 4096
  (θ := fun _ : Fin 1 => Real.pi / 1024) (m := 0) (by positivity) (W := 512) (by norm_num)
  (by push_cast; linarith [Real.pi_pos]) (0 : Fin 1)

end Transformer.MRCR
