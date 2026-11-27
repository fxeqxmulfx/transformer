module

public import Transformer.Grokking.FiniteRecovery.Series
public import Transformer.Grokking.DiffGPT.StepCarry

/-! First differences on a supplied DiffGPT level grid. The fractional
anchors, level rounding, step correction and vocabulary clipping follow
DiffGPT's native codec at 5a37c515, for lag one. Scales, zeros and initial
fractional seeds are inputs. The real-valued equations describe the
codec with those compiled values. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

/-- The three schemes select different fractional anchors on the same
first-difference equations. -/
inductive SeriesTokenizer where
  | lgt
  | mash
  | mashHalf

structure FirstDifferenceGrid where
  tokenizer : SeriesTokenizer
  scale : ℝ
  zero : ℤ
  seed : ℝ

namespace FirstDifferenceGrid

/-- LGT uses zero. MASH carries the initial fractional seed, and MASH-half
advances it by half a grid unit per step, modulo one. -/
noncomputable def fraction (grid : FirstDifferenceGrid) (n : ℕ) : ℝ :=
  match grid.tokenizer with
  | .lgt => 0
  | .mash => Int.fract grid.seed
  | .mashHalf => Int.fract (grid.seed + (n : ℝ) / 2)

noncomputable def level (grid : FirstDifferenceGrid) (n : ℕ) (x : ℝ) : ℤ :=
  ⌊grid.scale * x - grid.fraction n + 1 / 2⌋

/-- The step offset includes the integer part of the change in the
fractional anchor (DiffGPT native codec, anchor). -/
noncomputable def offset (grid : FirstDifferenceGrid) (n : ℕ) : ℤ :=
  grid.zero + ⌊grid.fraction (n + 1) - grid.fraction n⌋

noncomputable def step (grid : FirstDifferenceGrid) (n : ℕ) (x y : ℝ) : ℤ :=
  grid.offset n + grid.level (n + 1) y - grid.level n x

/-- The decoder carries a reconstructed level from the known prefix. -/
noncomputable def decodeLevel (grid : FirstDifferenceGrid) (n : ℕ)
    (previous token : ℤ) : ℤ :=
  previous + token - grid.offset n

noncomputable def valueAtLevel (grid : FirstDifferenceGrid) (n : ℕ) (z : ℤ) : ℝ :=
  ((z : ℝ) + grid.fraction n) / grid.scale

theorem decodeLevel_step (grid : FirstDifferenceGrid) (n : ℕ) (x y : ℝ) :
    grid.decodeLevel n (grid.level n x) (grid.step n x y) = grid.level (n + 1) y := by
  unfold decodeLevel step
  omega

/-- Before clipping, one encode/decode step has at most half a grid unit
of error in the raw value when the previous grid level is correct. -/
theorem abs_decode_step_sub_le (grid : FirstDifferenceGrid) (n : ℕ)
    (x y : ℝ) (hscale : 0 < grid.scale) :
    |grid.valueAtLevel (n + 1) (grid.decodeLevel n (grid.level n x) (grid.step n x y)) - y| ≤
      (1 / 2) / grid.scale := by
  rw [decodeLevel_step, valueAtLevel]
  have hsub :
      ((grid.level (n + 1) y : ℝ) + grid.fraction (n + 1)) / grid.scale - y =
        ((grid.level (n + 1) y : ℝ) + grid.fraction (n + 1) - grid.scale * y) / grid.scale := by
    field_simp
  rw [hsub, abs_div, abs_of_pos hscale]
  apply div_le_div_of_nonneg_right _ hscale.le
  rw [abs_le]
  have hlo := Int.lt_floor_add_one (grid.scale * y - grid.fraction (n + 1) + 1 / 2)
  have hhi := Int.floor_le (grid.scale * y - grid.fraction (n + 1) + 1 / 2)
  unfold level
  constructor <;> linarith

/-- The vocabulary is finite even if a raw series leaves its declared domain. -/
def clipToken {vocab : ℕ} (hv : 0 < vocab) (z : ℤ) : Fin vocab :=
  ⟨min z.toNat (vocab - 1), (min_le_right _ _).trans_lt (by omega)⟩

theorem clipToken_val_eq {vocab : ℕ} (hv : 0 < vocab) {z : ℤ}
    (h0 : 0 ≤ z) (hV : z < vocab) : ((clipToken hv z).val : ℤ) = z := by
  change (↑(min z.toNat (vocab - 1)) : ℤ) = z
  rw [min_eq_left (by omega), Int.toNat_of_nonneg h0]

/-- A word holds the encoded steps of the past. The right endpoint of
every step is part of the supplied prefix. -/
noncomputable def contextWord {T vocab : ℕ} (grid : FirstDifferenceGrid) (hv : 0 < vocab)
    (past : Fin (T + 1) → ℝ) : Word (Fin vocab) T :=
  fun j => clipToken hv (grid.step j.val (past j.castSucc) (past j.succ))

/-- The decoder reconstructs the last level from the anchor and the encoded
prefix, including any vocabulary clipping that took place in the prefix. -/
noncomputable def prefixLevel {T vocab : ℕ} (grid : FirstDifferenceGrid) (hv : 0 < vocab)
    (past : Fin (T + 1) → ℝ) : ℤ :=
  grid.level 0 (past 0) + ∑ j : Fin T, (((grid.contextWord hv past j).val : ℤ) - grid.offset j.val)

theorem prefixLevel_eq_of_no_clip {T vocab : ℕ} (grid : FirstDifferenceGrid) (hv : 0 < vocab)
    (past : Fin (T + 1) → ℝ)
    (h : ∀ j : Fin T, 0 ≤ grid.step j.val (past j.castSucc) (past j.succ) ∧
      grid.step j.val (past j.castSucc) (past j.succ) < vocab) :
    grid.prefixLevel hv past = grid.level T (past (Fin.last T)) := by
  let levelAt (j : Fin (T + 1)) := grid.level j.val (past j)
  have hs : (∑ j : Fin T, (((grid.contextWord hv past j).val : ℤ) - grid.offset j.val)) =
      ∑ j : Fin T, (levelAt j.succ - levelAt j.castSucc) := by
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [contextWord, clipToken_val_eq hv (h j).1 (h j).2]
    simp only [step, levelAt, Fin.val_succ, Fin.val_castSucc]
    ring
  rw [prefixLevel, hs, Finset.sum_sub_distrib]
  have htel := (Fin.sum_univ_succ levelAt).symm.trans (Fin.sum_univ_castSucc levelAt)
  change levelAt 0 + ((∑ j : Fin T, levelAt j.succ) - ∑ j : Fin T, levelAt j.castSucc) =
    levelAt (Fin.last T)
  omega

/-- For a batch, each site supplies its own compiled grid and past. The
raw target is indexed by the grid level at the next position. -/
noncomputable def windowEncoding {I : Type*} {T vocab : ℕ} (hv : 0 < vocab)
    (grid : I → FirstDifferenceGrid) (past : I → Fin (T + 1) → ℝ) :
    SeriesEncoding I (Word (Fin vocab) T) (Fin vocab) ℤ where
  context i := (grid i).contextWord hv (past i)
  decode i token := (grid i).decodeLevel T ((grid i).prefixLevel hv (past i)) token.val

/-- With no clipping, the encoded next step decodes to its grid level,
using only the grid and the observed prefix as decoder state. -/
theorem windowEncoding_decode_step {I : Type*} {T vocab : ℕ} (hv : 0 < vocab)
    (grid : I → FirstDifferenceGrid) (past : I → Fin (T + 1) → ℝ) (i : I) (future : ℝ)
    (hprefix : ∀ j : Fin T, 0 ≤ (grid i).step j.val (past i j.castSucc) (past i j.succ) ∧
      (grid i).step j.val (past i j.castSucc) (past i j.succ) < vocab)
    (h0 : 0 ≤ (grid i).step T (past i (Fin.last T)) future)
    (hV : (grid i).step T (past i (Fin.last T)) future < vocab) :
    (windowEncoding hv grid past).decode i
      (clipToken hv ((grid i).step T (past i (Fin.last T)) future)) =
        (grid i).level (T + 1) future := by
  unfold windowEncoding
  dsimp only
  rw [clipToken_val_eq hv h0 hV, prefixLevel_eq_of_no_clip _ _ _ hprefix, decodeLevel_step]

/-- The integer grid decoder is injective at every site, so the decoded
coverage lemma applies to all three tokenizers on this output domain. -/
theorem windowEncoding_decode_injective {I : Type*} {T vocab : ℕ} (hv : 0 < vocab)
    (grid : I → FirstDifferenceGrid) (past : I → Fin (T + 1) → ℝ) (i : I) :
    Function.Injective ((windowEncoding hv grid past).decode i) := by
  intro a b h
  apply Fin.ext
  change (grid i).decodeLevel T ((grid i).prefixLevel hv (past i)) a.val =
      (grid i).decodeLevel T ((grid i).prefixLevel hv (past i)) b.val at h
  unfold decodeLevel at h
  omega

/-- Aligned pasts give one grid of tokens, using the codec of each series
on its own time axis. -/
noncomputable def jointContext {R : Type*} {T vocab : ℕ} (hv : 0 < vocab)
    (grid : R → FirstDifferenceGrid) (past : R → Fin (T + 1) → ℝ) :
    R × Fin T → Fin vocab :=
  fun (r, t) => (grid r).contextWord hv (past r) t

private def unitLGT : FirstDifferenceGrid := ⟨.lgt, 1, 128, 0⟩

/-- The same LGT token 131 represents a step of three in both windows,
while the two decoder states recover different absolute levels. -/
example :
    unitLGT.decodeLevel 1 10 131 = 13 ∧ unitLGT.decodeLevel 1 100 131 = 103 := by
  norm_num [unitLGT, decodeLevel, offset, fraction, Int.floor_zero]

example := windowEncoding_decode_step (T := 0) (vocab := 256) (by decide)
  (fun _ : Unit => unitLGT) (fun _ _ => 0) () 0 (fun j => j.elim0)
  (by norm_num [unitLGT, step, offset, level, fraction, Int.floor_zero])
  (by norm_num [unitLGT, step, offset, level, fraction, Int.floor_zero])

example (n : ℕ) (x : ℝ) :
    (FirstDifferenceGrid.mk .mash 1 128 (1 / 4)).decodeLevel n
      ((FirstDifferenceGrid.mk .mash 1 128 (1 / 4)).level n x)
      ((FirstDifferenceGrid.mk .mash 1 128 (1 / 4)).step n x (x + 1)) =
        (FirstDifferenceGrid.mk .mash 1 128 (1 / 4)).level (n + 1) (x + 1) :=
  decodeLevel_step _ _ _ _

example (n : ℕ) (x y : ℝ) :
    let grid := FirstDifferenceGrid.mk .mashHalf 2 128 0
    |grid.valueAtLevel (n + 1) (grid.decodeLevel n (grid.level n x) (grid.step n x y)) - y| ≤
      (1 / 2) / grid.scale :=
  abs_decode_step_sub_le _ _ _ _ (by norm_num)

end FirstDifferenceGrid
end Transformer.Grokking.FiniteRecovery
