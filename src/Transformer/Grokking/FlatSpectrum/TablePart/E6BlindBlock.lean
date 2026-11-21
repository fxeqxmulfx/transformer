module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6BlindSystem

/-!
# PT6: four columns of the certificate of `⌊|a − b| / 2⌋`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The certificate of
this table is symmetric and keeps its form under the flip `x ↦ 96 − x`, so a cell is dead when any
of its four images is unseen (`deadCell`). The columns 38, 43, 53 and 58 keep two live rows each
(`block_dead`), and an entry of `Kᴴ Y = Yᴴ K` on two such columns is an equation in four cells
(`block_eq`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- The value `ψ(1)` of the character. -/
local notation "ζ" => (ZMod.stdAddChar (1 : ZMod 97) : ℂ)

lemma blindTable_comm (a b : ZMod 97) : blindTable b a = blindTable a b := by
  unfold blindTable
  rw [natDist_comm]

lemma blindTable_flip : ∀ a b : ZMod 97, blindTable (-1 - a) (-1 - b) = blindTable a b := by
  decide +kernel

example := certificate_symmetries det_blind_ne_zero blindTable_comm
  (α := fun _ => 1) (β := fun _ => 1) (by simp) (by simp)
  (fun a b => by simp [blindTable_flip])
  (polarFactor_isContraction_trace (tableKernel blindTable)).1
  (polarFactor_isContraction_trace (tableKernel blindTable)).2

/-- The cell `(a, b)` or one of its images under the transpose and the flip is unseen. -/
def deadCell (a b : ℕ) : Bool :=
  !halfMask.testBit (97 * a + b) || !halfMask.testBit (97 * b + a) ||
    !halfMask.testBit (97 * (96 - a) + (96 - b)) ||
    !halfMask.testBit (97 * (96 - b) + (96 - a))

/-- **The columns 38, 58, 43 and 53 of the certificate have two live rows each.** -/
theorem block_dead : ∀ l, l < 97 →
    (l = 42 ∨ l = 47 ∨ deadCell l 38 = true) ∧ (l = 54 ∨ l = 49 ∨ deadCell l 58 = true) ∧
      (l = 32 ∨ l = 59 ∨ deadCell l 43 = true) ∧ (l = 64 ∨ l = 37 ∨ deadCell l 53 = true) := by
  decide +kernel

lemma zero_of_deadCell {Y : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ}
    (hV : Vanishes Y halfMask) (hsymm : ∀ u v, Y v u = Y u v)
    (hflip : ∀ a b, a < 97 → b < 97 →
      Y (line 97 a) (line 97 b) = Y (line 97 (96 - a)) (line 97 (96 - b)))
    {a b : ℕ} (ha : a < 97) (hb : b < 97) (h : deadCell a b = true) :
    Y (line 97 a) (line 97 b) = 0 := by
  simp only [deadCell, Bool.or_eq_true, Bool.not_eq_true'] at h
  rcases h with ((h | h) | h) | h
  · exact hV a b ha hb h
  · rw [← hsymm (line 97 a) (line 97 b)]
    exact hV b a hb ha h
  · rw [hflip a b ha hb]
    exact hV _ _ (by omega) (by omega) h
  · rw [← hsymm (line 97 a) (line 97 b), hflip b a hb ha]
    exact hV _ _ (by omega) (by omega) h

example : (0 : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ) (line 97 0) (line 97 38) = 0 :=
  zero_of_deadCell (fun _ _ _ _ _ => rfl) (fun _ _ => rfl) (fun _ _ _ _ => rfl) (by norm_num)
    (by norm_num) (by decide +kernel)

/-- Only the rows `l₁` and `l₂` of column `c` of `Y` can be nonzero. -/
def TwoRows (Y : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ) (c l₁ l₂ : ℕ) : Prop :=
  l₁ ≠ l₂ ∧ l₁ < 97 ∧ l₂ < 97 ∧ ∀ l, l < 97 → l = l₁ ∨ l = l₂ ∨ Y (line 97 l) (line 97 c) = 0

lemma twoRows_of_dead {Y : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ} {c l₁ l₂ : ℕ}
    (hc : c < 97) (hne : l₁ ≠ l₂) (h₁ : l₁ < 97) (h₂ : l₂ < 97)
    (hz : ∀ a b, a < 97 → b < 97 → deadCell a b = true → Y (line 97 a) (line 97 b) = 0)
    (h : ∀ l, l < 97 → l = l₁ ∨ l = l₂ ∨ deadCell l c = true) : TwoRows Y c l₁ l₂ :=
  ⟨hne, h₁, h₂, fun l hl => (h l hl).imp_right (Or.imp_right (hz l c hl hc))⟩

example : TwoRows (0 : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ) 38 42 47 :=
  twoRows_of_dead (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (fun _ _ _ _ _ => rfl) fun l hl => (block_dead l hl).1

lemma sum_two_cells {F : (Fin 1 → ZMod 97) → ℂ} {l₁ l₂ : ℕ} (hne : l₁ ≠ l₂) (h₁ : l₁ < 97)
    (h₂ : l₂ < 97) (h : ∀ l, l < 97 → l = l₁ ∨ l = l₂ ∨ F (line 97 l) = 0) :
    ∑ u, F u = F (line 97 l₁) + F (line 97 l₂) := by
  refine Finset.sum_eq_add _ _ (fun heq => hne ?_) (fun c _ hc => ?_) (by simp) (by simp)
  · have hv := congrArg (fun u : Fin 1 → ZMod 97 => (u 0).val) heq
    simpa [line, ZMod.val_natCast_of_lt h₁, ZMod.val_natCast_of_lt h₂] using hv
  · have hv := ZMod.val_lt (c 0)
    rcases h _ hv with h' | h' | h'
    · exact absurd (by rw [← line_val c, h']) hc.1
    · exact absurd (by rw [← line_val c, h']) hc.2
    · rwa [line_val] at h'

example : ∑ _u : Fin 1 → ZMod 97, (0 : ℂ) = 0 + 0 :=
  sum_two_cells (F := fun _ => 0) (l₁ := 1) (l₂ := 2) (by norm_num) (by norm_num) (by norm_num)
    fun _ _ => Or.inr (Or.inr rfl)

lemma blindKernel_line {a b : ℕ} (ha : a < 97) (hb : b < 97) :
    tableKernel blindTable (line 97 a) (line 97 b) = ζ ^ (natDist a b / 2) := by
  simp only [tableKernel, blindTable, line, of_apply, ZMod.val_natCast_of_lt ha,
    ZMod.val_natCast_of_lt hb]
  exact stdAddChar_natCast _

example : tableKernel blindTable (line 97 1) (line 97 2) = ζ ^ (natDist 1 2 / 2) :=
  blindKernel_line (by norm_num) (by norm_num)

lemma star_zeta : star ζ = ζ ^ 96 := by
  have hn : ‖ζ‖ = 1 := by simp
  have h1 : ζ * star ζ = 1 := by
    rw [Complex.star_def, Complex.mul_conj, Complex.normSq_eq_norm_sq, hn]
    simp
  have h97 : ζ ^ 97 = 1 := by
    rw [← stdAddChar_natCast, ZMod.natCast_self]
    simp
  calc star ζ = star ζ * ζ ^ 97 := by rw [h97, mul_one]
    _ = (ζ * star ζ) * ζ ^ 96 := by ring
    _ = ζ ^ 96 := by rw [h1, one_mul]

/-- The entry `(i, j)` of `Kᴴ Y = Yᴴ K`, when the columns `i` and `j` have two live rows each. -/
lemma block_eq {Y : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ}
    (hH : (tableKernel blindTable)ᴴ * Y = Yᴴ * tableKernel blindTable)
    {i j m₁ m₂ l₁ l₂ : ℕ} (hi : i < 97) (hj : j < 97)
    (hI : TwoRows Y i m₁ m₂) (hJ : TwoRows Y j l₁ l₂) (e₁ e₂ f₁ f₂ : ℕ)
    (he₁ : natDist l₁ i / 2 = e₁) (he₂ : natDist l₂ i / 2 = e₂)
    (hf₁ : natDist m₁ j / 2 = f₁) (hf₂ : natDist m₂ j / 2 = f₂) :
    (ζ ^ 96) ^ e₁ * Y (line 97 l₁) (line 97 j) + (ζ ^ 96) ^ e₂ * Y (line 97 l₂) (line 97 j) =
      star (Y (line 97 m₁) (line 97 i)) * ζ ^ f₁ +
        star (Y (line 97 m₂) (line 97 i)) * ζ ^ f₂ := by
  obtain ⟨hne, hl₁, hl₂, hlive⟩ := hJ
  obtain ⟨hne', hm₁, hm₂, hlive'⟩ := hI
  have h : ((tableKernel blindTable)ᴴ * Y) (line 97 i) (line 97 j) =
      (Yᴴ * tableKernel blindTable) (line 97 i) (line 97 j) := by rw [hH]
  simp only [Matrix.mul_apply, Matrix.conjTranspose_apply] at h
  have hL := sum_two_cells
    (F := fun u => star (tableKernel blindTable u (line 97 i)) * Y u (line 97 j)) hne hl₁ hl₂
    fun l hl => (hlive l hl).imp_right (Or.imp_right fun h' => by simp [h'])
  have hR := sum_two_cells
    (F := fun u => star (Y u (line 97 i)) * tableKernel blindTable u (line 97 j)) hne' hm₁ hm₂
    fun l hl => (hlive' l hl).imp_right (Or.imp_right fun h' => by simp [h'])
  rw [hL, hR, blindKernel_line hl₁ hi, blindKernel_line hl₂ hi, blindKernel_line hm₁ hj,
    blindKernel_line hm₂ hj, he₁, he₂, hf₁, hf₂, star_pow, star_pow, star_zeta] at h
  exact h

example := block_eq (Y := 0) (by simp) (i := 38) (j := 38) (m₁ := 42) (m₂ := 47) (l₁ := 42)
  (l₂ := 47) (by norm_num) (by norm_num)
  ⟨by norm_num, by norm_num, by norm_num, fun _ _ => Or.inr (Or.inr rfl)⟩
  ⟨by norm_num, by norm_num, by norm_num, fun _ _ => Or.inr (Or.inr rfl)⟩ 2 4 2 4 rfl rfl rfl rfl

end Transformer.Grokking.FlatSpectrum.TablePart.E6
