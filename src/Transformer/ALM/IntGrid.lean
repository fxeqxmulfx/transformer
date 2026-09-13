/-
# Integer keys live on a grid, and their breakpoints on a half-grid

The arithmetic behind `Transformer.ALM.FloatLattice`, separated from it
because it says nothing about floating point: a sorted family of integer keys
steps by at least one, so consecutive breakpoints are at least one apart, and
every breakpoint is a multiple of `1/2`.  An integer query is therefore either
exactly on a breakpoint or a clear `1/2` from it, which is the gap a rounded
comparison has to fit inside.

Source: `transformer_vm/attention/hull2d_cht.h`, lines 60-70 (`isect`), and
`Transformer.ALM.ScalarInt` for the integrality of the stored keys.
-/

import Transformer.ALM.Defs

namespace Transformer
namespace ALM

/-! ### Integer keys step by at least one -/

/-- Two integer keys in increasing order differ by at least `1`. -/
lemma step_add_one_le {K : ℕ → ℝ} (hstep : ∀ j, K j < K (j + 1))
    (hKint : ∀ j, ∃ z : ℤ, K j = (z : ℝ)) (j : ℕ) : K j + 1 ≤ K (j + 1) := by
  obtain ⟨za, ha⟩ := hKint j
  obtain ⟨zb, hb⟩ := hKint (j + 1)
  have hlt : (za : ℝ) < (zb : ℝ) := by rw [← ha, ← hb]; exact hstep j
  have hz : za + 1 ≤ zb := by exact_mod_cast hlt
  rw [ha, hb]
  exact_mod_cast hz

/-- And distant ones by at least as much. -/
lemma key_add_one_le {K : ℕ → ℝ} (hstep : ∀ j, K j < K (j + 1))
    (hKint : ∀ j, ∃ z : ℤ, K j = (z : ℝ)) (a : ℕ) : ∀ b, a < b → K a + 1 ≤ K b := by
  intro b
  induction b with
  | zero => exact fun h => absurd h (Nat.not_lt_zero a)
  | succ b ih =>
    intro h
    rcases Nat.lt_or_ge a b with hlt | hge
    · exact le_trans (ih hlt) (hstep b).le
    · have hab : a = b := by omega
      subst hab
      exact step_add_one_le hstep hKint a

/-- So the midpoints are a unit apart, which is what keeps the computed
comparisons monotone however they round. -/
lemma mid_add_one_le {K : ℕ → ℝ} (hstep : ∀ j, K j < K (j + 1))
    (hKint : ∀ j, ∃ z : ℤ, K j = (z : ℝ)) {a b : ℕ} (hab : a < b) :
    (K a + K (a + 1)) / 2 + 1 ≤ (K b + K (b + 1)) / 2 := by
  have h1 := key_add_one_le hstep hKint a b hab
  have h2 := key_add_one_le hstep hKint (a + 1) (b + 1) (by omega)
  linarith

/-! ### An integer never lands strictly inside a half-integer gap -/

/-- The breakpoint of two integer keys is a multiple of `1/2`. -/
lemma mid_half_int {K : ℕ → ℝ} (hKint : ∀ j, ∃ z : ℤ, K j = (z : ℝ)) (j : ℕ) :
    ∃ z : ℤ, (K j + K (j + 1)) / 2 = (z : ℝ) / 2 := by
  obtain ⟨za, ha⟩ := hKint j
  obtain ⟨zb, hb⟩ := hKint (j + 1)
  exact ⟨za + zb, by rw [ha, hb]; push_cast; ring⟩

/-- A half-integer below `a + 1/2` is at most `a`: there is nothing in
between, which turns a comparison that is only accurate to `1/2` into an exact
one. -/
lemma half_int_le_of_lt_add_half {z a : ℤ} (h : (z : ℝ) / 2 < (a : ℝ) + 1 / 2) :
    (z : ℝ) / 2 ≤ (a : ℝ) := by
  have hr : (z : ℝ) < 2 * (a : ℝ) + 1 := by linarith
  have hz : z < 2 * a + 1 := by exact_mod_cast hr
  have hle : ((z : ℤ) : ℝ) ≤ ((2 * a : ℤ) : ℝ) := by exact_mod_cast (by omega : z ≤ 2 * a)
  push_cast at hle
  linarith

/-- And symmetrically, an integer below a half-integer plus `1/2` is at most
that half-integer. -/
lemma le_half_int_of_lt_add_half {z a : ℤ} (h : (a : ℝ) < (z : ℝ) / 2 + 1 / 2) :
    (a : ℝ) ≤ (z : ℝ) / 2 := by
  have hr : (2 * a : ℝ) < (z : ℝ) + 1 := by linarith
  have hz : 2 * a < z + 1 := by exact_mod_cast hr
  have hle : ((2 * a : ℤ) : ℝ) ≤ ((z : ℤ) : ℝ) := by exact_mod_cast (by omega : 2 * a ≤ z)
  push_cast at hle
  linarith

/-- The hypotheses are satisfiable, and the conclusions have content: the keys
`2j` are sorted integers whose breakpoints `2j + 1` are a full two apart. -/
example : (∀ j : ℕ, 2 * (j : ℝ) < 2 * ((j + 1 : ℕ) : ℝ)) ∧
    (∀ j : ℕ, ∃ z : ℤ, 2 * (j : ℝ) = (z : ℝ)) ∧
    (2 * ((0 : ℕ) : ℝ) + 2 * ((1 : ℕ) : ℝ)) / 2 = 1 := by
  refine ⟨fun j => by push_cast; linarith, fun j => ⟨2 * (j : ℤ), by push_cast; ring⟩, ?_⟩
  norm_num

end ALM
end Transformer
