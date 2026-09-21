/-
# Round-to-nearest: the dead zone and absorption

`Q` rounds to a nearest point of a set `G` of representable numbers (IEEE
754-2008, §4.3.1, roundTiesToEven and roundTiesToAway alike: the tie rule is
left free).  Two consequences hold for every such rounding, whatever the
format:

* *the dead zone* — if no point of `G` other than `0` is closer to `0` than
  `σ`, then `[0, σ/2)` rounds to `0` (`IsNearest.eq_zero`);
* *absorption* — if the next point of `G` above `a` is `g` away, then `a + t`
  rounds back to `a` for `0 ≤ t < g/2` (`IsNearest.add_eq`).

The first bounds how small a stored weight can be, the second how small an
increment an accumulator can still take.
-/

import Mathlib.Data.Set.Finite.Lemmas
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace Transformer
namespace Precision

/-- `Q` rounds every real number to a nearest point of the grid `G`. -/
def IsNearest (G : Set ℝ) (Q : ℝ → ℝ) : Prop :=
  ∀ x, Q x ∈ G ∧ ∀ z ∈ G, |Q x - x| ≤ |z - x|

/-- **The dead zone of round-to-nearest.**  If `0` is a grid point and every
other grid point has absolute value at least `σ`, everything in `[0, σ/2)`
rounds to `0`. -/
theorem IsNearest.eq_zero {G : Set ℝ} {Q : ℝ → ℝ} {σ : ℝ} (hQ : IsNearest G Q) (h0 : (0 : ℝ) ∈ G)
    (hσ : ∀ y ∈ G, y ≠ 0 → σ ≤ |y|) {x : ℝ} (hx0 : 0 ≤ x) (hx : x < σ / 2) : Q x = 0 := by
  by_contra hne
  have h1 := hσ _ (hQ x).1 hne
  have h2 := (hQ x).2 0 h0
  have h3 : |Q x| - x ≤ |Q x - x| := by
    have := abs_sub_abs_le_abs_sub (Q x) x; rwa [abs_of_nonneg hx0] at this
  rw [zero_sub, abs_neg, abs_of_nonneg hx0] at h2
  linarith

/-- A number of the grid rounds to itself. -/
theorem IsNearest.eq_self {G : Set ℝ} {Q : ℝ → ℝ} (hQ : IsNearest G Q) {x : ℝ} (hx : x ∈ G) :
    Q x = x := by
  have := (hQ x).2 x hx
  rw [sub_self, abs_zero] at this
  exact sub_eq_zero.1 (abs_nonpos_iff.1 this)

/-- A finite nonempty grid has a nearest rounding. -/
theorem exists_isNearest {G : Set ℝ} (hfin : G.Finite) (hne : G.Nonempty) : ∃ Q, IsNearest G Q := by
  choose Q hQ using fun x => Set.exists_min_image G (fun z => |z - x|) hfin hne
  exact ⟨Q, fun x => hQ x⟩

/-- The hypotheses of `IsNearest.eq_zero` are satisfiable: the grid `{0, 1}`
with `σ = 1` has a nearest rounding. -/
example : (∃ Q, IsNearest {0, 1} Q) ∧ (0 : ℝ) ∈ ({0, 1} : Set ℝ) ∧
    ∀ y ∈ ({0, 1} : Set ℝ), y ≠ 0 → (1 : ℝ) ≤ |y| :=
  ⟨exists_isNearest (Set.toFinite _) ⟨0, by simp⟩, by simp,
    by rintro y (rfl | rfl) h <;> simp_all⟩

/-- **Absorption.**  If the grid point above `a` is at least `g` away, rounding
`a + t` to nearest returns `a` for `0 ≤ t < g/2`. -/
theorem IsNearest.add_eq {G : Set ℝ} {Q : ℝ → ℝ} (hQ : IsNearest G Q) {a g t : ℝ} (ha : a ∈ G)
    (hg : ∀ z ∈ G, a < z → a + g ≤ z) (ht0 : 0 ≤ t) (ht : 2 * t < g) : Q (a + t) = a := by
  have h1 := (hQ (a + t)).2 a ha
  have h2 := hg _ (hQ (a + t)).1
  rw [show a - (a + t) = -t by ring, abs_neg, abs_of_nonneg ht0] at h1
  have h3 := abs_le.1 h1
  by_contra hne
  rcases lt_or_gt_of_ne hne with hlt | hlt
  · linarith
  · linarith [h2 hlt]

/-- The hypotheses of `IsNearest.add_eq` are satisfiable: the grid `{0, 1}`,
`a = 0`, gap `1`, increment `1/4`. -/
example : (∃ Q, IsNearest {0, 1} Q) ∧ (0 : ℝ) ∈ ({0, 1} : Set ℝ) ∧
    (∀ z ∈ ({0, 1} : Set ℝ), 0 < z → 0 + 1 ≤ z) ∧ (0 : ℝ) ≤ 1 / 4 ∧ 2 * (1 / 4 : ℝ) < 1 :=
  ⟨exists_isNearest (Set.toFinite _) ⟨0, by simp⟩, by simp,
    by rintro z (rfl | rfl) h <;> first | exact absurd h (lt_irrefl _) | norm_num, by norm_num, by norm_num⟩

end Precision
end Transformer
