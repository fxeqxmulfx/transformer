/-
# Every finite format blinds attention

Store the softmax weights of a head in any format with finitely many numbers,
rounded to nearest.  Nothing about the format is assumed: no `0`, no exponent,
no spacing.  Near `0` a finite set has one nearest point `c` from the right, so
rounding is constant on some interval `(0, τ)` (`IsNearest.exists_const`).
Scores in a window of width `D` put every weight below `e^D / n`
(`softmax_le`), so once `n > e^D / τ` every weight is stored as the same `c`,
and the head outputs

  `c · Σ_j v_j`

whatever its scores are (`qAttn_blind`).  The queries and keys no longer reach
the output: the head cannot tell which token it attends to.

The two kinds of formats differ only in `c`.  If `0` is a number of the format
(fixed point, IEEE floats, the integer quants, `MXFP4`), `c = 0` and the output
is `0`: the signal is lost.  If it is not (the codebook of `IQ4_NL`), `c ≠ 0`
and the output is an unweighted sum of the values: the signal is still there,
but it is no longer attention.

The dispersion is that of Veličković, Perivolaropoulos, Barbero, Pascanu,
arXiv:2410.01104, §2; the statement for an arbitrary finite format is this
repository's own.
-/

import Transformer.Precision.Basic
import Transformer.Precision.Nearest
import Mathlib.Order.Filter.Finite
import Mathlib.Topology.Order.LeftRightNhds

open scoped BigOperators Topology

namespace Transformer
namespace Precision

/-- **Round-to-nearest onto a finite set is constant near `0` from the right.**
The constant is the point of least absolute value, the positive one on a tie,
so it is `0` whenever `0` is in the set. -/
theorem IsNearest.exists_const {G : Set ℝ} {Q : ℝ → ℝ} (hQ : IsNearest G Q) (hfin : G.Finite)
    (hne : G.Nonempty) :
    ∃ c ∈ G, ((0 : ℝ) ∈ G → c = 0) ∧ ∃ τ > 0, ∀ x, 0 < x → x < τ → Q x = c := by
  obtain ⟨g0, hg0, hmin⟩ := Set.exists_min_image G (fun g => |g|) hfin hne
  obtain ⟨c, hc, habs, hneg⟩ : ∃ c ∈ G, |c| = |g0| ∧ (c < 0 → -c ∉ G) := by
    by_cases h : g0 < 0 ∧ -g0 ∈ G
    · exact ⟨-g0, h.2, abs_neg g0, fun h' => absurd h' (by linarith [h.1])⟩
    · exact ⟨g0, hg0, rfl, fun h1 h2 => h ⟨h1, h2⟩⟩
  have key : ∀ g ∈ G, ∀ᶠ x in 𝓝[>] (0 : ℝ), g ≠ c → |c - x| < |g - x| := by
    intro g hg
    by_cases hgc : g = c
    · exact Filter.Eventually.of_forall fun _ h => absurd hgc h
    refine Filter.Eventually.mono ?_ fun _ h _ => h
    rcases (hmin g hg).lt_or_eq with hlt | heq
    · have : ∀ᶠ x in 𝓝 (0 : ℝ), |c - x| < |g - x| :=
        ContinuousAt.eventually_lt (by fun_prop) (by fun_prop) (by simpa [habs] using hlt)
      exact this.filter_mono nhdsWithin_le_nhds
    · have hg' : g = -c := by
        rcases abs_eq_abs.1 ((show |g0| = |g| from heq).symm.trans habs.symm) with h | h
        · exact absurd h hgc
        · exact h
      have hc0 : 0 < c := by
        rcases lt_trichotomy c 0 with h | h | h
        · exact absurd (hg' ▸ hg) (hneg h)
        · exact absurd (by rw [hg', h, neg_zero]) hgc
        · exact h
      filter_upwards [self_mem_nhdsWithin] with x (hx : 0 < x)
      rw [hg', show -c - x = -(c + x) by ring, abs_neg, abs_of_pos (show 0 < c + x by linarith)]
      exact abs_lt.2 ⟨by linarith, by linarith⟩
  obtain ⟨τ, hτ, hsub⟩ := mem_nhdsGT_iff_exists_Ioo_subset.1 (hfin.eventually_all.2 key)
  refine ⟨c, hc, fun h0 => abs_eq_zero.1 (habs.trans (le_antisymm ?_ (abs_nonneg _))),
    τ, hτ, fun x hx0 hxτ => ?_⟩
  · simpa using hmin 0 h0
  by_contra hne'
  have h1 := hsub ⟨hx0, hxτ⟩ (Q x) (hQ x).1 hne'
  have h2 := (hQ x).2 c hc
  linarith

/-- The hypotheses of `IsNearest.exists_const` are satisfiable: the grid
`{-1, 1}` has a nearest rounding. -/
example : (∃ Q, IsNearest {-1, 1} Q) ∧ ({-1, 1} : Set ℝ).Finite ∧ ({-1, 1} : Set ℝ).Nonempty :=
  ⟨exists_isNearest (Set.toFinite _) ⟨1, by simp⟩, Set.toFinite _, ⟨1, by simp⟩⟩

variable {n : ℕ} {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- **Every finite format blinds attention.**  For a format with finitely many
numbers, rounded to nearest, and a score window of width `D`, there is a
length `N` and a number `c` of the format such that beyond `N` tokens every
head outputs `c · Σ_j v_j`, whatever its scores: the output no longer depends
on the queries and keys.  If `0` is a number of the format, `c = 0`. -/
theorem qAttn_blind {G : Set ℝ} {Q : ℝ → ℝ} (hQ : IsNearest G Q) (hfin : G.Finite)
    (hne : G.Nonempty) (D : ℝ) :
    ∃ c ∈ G, ((0 : ℝ) ∈ G → c = 0) ∧ ∃ N : ℕ, ∀ n ≥ N, ∀ s : Idx n → ℝ,
      (∀ i j, s i - s j ≤ D) → ∀ v : Idx n → E, qAttn Q s v = c • ∑ i, v i := by
  obtain ⟨c, hc, hc0, τ, hτ, hQc⟩ := hQ.exists_const hfin hne
  refine ⟨c, hc, hc0, ⌈Real.exp D / τ⌉₊ + 1, fun n hn s hs v => ?_⟩
  rw [qAttn, Finset.smul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hn0 : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  have hN : Real.exp D / τ < n := by
    have : ((⌈Real.exp D / τ⌉₊ + 1 : ℕ) : ℝ) ≤ n := by exact_mod_cast hn
    push_cast at this
    linarith [Nat.le_ceil (Real.exp D / τ)]
  rw [div_lt_iff₀ hτ] at hN
  have hlt : softmax s i < τ :=
    (softmax_le hs i).trans_lt (by rw [div_lt_iff₀ hn0]; linarith)
  rw [hQc (softmax s i) (div_pos (Real.exp_pos _) (softmax_denom_pos s i)) hlt]

/-- The hypotheses of `qAttn_blind` are satisfiable: the grid `{0, 1}` has a
nearest rounding. -/
example : (∃ Q, IsNearest {0, 1} Q) ∧ ({0, 1} : Set ℝ).Finite ∧ ({0, 1} : Set ℝ).Nonempty :=
  ⟨exists_isNearest (Set.toFinite _) ⟨0, by simp⟩, Set.toFinite _, ⟨0, by simp⟩⟩

end Precision
end Transformer
