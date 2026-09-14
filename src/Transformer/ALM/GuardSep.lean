/-
# What passing the guard buys: the separation the comparison needs

`Transformer.ALM.ScoreGuard` says what `grid.rs::off_the_grid` measures and how
little of it depends on the query scale.  It does not say what a query that
passes it has *earned*, and that is the only reason the check is in the
runtime: the chain is supposed to run from "the guard returned `false`" to "the
retrieval is the specified one", and until now it stopped one link short.

`Transformer.ALM.FloatHull.cmp_of_sep` is the other end of that link.  It
decides a comparison correctly as soon as the two operands are farther apart
than the two rounding errors put together, `δ₁ + δ₂ < |a - b|`, and asks no
questions about where the numbers came from.  This file discharges that
hypothesis from the guard.  Two ingredients and nothing else: the errors of
round-to-nearest are each at most half the spacing at that magnitude
(`ulpOf p E / 2`), and two distinct parabolic integer scores scaled by `σ` are
at least `σ` apart, because `one_le_sScore_sub` puts them a unit apart before
the scale.  A guard that passes says `ulp(score) ≤ σ`; the sum of the errors is
then at most `σ`, and the operands are at least `σ` apart.

At most, and at least.  The shipped test is `ulp(score) > margin`, so passing
it leaves the two bounds able to meet, and on that boundary the comparison can
return a tie where the exact one is strict — `guard_le_no_inversion` is what
survives there, and it is not nothing: the order is never *inverted*, only
flattened, so the merge walk collects one key too many and never the wrong
one.  One bit of room, `ulp(score) < margin`, and `cmp_of_guard` gives the full
equivalence.  `clean_strict_at_every_scale` says where that bit comes from: a
query with a factor of two to spare at unit scale has it at every scale.

Source: `todo3.md` §4 and §4a; `vm-rs/alm-hull/src/grid.rs`, `off_the_grid` and
its docstring's appeal to `ALM.FloatGrid.fp_eval_exact_of_grid`.
-/

import Transformer.ALM.ScoreGuard
import Transformer.ALM.FloatHull

namespace Transformer
namespace ALM

/-! ### One binade up -/

/-- The spacing doubles from one binade to the next. -/
lemma ulpOf_succ_exp (p : ℕ) (e : ℤ) : ulpOf p (e + 1) = 2 * ulpOf p e := by
  unfold ulpOf
  rw [show e + 1 - (p : ℤ) + 1 = (e - (p : ℤ) + 1) + 1 by ring,
    zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), zpow_one]
  ring

/-- Spacings are powers of two, so there is no room between `< 1` and `≤ 1/2`:
a query that is clean at all is clean by a factor of two. -/
lemma ulpOf_lt_one_iff (p : ℕ) (e : ℤ) : ulpOf p e < 1 ↔ 2 * ulpOf p e ≤ 1 := by
  rw [← ulpOf_succ_exp]
  unfold ulpOf
  rw [show (1 : ℝ) = (2 : ℝ) ^ (0 : ℤ) by norm_num,
    zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2),
    zpow_le_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
  omega

/-- Doubling a number moves it up exactly one binade. -/
lemma isExp_two_mul {e : ℤ} {x : ℝ} (h : IsExp e x) : IsExp (e + 1) (2 * x) := by
  have hstep : ∀ g : ℤ, (2 : ℝ) ^ (g + 1) = 2 * (2 : ℝ) ^ g := fun g => by
    rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), zpow_one]; ring
  have habs : |2 * x| = 2 * |x| := by rw [abs_mul]; norm_num
  refine ⟨?_, ?_⟩
  · rw [habs, hstep e]; linarith [h.1]
  · rw [habs, hstep (e + 1)]; linarith [h.2]

/-- **The strict guard is scale-free too.**  `clean_at_every_scale` carries
`2·ulp ≤ 1` at unit scale to `ulp(σs) ≤ σ`; the same argument one binade up
carries the strict `2·ulp < 1` to the strict `ulp(σs) < σ`, which is the bit of
room `cmp_of_guard` needs and the shipped comparison does not insist on. -/
theorem clean_strict_at_every_scale {p : ℕ} {e f E : ℤ} {σ s : ℝ}
    (hσ : IsExp f σ) (hs : IsExp e s) (hσpos : 0 < σ)
    (hclean : 2 * ulpOf p e < 1) (hE : IsExp E (σ * s)) : ulpOf p E < σ := by
  have hs' : IsExp (e + 1) (2 * s) := isExp_two_mul hs
  have hE' : IsExp (E + 1) (σ * (2 * s)) := by
    rw [show σ * (2 * s) = 2 * (σ * s) by ring]
    exact isExp_two_mul hE
  have hcl : 2 * ulpOf p (e + 1) ≤ 1 := by
    refine (ulpOf_lt_one_iff p (e + 1)).mp ?_
    rw [ulpOf_succ_exp]
    exact hclean
  have h := clean_at_every_scale hσ hs' hσpos hcl hE'
  rw [ulpOf_succ_exp] at h
  linarith [ulpOf_pos p E]

/-! ### What the keys contribute -/

/-- **The scale is the whole separation.**  Two parabolic integer scores that
differ at all differ by a unit (`one_le_sScore_sub`), so after the
hard-attention scale they differ by at least `σ` — which is exactly the margin
`grid.rs` compares the spacing against, the query's own second coordinate. -/
theorem scaled_unit_sep {q j k : ℤ} {σ : ℝ} (hσ : 0 < σ)
    (h : sScore q j < sScore q k) : σ ≤ |σ * sScore q k - σ * sScore q j| := by
  have h1 := one_le_sScore_sub h
  have hnn : 0 ≤ σ * sScore q k - σ * sScore q j := by nlinarith
  rw [abs_of_nonneg hnn]
  nlinarith

/-! ### The link -/

/-- **A guard passed with a bit to spare decides the comparison.**  The two
rounding errors are each at most half the spacing at the winner's magnitude, so
together at most `ulp(score)`; the guard says that is below the margin, and the
margin is below the true separation.  That is `cmp_of_sep`'s hypothesis, and
the computed comparison is therefore the exact one. -/
theorem cmp_of_guard {p : ℕ} {E : ℤ} {σ a b a' b' δ₁ δ₂ : ℝ}
    (hguard : ulpOf p E < σ) (ha : |a' - a| ≤ δ₁) (hb : |b' - b| ≤ δ₂)
    (hδ₁ : δ₁ ≤ ulpOf p E / 2) (hδ₂ : δ₂ ≤ ulpOf p E / 2) (hsep : σ ≤ |a - b|) :
    a' ≤ b' ↔ a ≤ b :=
  cmp_of_sep ha hb (by linarith)

/-- **And the machine's own comparison, end to end.**  Two stored integer keys,
the scores the head computes for them at a scale `σ`, the rounding the runtime
does, and a guard that passed: the computed scores are then in the order the
exact ones are.  This is the statement `grid.rs::off_the_grid` exists to make,
written out — from the runtime check to "the retrieval is the specified one",
with `clean_strict_at_every_scale` supplying the guard from the unit-scale
wall of `Transformer.ALM.ScoreWall`. -/
theorem retrieval_survives_the_guard {p : ℕ} {E : ℤ} {q j k : ℤ} {σ aj ak δ₁ δ₂ : ℝ}
    (hσ : 0 < σ) (hguard : ulpOf p E < σ)
    (haj : |aj - σ * sScore q j| ≤ δ₁) (hak : |ak - σ * sScore q k| ≤ δ₂)
    (hδ₁ : δ₁ ≤ ulpOf p E / 2) (hδ₂ : δ₂ ≤ ulpOf p E / 2)
    (hlt : sScore q j < sScore q k) : aj < ak := by
  have hsep : σ ≤ |σ * sScore q k - σ * sScore q j| := scaled_unit_sep hσ hlt
  have hiff := cmp_of_guard (p := p) (E := E) hguard hak haj hδ₂ hδ₁ hsep
  refine lt_of_not_ge fun hge => ?_
  have := hiff.mp hge
  nlinarith

/-- **What the shipped non-strict guard still gives.**  `off_the_grid` fires on
`ulp(score) > |margin|`, so passing it allows `δ₁ + δ₂ = σ = |a - b|` and the
two computed scores can land equal where the exact ones are not.  The order is
never inverted, though: the loser never comes out strictly ahead.  So on that
boundary the merge walk collects a key too many — a tie where there is none,
which `Transformer.ALM.TieBreak` then resolves — and never answers with the
wrong key. -/
theorem guard_le_no_inversion {p : ℕ} {E : ℤ} {σ a b a' b' δ₁ δ₂ : ℝ}
    (hguard : ulpOf p E ≤ σ) (ha : |a' - a| ≤ δ₁) (hb : |b' - b| ≤ δ₂)
    (hδ₁ : δ₁ ≤ ulpOf p E / 2) (hδ₂ : δ₂ ≤ ulpOf p E / 2) (hsep : σ ≤ b - a) :
    a' ≤ b' := by
  rw [abs_le] at ha hb
  linarith [ha.2, hb.1]

/-! ### The hypotheses are satisfiable -/

/-- A score of `2^50` at double precision leaves a spacing of `2^-2`, so it has
the factor of two `clean_strict_at_every_scale` asks for, and the scale `2^34`
is a binade like any other. -/
example : IsExp 50 ((2 : ℝ) ^ (50 : ℤ)) ∧ IsExp 34 ((2 : ℝ) ^ (34 : ℤ)) ∧
    (0 : ℝ) < (2 : ℝ) ^ (34 : ℤ) ∧ 2 * ulpOf 53 50 < 1 := by
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, by positivity, ?_⟩
  · rw [abs_of_pos (by positivity)]
  · rw [abs_of_pos (by positivity), zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
    omega
  · rw [abs_of_pos (by positivity)]
  · rw [abs_of_pos (by positivity), zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)]
    omega
  · unfold ulpOf
    norm_num

/-- And the whole chain at unit scale on the two smallest keys: `q = 1` scores
`0` at `k = 0` and `1` at `k = 1`, the guard at `E = 0` reports a spacing of
`2^-52` against a margin of `1`, and a half-ulp of rounding on each score
leaves the order alone.  These are the hypotheses of `scaled_unit_sep`,
`cmp_of_guard`, `retrieval_survives_the_guard` and `guard_le_no_inversion`. -/
example : (0 : ℝ) < 1 ∧ sScore 1 0 < sScore 1 1 ∧ ulpOf 53 0 < 1 ∧
    |(0 : ℝ) - 1 * sScore 1 0| ≤ ulpOf 53 0 / 2 ∧
    |(1 : ℝ) - 1 * sScore 1 1| ≤ ulpOf 53 0 / 2 ∧
    (1 : ℝ) ≤ |1 * sScore 1 1 - 1 * sScore 1 0| ∧ (1 : ℝ) ≤ 1 * sScore 1 1 - 1 * sScore 1 0 := by
  have h0 : sScore 1 0 = 0 := by simp [sScore]
  have h1 : sScore 1 1 = 1 := by norm_num [sScore]
  refine ⟨by norm_num, by rw [h0, h1]; norm_num, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp only [h0, h1, ulpOf] <;> norm_num

end ALM
end Transformer
