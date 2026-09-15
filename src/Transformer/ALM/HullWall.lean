/-
# Past the wall the perturbation is not stored, so the marked key is the lift

`Transformer.ALM.HullMark` reads the key the compiler emits as `markKey δ k`,
a lift whose intercept carries the recency offset
`LATEST_ALPHA * inv_log_pos(p)`, and shows that offset is too small to revive
an erase.  `Transformer.ALM.ScoreWall` reads the same intercept as a binary64
number and shows that past `2^53` the format has no odd integers left.  Put
together they say something neither says alone: the same smallness that makes
the offset harmless to the hull makes it *invisible* to the format.

Past the wall two distinct representables are at least two apart
(`two_le_dist_of_isBinary`), and the whole offset the compiler adds is under
one (`marked_sep_of_shipped`).  So a key whose square is past the wall and
whose intercept is a double at all has `δ = 0` exactly:
`markKey_eq_liftKey_of_wall`.  The family the head then holds is `Lifted`, not
merely `Marked` (`lifted_of_marked_of_wall`) — which is good news for the
erase argument, since `Transformer.ALM.HullLift` is the stronger statement, and
bad news for everything the offset was added to do.  `marks_tie_past_the_wall`
is that half: two writes to one logical key, separated by the compiler at
`0.3 * inv_log_pos`, arrive at the head as the *same point*.  `HullMeta.last_seq`
is then not a fallback for an unlikely tie, it is the only thing deciding the
answer.

Which heads this is about is measured, not guessed: `todo3.md` §4b reports the
heads keyed on 32-bit WASM values, and `the_wasm_heads_are_past_the_wall`
carries one of its numbers.

Source: `todo3.md` §2a and §4b; `alm-compile/src/graph.rs` lines 231-264
(`embed_key`); `alm-hull/src/gap.rs`, which counts what is left of the
separation at run time.
-/

import Transformer.ALM.HullMark
import Transformer.ALM.ScoreWall

namespace Transformer
namespace ALM

/-! ### The offset is smaller than the format's own step -/

/-- **A marked key past the wall is the lift itself.**  Both intercepts are
representable and both are past `2^53` in magnitude, so if they differed they
would differ by at least two — and the offset is under two by hypothesis and
under one in the shipped compiler.  So they do not differ: the `δ` the
compiler computed is not in the number the head receives.

Source: `ScoreWall.two_le_dist_of_isBinary`; `todo3.md` §4b. -/
theorem markKey_eq_liftKey_of_wall {p : ℕ} {δ k : ℝ}
    (hlift : IsBinary p (-(k ^ 2))) (hmark : IsBinary p (-(k ^ 2) + δ))
    (hw : (2 : ℝ) ^ p < k ^ 2 - δ) (hδ0 : 0 ≤ δ) (hδ2 : δ < 2) :
    markKey δ k = liftKey k := by
  have hpos : (0 : ℝ) < k ^ 2 - δ := lt_trans (by positivity) hw
  have hk2 : |(-(k ^ 2) : ℝ)| = k ^ 2 := by
    rw [abs_neg, abs_of_nonneg (sq_nonneg k)]
  have h2 : |(-(k ^ 2) + δ : ℝ)| = k ^ 2 - δ := by
    rw [show (-(k ^ 2) + δ : ℝ) = -(k ^ 2 - δ) by ring, abs_neg, abs_of_nonneg hpos.le]
  have heq : -(k ^ 2) + δ = -(k ^ 2) := by
    by_contra hne
    have hd := two_le_dist_of_isBinary hmark hlift (by rw [h2]; exact hw)
      (by rw [hk2]; linarith) hne
    rw [show (-(k ^ 2) + δ - -(k ^ 2) : ℝ) = δ by ring, abs_of_nonneg hδ0] at hd
    linarith
  unfold markKey liftKey
  rw [heq]

/-- The hypotheses are satisfiable, at the only `δ` that can satisfy them: the
key `2^27`, whose square is `2^54`, with no offset at all. -/
example : IsBinary 53 (-((134217728 : ℝ) ^ 2)) ∧
    IsBinary 53 (-((134217728 : ℝ) ^ 2) + 0) ∧
    (2 : ℝ) ^ 53 < (134217728 : ℝ) ^ 2 - 0 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 2 := by
  refine ⟨⟨-134217728, 27, by norm_num, by norm_num⟩,
    ⟨-134217728, 27, by norm_num, by norm_num⟩, by norm_num, le_refl 0, by norm_num⟩

/-- And the wall hypothesis is what does the work: below it a nonzero offset is
a perfectly good double, and the marked key is *not* the lift.  `-8.5` is
`-17 * 2⁻¹`, and `markKey 0.5 3 = (6, -8.5)` while `liftKey 3 = (6, -9)`. -/
example : IsBinary 53 (-((3 : ℝ) ^ 2) + 0.5) ∧ markKey 0.5 (3 : ℝ) ≠ liftKey 3 := by
  refine ⟨⟨-17, -1, by norm_num, by norm_num⟩, ?_⟩
  unfold markKey liftKey
  norm_num

/-! ### So the writes the compiler separated arrive tied -/

/-- **Two writes to one logical key land on one point.**  The compiler adds
`LATEST_ALPHA * inv_log_pos(p)` to `ky` precisely so that the later write
outscores the earlier one; past the wall both offsets round away and the two
keys are equal as points, so the head sees a tie and `HullMeta.last_seq` — the
counter `todo3.md` §2a calls a fallback — is the whole tie-break.

Source: `todo3.md` §2a; `alm-hull/src/gap.rs`, whose `NOISE` count is this
statement measured. -/
theorem marks_tie_past_the_wall {p : ℕ} {δ δ' k : ℝ}
    (hlift : IsBinary p (-(k ^ 2)))
    (hmark : IsBinary p (-(k ^ 2) + δ)) (hmark' : IsBinary p (-(k ^ 2) + δ'))
    (hw : (2 : ℝ) ^ p < k ^ 2 - δ) (hw' : (2 : ℝ) ^ p < k ^ 2 - δ')
    (hδ0 : 0 ≤ δ) (hδ2 : δ < 2) (hδ'0 : 0 ≤ δ') (hδ'2 : δ' < 2) :
    markKey δ k = markKey δ' k :=
  (markKey_eq_liftKey_of_wall hlift hmark hw hδ0 hδ2).trans
    (markKey_eq_liftKey_of_wall hlift hmark' hw' hδ'0 hδ'2).symm

/-- The hypotheses are satisfiable and the conclusion is not idle: the two
offsets are the same only because the wall forces them to be, and the same two
writes below the wall are two distinct points. -/
example : markKey 0 (134217728 : ℝ) = markKey 0 (134217728 : ℝ) ∧
    markKey 0 (3 : ℝ) ≠ markKey 0.5 (3 : ℝ) := by
  refine ⟨rfl, ?_⟩
  unfold markKey
  norm_num

/-! ### And the family degenerates to the lifted one -/

/-- **A marked family past the wall is a lifted family.**  Every line of it is
`markKey δ z` with `δ ≤ A < 2`, every intercept is a double, and every key's
square clears the wall by more than `A`; so every `δ` is zero and the container
holds the pure lift.  `Transformer.ALM.HullLift.not_eraseStep_of_lift` then
applies with no separation hypothesis at all — the perturbed argument of
`HullMark` is not needed past the wall, because there is nothing left to
perturb.

Source: `todo3.md` §4b. -/
theorem lifted_of_marked_of_wall {A : ℝ} {s : Finset (ℝ × ℝ)} (hA : A < 2)
    (hs : Marked A s) (hrep : ∀ l ∈ s, IsBinary 53 l.2)
    (hsq : ∀ l ∈ s, IsBinary 53 (-((l.1 / 2) ^ 2)))
    (hwall : ∀ l ∈ s, (2 : ℝ) ^ 53 < (l.1 / 2) ^ 2 - A) :
    Lifted s := by
  intro l hl
  obtain ⟨z, δ, hδ0, hδA, rfl⟩ := hs.perturbed l hl
  have hhalf : (markKey δ (z : ℝ)).1 / 2 = (z : ℝ) := by
    rw [markKey_fst]; ring
  refine ⟨(z : ℝ), markKey_eq_liftKey_of_wall (p := 53) (k := (z : ℝ)) (δ := δ) ?_ ?_ ?_ hδ0 ?_⟩
  · exact hhalf ▸ hsq _ hl
  · exact hrep _ hl
  · have := hwall _ hl
    rw [hhalf] at this
    linarith
  · linarith

/-- The hypotheses are satisfiable: the one-line family `{liftKey 2^27}`, whose
only offset is already zero. -/
example : Marked 0 ({markKey 0 ((134217728 : ℤ) : ℝ)} : Finset (ℝ × ℝ)) := by
  constructor
  · intro l hl
    simp only [Finset.mem_singleton] at hl
    exact ⟨134217728, 0, le_refl 0, le_refl 0, hl⟩
  · intro l hl l' hl' _
    simp only [Finset.mem_singleton] at hl hl'
    rw [hl, hl']

/-! ### And which heads are past it -/

/-- The offset the shipped compiler adds is under two, so the hypothesis
`δ < 2` above is discharged by `marked_sep_of_shipped` for every position the
model can reach. -/
theorem the_shipped_spread_is_under_two : (0.3 : ℝ) / Real.log 2 < 2 :=
  lt_trans marked_sep_of_shipped (by norm_num)

/-- **And the heads §4b names are past the wall by fifty-eight times over.**
`538 976 288 = 0x20202020` is one of the 32-bit values the WASM heads are keyed
on from the first token; its square clears `2^53` with room for any offset the
compiler could add.  So on those heads the recency term is not small, it is
absent.

Source: `todo3.md` §4b. -/
theorem the_wasm_heads_are_past_the_wall :
    (2 : ℝ) ^ 53 < (538976288 : ℝ) ^ 2 - 2 ∧ (2 : ℝ) ^ 53 < (673720322 : ℝ) ^ 2 - 2 := by
  constructor <;> norm_num

end ALM
end Transformer
