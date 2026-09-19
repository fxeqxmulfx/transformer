/-
# `α(ε)` at a pairwise-orthogonal frame

arXiv:2410.06833v1, "Dynamic metastability in the self-attention model"
(Geshkovski, Koubbi, Polyanskiy, Rigollet), `eq: alpha.dist` and `hyp: init`.

The deterministic half of `coro: cm`: a configuration whose points sit within
`√ε` of a pairwise-orthogonal frame is `(β, ε)`-separated as soon as `β` is
large enough that `γ(β) > 0`.  All of the content is the estimate

  `α(ε) ≤ 4ε + 4√ε`

for such a frame, and that estimate is where this development parts company
with the source — see `αDist_le_of_orthogonal`.

The probabilistic half, and the corollary itself, are in
`Metastability.InitialUniform`.
-/

import Transformer.Metastability.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

/-- A point of the cap `𝒮_w(c) = {x : ⟨x, w⟩ ≥ 1 - c}` is within `√(2c)` of
its centre, because `‖x - w‖² = ‖x‖² - 2⟨x, w⟩ + ‖w‖² = 2 - 2⟨x, w⟩ ≤ 2c`.

Source: arXiv:2410.06833v1, `eq: cones`. -/
theorem norm_sub_le_of_mem_sphericalCap (d : ℕ) (w x : SSphere d) (c : ℝ) (hc : 0 ≤ c)
    (hx : x ∈ sphericalCap d w c) :
    ‖(x : EucSpace d) - (w : EucSpace d)‖ ≤ Real.sqrt (2 * c) := by
  have hxn : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hwn : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hinner : 1 - c ≤ inner (𝕜 := ℝ) ((x : EucSpace d)) ((w : EucSpace d)) := hx
  rw [Real.le_sqrt (norm_nonneg _) (by linarith), norm_sub_sq_real, hxn, hwn]
  linarith

/-- **The estimate behind `eq: alpha.dist`.**  For `w_1,…,w_k` pairwise
orthogonal and `0 ≤ ε`,

  `α(ε) = sup { ⟨x, y⟩ : x ∈ 𝒮_i(2ε), y ∈ 𝒮_j(2ε), i ≠ j } ≤ 4ε + 4√ε`.

Expand `⟨x, y⟩` around `⟨w_i, w_j⟩ = 0`: the three remaining terms are bounded
by `‖x - w_i‖ ‖y - w_j‖`, `‖x - w_i‖` and `‖y - w_j‖`, and a point of the cap
of height `1 - 2ε` is within `2√ε` of its centre.

**What the source says and what is changed here.**  The proof of `coro: cm`
asserts `α(ε) ≤ ε² + 2ε`, which is the bound the same expansion gives from
`‖x - w_i‖ ≤ ε`.  Membership of `𝒮_i(2ε)` does not give that: the cap of
height `1 - 2ε` has Euclidean radius `2√ε`, since
`‖x - w_i‖² = 2 - 2⟨x, w_i⟩ ≤ 4ε`.  The difference is not slack.  At an
orthogonal frame the supremum is attained and equals `sin 2θ` with
`cos θ = 1 - 2ε`, that is `4(1 - 2ε)√(ε(1 - ε))`; for every `ε` small enough
this exceeds `ε² + 2ε`, so the source's bound is false, not merely loose.
What is proved here is `4ε + 4√ε`, which is of the right order `4√ε`.

Source: arXiv:2410.06833v1, `eq: alpha.dist`, and the proof of `coro: cm`. -/
theorem αDist_le_of_orthogonal (d k : ℕ) (ε : ℝ) (hε : 0 ≤ ε) (w : Idx k → SSphere d)
    (hw : ∀ i j : Idx k, i ≠ j →
      inner (𝕜 := ℝ) ((w i : EucSpace d)) ((w j : EucSpace d)) = 0) :
    αDist d k w ε ≤ 4 * ε + 4 * Real.sqrt ε := by
  have hroot : Real.sqrt (2 * (2 * ε)) = 2 * Real.sqrt ε := by
    rw [show (2 : ℝ) * (2 * ε) = 2 ^ 2 * ε by ring, Real.sqrt_mul (by positivity),
      Real.sqrt_sq (by norm_num)]
  unfold αDist
  refine Real.sSup_le ?_ (by positivity)
  rintro c ⟨i, j, hij, x, hx, y, hy, rfl⟩
  have hxi := norm_sub_le_of_mem_sphericalCap d (w i) x (2 * ε) (by linarith) hx
  have hyj := norm_sub_le_of_mem_sphericalCap d (w j) y (2 * ε) (by linarith) hy
  rw [hroot] at hxi hyj
  have hwi : ‖(w i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (w i).2
  have hwj : ‖(w j : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (w j).2
  have hexp : inner (𝕜 := ℝ) ((x : EucSpace d)) ((y : EucSpace d))
      = inner (𝕜 := ℝ) ((x : EucSpace d) - (w i : EucSpace d))
            ((y : EucSpace d) - (w j : EucSpace d))
        + inner (𝕜 := ℝ) ((x : EucSpace d) - (w i : EucSpace d)) ((w j : EucSpace d))
        + inner (𝕜 := ℝ) ((w i : EucSpace d)) ((y : EucSpace d) - (w j : EucSpace d))
        + inner (𝕜 := ℝ) ((w i : EucSpace d)) ((w j : EucSpace d)) := by
    simp only [inner_sub_left, inner_sub_right]
    ring
  have h1 : inner (𝕜 := ℝ) ((x : EucSpace d) - (w i : EucSpace d))
      ((y : EucSpace d) - (w j : EucSpace d))
      ≤ ‖(x : EucSpace d) - (w i : EucSpace d)‖ * ‖(y : EucSpace d) - (w j : EucSpace d)‖ :=
    real_inner_le_norm _ _
  have h2 : inner (𝕜 := ℝ) ((x : EucSpace d) - (w i : EucSpace d)) ((w j : EucSpace d))
      ≤ ‖(x : EucSpace d) - (w i : EucSpace d)‖ * ‖(w j : EucSpace d)‖ :=
    real_inner_le_norm _ _
  have h3 : inner (𝕜 := ℝ) ((w i : EucSpace d)) ((y : EucSpace d) - (w j : EucSpace d))
      ≤ ‖(w i : EucSpace d)‖ * ‖(y : EucSpace d) - (w j : EucSpace d)‖ :=
    real_inner_le_norm _ _
  rw [hwj, mul_one] at h2
  rw [hwi, one_mul] at h3
  have hprod : ‖(x : EucSpace d) - (w i : EucSpace d)‖
      * ‖(y : EucSpace d) - (w j : EucSpace d)‖ ≤ 4 * ε := by
    have h := mul_le_mul hxi hyj (norm_nonneg _) (by positivity)
    rw [show 2 * Real.sqrt ε * (2 * Real.sqrt ε)
        = 4 * (Real.sqrt ε * Real.sqrt ε) by ring, Real.mul_self_sqrt hε] at h
    exact h
  rw [hexp, hw i j hij]
  linarith

/-- **The deterministic half of `coro: cm`.**  If every `x_i` is within `√ε`
of the `i`-th vector of a pairwise-orthogonal frame `w_1,…,w_n`, and

  `4√ε + 12ε + β⁻¹ log(2n²/ε) < 1`,

then `(x_1,…,x_n)` is `(β, ε)`-separated in the sense of `hyp: init`, with
`k = n` and the frame as the centres.

Both conditions of `hyp: init` come out of the same estimate.  For the first,
`‖x_i - w_i‖² = 2 - 2⟨x_i, w_i⟩ ≤ ε` puts `x_i` in `𝒮_i(ε)`, with room to
spare.  For the second, `αDist_le_of_orthogonal` bounds `α(ε)` by `4ε + 4√ε`,
and `γ(β) = 1 - α - 8ε - β⁻¹ log(2n²/ε)` is then positive exactly when the
displayed inequality holds.

The `4√ε` is what the source writes as `ε²`; see `αDist_le_of_orthogonal`.

Source: arXiv:2410.06833v1, `hyp: init`, `eq: gamma`, and the proof of
`coro: cm`. -/
theorem isSeparated_of_near_orthogonal (d n : ℕ) (β ε : ℝ) (hε : 0 ≤ ε)
    (X w : Idx n → SSphere d)
    (hw : ∀ i j : Idx n, i ≠ j →
      inner (𝕜 := ℝ) ((w i : EucSpace d)) ((w j : EucSpace d)) = 0)
    (hclose : ∀ i : Idx n, ‖(X i : EucSpace d) - (w i : EucSpace d)‖ ≤ Real.sqrt ε)
    (hcond : 4 * Real.sqrt ε + 12 * ε + β⁻¹ * Real.log (2 * (n : ℝ) ^ 2 / ε) < 1) :
    isSeparated d n β ε X := by
  refine ⟨n, le_rfl, w, fun i => ⟨i, ?_⟩, ?_⟩
  · have hxn : ‖(X i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X i).2
    have hwn : ‖(w i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (w i).2
    have hsq : ‖(X i : EucSpace d) - (w i : EucSpace d)‖ ^ 2 ≤ ε := by
      have h := mul_self_le_mul_self
        (norm_nonneg ((X i : EucSpace d) - (w i : EucSpace d))) (hclose i)
      rw [Real.mul_self_sqrt hε] at h
      rw [pow_two]
      exact h
    rw [norm_sub_sq_real, hxn, hwn] at hsq
    show (1 : ℝ) - ε ≤ inner (𝕜 := ℝ) ((X i : EucSpace d)) ((w i : EucSpace d))
    linarith
  · have hα := αDist_le_of_orthogonal d n ε hε w hw
    unfold γβ
    linarith

/-! ### The hypotheses are satisfiable -/

/-- The hypotheses of `αDist_le_of_orthogonal` and of
`isSeparated_of_near_orthogonal` are satisfiable: the single token of `𝕊^0`,
its own frame, at `ε = 10⁻⁴` and `β = 10⁵`. -/
example :
    ∃ (X w : Idx 1 → SSphere 1) (ε β : ℝ), 0 ≤ ε ∧
      (∀ i j : Idx 1, i ≠ j →
        inner (𝕜 := ℝ) ((w i : EucSpace 1)) ((w j : EucSpace 1)) = 0) ∧
      (∀ i : Idx 1, ‖(X i : EucSpace 1) - (w i : EucSpace 1)‖ ≤ Real.sqrt ε) ∧
      4 * Real.sqrt ε + 12 * ε
        + β⁻¹ * Real.log (2 * (1 : ℝ) ^ 2 / ε) < 1 := by
  refine ⟨fun _ => ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩,
    fun _ => ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩, 1 / 10000, 100000,
    by norm_num, fun i j hij => absurd (Subsingleton.elim i j) hij,
    fun _ => by simp, ?_⟩
  have hroot : Real.sqrt (1 / 10000 : ℝ) = 1 / 100 := by
    rw [show (1 / 10000 : ℝ) = (1 / 100) ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hlog : Real.log (2 * (1 : ℝ) ^ 2 / (1 / 10000)) ≤ 19999 := by
    have h := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 2 * (1 : ℝ) ^ 2 / (1 / 10000) by
      norm_num)
    norm_num at h ⊢
    linarith
  rw [hroot]
  norm_num
  linarith

end Metastability
end Transformer
