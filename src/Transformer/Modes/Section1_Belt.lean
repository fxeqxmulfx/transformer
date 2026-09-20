/-
# The number of modes of a Gaussian KDE — the shape of the belt

§1.1 of arXiv:2412.09080v3: "almost all modes lie in **two intervals** of
length `Θ(√(log β))`", the prose reading of point 2 of `thm:main-result`, and
the belt-width remark that computes the length.

**What the source says and what is carried here.**

* The belt `t² ∈ [2 log n - 3 log β, 2 log n - log β]` is two intervals only
  where its inner edge is non-negative: for `β ≫ n^{2/3}` the number
  `2 log n - 3 log β` is negative, and the belt is the single interval
  `[-√(2 log n - log β), √(2 log n - log β)]`.  Point 2 of `thm:main-result`
  — an upper bound on what lies *outside* the belt — stays true either way,
  which is why it is stated over the set and not over the two intervals.
  `belt_eq_union` is the decomposition where it holds.

* The belt-width remark computes the length: the edges are `√U` and `√L` with
  `U - L = 2 log β`, so when both are `Θ(log β)` — the regime
  `2 log n - log β ≍ log β ≍ log n` the source restricts to "for the sake of
  clarity" — the difference is `Θ(√(log β))`.  That is
  `sqrt_sub_sqrt_isTheta`, and it is proved.

Source: arXiv:2412.09080v3, `thm:main-result` point 2, the belt-width remark
of §1.1.
-/

import Transformer.Modes.Section1_Main

open Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-- `t² ∈ [L, U]` says exactly that `|t|` lies between the two square roots. -/
theorem sq_mem_Icc_iff {L U t : ℝ} (hU : 0 ≤ U) :
    t ^ 2 ∈ Set.Icc L U ↔ |t| ∈ Set.Icc (Real.sqrt L) (Real.sqrt U) := by
  simp only [Set.mem_Icc, ← Real.sqrt_sq_eq_abs]
  rw [Real.sqrt_le_sqrt_iff hU, Real.sqrt_le_sqrt_iff (sq_nonneg t)]

/-- `{t : t² ∈ [L, U]}` is the union of two intervals of equal length
`√U - √L`, symmetric about the origin. -/
theorem setOf_sq_mem_Icc {L U : ℝ} (hL : 0 ≤ L) (hLU : L ≤ U) :
    {t : ℝ | t ^ 2 ∈ Set.Icc L U} =
      Set.Icc (-Real.sqrt U) (-Real.sqrt L) ∪ Set.Icc (Real.sqrt L) (Real.sqrt U) := by
  have hU : 0 ≤ U := hL.trans hLU
  have hsqrt : Real.sqrt L ≤ Real.sqrt U := Real.sqrt_le_sqrt hLU
  ext t
  rw [Set.mem_ofPred_eq, sq_mem_Icc_iff hU, Set.mem_Icc, Set.mem_union, Set.mem_Icc, Set.mem_Icc]
  constructor
  · rintro ⟨h1, h2⟩
    rcases le_or_gt t 0 with ht | ht
    · rw [abs_of_nonpos ht] at h1 h2
      exact Or.inl ⟨by linarith, by linarith⟩
    · rw [abs_of_nonneg ht.le] at h1 h2
      exact Or.inr ⟨h1, h2⟩
  · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩)
    · have ht : t ≤ 0 := h2.trans (neg_nonpos.mpr (Real.sqrt_nonneg L))
      rw [abs_of_nonpos ht]
      exact ⟨by linarith, by linarith⟩
    · have ht : 0 ≤ t := (Real.sqrt_nonneg L).trans h1
      rw [abs_of_nonneg ht]
      exact ⟨h1, h2⟩

/-- **The belt is two intervals**, as the source's "two intervals of length
`Θ(√(log β))`" says, whenever its inner edge is non-negative.

Source: arXiv:2412.09080v3, `thm:main-result`, point 2. -/
theorem belt_eq_union (n : ℕ) (β : ℝ)
    (hL : 0 ≤ 2 * Real.log n - 3 * Real.log β) (hβ : 0 ≤ Real.log β) :
    belt n β =
      Set.Icc (-Real.sqrt (2 * Real.log n - Real.log β))
          (-Real.sqrt (2 * Real.log n - 3 * Real.log β)) ∪
        Set.Icc (Real.sqrt (2 * Real.log n - 3 * Real.log β))
          (Real.sqrt (2 * Real.log n - Real.log β)) :=
  setOf_sq_mem_Icc hL (by linarith)

/-- The hypotheses of `belt_eq_union` are satisfiable: `n = 8`, `β = 2` has
inner edge `2 log 8 - 3 log 2 = 3 log 2 > 0`. -/
example : 0 ≤ 2 * Real.log 8 - 3 * Real.log 2 ∧ 0 ≤ Real.log 2 := by
  have h8 : Real.log 8 = 3 * Real.log 2 := by
    rw [show (8 : ℝ) = 2 ^ (3 : ℕ) by norm_num, Real.log_pow]
    push_cast
    ring
  have h2 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  exact ⟨by rw [h8]; linarith, h2⟩

/-- **The length of each interval is `Θ(√(log β))`.**  The belt's two edges
are `√U` and `√L` with `U - L = 2 log β`; when both edges are `Θ(log β)` — the
regime `2 log n - log β ≍ log β ≍ log n` the source restricts to "for the sake
of clarity" — their difference is `Θ(√(log β))`.

Source: arXiv:2412.09080v3, the belt-width remark of §1.1. -/
theorem sqrt_sub_sqrt_isTheta {u l g : ℕ → ℝ} (hu : u =Θ[atTop] g) (hl : l =Θ[atTop] g)
    (hd : (fun k => u k - l k) =Θ[atTop] g) (hg : ∀ᶠ k in atTop, 0 < g k)
    (hl0 : ∀ᶠ k in atTop, 0 ≤ l k) (hlu : ∀ᶠ k in atTop, l k ≤ u k) :
    (fun k => Real.sqrt (u k) - Real.sqrt (l k)) =Θ[atTop] fun k => Real.sqrt (g k) := by
  have hu0 : ∀ᶠ k in atTop, 0 ≤ u k := by filter_upwards [hl0, hlu] with k h1 h2 using h1.trans h2
  have hupos : ∀ᶠ k in atTop, 0 < u k := by
    filter_upwards [hg, hu0, hu.isBigO_symm.eq_zero_imp] with k hgk huk himp
    exact lt_of_le_of_ne huk fun h => hgk.ne' (himp h.symm)
  have hsu : (fun k => Real.sqrt (u k)) =Θ[atTop] fun k => Real.sqrt (g k) := by
    simpa only [Real.sqrt_eq_rpow] using
      hu.rpow (r := 1 / 2) hu0 (hg.mono fun _ h => h.le)
  have hsl : (fun k => Real.sqrt (l k)) =Θ[atTop] fun k => Real.sqrt (g k) := by
    simpa only [Real.sqrt_eq_rpow] using
      hl.rpow (r := 1 / 2) hl0 (hg.mono fun _ h => h.le)
  have hsum : (fun k => Real.sqrt (u k) + Real.sqrt (l k)) =Θ[atTop] fun k => Real.sqrt (g k) := by
    refine ⟨hsu.isBigO.add hsl.isBigO, ?_⟩
    refine hsu.isBigO_symm.trans (IsBigO.of_bound 1 ?_)
    filter_upwards [hl0] with k hk
    have h : ‖Real.sqrt (u k)‖ ≤ ‖Real.sqrt (u k) + Real.sqrt (l k)‖ := by
      rw [Real.norm_of_nonneg (Real.sqrt_nonneg _),
        Real.norm_of_nonneg (by positivity : (0 : ℝ) ≤ Real.sqrt (u k) + Real.sqrt (l k))]
      have := Real.sqrt_nonneg (l k)
      linarith
    simpa using h
  have heq : (fun k => Real.sqrt (u k) - Real.sqrt (l k)) =ᶠ[atTop]
      fun k => (u k - l k) / (Real.sqrt (u k) + Real.sqrt (l k)) := by
    filter_upwards [hupos, hl0] with k hukpos hlk
    have hsu0 : 0 < Real.sqrt (u k) + Real.sqrt (l k) := by
      have := Real.sqrt_pos.mpr hukpos
      have := Real.sqrt_nonneg (l k)
      linarith
    rw [eq_div_iff hsu0.ne']
    linear_combination Real.sq_sqrt hukpos.le - Real.sq_sqrt hlk
  refine (heq.trans_isTheta (hd.div hsum)).trans_eventuallyEq ?_
  filter_upwards with k
  exact Real.div_sqrt

/-- The hypotheses of `sqrt_sub_sqrt_isTheta` are satisfiable: two edges
`u = 2g` and `l = g` that are both `Θ(g)`, as are they are in the source's
regime `2 log n - 3 log β ≍ 2 log n - log β ≍ log β`. -/
example : (fun k : ℕ => 2 * ((k : ℝ) + 1)) =Θ[atTop] (fun k : ℕ => (k : ℝ) + 1) ∧
    (fun k : ℕ => ((k : ℝ) + 1)) =Θ[atTop] (fun k : ℕ => (k : ℝ) + 1) ∧
    (fun k : ℕ => 2 * ((k : ℝ) + 1) - ((k : ℝ) + 1)) =Θ[atTop] (fun k : ℕ => (k : ℝ) + 1) ∧
    (∀ᶠ k : ℕ in atTop, (0 : ℝ) < (k : ℝ) + 1) ∧
    (∀ᶠ k : ℕ in atTop, (0 : ℝ) ≤ (k : ℝ) + 1) ∧
    (∀ᶠ k : ℕ in atTop, ((k : ℝ) + 1) ≤ 2 * ((k : ℝ) + 1)) := by
  refine ⟨(isTheta_refl _ _).const_mul_left two_ne_zero, isTheta_refl _ _, ?_,
    .of_forall fun k => by positivity, .of_forall fun k => by positivity,
    .of_forall fun k => by linarith [Nat.cast_nonneg (α := ℝ) k]⟩
  refine Filter.EventuallyEq.trans_isTheta ?_ (isTheta_refl (fun k : ℕ => (k : ℝ) + 1) atTop)
  filter_upwards with k
  ring

end Modes
end Transformer
