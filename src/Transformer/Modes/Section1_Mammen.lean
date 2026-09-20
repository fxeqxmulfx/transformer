/-
# The number of modes of a Gaussian KDE — modes in a fixed interval

§1.1 of arXiv:2412.09080v3, `thm:mammen` (Mammen, *Statist. Probab. Lett.* 1995,
Thm. 1): the expected number of modes of `eq:gkde` in a **fixed** interval
`[a, b]`, in the four bandwidth regimes.

**What the source says and what is carried here.**

* "Asymptotically as `n → ∞`" with a bandwidth compared to a power of `n` is
  carried, as everywhere in this repository, along a sequence: `N : ℕ → ℕ` are
  the sample sizes, `B : ℕ → ℝ` the bandwidth parameters, and `≪`, `≍`, `≲` are
  `=o[atTop]`, `=Θ[atTop]`, `=O[atTop]`.

* `1{0 ∈ [a, b]} + o(1)`, `Θ(1)`, `Θ(n^{-1/2}β^{5/4})`, `Θ(√β)` are the four
  conclusions, in that order.  The identity `Θ(n^{-1/2}β^{5/4}) = o(√β)` that
  the source appends to the third is `isLittleO_mammen_mid_sqrt`, and it is
  proved: it is arithmetic on the regime `β ≪ n^{2/3}`, not a statement about
  modes.

* **Deviation.**  The source says "a fixed interval `[a, b]`" in all four
  bullets.  The second and the third carry here the extra hypothesis
  `0 ∈ [a, b]`, and all but the first the extra hypothesis `a < b`.  Both are
  needed:  a degenerate interval holds at most one mode, and an interval at
  positive distance from the origin holds `o(1)` of them in the second and
  third regimes — by the paper's own account of where the modes are, the
  Kac–Rice density `√β e^{-A_t}` with `A_t ≍ β^{-3/2} n t² e^{-t²/2}`
  (`rmk:empirical`, the belt-width remark) is exponentially small at a fixed
  `t ≠ 0` as soon as `β ≪ n^{2/3}`, so the count there tends to `0` while
  `Θ(1)` and `Θ(n^{-1/2}β^{5/4})` do not.  In the fourth regime `β ≳ n^{2/3}`
  no such hypothesis is needed: there `A_t = O(1)` at every fixed `t`, and the
  modes have density `≍ √β` throughout the interval.

Source: arXiv:2412.09080v3, `thm:mammen`.
-/

import Transformer.Modes.Section1_KDE
import Transformer.Modes.Growth

open Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-! ### `β ≪ n^{2/5}`: the single mode at the origin -/

/-- **Theorem (thm:mammen), first bullet.**  If `β ≪ n^{2/5}`, the expected
number of modes of `P̂_n` in a fixed `[a, b]` is `1{0 ∈ [a, b]} + o(1)`.

Not proved here.

Source: arXiv:2412.09080v3, `thm:mammen`. -/
theorem mammen_lt (a b : ℝ) (hab : a ≤ b) (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) (hB : ∀ k, 0 < B k)
    (hreg : B =o[atTop] fun k => (N k : ℝ) ^ ((2 : ℝ) / 5)) :
    (fun k => expectedModesReal (B k) (N k) (Set.Icc a b) -
        (if (0 : ℝ) ∈ Set.Icc a b then 1 else 0)) =o[atTop] fun _ => (1 : ℝ) := by
  sorry

/-- The hypotheses of `mammen_lt` are satisfiable: `n = k + 1` samples and a
constant bandwidth parameter. -/
example : (0 : ℝ) ≤ 1 ∧ Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    (∀ _ : ℕ, (0 : ℝ) < 1) ∧
    (fun _ : ℕ => (1 : ℝ)) =o[atTop] fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 5) := by
  refine ⟨zero_le_one, by push_cast; exact tendsto_natSucc_atTop, fun _ => one_pos, ?_⟩
  have := isLittleO_rpow_rpow_nat (a := 0) (b := (2 : ℝ) / 5) (by norm_num)
  simpa using this

/-! ### `β ≍ n^{2/5}`: the critical bandwidth -/

/-- **Theorem (thm:mammen), second bullet.**  If `β ≍ n^{2/5}`, the expected
number of modes of `P̂_n` in a fixed `[a, b] ∋ 0` is `Θ(1)`.

Not proved here.  See the file header for the added hypotheses.

Source: arXiv:2412.09080v3, `thm:mammen`. -/
theorem mammen_crit (a b : ℝ) (hab : a < b) (h0 : (0 : ℝ) ∈ Set.Icc a b)
    (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) (hB : ∀ k, 0 < B k)
    (hreg : B =Θ[atTop] fun k => (N k : ℝ) ^ ((2 : ℝ) / 5)) :
    (fun k => expectedModesReal (B k) (N k) (Set.Icc a b)) =Θ[atTop] fun _ => (1 : ℝ) := by
  sorry

/-- The hypotheses of `mammen_crit` are satisfiable: `β = n^{2/5}` itself. -/
example : (-1 : ℝ) < 1 ∧ (0 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 ∧
    Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    (∀ k : ℕ, (0 : ℝ) < ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 5)) ∧
    (fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 5)) =Θ[atTop]
      fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 5) := by
  refine ⟨by norm_num, by norm_num, by push_cast; exact tendsto_natSucc_atTop,
    fun k => Real.rpow_pos_of_pos (by positivity) _, isTheta_refl _ _⟩

/-! ### `n^{2/5} ≪ β ≪ n^{2/3}`: the count grows, but below `√β` -/

/-- **Theorem (thm:mammen), third bullet.**  If `n^{2/5} ≪ β ≪ n^{2/3}`, the
expected number of modes of `P̂_n` in a fixed `[a, b] ∋ 0` is
`Θ(n^{-1/2} β^{5/4})`.

Not proved here.  See the file header for the added hypotheses.

Source: arXiv:2412.09080v3, `thm:mammen`. -/
theorem mammen_mid (a b : ℝ) (hab : a < b) (h0 : (0 : ℝ) ∈ Set.Icc a b)
    (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) (hB : ∀ k, 0 < B k)
    (hlow : (fun k => (N k : ℝ) ^ ((2 : ℝ) / 5)) =o[atTop] B)
    (hupp : B =o[atTop] fun k => (N k : ℝ) ^ ((2 : ℝ) / 3)) :
    (fun k => expectedModesReal (B k) (N k) (Set.Icc a b)) =Θ[atTop]
      fun k => (N k : ℝ) ^ (-(1 : ℝ) / 2) * B k ^ ((5 : ℝ) / 4) := by
  sorry

/-- **`Θ(n^{-1/2} β^{5/4}) = o(√β)`**, the identity the source appends to the
third bullet.  It holds for the sole reason that `β ≪ n^{2/3}`, so it is proved
here rather than assumed.

Source: arXiv:2412.09080v3, `thm:mammen`, third bullet. -/
theorem isLittleO_mammen_mid_sqrt (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) (hB : ∀ k, 0 < B k)
    (hupp : B =o[atTop] fun k => (N k : ℝ) ^ ((2 : ℝ) / 3)) :
    (fun k => (N k : ℝ) ^ (-(1 : ℝ) / 2) * B k ^ ((5 : ℝ) / 4)) =o[atTop]
      fun k => Real.sqrt (B k) := by
  have hpow : (fun k => B k ^ ((3 : ℝ) / 4)) =o[atTop]
      fun k => ((N k : ℝ) ^ ((2 : ℝ) / 3)) ^ ((3 : ℝ) / 4) :=
    hupp.rpow (by norm_num) (Filter.Eventually.of_forall fun k => Real.rpow_nonneg
      (Nat.cast_nonneg _) _)
  have hmul := (isBigO_refl (fun k => (N k : ℝ) ^ (-(1 : ℝ) / 2) * B k ^ ((1 : ℝ) / 2))
    atTop).mul_isLittleO hpow
  refine hmul.congr' ?_ ?_
  · filter_upwards with k
    rw [mul_assoc, ← Real.rpow_add (hB k)]
    norm_num
  · filter_upwards [hN.eventually_gt_atTop 0] with k hk
    rw [← Real.rpow_mul hk.le, mul_right_comm, ← Real.rpow_add hk]
    norm_num [Real.sqrt_eq_rpow]

/-- The hypotheses of `mammen_mid` and `isLittleO_mammen_mid_sqrt` are
satisfiable: `β = √n` sits strictly between `n^{2/5}` and `n^{2/3}`. -/
example : Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    (∀ k : ℕ, (0 : ℝ) < ((k + 1 : ℕ) : ℝ) ^ ((1 : ℝ) / 2)) ∧
    (fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 5)) =o[atTop]
      (fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((1 : ℝ) / 2)) ∧
    (fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((1 : ℝ) / 2)) =o[atTop]
      fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) := by
  refine ⟨by push_cast; exact tendsto_natSucc_atTop,
    fun k => Real.rpow_pos_of_pos (by positivity) _, ?_, ?_⟩
  · simpa using isLittleO_rpow_rpow_nat (a := (2 : ℝ) / 5) (b := (1 : ℝ) / 2) (by norm_num)
  · simpa using isLittleO_rpow_rpow_nat (a := (1 : ℝ) / 2) (b := (2 : ℝ) / 3) (by norm_num)

/-! ### `n^{2/3} ≲ β ≪ n²/log⁶ n`: the count is `√β` -/

/-- **Theorem (thm:mammen), fourth bullet.**  If `n^{2/3} ≲ β ≪ n²/log⁶ n`,
the expected number of modes of `P̂_n` in a fixed `[a, b]` is `Θ(√β)`.

Not proved here.

Source: arXiv:2412.09080v3, `thm:mammen`. -/
theorem mammen_gt (a b : ℝ) (hab : a < b) (N : ℕ → ℕ) (B : ℕ → ℝ)
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) (hB : ∀ k, 0 < B k)
    (hlow : (fun k => (N k : ℝ) ^ ((2 : ℝ) / 3)) =O[atTop] B)
    (hupp : B =o[atTop] fun k => (N k : ℝ) ^ 2 / Real.log (N k) ^ 6) :
    (fun k => expectedModesReal (B k) (N k) (Set.Icc a b)) =Θ[atTop]
      fun k => Real.sqrt (B k) := by
  sorry

/-- The hypotheses of `mammen_gt` are satisfiable: `β = n` sits between
`n^{2/3}` and `n²/log⁶ n`. -/
example : (-1 : ℝ) < 1 ∧ Tendsto (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) atTop atTop ∧
    (∀ k : ℕ, (0 : ℝ) < ((k + 1 : ℕ) : ℝ)) ∧
    (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) =O[atTop]
      (fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ ((2 : ℝ) / 3) * ((k + 1 : ℕ) : ℝ) ^ ((1 : ℝ) / 3)) ∧
    (fun k : ℕ => ((k + 1 : ℕ) : ℝ)) =o[atTop]
      fun k : ℕ => ((k + 1 : ℕ) : ℝ) ^ 2 / Real.log ((k + 1 : ℕ) : ℝ) ^ 6 := by
  refine ⟨by norm_num, by push_cast; exact tendsto_natSucc_atTop, fun k => by positivity, ?_, ?_⟩
  · refine (isBigO_refl _ _).congr' (by rfl) ?_
    filter_upwards with k
    rw [← Real.rpow_add (by positivity)]
    norm_num
  · have := isLittleO_rpow_sq_div_log_nat (a := 1) (by norm_num)
    simpa using this

end Modes
end Transformer
