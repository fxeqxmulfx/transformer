module

public import Transformer.Precision.Binary32Error

/-!
# Product and reduction rounding in an explicit dot-product order

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C1.
Products and accumulator additions have separate rounding maps. With nearest
accumulator rounding, an identity product map models a single correctly rounded
fused multiply-add per step; two nearest maps model separate multiply/add.
This sequential order is explicit, not an assertion about a selected GPU kernel.
-/

@[expose] public section
noncomputable section

namespace Transformer.Precision

open scoped BigOperators

/-- Sequential dot product, with every product and accumulator rounding visible. -/
def roundedDot (mulRound addRound : ℝ → ℝ) (a b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => addRound (roundedDot mulRound addRound a b n + mulRound (a n * b n))

/-- C1: local errors from every product and addition accumulate additively.
The addition premise is evaluated at the actual rounded partial sum. -/
theorem roundedDot_error (mulRound addRound : ℝ → ℝ) (a b P A : ℕ → ℝ) (n : ℕ)
    (hp : ∀ k < n, |mulRound (a k * b k) - a k * b k| ≤ P k)
    (ha : ∀ k < n, |addRound (roundedDot mulRound addRound a b k + mulRound (a k * b k)) -
      (roundedDot mulRound addRound a b k + mulRound (a k * b k))| ≤ A k) :
    |roundedDot mulRound addRound a b n - ∑ k ∈ Finset.range n, a k * b k| ≤
      ∑ k ∈ Finset.range n, (P k + A k) := by
  induction n with
  | zero => simp [roundedDot]
  | succ n ih =>
      have hi := ih (fun k hk => hp k (by omega)) (fun k hk => ha k (by omega))
      rw [roundedDot, Finset.sum_range_succ, Finset.sum_range_succ]
      have heq : addRound (roundedDot mulRound addRound a b n + mulRound (a n * b n)) -
          ((∑ k ∈ Finset.range n, a k * b k) + a n * b n) =
          (addRound (roundedDot mulRound addRound a b n + mulRound (a n * b n)) -
            (roundedDot mulRound addRound a b n + mulRound (a n * b n))) +
          ((roundedDot mulRound addRound a b n - ∑ k ∈ Finset.range n, a k * b k) +
            (mulRound (a n * b n) - a n * b n)) := by ring
      rw [heq]
      exact (abs_add_le _ _).trans ((add_le_add (ha n (by omega))
        ((abs_add_le _ _).trans (add_le_add hi (hp n (by omega))))).trans_eq (by ring))

/-- C1: width n exposes the reduction cost explicitly. Separate product and
addition errors are allowed; a fused multiply-add has P=0 in this model. -/
theorem roundedDot_uniform_error (mulRound addRound : ℝ → ℝ) (a b : ℕ → ℝ)
    (P A : ℝ) (n : ℕ)
    (hp : ∀ k < n, |mulRound (a k * b k) - a k * b k| ≤ P)
    (ha : ∀ k < n, |addRound (roundedDot mulRound addRound a b k + mulRound (a k * b k)) -
      (roundedDot mulRound addRound a b k + mulRound (a k * b k))| ≤ A) :
    |roundedDot mulRound addRound a b n - ∑ k ∈ Finset.range n, a k * b k| ≤ (n : ℝ) * (P + A) := by
  simpa [mul_add] using roundedDot_error mulRound addRound a b (fun _ => P) (fun _ => A) n hp ha

/-- C1: representable bracketing at each actual primitive input discharges
the error premises. Different product and accumulator formats are permitted. -/
theorem roundedDot_bracket_error {G H : Set ℝ} {mulRound addRound : ℝ → ℝ}
    (hp : IsNearest G mulRound) (ha : IsNearest H addRound)
    (a b pLo pHi aLo aHi : ℕ → ℝ) (n : ℕ)
    (hprod : ∀ k < n, pLo k ∈ G ∧ pHi k ∈ G ∧ pLo k ≤ a k * b k ∧ a k * b k ≤ pHi k)
    (hadd : ∀ k < n, aLo k ∈ H ∧ aHi k ∈ H ∧
      aLo k ≤ roundedDot mulRound addRound a b k + mulRound (a k * b k) ∧
      roundedDot mulRound addRound a b k + mulRound (a k * b k) ≤ aHi k) :
    |roundedDot mulRound addRound a b n - ∑ k ∈ Finset.range n, a k * b k| ≤
      ∑ k ∈ Finset.range n, ((pHi k - pLo k) / 2 + (aHi k - aLo k) / 2) := by
  apply roundedDot_error
  · intro k hk
    obtain ⟨hl, hu, hxl, hxu⟩ := hprod k hk
    exact hp.error_of_bracket hl hu hxl hxu
  · intro k hk
    obtain ⟨hl, hu, hxl, hxu⟩ := hadd k hk
    exact ha.error_of_bracket hl hu hxl hxu

/-- C1: an actual binary32 nearest rounding loses a positive quarter-ulp
increment above one, so the operation-error bounds have nonzero witnesses. -/
theorem binary32_quarter_ulp {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q) :
    Q (1 + 1 / (2 : ℝ) ^ 25) = 1 := by
  have h1 : (1 : ℝ) ∈ grid 8 23 := by
    convert binary32_value_mem false 127 0 (by decide) (by decide) using 1
    norm_num [binary32Value]
  apply hQ.add_eq h1
  · intro z hz hz1
    have hg := (grid_hasSignificand 8 23).gap h1 hz (s := -23)
      (by norm_num) hz1
    exact hg
  · positivity
  · norm_num

example := roundedDot_error id id (fun k => (k : ℝ)) (fun _ => 1)
  (fun _ => 0) (fun _ => 0) 4 (by intro k hk; simp) (by intro k hk; simp)
example := roundedDot_uniform_error id id (fun k => (k : ℝ)) (fun _ => 1)
  0 0 4 (by intro k hk; simp) (by intro k hk; simp)

example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    |Q (1 + 1 / (2 : ℝ) ^ 25) - (1 + 1 / (2 : ℝ) ^ 25)| = 1 / (2 : ℝ) ^ 25 := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  refine ⟨Q, hQ, ?_⟩
  rw [binary32_quarter_ulp hQ]
  norm_num

example : ∃ Q : ℝ → ℝ, IsNearest (grid 8 23) Q ∧
    |roundedDot Q Q (fun _ => 0) (fun _ => 0) 1 - ∑ _ ∈ Finset.range 1, (0 : ℝ) * 0| ≤
      ∑ _ ∈ Finset.range 1, ((0 - 0 : ℝ) / 2 + (0 - 0 : ℝ) / 2) := by
  obtain ⟨Q, hQ⟩ := exists_isNearest_grid (E := 8) (by decide) 23
  have hz := hQ.eq_self (zero_mem_grid (E := 8) (by decide) 23)
  refine ⟨Q, hQ, roundedDot_bracket_error hQ hQ (fun _ => 0) (fun _ => 0)
    (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) 1 ?_ ?_⟩
  · intro k hk
    exact ⟨zero_mem_grid (by decide) 23, zero_mem_grid (by decide) 23, by norm_num, by norm_num⟩
  · intro k hk
    have hk0 : k = 0 := by omega
    subst k
    exact ⟨zero_mem_grid (by decide) 23, zero_mem_grid (by decide) 23,
      by simp [roundedDot, hz], by simp [roundedDot, hz]⟩

end Transformer.Precision
