module

public import Transformer.Clusters.Section12_DiscreteEscape
public import Mathlib.Algebra.Ring.GeomSum

/-!
# Finite geometric errors and exact affine growth in the discrete asymptotic argument

ArXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, in the
discrete convention of §3, `r:discreterescaling`.
The original head pull error is inversely proportional to the query.
Once that query grows geometrically, its accumulated error is bounded
by an explicit finite geometric sum. The bounds below give both signs
of this elementary discrete estimate, without a convergence premise.

A separate exact affine recursion bounds squared original coordinates.
For w_next ≥ βw-C with β>1, subtracting C/(β-1) gives the full β^k
lower bound. A positive initial offset consequently gives a strictly
positive geometric prefactor. Applying this estimate after a sufficiently
late original layer will retain the full original layer factor.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Clusters

/-- Finite nonnegative geometric errors have an explicit uniform sum bound.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, the discrete summation step. -/
theorem discrete_geometric_sum_le (r : ℝ) (hr : 0 ≤ r) (hr1 : r < 1) (N : ℕ) :
    ∑ j ∈ Finset.range N, r ^ j ≤ 1 / (1 - r) := by
  have he := geom_sum_mul_neg r N
  have hp := pow_nonneg hr N
  apply (le_div_iff₀ (by linarith : 0 < 1 - r)).mpr
  nlinarith

/-- A finite geometric upper error gives a uniform discrete upper bound.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, the discrete accumulated-error estimate. -/
theorem discrete_geometric_error_upper (f : ℕ → ℝ) (C r : ℝ) (hC : 0 ≤ C)
    (hr : 0 ≤ r) (hr1 : r < 1) (hstep : ∀ k, f (k + 1) ≤ f k + C * r ^ k) :
    ∀ N : ℕ, f N ≤ f 0 + C / (1 - r) := by
  have hsum : ∀ N : ℕ, f N ≤ f 0 + C * ∑ j ∈ Finset.range N, r ^ j := by
    intro N
    induction N with
    | zero => simp only [Finset.range_zero, Finset.sum_empty, mul_zero, add_zero]; exact le_rfl
    | succ N ih =>
        rw [Finset.sum_range_succ]
        have hs := hstep N
        nlinarith
  intro N
  have hm := mul_le_mul_of_nonneg_left (discrete_geometric_sum_le r hr hr1 N) hC
  have he : C * (1 / (1 - r)) = C / (1 - r) := by ring
  rw [he] at hm
  linarith [hsum N]

/-- A finite geometric lower error gives a uniform discrete lower bound.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, the reflected accumulated-error estimate. -/
theorem discrete_geometric_error_lower (f : ℕ → ℝ) (C r : ℝ) (hC : 0 ≤ C)
    (hr : 0 ≤ r) (hr1 : r < 1) (hstep : ∀ k, f k - C * r ^ k ≤ f (k + 1)) :
    ∀ N : ℕ, f 0 - C / (1 - r) ≤ f N := by
  have hn : ∀ k, -f (k + 1) ≤ -f k + C * r ^ k := fun k => by linarith [hstep k]
  intro N
  have he := discrete_geometric_error_upper (fun k => -f k) C r hC hr hr1 hn N
  linarith

/-- The exact affine lower recursion retains the full geometric factor after subtracting its fixed point.
Source: arXiv:2305.05465v6, §7, `l:auxiliary` and `l:exactasymptotic`, in the discrete convention of §3. -/
theorem discrete_affine_lower (w : ℕ → ℝ) (β C : ℝ) (hβ : 1 < β)
    (hstep : ∀ k, β * w k - C ≤ w (k + 1)) :
    ∀ N : ℕ, C / (β - 1) + β ^ N * (w 0 - C / (β - 1)) ≤ w N := by
  let D := C / (β - 1)
  have hD : D * (β - 1) = C := div_mul_cancel₀ C (by linarith)
  have hβ0 : 0 ≤ β := by linarith
  intro N
  induction N with
  | zero => simp only [pow_zero, one_mul]; linarith
  | succ N ih =>
      have hm := mul_le_mul_of_nonneg_left ih hβ0
      have hs := hstep N
      rw [pow_succ]
      nlinarith

/-- The same full-factor affine estimate starts at any sufficiently late original layer.
Source: arXiv:2305.05465v6, §7, proof of `l:exactasymptotic`, in discrete time. -/
theorem discrete_affine_lower_after (w : ℕ → ℝ) (β C : ℝ) (hβ : 1 < β) (T : ℕ)
    (hstep : ∀ k, T ≤ k → β * w k - C ≤ w (k + 1)) :
    ∀ N : ℕ, C / (β - 1) + β ^ N * (w T - C / (β - 1)) ≤ w (T + N) := by
  have hs : ∀ k, β * w (T + k) - C ≤ w (T + (k + 1)) := by
    intro k
    exact hstep (T + k) (by omega)
  exact discrete_affine_lower (fun k => w (T + k)) β C hβ hs

/-- A positive original offset above the affine fixed point gives a strictly positive full-factor prefactor.
Source: arXiv:2305.05465v6, §7, `l:auxiliary` and `l:exactasymptotic`, in discrete time. -/
theorem discrete_affine_positive_growth (w : ℕ → ℝ) (β C : ℝ) (hβ : 1 < β) (hC : 0 ≤ C)
    (hstep : ∀ k, β * w k - C ≤ w (k + 1)) (hinit : C / (β - 1) < w 0) :
    ∃ c : ℝ, 0 < c ∧ ∀ N : ℕ, c * β ^ N ≤ w N := by
  let c := w 0 - C / (β - 1)
  have hc : 0 < c := sub_pos.mpr hinit
  have hD : 0 ≤ C / (β - 1) := div_nonneg hC (by linarith)
  refine ⟨c, hc, ?_⟩
  intro N
  have he := discrete_affine_lower w β C hβ hstep N
  dsimp [c]
  nlinarith

/-- An actual positive squared recursion above its fixed point retains the full unsquared factor.
Source: arXiv:2305.05465v6, §7, `l:auxiliary` and `l:exactasymptotic`, in discrete time. -/
theorem discrete_positive_square_growth (f : ℕ → ℝ) (q b : ℝ) (hq : 1 < q) (hb : 0 ≤ b)
    (hpos : ∀ k, 0 ≤ f k) (hstep : ∀ k, q ^ 2 * f k ^ 2 - b ≤ f (k + 1) ^ 2)
    (hinit : b / (q ^ 2 - 1) < f 0 ^ 2) :
    ∃ c : ℝ, 0 < c ∧ ∀ N : ℕ, c * q ^ N ≤ f N := by
  have hq0 : 0 < q := by linarith
  have hq2 : 1 < q ^ 2 := by nlinarith
  obtain ⟨d, hd, hbound⟩ := discrete_affine_positive_growth (fun k => f k ^ 2) (q ^ 2) b hq2 hb hstep hinit
  refine ⟨Real.sqrt d, Real.sqrt_pos.mpr hd, ?_⟩
  intro N
  have he : (Real.sqrt d * q ^ N) ^ 2 = d * (q ^ 2) ^ N := by
    rw [mul_pow, Real.sq_sqrt hd.le, ← pow_mul, Nat.mul_comm N 2, pow_mul]
  have hs : (Real.sqrt d * q ^ N) ^ 2 ≤ f N ^ 2 := by rw [he]; exact hbound N
  exact (sq_le_sq₀ (mul_nonneg (Real.sqrt_nonneg d) (pow_pos hq0 N).le) (hpos N)).mp hs

/-- A nonconstant sequence with actual geometric increments jointly realizes
all nonnegative-error and both signed step-error premises. -/
example : ∃ f : ℕ → ℝ, (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (∀ k, f (k + 1) ≤ f k + (1 : ℝ) * (1 / 2) ^ k) ∧
    (∀ k, f k - (1 : ℝ) * (1 / 2) ^ k ≤ f (k + 1)) := by
  refine ⟨fun k => 1 - (1 / 2 : ℝ) ^ k, zero_le_one, by norm_num, by norm_num, ?_, ?_⟩
  · intro k
    change 1 - (1 / 2 : ℝ) ^ (k + 1) ≤ 1 - (1 / 2 : ℝ) ^ k + 1 * (1 / 2 : ℝ) ^ k
    rw [pow_succ]
    have hp : 0 ≤ (1 / 2 : ℝ) ^ k := by positivity
    nlinarith
  · intro k
    change 1 - (1 / 2 : ℝ) ^ k - 1 * (1 / 2 : ℝ) ^ k ≤ 1 - (1 / 2 : ℝ) ^ (k + 1)
    rw [pow_succ]
    have hp : 0 ≤ (1 / 2 : ℝ) ^ k := by positivity
    nlinarith

/-- A genuinely growing positive sequence jointly realizes the affine recursion,
nonnegative error, and strictly positive offset premises. -/
example : ∃ w : ℕ → ℝ, (1 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) / (2 - 1) < w 0 ∧
    (∀ k, (2 : ℝ) * w k - 1 ≤ w (k + 1)) ∧ (∀ k, 0 ≤ w k) ∧
    (1 : ℝ) / (2 ^ 2 - 1) < w 0 ^ 2 ∧
    (∀ k, (2 : ℝ) ^ 2 * w k ^ 2 - 1 ≤ w (k + 1) ^ 2) := by
  refine ⟨fun k => (2 : ℝ) * 2 ^ k, by norm_num, zero_le_one, by norm_num, ?_,
    fun _ => by positivity, by norm_num, ?_⟩
  · intro k
    change (2 : ℝ) * (2 * 2 ^ k) - 1 ≤ 2 * 2 ^ (k + 1)
    rw [pow_succ]
    nlinarith
  · intro k
    change (2 : ℝ) ^ 2 * (2 * 2 ^ k) ^ 2 - 1 ≤ (2 * 2 ^ (k + 1)) ^ 2
    rw [pow_succ (2 : ℝ) k]
    nlinarith

end Transformer.Clusters
