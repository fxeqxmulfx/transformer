/-
# The emergence of clusters in self-attention dynamics — `l:auxiliary`

§7 of arXiv:2305.05465v6: in `d = 1`, a token that is once above the constant
`A` of `A² = n² e^{-A²}` grows at least like `c e^t`; a token once below `-A`
decreases at least like `-c e^t`.

The proof is the source's, on the scalar curve `a(t) = x_i(t)`
(`exp_le_of_auxiliary_lt`): `e:pypyly` gives a positive speed above `A`, a
barrier turns it into linear growth, and then `(log a)' ≥ 1 - n/a²` has an
integrable defect.  A token below `-A` is a token above `A` of `-x`, which
solves the same equation: the scores `x_i x_l` are unchanged by `x ↦ -x`.
This is the source's involution, without the relabelling `i ↦ n + 1 - i`.

**Deviation from the source, a strengthening.**  `l:auxiliary` is stated for
the extreme tokens `x_n` and `x_1`, under the ordering of §7 and with
`t₀ ≥ 0`.  None of this is needed: both drift bounds (`sub_le_drift`,
`div_sub_le_drift`) hold for any token with `x_i > 0`, since the weight of the
largest coordinate is at least `1/n` whichever token looks at it.  The lemma is
therefore stated here for every token `i` and every `t₀`, which is also the
form in which `l:unboundedparticles` uses it.  The conclusion is proved for
every `t ≥ t₀` and stated, as in the source, for every sufficiently large `t`.

Source: arXiv:2305.05465v6, `l:auxiliary`.
-/

import Transformer.Clusters.Section7_Drift
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- **`l:auxiliary` for a scalar curve.**  If `a(t) = x_N(t)` follows the
drift of `e:Idnonresca` and starts above `A` at `t₀`, then `a(t) ≥ c e^t` for
every `t ≥ t₀`.

The proof follows the source: `e:pypyly` at `B = a(t₀) > A` gives a positive
speed `g`, a barrier gives the linear growth `a(t) ≥ B + (g/2)(t - t₀)`, and
then `(log a)' ≥ 1 - n/a²` is integrable, so `log a(t) - t` is bounded below.

Source: arXiv:2305.05465v6, `l:auxiliary`. -/
theorem exp_le_of_auxiliary_lt (A : ℝ) (hA : 0 < A)
    (hAeq : A ^ 2 = ((m + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2))
    (x : ℝ → Idx (m + 1) → ℝ) (N : Idx (m + 1))
    (hder : ∀ t, HasDerivAt (fun s => x s N)
      (∑ j, Perspective.softmaxWeight (fun l => x t N * x t l) j * x t j) t)
    (t₀ : ℝ) (hstart : A < x t₀ N) :
    ∃ c : ℝ, 0 < c ∧ ∀ t, t₀ ≤ t → c * Real.exp t ≤ x t N := by
  set a : ℝ → ℝ := fun s => x s N with ha_def
  set D : ℝ → ℝ := fun t =>
    ∑ j, Perspective.softmaxWeight (fun l => x t N * x t l) j * x t j with hD_def
  set B := a t₀
  have hB : 0 < B := hA.trans hstart
  push_cast at hAeq
  -- the speed at `B`
  set g := B / (m + 1) - (m + 1) * (Real.exp (-B ^ 2) / B)
  have hg : 0 < g := by
    have hAB : A ^ 2 < B ^ 2 := by nlinarith
    have he : Real.exp (-B ^ 2) < Real.exp (-A ^ 2) := Real.exp_lt_exp.2 (by linarith)
    have h1 : ((m : ℝ) + 1) ^ 2 * Real.exp (-B ^ 2) < B ^ 2 := by
      nlinarith [mul_lt_mul_of_pos_left he (by positivity : (0 : ℝ) < ((m : ℝ) + 1) ^ 2)]
    have h2 : g = (B ^ 2 - ((m : ℝ) + 1) ^ 2 * Real.exp (-B ^ 2)) / ((m + 1) * B) := by
      simp only [g]; field_simp
    rw [h2]
    exact div_pos (by linarith) (by positivity)
  set κ := g / 2
  have hκ : 0 < κ := by positivity
  set L : ℝ → ℝ := fun t => B + κ * (t - t₀)
  have hcont : Continuous a := continuous_iff_continuousAt.2 fun t => (hder t).continuousAt
  -- linear growth, by a barrier
  have hlin : ∀ t, t₀ ≤ t → L t ≤ a t := by
    intro t ht
    have hL : ∀ y, HasDerivAt L κ y := fun y => by
      show HasDerivAt (fun t => B + κ * (t - t₀)) κ y
      simpa using (((hasDerivAt_id y).sub_const t₀).const_mul κ).const_add B
    refine image_le_of_deriv_right_lt_deriv_boundary' (f' := fun _ => κ) (B' := D)
      (by fun_prop) (fun y _ => (hL y).hasDerivWithinAt) (by simp [L, B])
      hcont.continuousOn (fun y _ => (hder y).hasDerivWithinAt) ?_ ⟨ht, le_rfl⟩
    rintro y ⟨hy, -⟩ hLa
    have hBa : B ≤ a y := by
      rw [← hLa]; simp only [L]; nlinarith
    have hpos : 0 < a y := hB.trans_le hBa
    have hd := div_sub_le_drift (x y) N hpos
    have h1 : B / ((m : ℝ) + 1) ≤ a y / ((m : ℝ) + 1) :=
      div_le_div_of_nonneg_right hBa (by positivity)
    have h2 : Real.exp (-a y ^ 2) / a y ≤ Real.exp (-B ^ 2) / B :=
      div_le_div₀ (Real.exp_pos _).le (Real.exp_le_exp.2 (by nlinarith)) hB hBa
    have h3 := mul_le_mul_of_nonneg_left h2 (by positivity : (0 : ℝ) ≤ (m : ℝ) + 1)
    show κ < D y
    have : g ≤ D y := by simp only [g, D]; simp only [a] at h1 h3; linarith
    simp only [κ]; linarith
  -- the exponential rate: `G = log a - t - (n/κ) L⁻¹` does not decrease
  have hLpos : ∀ t, t₀ ≤ t → 0 < L t := fun t ht => by
    simp only [L]; nlinarith
  set G : ℝ → ℝ := fun t => Real.log (a t) - t - (m + 1) / κ * (L t)⁻¹
  have hG : ∀ y, t₀ ≤ y → HasDerivAt G
      (D y / a y - 1 - (m + 1) / κ * (-κ / L y ^ 2)) y := by
    intro y hy
    have hL : HasDerivAt L κ y := by
      show HasDerivAt (fun t => B + κ * (t - t₀)) κ y
      simpa using (((hasDerivAt_id y).sub_const t₀).const_mul κ).const_add B
    exact (((hder y).log ((hLpos y hy).trans_le (hlin y hy)).ne').sub (hasDerivAt_id y)).sub
      ((hL.fun_inv (hLpos y hy).ne').const_mul _)
  have hmono : MonotoneOn G (Set.Ici t₀) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg
      (f' := fun y => D y / a y - 1 - (m + 1) / κ * (-κ / L y ^ 2)) (convex_Ici t₀)
      (fun y hy => (hG y hy).continuousAt.continuousWithinAt) ?_ ?_
    · intro y hy
      rw [interior_Ici] at hy
      exact (hG y (le_of_lt hy)).hasDerivWithinAt
    · intro y hy
      rw [interior_Ici] at hy
      have hy' : t₀ ≤ y := le_of_lt hy
      have hLy := hLpos y hy'
      have hay := hlin y hy'
      have hpos : 0 < a y := hLy.trans_le hay
      have hd := sub_le_drift (x y) N hpos
      have h1 : 1 - (m + 1) / a y ^ 2 ≤ D y / a y := by
        rw [le_div_iff₀ hpos]
        have : (1 - ((m : ℝ) + 1) / a y ^ 2) * a y = a y - (m + 1) / a y := by
          field_simp
        rw [this]; exact hd
      have h2 : ((m : ℝ) + 1) / a y ^ 2 ≤ (m + 1) / L y ^ 2 :=
        div_le_div_of_nonneg_left (by positivity) (by positivity)
          (pow_le_pow_left₀ hLy.le hay 2)
      have h3 : (m + 1) / κ * (-κ / L y ^ 2) = -(((m : ℝ) + 1) / L y ^ 2) := by
        field_simp
      rw [h3]; linarith
  refine ⟨Real.exp (G t₀), Real.exp_pos _, fun t ht => ?_⟩
  have hGt := hmono (Set.mem_Ici.2 le_rfl) (Set.mem_Ici.2 ht) ht
  have hpos : 0 < a t := (hLpos t ht).trans_le (hlin t ht)
  have hinv : 0 ≤ (m + 1) / κ * (L t)⁻¹ := by
    have := hLpos t ht; positivity
  have hlog : G t₀ + t ≤ Real.log (a t) := by simp only [G] at hGt ⊢; linarith
  calc Real.exp (G t₀) * Real.exp t = Real.exp (G t₀ + t) := (Real.exp_add _ _).symm
    _ ≤ Real.exp (Real.log (a t)) := Real.exp_le_exp.2 hlog
    _ = a t := Real.exp_log hpos

/-- The hypotheses of `exp_le_of_auxiliary_lt` are satisfiable at `m = 0`: the
curve `x(t) = e^t (A + 1)`, with `t₀ = 0`. -/
example : ∃ A : ℝ, 0 < A ∧ A ^ 2 = ((0 + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2) ∧
    (∀ t, HasDerivAt (fun s => Real.exp s * (A + 1))
      (∑ j : Idx 1, Perspective.softmaxWeight
        (fun _ => Real.exp t * (A + 1) * (Real.exp t * (A + 1))) j *
          (Real.exp t * (A + 1))) t) ∧
    A < Real.exp 0 * (A + 1) := by
  obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant 1 one_pos
  refine ⟨A, hA, by simpa using hAeq, fun t => ?_, by simp⟩
  rw [← Finset.sum_mul, Perspective.sum_softmaxWeight one_pos, one_mul]
  exact (Real.hasDerivAt_exp t).mul_const _

/-- **Lemma (l:auxiliary), above `A`.**  Let `A > 0` satisfy
`A² = n² exp(-A²)`.  If `x_i(t₀) > A`, then there is `c₁ > 0` with
`x_i(t) ≥ c₁ e^t` for every sufficiently large `t`.

The source states this for `i = n` and `t₀ ≥ 0`, under the ordering of §7;
see the module docstring for why neither is needed.

Source: arXiv:2305.05465v6, `l:auxiliary`. -/
theorem exists_exp_lower_bound (A : ℝ) (hA : 0 < A)
    (hAeq : A ^ 2 = ((m + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2))
    (X : ℝ → Idx (m + 1) → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (i : Idx (m + 1)) (t₀ : ℝ) (hi : A < X t₀ i 0) :
    ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, c * Real.exp t ≤ X t i 0 := by
  obtain ⟨c, hc, hle⟩ := exp_le_of_auxiliary_lt A hA hAeq (fun t j => X t j 0) i
    (fun t => hasDerivAt_coord X hX t _) t₀ hi
  exact ⟨c, hc, eventually_atTop.2 ⟨t₀, hle⟩⟩

/-- **Lemma (l:auxiliary), below `-A`.**  Symmetrically, if `x_i(t₀) < -A`
then `x_i(t) ≤ -c₁ e^t` for every sufficiently large `t`.

The source states this for `i = 1` and `t₀ ≥ 0`, under the ordering of §7;
see the module docstring for why neither is needed.

Source: arXiv:2305.05465v6, `l:auxiliary`. -/
theorem exists_exp_upper_bound (A : ℝ) (hA : 0 < A)
    (hAeq : A ^ 2 = ((m + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2))
    (X : ℝ → Idx (m + 1) → EucSpace 1) (hX : IdNonrescaledDynamics X)
    (i : Idx (m + 1)) (t₀ : ℝ) (hi : X t₀ i 0 < -A) :
    ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, X t i 0 ≤ -(c * Real.exp t) := by
  -- a token of `-x`, with the same scores
  have hder : ∀ t, HasDerivAt (fun s => -X s i 0)
      (∑ j, Perspective.softmaxWeight (fun l => -X t i 0 * -X t l 0) j * -X t j 0) t := by
    intro t
    convert (hasDerivAt_coord X hX t i).fun_neg using 1
    simp only [mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib]
  obtain ⟨c, hc, hle⟩ := exp_le_of_auxiliary_lt A hA hAeq (fun t j => -X t j 0) i hder t₀
    (by linarith)
  exact ⟨c, hc, eventually_atTop.2 ⟨t₀, fun t ht => by linarith [hle t ht]⟩⟩

/-- The hypotheses of `exists_exp_lower_bound` are satisfiable at
`m = 0`: the one-token solution `x(t) = e^t (A+1)` starts above `A`, and the
constant `A` itself exists by `exists_auxiliary_constant`. -/
example : ∃ A : ℝ, 0 < A ∧ A ^ 2 = ((0 + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2) ∧
    IdNonrescaledDynamics (n := 1)
        (fun t _ => Real.exp t • (EuclideanSpace.single 0 (A + 1) : EucSpace 1)) ∧
      A < (Real.exp 0 • (EuclideanSpace.single 0 (A + 1) : EucSpace 1)) 0 := by
  obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant 1 one_pos
  refine ⟨A, hA, by simpa using hAeq, idNonrescaledDynamics_single _, ?_⟩
  simp

/-- The hypotheses of `exists_exp_upper_bound` are satisfiable at
`m = 0`: the one-token solution `x(t) = -e^t (A+1)` starts below `-A`. -/
example : ∃ A : ℝ, 0 < A ∧ A ^ 2 = ((0 + 1 : ℕ) : ℝ) ^ 2 * Real.exp (-A ^ 2) ∧
    IdNonrescaledDynamics (n := 1)
        (fun t _ => Real.exp t • (EuclideanSpace.single 0 (-(A + 1)) : EucSpace 1)) ∧
      (Real.exp 0 • (EuclideanSpace.single 0 (-(A + 1)) : EucSpace 1)) 0 < -A := by
  obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant 1 one_pos
  refine ⟨A, hA, by simpa using hAeq, idNonrescaledDynamics_single _, ?_⟩
  simp

end Clusters
end Transformer
