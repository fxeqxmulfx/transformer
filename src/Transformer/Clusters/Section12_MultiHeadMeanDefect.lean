/-
# Vanishing off-diagonal weights from the actual weighted-head mean defect

ArXiv:2305.05465v6, §7, `l:boundedxn`, for the original
separate heads in §12. A probability average of anchors below A
has a deficit at least δ times the mass assigned to an anchor
at most A-δ. Positive original value coefficients preserve that
bound for each selected head inside the actual weighted head sum.

When the self anchor tends to zero and every other limiting anchor
is negative, all anchors eventually lie below the absolute self
anchor. A scaled weighted drift tending to zero makes every selected
head's off-diagonal probability tend to zero. Finite normalization
then makes the self probability tend to one. No convergence of the
scaled query is required for this one-sided case.
-/

import Transformer.Clusters.Section12_MultiHeadMean
import Transformer.Clusters.Section7_UnboundedParticles

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {m H : ℕ}

/-- A head's probability deficit controls its mass on an anchor separated below a common upper bound.
Source: arXiv:2305.05465v6, §7, the probability-average comparison in `l:boundedxn`, for §12. -/
theorem scalarHeadAverage_defect_ge (c : ℝ) (y : Idx (m + 1) → ℝ) (r A δ : ℝ)
    (j : Idx (m + 1)) (hy : ∀ l, y l ≤ A) (hj : y j + δ ≤ A) :
    δ * Perspective.softmaxWeight (fun l => c * r * y l) j ≤ A - scalarHeadAverage c y r := by
  have he : (∑ l, Perspective.softmaxWeight (fun b => c * r * y b) l * (A - y l)) =
      A - scalarHeadAverage c y r := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, scalarHeadAverage,
      Perspective.sum_softmaxWeight (Nat.succ_pos m), one_mul]
  calc
    δ * Perspective.softmaxWeight (fun l => c * r * y l) j ≤
        Perspective.softmaxWeight (fun l => c * r * y l) j * (A - y j) := by
      nlinarith [Perspective.softmaxWeight_nonneg (fun l => c * r * y l) j]
    _ ≤ ∑ l, Perspective.softmaxWeight (fun b => c * r * y b) l * (A - y l) :=
      Finset.single_le_sum (f := fun l => Perspective.softmaxWeight (fun b => c * r * y b) l * (A - y l))
        (fun l _ => mul_nonneg (Perspective.softmaxWeight_nonneg _ _) (by linarith [hy l]))
        (Finset.mem_univ j)
    _ = A - scalarHeadAverage c y r := he

/-- The actual weighted sum controls each original head's separated-anchor mass.
Source: arXiv:2305.05465v6, §12's original head sum and §7, proof of `l:boundedxn`. -/
theorem multiHeadScalarDrift_defect_ge (c v : Idx H → ℝ) (hv : ∀ a, 0 ≤ v a)
    (y : Idx (m + 1) → ℝ) (r A δ : ℝ) (j : Idx (m + 1)) (h : Idx H)
    (hy : ∀ l, y l ≤ A) (hj : y j + δ ≤ A) :
    v h * (δ * Perspective.softmaxWeight (fun l => c h * r * y l) j) ≤
      (∑ a, v a) * A - multiHeadScalarDrift c v y r := by
  have hdef : ∀ a, 0 ≤ A - scalarHeadAverage (c a) y r := by
    intro a
    simpa only [zero_mul] using scalarHeadAverage_defect_ge (c a) y r A 0 j hy (by simpa using hy j)
  calc
    v h * (δ * Perspective.softmaxWeight (fun l => c h * r * y l) j) ≤
        v h * (A - scalarHeadAverage (c h) y r) :=
      mul_le_mul_of_nonneg_left (scalarHeadAverage_defect_ge (c h) y r A δ j hy hj) (hv h)
    _ ≤ ∑ a, v a * (A - scalarHeadAverage (c a) y r) :=
      Finset.single_le_sum (fun a _ => mul_nonneg (hv a) (hdef a)) (Finset.mem_univ h)
    _ = (∑ a, v a) * A - multiHeadScalarDrift c v y r := by
      simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, multiHeadScalarDrift]

/-- A zero self anchor and strictly negative off-diagonal limits force every head's off-diagonal mass to zero.
Source: arXiv:2305.05465v6, §7, `l:boundedxn`, with the weighted original head sum from §12. -/
theorem tendsto_multiHead_oneSided_off (c v : Idx H → ℝ) (hv : ∀ a, 0 < v a)
    (z : ℕ → Idx (m + 1) → ℝ) (r : ℕ → ℝ) (i : Idx (m + 1))
    (hz : Tendsto (fun k => z k i) atTop (nhds 0))
    (hneg : ∀ j, j ≠ i → ∃ γ : ℝ, γ < 0 ∧ Tendsto (fun k => z k j) atTop (nhds γ))
    (hzero : Tendsto (fun k => multiHeadScalarDrift c v (z k) (r k)) atTop (nhds 0))
    (h : Idx H) (j : Idx (m + 1)) (hji : j ≠ i) :
    Tendsto (fun k => Perspective.softmaxWeight (fun l => c h * r k * z k l) j) atTop (nhds 0) := by
  obtain ⟨γ, hγ, ht⟩ := hneg j hji
  let δ := -γ / 2
  have hδ : 0 < δ := by dsimp [δ]; linarith
  have hupper : ∀ᶠ k in atTop, ∀ l, z k l ≤ |z k i| := by
    apply Filter.eventually_all.mpr
    intro l
    by_cases hli : l = i
    · subst l
      exact Eventually.of_forall fun k => le_abs_self _
    · obtain ⟨a, ha, hta⟩ := hneg l hli
      filter_upwards [(tendsto_order.mp hta).2 0 ha] with k hk
      exact hk.le.trans (abs_nonneg _)
  have hjbound : ∀ᶠ k in atTop, z k j + δ ≤ |z k i| := by
    filter_upwards [(tendsto_order.mp ht).2 (γ / 2) (by linarith)] with k hk
    dsimp [δ]
    linarith [abs_nonneg (z k i)]
  have hbound : ∀ᶠ k in atTop,
      Perspective.softmaxWeight (fun l => c h * r k * z k l) j ≤
        ((∑ a, v a) * |z k i| - multiHeadScalarDrift c v (z k) (r k)) / (v h * δ) := by
    filter_upwards [hupper, hjbound] with k hk hj
    apply (le_div_iff₀ (mul_pos (hv h) hδ)).mpr
    have he := multiHeadScalarDrift_defect_ge c v (fun a => (hv a).le) (z k) (r k) |z k i| δ j h hk hj
    nlinarith
  have hlim : Tendsto (fun k =>
      ((∑ a, v a) * |z k i| - multiHeadScalarDrift c v (z k) (r k)) / (v h * δ)) atTop (nhds 0) := by
    simpa only [abs_zero, mul_zero, sub_zero, zero_div] using
      ((hz.abs.const_mul (∑ a, v a)).sub hzero).div_const (v h * δ)
  exact squeeze_zero' (Eventually.of_forall fun k => Perspective.softmaxWeight_nonneg _ _) hbound hlim

/-- Finite probability normalization supplies the self weight once all other weights vanish.
Source: arXiv:2305.05465v6, §7, normalization at the end of `l:boundedxn`. -/
theorem tendsto_softmaxWeight_self_of_off_zero (u : ℕ → Idx (m + 1) → ℝ) (i : Idx (m + 1))
    (hoff : ∀ j, j ≠ i → Tendsto (fun k => Perspective.softmaxWeight (u k) j) atTop (nhds 0)) :
    Tendsto (fun k => Perspective.softmaxWeight (u k) i) atTop (nhds 1) := by
  have hs : Tendsto (fun k => ∑ j ∈ Finset.univ.erase i, Perspective.softmaxWeight (u k) j)
      atTop (nhds 0) := by
    simpa only [Finset.sum_const_zero] using tendsto_finsetSum (Finset.univ.erase i)
      (fun j hj => hoff j (Finset.ne_of_mem_erase hj))
  have he : ∀ k, Perspective.softmaxWeight (u k) i =
      1 - ∑ j ∈ Finset.univ.erase i, Perspective.softmaxWeight (u k) j := by
    intro k
    have hsum := Finset.add_sum_erase Finset.univ (Perspective.softmaxWeight (u k)) (Finset.mem_univ i)
    rw [Perspective.sum_softmaxWeight (Nat.succ_pos m)] at hsum
    linarith
  simpa only [he, sub_zero] using (tendsto_const_nhds (x := (1 : ℝ))).sub hs

/-- Nonconstant one-sided anchors, two different original positive head weights,
and a growing query jointly satisfy the mean-defect and off-diagonal limit premises. -/
example : ∃ (c v : Idx 2 → ℝ) (z : ℕ → Idx 2 → ℝ) (r : ℕ → ℝ),
    (∀ h, 0 < v h) ∧ (∀ k l, z k l ≤ 0) ∧ (∀ k, z k 1 + 1 ≤ 0) ∧
    Tendsto (fun k => z k 0) atTop (nhds 0) ∧
    (∀ j : Idx 2, j ≠ 0 → ∃ γ : ℝ, γ < 0 ∧ Tendsto (fun k => z k j) atTop (nhds γ)) ∧
    Tendsto (fun k => multiHeadScalarDrift c v (z k) (r k)) atTop (nhds 0) ∧
    (∀ h j, j ≠ (0 : Idx 2) → Tendsto
      (fun k => Perspective.softmaxWeight (fun l => c h * r k * z k l) j) atTop (nhds 0)) := by
  let c : Idx 2 → ℝ := ![1, 2]
  let v : Idx 2 → ℝ := ![1, 3]
  let z : ℕ → Idx 2 → ℝ := fun _ => ![0, -1]
  let r : ℕ → ℝ := fun k => k
  have hp : ∀ h : Idx 2, Tendsto (fun k => Perspective.softmaxWeight
      (fun l => c h * r k * z k l) 1) atTop (nhds 0) := by
    intro h
    have hc : 0 < c h := by fin_cases h <;> norm_num [c]
    have hg : Tendsto (fun k : ℕ => c h * (k : ℝ)) atTop atTop :=
      (tendsto_natCast_atTop_atTop (R := ℝ)).const_mul_atTop hc
    have hl : Tendsto (fun k : ℕ => Real.exp (-(c h * (k : ℝ)))) atTop (nhds 0) :=
      Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp hg)
    apply squeeze_zero (fun k => Perspective.softmaxWeight_nonneg _ _) _ hl
    intro k
    have he := softmaxWeight_le_exp_sub (fun l => c h * r k * z k l) 0 1
    simpa [z, r] using he
  refine ⟨c, v, z, r, ?_, ?_, ?_, tendsto_const_nhds, ?_, ?_, ?_⟩
  · intro h; fin_cases h <;> norm_num [v]
  · intro k l; fin_cases l <;> norm_num [z]
  · intro k; norm_num [z]
  · intro j hj; fin_cases j <;> norm_num at hj
    exact ⟨-1, by norm_num, tendsto_const_nhds⟩
  · have he : ∀ k, multiHeadScalarDrift c v (z k) (r k) =
        -(∑ h, v h * Perspective.softmaxWeight (fun l => c h * r k * z k l) 1) := by
      intro k
      simp [multiHeadScalarDrift, scalarHeadAverage, z, Fin.sum_univ_two]
    simpa only [he, Finset.sum_const_zero, mul_zero, neg_zero] using
      (tendsto_finsetSum Finset.univ fun h _ => (hp h).const_mul (v h)).neg

  · intro h j hj
    fin_cases j <;> norm_num at hj
    exact hp h

end Transformer.Clusters
