/-
# Limits and the unique zero of the original weighted multi-head scalar mean

ArXiv:2305.05465v6, §7, proof of `l:boundedother`, for the
separate positive original heads of §12, `sec:conclusion`.
Finite softmax weights, head averages, and their original value-weighted
sum are continuous under converging coordinates and queries. The
summed mean is strictly increasing when the rate configuration has
two different values and the actual head coefficients are positive.

When a rate is positive and another is negative, the proved single-head
mean has a finite sign-changing bracket. Each original head changes
query by its own positive query-key coefficient. A common finite bracket
for their separate signs makes the actual weighted sum change sign.
Continuity supplies its zero and strict monotonicity makes it unique.
No common effective head replaces the original finite head sum.
-/

import Transformer.Clusters.Section12_MultiHeadScalar
import Transformer.Clusters.Section7_BoundedOtherMean

open scoped BigOperators
open Real Filter Topology Set

namespace Transformer.Clusters

variable {m H : ℕ}

/-- A genuine finite head softmax preserves limits of its original scores over any source filter.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for each original head of §12. -/
theorem tendsto_softmaxWeight_finite {α : Type*} {F : Filter α}
    {u : α → Idx (m + 1) → ℝ} {γ : Idx (m + 1) → ℝ}
    (hu : ∀ j, Tendsto (fun a => u a j) F (nhds (γ j))) (j : Idx (m + 1)) :
    Tendsto (fun a => Perspective.softmaxWeight (u a) j) F
      (nhds (Perspective.softmaxWeight γ j)) := by
  unfold Perspective.softmaxWeight
  exact ((hu j).rexp).div (tendsto_finsetSum _ fun l _ => (hu l).rexp)
    (Perspective.softmaxPartition_pos (Nat.succ_pos m) γ).ne'

/-- Each original head average preserves the limits of its original anchors and query.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for §12's separate head coefficients. -/
theorem tendsto_scalarHeadAverage {α : Type*} {F : Filter α} (c : ℝ)
    {y : α → Idx (m + 1) → ℝ} {γ : Idx (m + 1) → ℝ} {r : α → ℝ} {θ : ℝ}
    (hy : ∀ j, Tendsto (fun a => y a j) F (nhds (γ j))) (hr : Tendsto r F (nhds θ)) :
    Tendsto (fun a => scalarHeadAverage c (y a) (r a)) F (nhds (scalarHeadAverage c γ θ)) := by
  have hs : ∀ j, Tendsto (fun a => c * r a * y a j) F (nhds (c * θ * γ j)) :=
    fun j => (hr.const_mul c).mul (hy j)
  unfold scalarHeadAverage
  exact tendsto_finsetSum _ fun j _ => (tendsto_softmaxWeight_finite hs j).mul (hy j)

/-- The actual weighted finite head sum preserves the limits of anchors and query.
Source: arXiv:2305.05465v6, §12's original head sum and §7, proof of `l:boundedother`. -/
theorem tendsto_multiHeadScalarDrift {α : Type*} {F : Filter α} (c v : Idx H → ℝ)
    {y : α → Idx (m + 1) → ℝ} {γ : Idx (m + 1) → ℝ} {r : α → ℝ} {θ : ℝ}
    (hy : ∀ j, Tendsto (fun a => y a j) F (nhds (γ j))) (hr : Tendsto r F (nhds θ)) :
    Tendsto (fun a => multiHeadScalarDrift c v (y a) (r a)) F
      (nhds (multiHeadScalarDrift c v γ θ)) := by
  unfold multiHeadScalarDrift
  exact tendsto_finsetSum _ fun h _ => (tendsto_scalarHeadAverage (c h) hy hr).const_mul (v h)

/-- The actual weighted original mean is continuous in its scalar query.
Source: arXiv:2305.05465v6, §7, proof of `l:boundedother`, for §12's original head sum. -/
theorem continuous_multiHeadScalarDrift (c v : Idx H → ℝ) (γ : Idx (m + 1) → ℝ) :
    Continuous (multiHeadScalarDrift c v γ) := by
  apply continuous_iff_continuousAt.mpr
  intro θ
  exact tendsto_multiHeadScalarDrift c v (y := fun _ : ℝ => γ) (r := fun r : ℝ => r)
    (fun _ => tendsto_const_nhds) (continuous_id.tendsto θ)

/-- Positive original head coefficients make the weighted mean strictly increasing for nonconstant rates.
Source: arXiv:2305.05465v6, §7, the positive derivative of the mean in `l:boundedother`, for §12. -/
theorem multiHeadScalarDrift_strictMono (c v : Idx H → ℝ)
    (hc : ∀ h, 0 < c h) (hv : ∀ h, 0 < v h)
    (γ : Idx (m + 1) → ℝ) {p q : Idx (m + 1)} (hpq : γ p ≠ γ q) (h : Idx H) :
    StrictMono (multiHeadScalarDrift c v γ) := by
  have hh : ∀ a, StrictMono (scalarHeadAverage (c a) γ) := by
    intro a r s hrs
    change softmaxMean γ (c a * r) < softmaxMean γ (c a * s)
    exact softmaxMean_strictMono γ hpq (mul_lt_mul_of_pos_left hrs (hc a))
  intro r s hrs
  apply Finset.sum_lt_sum
  · exact fun a _ => (mul_lt_mul_of_pos_left (hh a hrs) (hv a)).le
  · exact ⟨h, Finset.mem_univ h, mul_lt_mul_of_pos_left (hh h hrs) (hv h)⟩

/-- The weighted sum of the separate original positive heads has exactly one zero for rates of both signs.
Source: arXiv:2305.05465v6, §7, the root in `l:boundedother`, extended to §12's original head sum. -/
theorem existsUnique_multiHeadScalarDrift_root (c v : Idx H → ℝ) (hc : ∀ h, 0 < c h)
    (hv : ∀ h, 0 < v h) (γ : Idx (m + 1) → ℝ) {p q : Idx (m + 1)}
    (hp : 0 < γ p) (hq : γ q < 0) (h : Idx H) :
    ∃! θ : ℝ, multiHeadScalarDrift c v γ θ = 0 := by
  obtain ⟨α, hα⟩ := exists_softmaxMean_root γ hp hq
  let T := 1 + ∑ a, |α / c a|
  have hT : 0 < T := by
    have hs : 0 ≤ ∑ a, |α / c a| := Finset.sum_nonneg fun a _ => abs_nonneg _
    dsimp [T]
    linarith
  have hbracket : ∀ a, -T < α / c a ∧ α / c a < T := by
    intro a
    have hs := Finset.single_le_sum (f := fun a => |α / c a|) (fun a _ => abs_nonneg _) (Finset.mem_univ a)
    have hb := abs_le.mp hs
    dsimp [T]
    constructor <;> linarith [hb.1, hb.2]
  have hneg : ∀ a, scalarHeadAverage (c a) γ (-T) < 0 := by
    intro a
    change softmaxMean γ (c a * (-T)) < 0
    apply (hα _).1
    have hs := (lt_div_iff₀ (hc a)).mp (hbracket a).1
    nlinarith
  have hpos : ∀ a, 0 < scalarHeadAverage (c a) γ T := by
    intro a
    change 0 < softmaxMean γ (c a * T)
    apply (hα _).2
    have hs := (div_lt_iff₀ (hc a)).mp (hbracket a).2
    nlinarith
  have hpG : 0 < multiHeadScalarDrift c v γ T :=
    (mul_pos (hv h) (hpos h)).trans_le (Finset.single_le_sum
      (fun a _ => (mul_pos (hv a) (hpos a)).le) (Finset.mem_univ h))
  have hnG : multiHeadScalarDrift c v γ (-T) < 0 := by
    have hs : 0 < ∑ a, v a * (-scalarHeadAverage (c a) γ (-T)) :=
      (mul_pos (hv h) (by linarith [hneg h])).trans_le (Finset.single_le_sum
        (fun a _ => (mul_pos (hv a) (by linarith [hneg a])).le) (Finset.mem_univ h))
    have he : (∑ a, v a * (-scalarHeadAverage (c a) γ (-T))) = -multiHeadScalarDrift c v γ (-T) := by
      simp only [multiHeadScalarDrift, mul_neg, Finset.sum_neg_distrib]
    rw [he] at hs
    linarith
  obtain ⟨θ, _, hθ⟩ := intermediate_value_Icc (show -T ≤ T by linarith)
    (continuous_multiHeadScalarDrift c v γ).continuousOn (⟨hnG.le, hpG.le⟩ :
      (0 : ℝ) ∈ Icc (multiHeadScalarDrift c v γ (-T)) (multiHeadScalarDrift c v γ T))
  refine ⟨θ, hθ, ?_⟩
  intro a ha
  have hneq : γ p ≠ γ q := by linarith
  exact (multiHeadScalarDrift_strictMono c v hc hv γ hneq h).injective (ha.trans hθ.symm)

/-- Asymmetric rates, different genuine positive head coefficients, and nonconstant
converging anchors and query jointly realize the limit and unique-root premises. -/
example : ∃ (γ : Idx 3 → ℝ) (c v : Idx 2 → ℝ) (y : ℕ → Idx 3 → ℝ) (r : ℕ → ℝ),
    (∀ h, 0 < c h) ∧ (∀ h, 0 < v h) ∧ 0 < γ 2 ∧ γ 0 < 0 ∧
    (∀ j, Tendsto (fun k => y k j) atTop (nhds (γ j))) ∧ Tendsto r atTop (nhds 1) := by
  let γ : Idx 3 → ℝ := ![-1, 0, 2]
  let e := fun k : ℕ => (1 / 2 : ℝ) ^ k
  have he : Tendsto e atTop (nhds 0) := tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  refine ⟨γ, ![1, 2], ![1, 3], fun k j => γ j + e k, fun k => 1 + e k,
    ?_, ?_, by norm_num [γ], by norm_num [γ], ?_, ?_⟩
  · intro h; fin_cases h <;> norm_num
  · intro h; fin_cases h <;> norm_num
  · intro j
    simpa only [add_zero] using (tendsto_const_nhds (x := γ j)).add he
  · simpa only [add_zero] using (tendsto_const_nhds (x := (1 : ℝ))).add he

end Transformer.Clusters
