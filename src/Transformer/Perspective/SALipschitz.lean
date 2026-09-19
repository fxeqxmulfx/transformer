/-
# Step 1 of Appendix D — the drift of `SA` is Lipschitz

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`eq: lip.1`–`eq: lip.3`: the vector field `Perspective.saField` of `eq: SA` is
Lipschitz on tuples of unit vectors, with the constant `10 n max{1, β}` that
`eq: stability.4ortho` runs Grönwall with.

What the argument costs, term by term, for `β ≥ 0`:

* the scores `β⟨y_i, y_j⟩` move by at most `2 β δ` when the tuple moves by `δ`;
* hence the attention weights move by at most `8 β δ` in total variation
  (`Perspective.sum_abs_softmaxWeight_sub_le`);
* hence the attention average moves by at most `δ + 8 β δ` — the first `δ` is
  `eq: lip.2`, the rest is `eq: lip.3`;
* hence the drift moves by at most `(3 + 8 β) δ`
  (`Transformer.norm_proj_sub_proj_le`).

And `3 + 8 β ≤ 10 n max{1, β}` as soon as `n ≥ 2`.  At `n = 1` it is false —
`3 + 8 β > 10` for `β` just below `1` — but there the field vanishes
identically (`Perspective.saField_of_one`), so any constant works.  This is the
only place where the survey's `n` is needed; the estimate above is uniform in
`n`, whereas the survey's passage through the Euclidean norm pays `√n` twice.
-/

import Transformer.Perspective.SAField

open scoped BigOperators NNReal
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)


/-- **The drift of `eq: SA` is `10 n max{1, β}`-Lipschitz on tuples of unit
vectors.**

This is Step 1 of Appendix D, `eq: lip.1`–`eq: lip.3`, stated as a Lipschitz
bound on the vector field rather than as an integral inequality along
trajectories.

**What the source says and what is changed here.**  The sign condition
`0 ≤ β` is added.  The survey's `β` is an inverse temperature and is positive
throughout, but the constant `10 n max{1, β}` does depend on it: for `β < 0`
the drift's actual Lipschitz constant grows like `8 |β|` while
`10 n max{1, β} = 10 n` stays bounded, so the bound is false for `β` negative
and large.

Source: arXiv:2312.10794v5, Appendix D, `eq: lip.1`–`eq: lip.3`. -/
theorem lipschitzOnWith_saField (β : ℝ) (hβ : 0 ≤ β) (hn : 0 < n) :
    LipschitzOnWith (Real.toNNReal (10 * n * max 1 β)) (saField d n β)
      (unitTuples d n) := by
  have hmax1 : (1 : ℝ) ≤ max 1 β := le_max_left _ _
  have hmaxβ : β ≤ max 1 β := le_max_right _ _
  have hn' : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hKnn : (0 : ℝ) ≤ 10 * (n : ℝ) * max 1 β := by positivity
  have hK : ((Real.toNNReal (10 * (n : ℝ) * max 1 β) : ℝ≥0) : ℝ)
      = 10 * (n : ℝ) * max 1 β := Real.coe_toNNReal _ hKnn
  rw [lipschitzOnWith_iff_dist_le_mul]
  intro Y hY Z hZ
  have hd0 : (0 : ℝ) ≤ dist Y Z := dist_nonneg
  rw [hK, dist_pi_le_iff (mul_nonneg hKnn hd0)]
  intro i
  set δ := dist Y Z with hδ
  have hstep : ∀ j : Idx n, ‖Y j - Z j‖ ≤ δ := by
    intro j
    rw [← dist_eq_norm]
    exact dist_le_pi_dist Y Z j
  rcases eq_or_ne n 1 with rfl | hn1
  · rw [saField_of_one d β hY i, saField_of_one d β hZ i, dist_self]
    exact mul_nonneg hKnn hd0
  have hn2 : (2 : ℝ) ≤ (n : ℝ) := by
    have : 2 ≤ n := by omega
    exact_mod_cast this
  -- the scores move by at most `2 β δ`
  have hscore : ∀ j : Idx n,
      |saScore d n β Y i j - saScore d n β Z i j| ≤ β * (2 * δ) := by
    intro j
    have hexpand : inner (𝕜 := ℝ) (Y i) (Y j) - inner (𝕜 := ℝ) (Z i) (Z j)
        = inner (𝕜 := ℝ) (Y i - Z i) (Y j) + inner (𝕜 := ℝ) (Z i) (Y j - Z j) := by
      simp only [inner_sub_left, inner_sub_right]
      ring
    have h1 : |inner (𝕜 := ℝ) (Y i - Z i) (Y j)| ≤ δ := by
      have h := abs_real_inner_le_norm (Y i - Z i) (Y j)
      rw [hY j, mul_one] at h
      exact h.trans (hstep i)
    have h2 : |inner (𝕜 := ℝ) (Z i) (Y j - Z j)| ≤ δ := by
      have h := abs_real_inner_le_norm (Z i) (Y j - Z j)
      rw [hZ i, one_mul] at h
      exact h.trans (hstep j)
    have hb : |inner (𝕜 := ℝ) (Y i) (Y j) - inner (𝕜 := ℝ) (Z i) (Z j)| ≤ 2 * δ := by
      rw [hexpand]
      refine (abs_add_le _ _).trans ?_
      linarith
    simp only [saScore, ← mul_sub, abs_mul, abs_of_nonneg hβ]
    exact mul_le_mul_of_nonneg_left hb hβ
  -- hence the attention weights move by at most `8 β δ` in total variation
  have hw : ∑ j : Idx n,
      |softmaxWeight (saScore d n β Y i) j - softmaxWeight (saScore d n β Z i) j|
        ≤ 4 * (β * (2 * δ)) :=
    sum_abs_softmaxWeight_sub_le hn _ _ _ (mul_nonneg hβ (by linarith)) hscore
  -- hence the attention average moves by at most `δ + 8 β δ`
  have hmdiff : ‖(∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j)
      - ∑ j : Idx n, softmaxWeight (saScore d n β Z i) j • Z j‖ ≤ δ + 8 * β * δ := by
    have hsplit : (∑ j : Idx n, softmaxWeight (saScore d n β Y i) j • Y j)
        - ∑ j : Idx n, softmaxWeight (saScore d n β Z i) j • Z j
        = ∑ j : Idx n, (softmaxWeight (saScore d n β Y i) j • (Y j - Z j)
            + (softmaxWeight (saScore d n β Y i) j
                - softmaxWeight (saScore d n β Z i) j) • Z j) := by
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [smul_sub, sub_smul]
      abel
    rw [hsplit]
    refine (norm_sum_le _ _).trans ?_
    have hterm : ∀ j : Idx n,
        ‖softmaxWeight (saScore d n β Y i) j • (Y j - Z j)
            + (softmaxWeight (saScore d n β Y i) j
                - softmaxWeight (saScore d n β Z i) j) • Z j‖
          ≤ softmaxWeight (saScore d n β Y i) j * δ
            + |softmaxWeight (saScore d n β Y i) j
                - softmaxWeight (saScore d n β Z i) j| := by
      intro j
      have e1 : ‖softmaxWeight (saScore d n β Y i) j • (Y j - Z j)‖
          ≤ softmaxWeight (saScore d n β Y i) j * δ := by
        rw [norm_smul, Real.norm_eq_abs,
          abs_of_nonneg (softmaxWeight_nonneg (saScore d n β Y i) j)]
        exact mul_le_mul_of_nonneg_left (hstep j)
          (softmaxWeight_nonneg (saScore d n β Y i) j)
      have e2 : ‖(softmaxWeight (saScore d n β Y i) j
              - softmaxWeight (saScore d n β Z i) j) • Z j‖
          = |softmaxWeight (saScore d n β Y i) j
              - softmaxWeight (saScore d n β Z i) j| := by
        rw [norm_smul, Real.norm_eq_abs, hZ j, mul_one]
      refine (norm_add_le _ _).trans ?_
      rw [e2]
      linarith
    calc ∑ j : Idx n, ‖softmaxWeight (saScore d n β Y i) j • (Y j - Z j)
            + (softmaxWeight (saScore d n β Y i) j
                - softmaxWeight (saScore d n β Z i) j) • Z j‖
        ≤ ∑ j : Idx n, (softmaxWeight (saScore d n β Y i) j * δ
            + |softmaxWeight (saScore d n β Y i) j
                - softmaxWeight (saScore d n β Z i) j|) :=
          Finset.sum_le_sum fun j _ => hterm j
      _ = (∑ j : Idx n, softmaxWeight (saScore d n β Y i) j) * δ
            + ∑ j : Idx n, |softmaxWeight (saScore d n β Y i) j
                - softmaxWeight (saScore d n β Z i) j| := by
          rw [Finset.sum_add_distrib, Finset.sum_mul]
      _ ≤ 1 * δ + 4 * (β * (2 * δ)) := by
          rw [sum_softmaxWeight hn]
          linarith
      _ = δ + 8 * β * δ := by ring
  -- hence the drift moves by at most `(3 + 8 β) δ`
  have hbound := norm_proj_sub_proj_le (d := d) (hY i) (hZ i)
    (norm_saAvg_le d n β hn hZ i) (hstep i) hmdiff
  rw [dist_eq_norm]
  have hfinal : (3 : ℝ) + 8 * β ≤ 10 * (n : ℝ) * max 1 β := by nlinarith
  calc ‖saField d n β Y i - saField d n β Z i‖ ≤ (δ + 8 * β * δ) + 2 * δ := hbound
    _ = (3 + 8 * β) * δ := by ring
    _ ≤ 10 * (n : ℝ) * max 1 β * δ := by
        exact mul_le_mul_of_nonneg_right hfinal hd0

/-- The hypotheses of `lipschitzOnWith_saField` are satisfiable. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℕ) < 1 := ⟨zero_le_one, one_pos⟩

end Perspective
end Transformer
