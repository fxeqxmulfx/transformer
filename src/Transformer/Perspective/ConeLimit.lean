/-
# Cone collapse — exponential convergence of one weighted flow

Auxiliary to arXiv:2312.10794v5, *A mathematical perspective on Transformers*,
§6.1, `lem: hemisphere.clustering`.

For a weighted flow with weights in `[m, M]`, `m > 0`, started in the open
hemisphere around `w`, the width of the configuration in the chart
`y = x / ⟨x, w⟩` decays like `e^{-κ t}` (`Perspective.ConeWidth`), so every
velocity `ẏ_i = Σ_j b_ij (y_j - y_i)` does too, the `y_i` converge at that rate
to one point `y⋆` with `‖y⋆‖ ≥ 1`, and the particles `x_i = y_i / ‖y_i‖` to
`x⋆ = y⋆ / ‖y⋆‖` (`exists_expLimit_of_weightedFlow`).
-/

import Transformer.Perspective.ConeWidth

open scoped BigOperators Topology
open Real Filter

namespace Transformer
namespace Perspective

variable {d n : ℕ} {X : ℝ → SphereTuple d n} {a : ℝ → Idx n → Idx n → ℝ}

/-- Normalisation is `2`-Lipschitz away from the unit ball:
`‖u / ‖u‖ - v / ‖v‖‖ ≤ 2 ‖u - v‖` for `‖u‖ ≥ 1`, `v ≠ 0`. -/
theorem norm_normalize_sub_le {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {u v : E}
    (hu : 1 ≤ ‖u‖) (hv : 0 < ‖v‖) : ‖‖u‖⁻¹ • u - ‖v‖⁻¹ • v‖ ≤ 2 * ‖u - v‖ := by
  have hu0 : 0 < ‖u‖ := one_pos.trans_le hu
  have hsplit : ‖u‖⁻¹ • u - ‖v‖⁻¹ • v = ‖u‖⁻¹ • (u - v) + (‖u‖⁻¹ - ‖v‖⁻¹) • v := by
    rw [smul_sub, sub_smul]; abel
  have h2 : ‖(‖u‖⁻¹ - ‖v‖⁻¹) • v‖ ≤ ‖u‖⁻¹ * ‖u - v‖ := by
    rw [norm_smul, Real.norm_eq_abs, inv_sub_inv hu0.ne' hv.ne', abs_div, abs_mul, abs_of_pos hu0,
      abs_of_pos hv, div_mul_eq_mul_div, mul_div_mul_right _ _ hv.ne', div_eq_inv_mul]
    exact mul_le_mul_of_nonneg_left ((abs_norm_sub_norm_le v u).trans_eq (norm_sub_rev v u))
      (inv_nonneg.mpr hu0.le)
  have h1 : ‖‖u‖⁻¹ • (u - v)‖ = ‖u‖⁻¹ * ‖u - v‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hu0)]
  have hinv : ‖u‖⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hu
  rw [hsplit]
  refine (norm_add_le _ _).trans ?_
  rw [h1]
  nlinarith [norm_nonneg (u - v)]

/-- The hypotheses of `norm_normalize_sub_le` are satisfiable: `u = v = 1` in `ℝ`. -/
example : (1 : ℝ) ≤ ‖(1 : ℝ)‖ ∧ (0 : ℝ) < ‖(1 : ℝ)‖ := by norm_num

/-- **Exponential convergence of a weighted flow started in a hemisphere.**

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`, whose proof
"only makes use of the positivity of the coefficients `a_ij`"; the upper bound
`M` is the boundedness of the weights, automatic for the dynamics there. -/
theorem exists_expLimit_of_weightedFlow (hX : IsWeightedFlow X a) {m M : ℝ} (hm : 0 < m)
    (ha : ∀ t i j, m ≤ a t i j ∧ a t i j ≤ M) (hn : 0 < n) (w : SSphere d)
    (hw0 : ∀ i, 0 < inner (𝕜 := ℝ) (X 0 i : EucSpace d) (w : EucSpace d)) :
    ∃ (x_star : SSphere d) (C lam : ℝ), 0 < C ∧ 0 < lam ∧
      ∀ (i : Idx n) (t : ℝ), 0 ≤ t →
        ‖(X t i : EucSpace d) - x_star‖ ≤ C * exp (-(lam * t)) := by
  have : Nonempty (Idx n) := ⟨⟨0, hn⟩⟩
  have hne : (Finset.univ : Finset (Idx n)).Nonempty := Finset.univ_nonempty
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  set W : EucSpace d := (w : EucSpace d)
  set r₀ := Finset.univ.inf' hne fun i => inner (𝕜 := ℝ) (X 0 i : EucSpace d) W
  have hr₀ : 0 < r₀ := (Finset.lt_inf'_iff hne).2 fun i _ => hw0 i
  have hfl : ∀ t : ℝ, 0 ≤ t → ∀ i, r₀ ≤ inner (𝕜 := ℝ) (X t i : EucSpace d) W :=
    fun t ht i => floor_of_weightedFlow hX (fun t i j => hm.le.trans (ha t i j).1) hn W hr₀
      (fun i => Finset.inf'_le _ (Finset.mem_univ i)) ht i
  have hM : 0 ≤ M := hm.le.trans ((ha 0 ⟨0, hn⟩ ⟨0, hn⟩).1.trans (ha 0 ⟨0, hn⟩ ⟨0, hn⟩).2)
  set κ := 2 * m * r₀
  have hκ : 0 < κ := by positivity
  set D₀ := chartSpread W X
  have hD₀ : 0 ≤ D₀ := Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => norm_nonneg _
  set A := n * (M / r₀) * D₀
  have hA : 0 ≤ A := by positivity
  set y : Idx n → ℝ → EucSpace d := fun i t => hemiChart W (X t i : EucSpace d)
  have hwidth : ∀ t, 0 ≤ t → ∀ i k, ‖y i t - y k t‖ ≤ D₀ * exp (-(κ * t)) := fun t ht i k => by
    simpa [κ, mul_assoc] using
      norm_hemiChart_sub_le hX hm (fun t i j => (ha t i j).1) hw hr₀ hfl ht i k
  -- The velocity in the chart.
  have hvel : ∀ i t, ∃ v, 0 ≤ t → HasDerivAt (y i) v t ∧ ‖v‖ ≤ A * exp (-(κ * t)) := by
    intro i t
    by_cases ht : 0 ≤ t
    swap
    · exact ⟨0, fun h => absurd h ht⟩
    have hpos : ∀ j, 0 < inner (𝕜 := ℝ) (X t j : EucSpace d) W := fun j => hr₀.trans_le (hfl t ht j)
    refine ⟨_, fun _ => ⟨hasDerivAt_hemiChart hX W hpos i, ?_⟩⟩
    refine (norm_sum_le _ _).trans ?_
    have hterm : ∀ j, ‖(a t i j * (inner (𝕜 := ℝ) (X t j : EucSpace d) W
        / inner (𝕜 := ℝ) (X t i : EucSpace d) W)) • (y j t - y i t)‖
        ≤ M / r₀ * D₀ * exp (-(κ * t)) := by
      intro j
      have hb0 : 0 ≤ a t i j * (inner (𝕜 := ℝ) (X t j : EucSpace d) W
          / inner (𝕜 := ℝ) (X t i : EucSpace d) W) :=
        mul_nonneg (hm.le.trans (ha t i j).1) (div_nonneg (hpos j).le (hpos i).le)
      have hb : a t i j * (inner (𝕜 := ℝ) (X t j : EucSpace d) W
          / inner (𝕜 := ℝ) (X t i : EucSpace d) W) ≤ M / r₀ := by
        refine mul_le_mul (ha t i j).2 ?_ (div_nonneg (hpos j).le (hpos i).le) hM |>.trans_eq
          (mul_one_div M r₀)
        exact div_le_div₀ zero_le_one (height_le_one hw (X t j)) hr₀ (hfl t ht i)
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hb0, mul_assoc (M / r₀)]
      exact mul_le_mul hb (hwidth t ht j i) (norm_nonneg _) (by positivity)
    refine (Finset.sum_le_sum fun j _ => hterm j).trans_eq ?_
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  -- Increments: `‖y_i(t') - y_i(t)‖ ≤ (A / κ) e^{-κ t}` for `0 ≤ t ≤ t'`.
  have hinc : ∀ i t t', 0 ≤ t → t ≤ t' → ‖y i t' - y i t‖ ≤ A / κ * exp (-(κ * t)) := by
    intro i t t' ht htt'
    choose v hv using hvel i
    have hB : ∀ s, HasDerivAt (fun s => A / κ * (exp (-(κ * t)) - exp (-(κ * s))))
        (A * exp (-(κ * s))) s := by
      intro s
      have h := ((hasDerivAt_id s).const_mul κ).neg.exp.const_sub (exp (-(κ * t)))
        |>.const_mul (A / κ)
      refine h.congr_deriv ?_
      field_simp
      simp
    have key := image_norm_le_of_norm_deriv_right_le_deriv_boundary
      (f := fun s => y i s - y i t) (f' := v)
      (a := t) (b := t') ?_ ?_ (by simp) hB ?_ ⟨htt', le_rfl⟩
    · refine key.trans ?_
      have := exp_pos (-(κ * t'))
      have : 0 ≤ A / κ := div_nonneg hA hκ.le
      nlinarith
    · exact fun s hs => ((hv s (ht.trans hs.1)).1.sub_const _).continuousAt.continuousWithinAt
    · exact fun s hs => ((hv s (ht.trans hs.1)).1.sub_const _).hasDerivWithinAt
    · exact fun s hs => (hv s (ht.trans hs.1)).2
  -- The limit `y⋆` of one particle in the chart.
  set i₀ : Idx n := ⟨0, hn⟩
  have htend : Tendsto (fun N : ℝ => A / κ * exp (-(κ * N))) atTop (𝓝 0) := by
    simpa using (tendsto_exp_neg_atTop_nhds_zero.comp
      (tendsto_id.const_mul_atTop hκ)).const_mul (A / κ)
  have hcauchy : CauchySeq (y i₀) := by
    refine Metric.cauchySeq_iff'.2 fun ε hε => ?_
    obtain ⟨N, hN, hN0⟩ := ((htend.eventually (gt_mem_nhds hε)).and
      (eventually_ge_atTop 0)).exists
    exact ⟨N, fun t ht => (dist_eq_norm _ _).trans_lt ((hinc i₀ N t hN0 ht).trans_lt hN)⟩
  obtain ⟨ys, hys⟩ := cauchySeq_tendsto_of_complete hcauchy
  have hlim : ∀ t, 0 ≤ t → ‖y i₀ t - ys‖ ≤ A / κ * exp (-(κ * t)) := by
    intro t ht
    refine le_of_tendsto ((tendsto_const_nhds.sub hys).norm) ?_
    filter_upwards [eventually_ge_atTop t] with s hs
    rw [norm_sub_rev]
    exact hinc i₀ t s ht hs
  have hynorm : ∀ i t, 0 ≤ t → 1 ≤ ‖y i t‖ := by
    intro i t ht
    rw [(hemiChart_normalize (norm_coe_tuple (X t) i) (hr₀.trans_le (hfl t ht i))).1]
    exact one_le_inv₀ (hr₀.trans_le (hfl t ht i)) |>.2 (height_le_one hw (X t i))
  have hys1 : 1 ≤ ‖ys‖ :=
    ge_of_tendsto hys.norm ((eventually_ge_atTop 0).mono fun t ht => hynorm i₀ t ht)
  -- Back on the sphere.
  have hxs : ‖(‖ys‖⁻¹ • ys)‖ = 1 := by
    rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_norm, inv_mul_cancel₀ (by linarith)]
  refine ⟨⟨‖ys‖⁻¹ • ys, mem_sphere_zero_iff_norm.mpr hxs⟩, 2 * (D₀ + A / κ) + 1, κ,
    by positivity, hκ, fun i t ht => ?_⟩
  have hxi := (hemiChart_normalize (norm_coe_tuple (X t) i) (hr₀.trans_le (hfl t ht i))).2
  change ‖(X t i : EucSpace d) - ‖ys‖⁻¹ • ys‖ ≤ _
  rw [← hxi, norm_sub_rev]
  have hyi : ‖y i t - ys‖ ≤ (D₀ + A / κ) * exp (-(κ * t)) := by
    calc ‖y i t - ys‖ ≤ ‖y i t - y i₀ t‖ + ‖y i₀ t - ys‖ := norm_sub_le_norm_sub_add_norm_sub _ _ _
      _ ≤ D₀ * exp (-(κ * t)) + A / κ * exp (-(κ * t)) := add_le_add (hwidth t ht i i₀) (hlim t ht)
      _ = (D₀ + A / κ) * exp (-(κ * t)) := by ring
  refine (norm_normalize_sub_le hys1 (one_pos.trans_le (hynorm i t ht))).trans ?_
  rw [norm_sub_rev]
  have := exp_pos (-(κ * t))
  nlinarith

/-- The hypotheses of `exists_expLimit_of_weightedFlow` are satisfiable: one
resting particle with weight `1`, `m = M = 1`, in the hemisphere around
itself. -/
example : IsWeightedFlow (fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) (fun _ _ _ => 1) ∧
    (∀ (_ : ℝ) (_ _ : Idx 1), (1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1) ∧ 0 < 1 ∧
    ∀ _ : Idx 1, 0 < inner (𝕜 := ℝ) ((basePoint 0 : SSphere 1) : EucSpace 1)
      ((basePoint 0 : SSphere 1) : EucSpace 1) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  refine ⟨fun t _ => ?_, fun _ _ _ => ⟨le_rfl, le_rfl⟩, one_pos, fun _ => ?_⟩
  · simpa [proj, hx] using hasDerivAt_const t ((basePoint 0 : SSphere 1) : EucSpace 1)
  · simp [hx]

end Perspective
end Transformer
