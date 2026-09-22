/-
# Cone collapse — the width of the configuration in the chart decays

Auxiliary to arXiv:2312.10794v5, *A mathematical perspective on Transformers*,
§6.1, `lem: hemisphere.clustering`.

In the chart `y = x / ⟨x, w⟩` of `Perspective.ConeChart`, a weighted flow with
weights `a_ij ≥ m > 0` whose heights stay in `[r₀, 1]` is the consensus
dynamics `ẏ_i = Σ_j b_ij (y_j - y_i)` with `b_ij = a_ij s_j / s_i ≥ m r₀`.  In
any direction `u`, the width `max_i ⟨y_i, u⟩ - min_i ⟨y_i, u⟩` then contracts at
rate `2 m r₀`: the top particle is pulled down by the bottom one at least at
rate `m r₀`, and the bottom one up by the top one.  Hence
`‖y_i(t) - y_k(t)‖ ≤ D₀ e^{-2 m r₀ t}` (`norm_hemiChart_sub_le`).
-/

import Transformer.Perspective.ConeChart
import Transformer.Perspective.MaxCurve
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {d n : ℕ} {X : ℝ → SphereTuple d n} {a : ℝ → Idx n → Idx n → ℝ}

/-- The spread of the configuration in the chart at time `0`:
`D₀ = Σ_p Σ_q ‖y_p(0) - y_q(0)‖`. -/
noncomputable def chartSpread (w : EucSpace d) (X : ℝ → SphereTuple d n) : ℝ :=
  ∑ p : Idx n, ∑ q : Idx n,
    ‖hemiChart w (X 0 p : EucSpace d) - hemiChart w (X 0 q : EucSpace d)‖

theorem height_le_one {w : EucSpace d} (hw : ‖w‖ = 1) (x : SSphere d) :
    inner (𝕜 := ℝ) (x : EucSpace d) w ≤ 1 := by
  have := real_inner_le_norm (x : EucSpace d) w
  rwa [mem_sphere_zero_iff_norm.mp x.2, hw, mul_one] at this

/-- **Width decay in a fixed direction.**  For `t ≥ 0`,
`⟨y_i(t) - y_k(t), u⟩ ≤ D₀ ‖u‖ e^{-2 m r₀ t}`.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`. -/
theorem inner_hemiChart_sub_le (hX : IsWeightedFlow X a) {m r₀ : ℝ} (hm : 0 < m)
    (ha : ∀ t i j, m ≤ a t i j) {w : EucSpace d} (hw : ‖w‖ = 1) (hr₀ : 0 < r₀)
    (hfl : ∀ t : ℝ, 0 ≤ t → ∀ i, r₀ ≤ inner (𝕜 := ℝ) (X t i : EucSpace d) w)
    (u : EucSpace d) {t : ℝ} (ht : 0 ≤ t) (i k : Idx n) :
    inner (𝕜 := ℝ) (hemiChart w (X t i : EucSpace d) - hemiChart w (X t k : EucSpace d)) u
      ≤ chartSpread w X * ‖u‖ * exp (-(2 * m * r₀ * t)) := by
  set κ := 2 * m * r₀
  set s : ℝ → Idx n → ℝ := fun t j => inner (𝕜 := ℝ) (X t j : EucSpace d) w
  set g : Idx n → ℝ → ℝ := fun j t => inner (𝕜 := ℝ) (hemiChart w (X t j : EucSpace d)) u
  set b : ℝ → Idx n → Idx n → ℝ := fun t i j => a t i j * (s t j / s t i)
  set g' : Idx n → ℝ → ℝ := fun i t => ∑ j : Idx n, b t i j * (g j t - g i t)
  have hpos : ∀ t, 0 ≤ t → ∀ j, 0 < s t j := fun t ht j => hr₀.trans_le (hfl t ht j)
  have hb : ∀ t, 0 ≤ t → ∀ i j, m * r₀ ≤ b t i j := by
    intro t ht i j
    have hsi := hpos t ht i
    have hsi1 : s t i ≤ 1 := height_le_one hw (X t i)
    have hsj : r₀ ≤ s t j := hfl t ht j
    have h1 : s t j ≤ s t j / s t i := by
      rw [le_div_iff₀ hsi]; nlinarith [hpos t ht j]
    calc m * r₀ ≤ a t i j * s t j :=
          mul_le_mul (ha t i j) hsj hr₀.le (hm.le.trans (ha t i j))
      _ ≤ a t i j * (s t j / s t i) :=
          mul_le_mul_of_nonneg_left h1 (hm.le.trans (ha t i j))
  have hg : ∀ j t, 0 ≤ t → HasDerivAt (g j) (g' j t) t := by
    intro j t ht
    refine ((hasDerivAt_hemiChart hX w (hpos t ht) j).inner ℝ
      (hasDerivAt_const t u)).congr_deriv ?_
    simp only [inner_zero_right, zero_add, sum_inner, real_inner_smul_left, inner_sub_left, g', g,
      b, s]
  -- The pairs `(i, k)` and the weighted gaps `e^{κ t} (g_i - g_k)`.
  set F : Idx n × Idx n → ℝ → ℝ := fun p t => exp (κ * t) * (g p.1 t - g p.2 t)
  set F' : Idx n × Idx n → ℝ → ℝ := fun p t =>
    exp (κ * t) * κ * (g p.1 t - g p.2 t) + exp (κ * t) * (g' p.1 t - g' p.2 t)
  have hne : (Finset.univ : Finset (Idx n × Idx n)).Nonempty := ⟨(i, k), Finset.mem_univ _⟩
  set R : ℝ → ℝ := fun t => Finset.univ.sup' hne fun p => F p t
  have hmax : ∀ t p, F p t ≤ R t := fun t p => Finset.le_sup' (fun p => F p t) (Finset.mem_univ p)
  have hatt : ∀ t, ∃ p, R t = F p t := fun t => by
    obtain ⟨p, -, hp⟩ := Finset.exists_mem_eq_sup' hne fun p => F p t
    exact ⟨p, hp⟩
  have hFd : ∀ p t, 0 ≤ t → HasDerivAt (F p) (F' p t) t := by
    intro p t ht
    have he : HasDerivAt (fun t => exp (κ * t)) (exp (κ * t) * κ) t := by
      simpa using HasDerivAt.exp ((hasDerivAt_id t).const_mul κ)
    exact (he.mul ((hg p.1 t ht).sub (hg p.2 t ht))).congr_deriv
      (by simp only [F', Pi.sub_apply])
  have hnp : ∀ t, 0 ≤ t → ∀ p, F p t = R t → F' p t ≤ 0 := by
    rintro t ht ⟨i, k⟩ hp
    have he := exp_pos (κ * t)
    have htop : ∀ j, g j t ≤ g i t := fun j => by
      have := (hmax t (j, k)).trans_eq hp.symm
      simp only [F] at this
      nlinarith
    have hbot : ∀ j, g k t ≤ g j t := fun j => by
      have := (hmax t (i, j)).trans_eq hp.symm
      simp only [F] at this
      nlinarith
    have hD : 0 ≤ g i t - g k t := sub_nonneg.mpr (htop k)
    have hi : g' i t ≤ m * r₀ * (g k t - g i t) := by
      have hle : g' i t ≤ b t i k * (g k t - g i t) := by
        simp only [g']
        rw [← Finset.add_sum_erase _ _ (Finset.mem_univ k)]
        have : ∑ j ∈ Finset.univ.erase k, b t i j * (g j t - g i t) ≤ 0 :=
          Finset.sum_nonpos fun j _ => mul_nonpos_of_nonneg_of_nonpos
            ((mul_pos hm hr₀).le.trans (hb t ht i j)) (sub_nonpos.mpr (htop j))
        linarith
      nlinarith [hb t ht i k]
    have hk : m * r₀ * (g i t - g k t) ≤ g' k t := by
      have hle : b t k i * (g i t - g k t) ≤ g' k t :=
        Finset.single_le_sum (f := fun j => b t k j * (g j t - g k t))
          (fun j _ => mul_nonneg ((mul_pos hm hr₀).le.trans (hb t ht k j))
            (sub_nonneg.mpr (hbot j))) (Finset.mem_univ i)
      nlinarith [hb t ht k i]
    simp only [F']
    have : κ * (g i t - g k t) + (g' i t - g' k t) ≤ 0 := by simp only [κ]; nlinarith
    nlinarith
  have hRt : R t ≤ R 0 := max_curve_le_of_deriv_nonpos hFd hmax hatt hnp le_rfl ht
  obtain ⟨⟨p, q⟩, hpq⟩ := hatt 0
  have hR0 : R 0 ≤ chartSpread w X * ‖u‖ := by
    rw [hpq]
    simp only [F, mul_zero, exp_zero, one_mul, g, ← inner_sub_left]
    refine (real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_right ?_ (norm_nonneg u))
    exact (Finset.single_le_sum (f := fun q => ‖hemiChart w (X 0 p : EucSpace d)
        - hemiChart w (X 0 q : EucSpace d)‖) (fun _ _ => norm_nonneg _) (Finset.mem_univ q)).trans
      (Finset.single_le_sum (f := fun p => ∑ q : Idx n, ‖hemiChart w (X 0 p : EucSpace d)
        - hemiChart w (X 0 q : EucSpace d)‖) (fun _ _ => Finset.sum_nonneg fun _ _ =>
          norm_nonneg _) (Finset.mem_univ p))
  have hF := (hmax t (i, k)).trans (hRt.trans hR0)
  simp only [F, g] at hF
  rw [← inner_sub_left, ← le_div_iff₀' (exp_pos _), div_eq_mul_inv, ← exp_neg] at hF
  simpa [κ, mul_assoc] using hF

/-- **Width decay.**  For `t ≥ 0`, `‖y_i(t) - y_k(t)‖ ≤ D₀ e^{-2 m r₀ t}`.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`. -/
theorem norm_hemiChart_sub_le (hX : IsWeightedFlow X a) {m r₀ : ℝ} (hm : 0 < m)
    (ha : ∀ t i j, m ≤ a t i j) {w : EucSpace d} (hw : ‖w‖ = 1) (hr₀ : 0 < r₀)
    (hfl : ∀ t : ℝ, 0 ≤ t → ∀ i, r₀ ≤ inner (𝕜 := ℝ) (X t i : EucSpace d) w)
    {t : ℝ} (ht : 0 ≤ t) (i k : Idx n) :
    ‖hemiChart w (X t i : EucSpace d) - hemiChart w (X t k : EucSpace d)‖
      ≤ chartSpread w X * exp (-(2 * m * r₀ * t)) := by
  set v := hemiChart w (X t i : EucSpace d) - hemiChart w (X t k : EucSpace d)
  have h := inner_hemiChart_sub_le hX hm ha hw hr₀ hfl v ht i k
  rw [real_inner_self_eq_norm_mul_norm] at h
  rcases (norm_nonneg v).eq_or_lt with h0 | h0
  · rw [← h0]
    exact mul_nonneg (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => norm_nonneg _)
      (exp_pos _).le
  · nlinarith

/-- The hypotheses of `inner_hemiChart_sub_le` and `norm_hemiChart_sub_le` are
satisfiable: one resting particle with weight `1`, in the hemisphere around
itself, `m = r₀ = 1`. -/
example : IsWeightedFlow (fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) (fun _ _ _ => 1) ∧
    ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 ∧
    ∀ t : ℝ, 0 ≤ t → ∀ i : Idx 1, (1 : ℝ) ≤ inner (𝕜 := ℝ)
      (((fun _ _ => basePoint 0 : ℝ → SphereTuple 1 1) t i : EucSpace 1))
      ((basePoint 0 : SSphere 1) : EucSpace 1) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  refine ⟨fun t _ => ?_, hx, fun _ _ _ => ?_⟩
  · simpa [proj, real_inner_self_eq_norm_mul_norm, hx] using
      hasDerivAt_const t ((basePoint 0 : SSphere 1) : EucSpace 1)
  · simp [hx]

end Perspective
end Transformer
