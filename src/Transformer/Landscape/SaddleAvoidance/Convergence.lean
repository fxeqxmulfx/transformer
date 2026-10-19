module

public import Transformer.Landscape.SaddleAvoidance.Trap
public import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

/-!
# Saddle avoidance — gradient descent converges near a Łojasiewicz point

Along an orbit of gradient descent whose steps lower `f` by at least `η κ ‖∇f‖²`,
`κ > 0`, with a limit point `z` up to translations leaving `f` invariant, where `f`
is continuous, the values of `f` decrease to `f z` (`antitone_iterate_of_le_sub`,
`frequently_abs_sub_lt`, `le_and_tendsto_of_frequently`). Such a point exists when
every point of the orbit can be translated into one compact set
(`exists_frequently_norm_add_sub_lt`), as for a function on a torus.

If `f` satisfies the Łojasiewicz inequality at `z` with `0 ≤ a < 1`, the orbit comes,
up to a translation, close enough to `z` with `f` close enough to `f z` for the trap
of `iterate_mem_ball_of_le_sub`. From then on it has finite length, so it converges
(`exists_tendsto_of_isLojasiewiczAt_of_le_sub`), to `z` itself when `z` is a limit
point of the orbit (`tendsto_of_isLojasiewiczAt_of_le_sub`). This is Theorem 3.2 of
Absil, Mahony and Andrews (SIAM J. Optim. 16, 2005), up to translations. An
`L`-Lipschitz gradient with `η L < 1` gives the decrease with `κ = 1 - η L`
(`exists_tendsto_of_isLojasiewiczAt`, `tendsto_of_isLojasiewiczAt`).
-/

@[expose] public section

open Filter Metric
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η κ a : ℝ}

/-- **Gradient descent converges near a point where the Łojasiewicz inequality
holds**, along an orbit whose steps lower `f` by at least `η κ ‖∇f‖²`, `κ > 0`. Let
`0 < η`, and let `f` be continuous at `z` and satisfy the Łojasiewicz inequality there
with `0 ≤ a < 1`. If infinitely many points of the orbit of `x` lie near `z` up to
translations in a set `Λ` leaving `f` invariant, the orbit converges. -/
theorem exists_tendsto_of_isLojasiewiczAt_of_le_sub (hη : 0 < η) (hκ : 0 < κ) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) (ha0 : 0 ≤ a) (ha1 : a < 1) {z : E}
    (hz : IsLojasiewiczAt f z a) (hfc : ContinuousAt f z) {x : E}
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hcl : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨m, hm, ε, hε, hev⟩ := hz
  obtain ⟨r, hr, hloj⟩ := Metric.eventually_nhds_iff_ball.1 hev
  have hc : 0 < m * (1 - a) * κ := mul_pos (mul_pos hm (by linarith)) hκ
  obtain ⟨hfz, hlim⟩ := le_and_tendsto_of_frequently (antitone_iterate_of_le_sub hη.le hκ.le hdec)
    (frequently_abs_sub_lt hfc hΛ hcl)
  have hφ : Tendsto (fun k => (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a)) atTop (𝓝 0) := by
    have h := (hlim.sub_const (f z)).rpow_const (Or.inr (sub_nonneg.2 ha1.le))
    rwa [sub_self, Real.zero_rpow (sub_pos.2 ha1).ne'] at h
  obtain ⟨k0, ⟨v, hv, hk0⟩, hφk0, hfk0⟩ := ((hcl (r / 2) (half_pos hr)).and_eventually
    ((hφ.eventually (gt_mem_nhds (half_pos (mul_pos hc hr)))).and
      (hlim.eventually (gt_mem_nhds (lt_add_of_pos_right (f z) hε))))).exists
  have horb : ∀ n, (gradientStep f η)^[n] ((gradientStep f η)^[k0] x + v) =
      (gradientStep f η)^[n + k0] x + v := fun n => by
    rw [iterate_gradientStep_add (hΛ v hv), Function.iterate_add_apply]
  have hfo : ∀ n, f ((gradientStep f η)^[n] ((gradientStep f η)^[k0] x + v)) =
      f ((gradientStep f η)^[n + k0] x) := fun n => by rw [horb, hΛ v hv]
  have hfz' : ∀ n, f z ≤ f ((gradientStep f η)^[n] ((gradientStep f η)^[k0] x + v)) :=
    fun n => by
      rw [hfo]
      exact hfz _
  have hdec' : ∀ n, f ((gradientStep f η)^[n + 1] ((gradientStep f η)^[k0] x + v)) ≤
      f ((gradientStep f η)^[n] ((gradientStep f η)^[k0] x + v)) - η * κ *
        ‖gradient f ((gradientStep f η)^[n] ((gradientStep f η)^[k0] x + v))‖ ^ 2 := fun n => by
    rw [hfo, hfo, horb, gradient_add_eq (hΛ v hv), Nat.add_right_comm n 1 k0]
    exact hdec _
  have hε' : f ((gradientStep f η)^[k0] x + v) < f z + ε := by rwa [hΛ v hv]
  have hstart : m * (1 - a) * κ * ‖(gradientStep f η)^[k0] x + v - z‖ +
      (f ((gradientStep f η)^[k0] x + v) - f z) ^ (1 - a) < m * (1 - a) * κ * r := by
    rw [hΛ v hv]
    have := mul_lt_mul_of_pos_left hk0 hc
    linarith
  have hcauchy : CauchySeq fun n => (gradientStep f η)^[n + k0] x := by
    refine cauchySeq_of_le_tendsto_0' (fun n =>
      (f ((gradientStep f η)^[n + k0] x) - f z) ^ (1 - a) / (m * (1 - a) * κ))
      (fun n k hnk => ?_) ?_
    · have h := mul_norm_iterate_sub_le_of_le_sub hη hκ ha0 ha1 hm hdec' hloj hfz' hε' hstart hnk
      rw [hfo, hfo, horb, horb, add_sub_add_right_eq_sub] at h
      rw [dist_comm, dist_eq_norm, le_div_iff₀ hc]
      have := Real.rpow_nonneg (sub_nonneg.2 (hfz (k + k0))) (1 - a)
      linarith
    · have h := (hφ.comp (tendsto_add_atTop_nat k0)).div_const (m * (1 - a) * κ)
      rwa [zero_div] at h
  obtain ⟨w, hw⟩ := cauchySeq_tendsto_of_complete hcauchy
  exact ⟨w, (tendsto_add_atTop_iff_nat k0).1 hw⟩

/-- **Gradient descent converges near a point where the Łojasiewicz inequality
holds**, for an `L`-Lipschitz gradient and `η L < 1`:
`exists_tendsto_of_isLojasiewiczAt_of_le_sub` with `κ = 1 - η L`. -/
theorem exists_tendsto_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) (ha0 : 0 ≤ a) (ha1 : a < 1) {z : E}
    (hz : IsLojasiewiczAt f z a) {x : E}
    (hcl : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) :=
  exists_tendsto_of_isLojasiewiczAt_of_le_sub hη (by linarith) hΛ ha0 ha1 hz
    hf.continuous.continuousAt (iterate_succ_le_sub hf hL η x) hcl

/-- **Gradient descent converges to a limit point where the Łojasiewicz inequality
holds**, under the hypotheses of `exists_tendsto_of_isLojasiewiczAt_of_le_sub`. -/
theorem tendsto_of_isLojasiewiczAt_of_le_sub (hη : 0 < η) (hκ : 0 < κ) (ha0 : 0 ≤ a)
    (ha1 : a < 1) {z : E} (hz : IsLojasiewiczAt f z a) (hfc : ContinuousAt f z) {x : E}
    (hdec : ∀ k, f ((gradientStep f η)^[k + 1] x) ≤
      f ((gradientStep f η)^[k] x) - η * κ * ‖gradient f ((gradientStep f η)^[k] x)‖ ^ 2)
    (hcl : MapClusterPt z atTop fun k => (gradientStep f η)^[k] x) :
    Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z) := by
  have hfreq : ∀ ε > 0, ∃ᶠ k in atTop, (gradientStep f η)^[k] x ∈ ball z ε := fun ε hε =>
    mapClusterPt_iff_frequently.1 hcl _ (ball_mem_nhds z hε)
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt_of_le_sub hη hκ (Λ := {0}) (by simp) ha0
    ha1 hz hfc hdec fun ε hε => (hfreq ε hε).mono fun k hk =>
      ⟨0, Set.mem_singleton 0, by rw [add_zero, ← dist_eq_norm]; exact mem_ball.1 hk⟩
  have hwz : w = z := eq_of_forall_dist_le fun ε hε => by
    obtain ⟨k, hk, hkw⟩ := ((hfreq (ε / 2) (half_pos hε)).and_eventually
      (Metric.tendsto_nhds.1 hw _ (half_pos hε))).exists
    rw [mem_ball] at hk
    linarith [dist_triangle_left w z ((gradientStep f η)^[k] x)]
  rwa [hwz] at hw

/-- **Gradient descent converges to a limit point where the Łojasiewicz inequality
holds**, for an `L`-Lipschitz gradient and `η L < 1`:
`tendsto_of_isLojasiewiczAt_of_le_sub` with `κ = 1 - η L`. -/
theorem tendsto_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a)
    (ha1 : a < 1) {z : E} (hz : IsLojasiewiczAt f z a) {x : E}
    (hcl : MapClusterPt z atTop fun k => (gradientStep f η)^[k] x) :
    Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z) :=
  tendsto_of_isLojasiewiczAt_of_le_sub hη (by linarith) ha0 ha1 hz hf.continuous.continuousAt
    (iterate_succ_le_sub hf hL η x) hcl

/-- On `‖x‖² / 2` every orbit of gradient descent with `|1 - η| < 1` converges to the
critical point `0`. -/
theorem tendsto_iterate_gradientStep_half_mul_norm_sq (hη : |1 - η| < 1) (x : E) :
    Tendsto (fun k => (gradientStep (fun x : E => (1 / 2 : ℝ) * ‖x‖ ^ 2) η)^[k] x) atTop
      (𝓝 0) := by
  simp only [iterate_gradientStep_half_mul_norm_sq]
  simpa using (tendsto_pow_atTop_nhds_zero_of_abs_lt_one hη).smul_const x

/-- On the line the orbit of `1` for `η = 3/2`, `(-1/2)ᵏ`, converges to `0`. -/
example := tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 3 / 2) (by norm_num) 1

/-- `‖x‖² / 2` on the line satisfies the inequality at its critical point `0` with
`a = 1/2`, the orbit `2⁻ᵏ` of `1` for `η = 1/2` and `L = 1` converges to `0`, and
`Λ = {0}`. -/
example := exists_tendsto_of_isLojasiewiczAt (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1)
  ((contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero)
  (by rw [gradient_half_mul_norm_sq]; exact LipschitzWith.id) (η := 1 / 2) (by norm_num)
  (by norm_num) (Λ := {0}) (by simp) (by norm_num) (by norm_num) isLojasiewiczAt_half_mul_norm_sq
  (x := 1) fun ε hε => ((tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2)
    (by norm_num) 1).eventually (ball_mem_nhds 0 hε)).frequently.mono fun _ hk =>
      ⟨0, Set.mem_singleton 0, by simpa using hk⟩

example := tendsto_of_isLojasiewiczAt (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1)
  ((contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero)
  (by rw [gradient_half_mul_norm_sq]; exact LipschitzWith.id) (η := 1 / 2) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) isLojasiewiczAt_half_mul_norm_sq
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num)
    1).mapClusterPt

/-- With `η = 3/2`, beyond `η L < 1` for every Lipschitz constant `L ≥ 1` of the
gradient, the orbit `(-1/2)ᵏ` of `1` lowers `‖x‖² / 2` by `η κ ‖∇f‖²` at every step,
`κ = 1 - η / 2 = 1/4`, and converges to `0`. -/
example := exists_tendsto_of_isLojasiewiczAt_of_le_sub
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (by norm_num)
  (by norm_num) (Λ := {0}) (by simp) (by norm_num) (by norm_num) isLojasiewiczAt_half_mul_norm_sq
  (by fun_prop) (x := 1)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  fun ε hε => ((tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 3 / 2)
    (by norm_num) 1).eventually (ball_mem_nhds 0 hε)).frequently.mono fun _ hk =>
      ⟨0, Set.mem_singleton 0, by simpa using hk⟩

example := tendsto_of_isLojasiewiczAt_of_le_sub (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (η := 3 / 2) (κ := 1 - 3 / 2 / 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  isLojasiewiczAt_half_mul_norm_sq (by fun_prop)
  (fun _ => by rw [Function.iterate_succ_apply']; exact (half_mul_norm_sq_gradientStep _ _).le)
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 3 / 2) (by norm_num)
    1).mapClusterPt

end Transformer.Landscape.SaddleAvoidance
