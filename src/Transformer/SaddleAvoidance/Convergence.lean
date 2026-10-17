/-
# Saddle avoidance — gradient descent converges near a Łojasiewicz point

The values of a decreasing sequence that comes close to a value `c`
infinitely often stay above `c` and converge to it
(`le_and_tendsto_of_frequently`).  Along an orbit of gradient descent with a
limit point `z` up to translations leaving `f` invariant
(`frequently_abs_sub_lt`), the values of `f` therefore decrease to `f z`.
Such a point exists when every point of the orbit can be translated into one
compact set (`exists_frequently_norm_add_sub_lt`), as for a function on a
torus.

If `f` satisfies the Łojasiewicz inequality at `z` with `0 ≤ a < 1`, the
orbit comes, up to a translation, close enough to `z` with `f` close enough
to `f z` for the trap of `iterate_mem_ball`.  From then on it has finite
length, so it converges (`exists_tendsto_of_isLojasiewiczAt`), to `z` itself
when `z` is a limit point of the orbit (`tendsto_of_isLojasiewiczAt`).  This
is Theorem 3.2 of Absil, Mahony and Andrews (SIAM J. Optim. 16, 2005), up to
translations.
-/

import Transformer.SaddleAvoidance.Trap
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity
import Mathlib.Topology.Sequences

open Filter Metric
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

/-- **The values of a decreasing sequence near a value it comes close to
infinitely often.**  They stay above it and converge to it.  The index may be
discrete, as for gradient descent, or continuous, as for gradient flows. -/
theorem le_and_tendsto_of_frequently {ι : Type*} [SemilatticeSup ι] [Nonempty ι] {u : ι → ℝ}
    (hu : Antitone u) {c : ℝ}
    (hc : ∀ ε > 0, ∃ᶠ k in atTop, |u k - c| < ε) :
    (∀ k, c ≤ u k) ∧ Tendsto u atTop (𝓝 c) := by
  have hle : ∀ k, c ≤ u k := by
    intro m
    by_contra hm
    push Not at hm
    obtain ⟨k, hk, hkm⟩ := ((hc (c - u m) (by linarith)).and_eventually
      (eventually_ge_atTop m)).exists
    have := hu hkm
    rw [abs_lt] at hk
    linarith [hk.1]
  refine ⟨hle, Metric.tendsto_atTop.2 fun ε hε => ?_⟩
  obtain ⟨N, hN⟩ := (hc ε hε).exists
  refine ⟨N, fun n hn => ?_⟩
  rw [Real.dist_eq, abs_lt] at *
  constructor <;> linarith [hle n, hu hn]

section Translate

variable {E : Type*} [NormedAddCommGroup E]

/-- If infinitely many points of a sequence lie near `z` up to translations
in `Λ`, and `f` is continuous at `z` and invariant under `Λ`, then infinitely
many values of `f` along the sequence lie near `f z`. -/
theorem frequently_abs_sub_lt {f : E → ℝ} {z : E} (hcont : ContinuousAt f z) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {ι : Type*} [Preorder ι] {u : ι → E}
    (hcl : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖u k + v - z‖ < ε) :
    ∀ ε > 0, ∃ᶠ k in atTop, |f (u k) - f z| < ε := by
  intro ε hε
  obtain ⟨δ, hδ, hball⟩ := Metric.continuousAt_iff.1 hcont ε hε
  refine (hcl δ hδ).mono fun k ⟨v, hv, hk⟩ => ?_
  have h := hball (x := u k + v) (by rwa [dist_eq_norm])
  rwa [hΛ v hv, Real.dist_eq] at h

/-- **A limit point up to translations.**  If every term of a sequence can
be translated by a vector of `Λ` into a compact set `K`, some point of `K` is
approached by infinitely many terms up to such translations. -/
theorem exists_frequently_norm_add_sub_lt {K : Set E} (hK : IsCompact K) {Λ : Set E}
    {u : ℕ → E} (hu : ∀ k, ∃ v ∈ Λ, u k + v ∈ K) :
    ∃ z ∈ K, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖u k + v - z‖ < ε := by
  choose v hv hvK using hu
  obtain ⟨z, hzK, φ, hφ, hlim⟩ := hK.tendsto_subseq hvK
  refine ⟨z, hzK, fun ε hε => frequently_atTop.2 fun a => ?_⟩
  obtain ⟨j, hja, hj⟩ := ((hφ.tendsto_atTop.eventually (eventually_ge_atTop a)).and
    (Metric.tendsto_nhds.1 hlim ε hε)).exists
  exact ⟨φ j, hja, v (φ j), hv _, by rwa [← dist_eq_norm]⟩

end Translate

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {η a : ℝ}

/-- **Gradient descent converges near a point where the Łojasiewicz
inequality holds.**  Let `f` have an `L`-Lipschitz gradient, `0 < η`,
`η L < 1`, and let `f` satisfy the Łojasiewicz inequality at `z` with
`0 ≤ a < 1`.  If infinitely many points of the orbit of `x` lie near `z` up
to translations in a set `Λ` leaving `f` invariant, the orbit converges. -/
theorem exists_tendsto_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) (ha0 : 0 ≤ a) (ha1 : a < 1) {z : E}
    (hz : IsLojasiewiczAt f z a) {x : E}
    (hcl : ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨m, hm, ε, hε, hev⟩ := hz
  obtain ⟨r, hr, hloj⟩ := Metric.eventually_nhds_iff_ball.1 hev
  have hκ : 0 < m * (1 - a) * (1 - η * L) := mul_pos (mul_pos hm (by linarith)) (by linarith)
  obtain ⟨hfz, hlim⟩ := le_and_tendsto_of_frequently (antitone_iterate hf hL hη.le hηL.le x)
    (frequently_abs_sub_lt hf.continuous.continuousAt hΛ hcl)
  have hφ : Tendsto (fun k => (f ((gradientStep f η)^[k] x) - f z) ^ (1 - a)) atTop (𝓝 0) := by
    have h := (hlim.sub_const (f z)).rpow_const (Or.inr (sub_nonneg.2 ha1.le))
    rwa [sub_self, Real.zero_rpow (sub_pos.2 ha1).ne'] at h
  obtain ⟨k0, ⟨v, hv, hk0⟩, hφk0, hfk0⟩ := ((hcl (r / 2) (half_pos hr)).and_eventually
    ((hφ.eventually (gt_mem_nhds (half_pos (mul_pos hκ hr)))).and
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
  have hε' : f ((gradientStep f η)^[k0] x + v) < f z + ε := by rwa [hΛ v hv]
  have hstart : m * (1 - a) * (1 - η * L) * ‖(gradientStep f η)^[k0] x + v - z‖ +
      (f ((gradientStep f η)^[k0] x + v) - f z) ^ (1 - a) < m * (1 - a) * (1 - η * L) * r := by
    rw [hΛ v hv]
    have := mul_lt_mul_of_pos_left hk0 hκ
    linarith
  have hcauchy : CauchySeq fun n => (gradientStep f η)^[n + k0] x := by
    refine cauchySeq_of_le_tendsto_0' (fun n =>
      (f ((gradientStep f η)^[n + k0] x) - f z) ^ (1 - a) / (m * (1 - a) * (1 - η * L)))
      (fun n k hnk => ?_) ?_
    · have h := mul_norm_iterate_sub_le hf hL hη hηL ha0 ha1 hm hloj hfz' hε' hstart hnk
      rw [hfo, hfo, horb, horb, add_sub_add_right_eq_sub] at h
      rw [dist_comm, dist_eq_norm, le_div_iff₀ hκ]
      have := Real.rpow_nonneg (sub_nonneg.2 (hfz (k + k0))) (1 - a)
      linarith
    · have h := (hφ.comp (tendsto_add_atTop_nat k0)).div_const (m * (1 - a) * (1 - η * L))
      rwa [zero_div] at h
  obtain ⟨w, hw⟩ := cauchySeq_tendsto_of_complete hcauchy
  exact ⟨w, (tendsto_add_atTop_iff_nat k0).1 hw⟩

/-- **Gradient descent converges to a limit point where the Łojasiewicz
inequality holds**, under the hypotheses of
`exists_tendsto_of_isLojasiewiczAt`. -/
theorem tendsto_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 ≤ a)
    (ha1 : a < 1) {z : E} (hz : IsLojasiewiczAt f z a) {x : E}
    (hcl : MapClusterPt z atTop fun k => (gradientStep f η)^[k] x) :
    Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z) := by
  have hfreq : ∀ ε > 0, ∃ᶠ k in atTop, (gradientStep f η)^[k] x ∈ ball z ε := fun ε hε =>
    mapClusterPt_iff_frequently.1 hcl _ (ball_mem_nhds z hε)
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt hf hL hη hηL (Λ := {0}) (by simp) ha0 ha1
    hz fun ε hε => (hfreq ε hε).mono fun k hk =>
      ⟨0, Set.mem_singleton 0, by rw [add_zero, ← dist_eq_norm]; exact mem_ball.1 hk⟩
  have hwz : w = z := eq_of_forall_dist_le fun ε hε => by
    obtain ⟨k, hk, hkw⟩ := ((hfreq (ε / 2) (half_pos hε)).and_eventually
      (Metric.tendsto_nhds.1 hw _ (half_pos hε))).exists
    rw [mem_ball] at hk
    linarith [dist_triangle_left w z ((gradientStep f η)^[k] x)]
  rwa [hwz] at hw

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line with `η = 1/2`, `L = 1`, `a = 1/2`, `Λ = {0}`, the critical point `0`,
the orbit `2⁻ᵏ` of `1`, which converges to `0`, and the compact set
`[-1, 1]`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    Antitone (fun k => f ((gradientStep f (1 / 2))^[k] 1)) ∧
      (∀ ε > 0, ∃ᶠ k in atTop, |f ((gradientStep f (1 / 2))^[k] 1) - f 0| < ε) ∧
      ContinuousAt f 0 ∧ (∀ v ∈ ({0} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ ({0} : Set ℝ),
        ‖(gradientStep f (1 / 2))^[k] 1 + v - 0‖ < ε) ∧
      IsCompact (closedBall (0 : ℝ) 1) ∧
      (∀ k, ∃ v ∈ ({0} : Set ℝ), (gradientStep f (1 / 2))^[k] 1 + v ∈ closedBall (0 : ℝ) 1) ∧
      Differentiable ℝ f ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
      IsLojasiewiczAt f 0 (1 / 2) ∧
      MapClusterPt 0 atTop (fun k => (gradientStep f (1 / 2))^[k] (1 : ℝ)) := by
  intro f
  have hf : Differentiable ℝ f :=
    (contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero
  have hL : LipschitzWith 1 (gradient f) := by
    rw [show gradient f = id from gradient_half_mul_norm_sq]
    exact LipschitzWith.id
  have horb : ∀ k, (gradientStep f (1 / 2))^[k] (1 : ℝ) = (1 / 2) ^ k := fun k => by
    simp only [f, iterate_gradientStep_half_mul_norm_sq]
    norm_num
  have hlim : Tendsto (fun k => (gradientStep f (1 / 2))^[k] (1 : ℝ)) atTop (𝓝 0) := by
    simp only [horb]
    exact tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
  refine ⟨antitone_iterate hf hL (by norm_num) (by norm_num) 1, fun ε hε => ?_,
    hf.continuous.continuousAt, by simp, fun ε hε => ?_, isCompact_closedBall 0 1,
    fun k => ⟨0, Set.mem_singleton 0, ?_⟩, hf, hL, by norm_num, by norm_num, by norm_num,
    by norm_num, isLojasiewiczAt_half_mul_norm_sq, hlim.mapClusterPt⟩
  · exact (Metric.tendsto_nhds.1 ((hf.continuous.tendsto 0).comp hlim) ε hε).frequently.mono
      fun k hk => by rwa [Real.dist_eq] at hk
  · exact (Metric.tendsto_nhds.1 hlim ε hε).frequently.mono fun k hk =>
      ⟨0, Set.mem_singleton 0, by rwa [add_zero, ← dist_eq_norm]⟩
  · rw [add_zero, horb, mem_closedBall, dist_zero_right, norm_pow,
      Real.norm_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)]
    exact pow_le_one₀ (by norm_num) (by norm_num)

end Transformer.SaddleAvoidance
