/-
# Growth freezes the directions

**Not a statement of any paper.**  The stream of `Perspective.RawStream`,
`ẋ_i = c x_i + g_i(t)` with `‖g_i‖ ≤ M`, for `c > 0`: `mix[0] > 1` in every
channel of a parameter-golf block.

* The stream outgrows its drive: after `t₀` the direction of token `i` moves
  by at most `2 M / (c ‖x_i(t₀)‖)` for all time, whatever the attention, the
  feed-forward term and the injections do (`rawStream_frozen`).  The
  directions keep the configuration they had at `t₀` up to that much, a
  consensus as well as a spread one.
* So `blockDrive_spread` does not survive the norm.  On the sphere the pure
  injection `ẋ_i = Proj_{x_i}(z_i)` with two independent differences spreads
  within every window of length `τ` (`blockDrive_spread` at `A = B = L_G = 0`).
  Before the norm, for every `c > 0`, any injections and any `δ > 0`, some
  stream keeps all its directions `δ`-close at all times `t ≥ 0`
  (`not_rawStream_spread`).

A `mix[0]` that differs from channel to channel is a different matter, and not
an open one: its linear part turns the directions by itself, so no such gain is
a change of variable (`not_channelGain_gauge` of `Perspective.RawStack`), while a
gain that is scalar — or one gate per token — is one at every depth
(`normalize_rawStack`).

Left open: `c ≤ 0` (`mix[0] ≤ 1`), where growth no longer outpaces the drive
but the tokens have different norms, so the part of the drive they share no
longer cancels in a difference.

The U-Net skip of the decoder half, `x ← x + skip_weights ⊙ skip`, which adds
an earlier state of the stream rather than a bounded drive, is no longer open
as a question of the gauge: it is gauge-covariant (`ungauged_rec_skip` of
`Perspective.RawStackSkip`), with its gate divided by the gain accumulated over
the depth it jumps.  What stays open is freezing through it, and for a reason
that is now proved rather than guessed: at a fixed gap the rescaled gates do
not shrink with depth (`le_abs_skipWeight` of `Perspective.RawStackSkipDamp`),
so they are not summable the way the block outputs are.
-/

import Transformer.Perspective.RawStream
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Growth freezes the direction.**  Let `c > 0` and `ẋ = c x + g(t)` on
`[t₀, t₁]` (right derivatives, so that `g` may jump from block to block), with
`‖g‖ ≤ M`.  Then for every `t` of the window the direction of `x` has moved by
at most `2 M / (c ‖x(t₀)‖)` — a bound that does not depend on `t₁`.  In the
frame `e^{-c(t - t₀)} x(t)` that grows with the stream, the drive
`e^{-c(t - t₀)} g` has total length at most `M / c`.

Source: none — posed here; the residual stream of a pre-norm block with
`mix[0] = 1 + c` in every channel, whose attention, feed-forward term and
injection make up `g`. -/
theorem rawStream_frozen {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {x g : ℝ → E}
    {c M t₀ t₁ : ℝ} (hc : 0 < c) (hM : 0 ≤ M) (hcont : ContinuousOn x (Set.Icc t₀ t₁))
    (hx : ∀ s ∈ Set.Ico t₀ t₁, HasDerivWithinAt x (c • x s + g s) (Set.Ici s) s)
    (hg : ∀ s ∈ Set.Ico t₀ t₁, ‖g s‖ ≤ M) (h0 : x t₀ ≠ 0) :
    ∀ t ∈ Set.Icc t₀ t₁, ‖‖x t‖⁻¹ • x t - ‖x t₀‖⁻¹ • x t₀‖ ≤ 2 * M / (c * ‖x t₀‖) := by
  intro t ht
  have hx0 : 0 < ‖x t₀‖ := norm_pos_iff.2 h0
  have hexp : ∀ s, HasDerivAt (fun r => Real.exp (-c * (r - t₀)))
      (Real.exp (-c * (s - t₀)) * (-c * 1)) s := fun s =>
    (((hasDerivAt_id' (x := s)).sub_const t₀).const_mul (-c)).exp
  have hB : ∀ s, HasDerivAt (fun r => M / c * (1 - Real.exp (-c * (r - t₀))))
      (M * Real.exp (-c * (s - t₀))) s := fun s =>
    (((hexp s).const_sub 1).const_mul (M / c)).congr_deriv (by field_simp [hc.ne'])
  have hf := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (f := fun s => Real.exp (-c * (s - t₀)) • x s - x t₀)
    (f' := fun s => Real.exp (-c * (s - t₀)) • g s)
    ((ContinuousOn.fun_smul (fun s _ => (hexp s).continuousAt.continuousWithinAt) hcont).sub
      continuousOn_const)
    (fun s hs => (((hexp s).hasDerivWithinAt.fun_smul (hx s hs)).sub_const (x t₀)).congr_deriv
      (by module))
    (by simp) hB
    (fun s hs => by
      rw [norm_smul, Real.norm_of_nonneg (Real.exp_pos _).le, mul_comm]
      exact mul_le_mul_of_nonneg_right (hg s hs) (Real.exp_pos _).le) ht
  have hft : ‖Real.exp (-c * (t - t₀)) • x t - x t₀‖ ≤ M / c := hf.trans (by
    have := Real.exp_pos (-c * (t - t₀))
    have : 0 ≤ M / c := div_nonneg hM hc.le
    nlinarith)
  have hxt : x t = Real.exp (c * (t - t₀)) • (x t₀ + (Real.exp (-c * (t - t₀)) • x t - x t₀)) := by
    rw [add_sub_cancel, smul_smul, ← Real.exp_add,
      show c * (t - t₀) + -c * (t - t₀) = 0 by ring, Real.exp_zero, one_smul]
  rw [hxt, normalize_smul_of_pos (Real.exp_pos _), norm_sub_rev]
  calc _ ≤ 2 * ‖x t₀ - (x t₀ + (Real.exp (-c * (t - t₀)) • x t - x t₀))‖ / ‖x t₀‖ :=
        norm_normalize_sub_le_div h0
    _ = 2 * ‖Real.exp (-c * (t - t₀)) • x t - x t₀‖ / ‖x t₀‖ := by
        rw [sub_add_cancel_left, norm_neg]
    _ ≤ 2 * (M / c) / ‖x t₀‖ := by gcongr
    _ = 2 * M / (c * ‖x t₀‖) := by field_simp

/-- The hypotheses of `rawStream_frozen` are satisfiable: `c = 1`, no drive
(`M = 0`) and `x(t) = e^t e₀` on `[0, 1]`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧
    ContinuousOn (fun t => Real.exp t • (basePoint 0 : EucSpace 1)) (Set.Icc 0 1) ∧
    (∀ s ∈ Set.Ico (0 : ℝ) 1, HasDerivWithinAt (fun t => Real.exp t • (basePoint 0 : EucSpace 1))
      ((1 : ℝ) • (Real.exp s • (basePoint 0 : EucSpace 1)) + 0) (Set.Ici s) s) ∧
    (∀ s ∈ Set.Ico (0 : ℝ) 1, ‖(fun _ => (0 : EucSpace 1)) s‖ ≤ 0) ∧
    Real.exp 0 • (basePoint 0 : EucSpace 1) ≠ 0 :=
  ⟨one_pos, le_rfl, (Real.continuous_exp.smul continuous_const).continuousOn,
    fun s _ => ((Real.hasDerivAt_exp s).smul_const _).hasDerivWithinAt.congr_deriv (by simp),
    fun _ _ => by simp, by simp [basePoint]⟩

/-- **Before the norm, growth defeats the injection.**  The claim refuted:
`blockDrive_spread` for the stream behind the RMS norm — that for `c > 0`
(`mix[0] > 1`) and injections `z_i` there is `δ > 0` such that along every
nonvanishing stream `ẋ_i = c x_i + z_i` two directions are more than `δ` apart
at some time `t ≥ 0`.  It fails for every `c > 0` and every `z`, independent
differences included, although this is a block with no attention and no
feed-forward term, within the bounds `IsBoundedBlockOn` for any
`A, N, B, L_G ≥ 0` and `Z ≥ ‖z_i‖`: start all tokens at `R e₀` with `R` large;
by `rawStream_frozen` every direction stays within `2 M / (c R)` of `e₀`,
`M = Σ_k ‖z_k‖`.  On the sphere, once two differences `z_k - z_l`, `z_m - z_p`
are independent, the same drive spreads the tokens within every window of
length `τ` (`blockDrive_spread`).

The initial state is free, as in `blockDrive_spread`.  In parameter-golf the
stream starts at `x0`, of RMS norm `1`, and `z = mix[1] ⊙ x0` is tied to it;
that start the refutation does not reach.  What it shows is that no bound on
the drive forces the directions apart once `‖x_i‖` is large against `M / c`.

Source: none — posed here; it refutes the extension of `blockDrive_spread` to
the pre-norm stream of parameter-golf with `mix[0] > 1`. -/
theorem not_rawStream_spread {c : ℝ} (hc : 0 < c) (z : Idx n → EucSpace (d + 1)) :
    ¬ ∃ δ : ℝ, 0 < δ ∧ ∀ x : ℝ → Idx n → EucSpace (d + 1),
      (∀ t i, HasDerivAt (fun s => x s i) (c • x t i + z i) t) →
      (∀ t, 0 ≤ t → ∀ i, x t i ≠ 0) →
      ∃ t, 0 ≤ t ∧ ∃ i j : Idx n, δ < ‖‖x t i‖⁻¹ • x t i - ‖x t j‖⁻¹ • x t j‖ := by
  rintro ⟨δ, hδ, h⟩
  have he : ‖(basePoint d : EucSpace (d + 1))‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint d).2
  obtain ⟨M, hM_def⟩ : ∃ M : ℝ, M = ∑ k, ‖z k‖ := ⟨_, rfl⟩
  have hM : 0 ≤ M := hM_def ▸ Finset.sum_nonneg fun k _ => norm_nonneg _
  have hzM : ∀ i, ‖z i‖ ≤ M := fun i => hM_def ▸
    Finset.single_le_sum (fun k _ => norm_nonneg (z k)) (Finset.mem_univ i)
  have hε : 0 < min δ 1 := lt_min hδ one_pos
  obtain ⟨R, hR_def⟩ : ∃ R : ℝ, R = 4 * M / (c * min δ 1) + 1 := ⟨_, rfl⟩
  have hR : 0 < R := by rw [hR_def]; positivity
  have hkey : 4 * M / (c * R) < min δ 1 := by
    rw [div_lt_iff₀ (by positivity)]
    have : min δ 1 * (c * R) = 4 * M + min δ 1 * c := by
      rw [hR_def]; field_simp
    rw [this]
    nlinarith [mul_pos hε hc]
  have h24 : 2 * M / (c * R) ≤ 4 * M / (c * R) := by gcongr; norm_num
  obtain ⟨x, hx⟩ : ∃ x : ℝ → Idx n → EucSpace (d + 1), ∀ t i,
      x t i = Real.exp (c * t) • (R • (basePoint d : EucSpace (d + 1)) + c⁻¹ • z i) - c⁻¹ • z i :=
    ⟨_, fun _ _ => rfl⟩
  have hderiv : ∀ t i, HasDerivAt (fun s => x s i) (c • x t i + z i) t := fun t i => by
    rw [show (fun s => x s i) = fun s => Real.exp (c * s) •
      (R • (basePoint d : EucSpace (d + 1)) + c⁻¹ • z i) - c⁻¹ • z i from funext (hx · i), hx]
    refine ((((hasDerivAt_id' (x := t)).const_mul c).exp.smul_const _).sub_const
      (c⁻¹ • z i)).congr_deriv ?_
    rw [smul_sub, smul_smul c c⁻¹, mul_inv_cancel₀ hc.ne', one_smul, sub_add_cancel, smul_smul,
      mul_one, mul_comm (Real.exp (c * t)) c]
  have hnR : ‖R • (basePoint d : EucSpace (d + 1))‖ = R := by
    rw [norm_smul, Real.norm_of_nonneg hR.le, he, mul_one]
  have hx0 : ∀ i, x 0 i = R • (basePoint d : EucSpace (d + 1)) := fun i => by
    rw [hx, mul_zero, Real.exp_zero, one_smul, add_sub_cancel_right]
  have hfrozen : ∀ t, 0 ≤ t → ∀ i,
      ‖‖x t i‖⁻¹ • x t i - (basePoint d : EucSpace (d + 1))‖ ≤ 2 * M / (c * R) := fun t ht i => by
    have h' := rawStream_frozen (x := fun s => x s i) (g := fun _ => z i) hc hM
      (fun s _ => (hderiv s i).continuousAt.continuousWithinAt)
      (fun s _ => (hderiv s i).hasDerivWithinAt) (fun _ _ => hzM i)
      (by rw [hx0, ← norm_ne_zero_iff, hnR]; exact hR.ne') t ⟨ht, le_rfl⟩
    rwa [hx0, hnR, smul_smul, inv_mul_cancel₀ hR.ne', one_smul] at h'
  obtain ⟨t, ht, i, j, hij⟩ := h x hderiv (fun t ht i h0 => by
    have := hfrozen t ht i
    rw [h0, norm_zero, inv_zero, zero_smul, zero_sub, norm_neg, he] at this
    linarith [min_le_right δ 1])
  have htri := norm_sub_le_norm_sub_add_norm_sub (‖x t i‖⁻¹ • x t i)
    (basePoint d : EucSpace (d + 1)) (‖x t j‖⁻¹ • x t j)
  rw [norm_sub_rev (basePoint d : EucSpace (d + 1))] at htri
  have h4 : 2 * M / (c * R) + 2 * M / (c * R) = 4 * M / (c * R) := by ring
  linarith [hfrozen t ht i, hfrozen t ht j, min_le_left δ 1]

/-- The hypothesis of `not_rawStream_spread` is satisfiable: `c = 1`. -/
example : (0 : ℝ) < 1 := one_pos

end Perspective
end Transformer
