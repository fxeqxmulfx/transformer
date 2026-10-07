import Transformer.Modes.Section3_ChangeOfVar
import Transformer.Modes.Section5_PtBddDensity

/-
# The number of modes of a Gaussian KDE — density obstructions after whitening

The proved affine change of variables `eq:qt` transfers the counterexample
to `lem: pt.bdd` to the standardized sum used throughout §3. Neither a
continuous density nor a bounded density exists when `β > 2`, `0 < n < β+2`.

The regime `β = n = k+1` satisfies the source's growth assumptions, with
`c = 1`. At every sufficiently large index the obstruction applies at every
observation point, including `0 ∈ T`. The two eventual refutations below
keep the source's quantifier order and apply to every window function `ω`.

Finally, a uniform weighted approximation to a bounded function would itself
give a bounded density. This implication does not require continuity and
will refute the pointwise bounds in `lem:error-higher` even if the continuity
assertion inherited from `lem: pt.bdd` is removed.

Source: arXiv:2412.09080v3, §2.2, `eq:qt`; §3.2, `lem:error-higher`;
§4.1 and §5.5, `lem: pt.bdd`.
-/

open Real MeasureTheory Filter
open scoped ENNReal

namespace Transformer.Modes

/-- The standardized sum has no continuous density in the counterexample
range of `lem: pt.bdd`; the affine identity `eq:qt` preserves continuity.
Source: arXiv:2412.09080v3, §2.2, `eq:qt`; §4.1, `lem: pt.bdd`. -/
theorem not_continuous_density_scaledSum {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
      IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q := by
  rintro ⟨q, hq, hqP⟩
  have hβ0 : 0 < β := by linarith
  have hp := (isDensityOf_scaledSum_iff hβ0 t (by omega) q).mp hqP
  have hcont : Continuous fun z : ℝ × ℝ => sigmaDet β t ^ (-(1 : ℝ) / 2) *
      q (whiten (sigmaFst β t) (sigmaCov β t) (sigmaSnd β t)
        (z.1 - muFst n β t, z.2 - muSnd n β t)) := by
    apply continuous_const.mul
    apply hq.comp
    unfold whiten
    fun_prop
  apply not_continuous_density_sumGG' hn hβ hnβ t
  exact ⟨_, hcont.continuousOn, hp.map_eq⟩

/-- The continuity-transfer hypotheses hold in the regime `β = n`. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 5 ∧ (5 : ℝ) < 5 + 2 := by norm_num

/-- The standardized sum has no bounded density: `eq:qt` multiplies a
putative bound by the positive constant `(det Σ_t)^{-1/2}`.
Source: arXiv:2412.09080v3, §2.2, `eq:qt`; §5.5, `lem: pt.bdd`. -/
theorem not_bounded_density_scaledSum {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ,
      IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ∧
      ∃ B : ℝ, ∀ z, q z ≤ B := by
  rintro ⟨q, hqP, B, hq⟩
  have hβ0 : 0 < β := by linarith
  have hp := (isDensityOf_scaledSum_iff hβ0 t (by omega) q).mp hqP
  have hc : 0 < sigmaDet β t ^ (-(1 : ℝ) / 2) :=
    Real.rpow_pos_of_pos (sigmaDet_pos hβ0 t) _
  apply not_bounded_density_sumGG' hn hβ hnβ t
  exact ⟨_, sigmaDet β t ^ (-(1 : ℝ) / 2) * B, hp.map_eq,
    fun z => mul_le_mul_of_nonneg_left (hq _) hc.le⟩

/-- Boundedness transfer has simultaneous numerical witnesses. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 5 ∧ (5 : ℝ) < 5 + 2 := by norm_num

/-- The standardized continuous-density claim fails for every `n ≥ 5`
in the regime `β = n`, at every observation point.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`; §4.1, `lem: pt.bdd`. -/
theorem not_continuous_density_scaledSum_beta_eq_n {n : ℕ} (hn : 5 ≤ n) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
      IsDensityOf (Measure.pi fun _ : Fin n => lawY n t) (scaledSum n) q := by
  apply not_continuous_density_scaledSum (by omega)
  · exact_mod_cast (show 2 < n by omega)
  · norm_num

/-- Five samples meet the regime counterexample's sample-count assumption. -/
example : 5 ≤ (5 : ℕ) := le_rfl

/-- Every density in the same allowed regime is necessarily unbounded.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`; §5.5, `lem: pt.bdd`. -/
theorem not_bounded_density_scaledSum_beta_eq_n {n : ℕ} (hn : 5 ≤ n) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ,
      IsDensityOf (Measure.pi fun _ : Fin n => lawY n t) (scaledSum n) q ∧
      ∃ B : ℝ, ∀ z, q z ≤ B := by
  apply not_bounded_density_scaledSum (by omega)
  · exact_mod_cast (show 2 < n by omega)
  · norm_num

/-- The bounded-density regime hypothesis is satisfied at a finite count. -/
example : 5 ≤ (5 : ℕ) := le_rfl

/-- A weighted uniform approximation to a bounded target would make `q`
bounded. This contradicts the density obstruction independently of continuity.
Source: arXiv:2412.09080v3, §3.2, `eq:error-higher`, weights `1+‖x‖^s`. -/
theorem not_weighted_density_approximation {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ)
    {f w : ℝ × ℝ → ℝ} {M : ℝ} (hf : ∀ z, |f z| ≤ M) (hw : ∀ z, 1 ≤ w z) :
    ¬ ∃ q : ℝ × ℝ → ℝ,
      IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ∧
      ∃ E : ℝ, ∀ z, w z * |q z - f z| ≤ E := by
  rintro ⟨q, hqP, E, hE⟩
  apply not_bounded_density_scaledSum hn hβ hnβ t
  refine ⟨q, hqP, M + E, fun z => ?_⟩
  have herr : |q z - f z| ≤ E := by
    calc |q z - f z| = 1 * |q z - f z| := (one_mul _).symm
      _ ≤ w z * |q z - f z| := mul_le_mul_of_nonneg_right (hw z) (abs_nonneg _)
      _ ≤ E := hE z
  have hdiff := le_abs_self (q z - f z)
  have htarget := (le_abs_self (f z)).trans (hf z)
  linarith

/-- The approximation hypotheses hold simultaneously for a zero target
and unit weight, along with the source's numerical counterexample. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 5 ∧ (5 : ℝ) < 5 + 2 ∧
    (∀ z : ℝ × ℝ, |(fun _ => (0 : ℝ)) z| ≤ 0) ∧
    (∀ z : ℝ × ℝ, 1 ≤ (fun _ => (1 : ℝ)) z) := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_, fun _ => le_rfl⟩
  intro z
  norm_num

/-- Along `β = n = k+1`, continuous densities are eventually impossible
at every `t`; this is an allowed asymptotic regime in §3.2.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`; §4.1, `lem: pt.bdd`. -/
theorem eventually_not_continuous_density_succ :
    ∀ᶠ k : ℕ in atTop, ∀ t : ℝ, ¬ ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
      IsDensityOf (Measure.pi fun _ : Fin (k + 1) => lawY (k + 1) t)
        (scaledSum (k + 1)) q := by
  filter_upwards [eventually_ge_atTop (4 : ℕ)] with k hk
  intro t
  apply not_continuous_density_scaledSum (by omega)
  · exact_mod_cast (show 2 < k + 1 by omega)
  · norm_num

/-- The window `T` always contains the obstructed observation point `0`.
Thus eventually existing continuous densities on all of `T` are impossible.
Source: arXiv:2412.09080v3, §3.2, `lem:error-higher`, `cor:error-higher`. -/
theorem not_eventually_continuous_density_on_T_succ (ω : ℝ → ℝ) :
    ¬ ∀ᶠ k : ℕ in atTop, ∀ t ∈ intervalT (k + 1) (k + 1) (ω (k + 1)),
      ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
        IsDensityOf (Measure.pi fun _ : Fin (k + 1) => lawY (k + 1) t)
          (scaledSum (k + 1)) q := by
  intro h
  obtain ⟨k, hk, hnot⟩ := (h.and eventually_not_continuous_density_succ).exists
  exact hnot 0 (hk 0 (zero_mem_intervalT _ _ _))

/-- The eventual counterexample satisfies all original growth assumptions;
the obstruction is therefore present in the source's quantified range. -/
example : IsRegime 1 (fun k => k + 1) (fun k => ((k + 1 : ℕ) : ℝ)) ∧
    IsSlowGrowth (fun β => Real.sqrt (Real.log (Real.log β))) :=
  ⟨isRegime_succ, isSlowGrowth_sqrt_log_log⟩

end Transformer.Modes
