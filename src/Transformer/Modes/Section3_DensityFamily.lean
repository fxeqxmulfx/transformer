import Transformer.Modes.Section3_DensityInversion
import Transformer.Modes.Section3_ErrorKR

/-!
# Actual continuous density families on the smaller window

The two remaining Kac–Rice error claims on `T'` in §3.2 of
arXiv:2412.09080v3 ask for continuous families of actual normalized-sum
densities as well as quantitative integral bounds. The density condition
is proved here in every source regime, allowing `β` to vary with `n`.

The source defines `T'` to be empty for `β > n^(2/3)`. Otherwise its
bandwidth cap and `n → ∞` imply `n > 4(β + 1)` eventually. The proved
characteristic-function integrability and density inversion therefore
apply at every translation in the entire smaller window. This uses the
corrected bandwidth-dependent threshold, not the false fixed-five claim.

The threshold uses the elementary ratio `n^(2/3)/n = n^(-1/3)`.
Its limit is zero, while the additive constant four is eventually smaller
than a fixed fraction of `n`. This proves one sample-size threshold
before considering translations, as required by the family quantifiers.

Each density in the resulting family is continuous and bounded in its
spatial argument. The bounds here may depend on the translation; no
uniform Edgeworth remainder or Kac–Rice integral estimate is asserted.
Those quantitative conclusions remain the two open statements in
`Section3_ErrorKR.lean`. In particular the `β = n` counterexample on `T`
does not refute these existence statements on the smaller window.

The finite-bandwidth examples use a density family on all translations,
with seventeen summands and bandwidth three, so they do not depend on
an empty-window case. The eventual claims also have the source's
already verified regime as a concrete witness to all their hypotheses.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped Topology ENNReal

namespace Transformer.Modes

/-- The `T'` bandwidth cap eventually implies the proved inversion
threshold: `4(n^(2/3) + 1) < n`. The sample size alone must tend to infinity.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5. -/
theorem eventually_density_threshold {N : ℕ → ℕ}
    (hN : Tendsto (fun k => (N k : ℝ)) atTop atTop) :
    ∀ᶠ k in atTop, 4 * ((N k : ℝ) ^ ((2 : ℝ) / 3) + 1) < (N k : ℝ) := by
  have hp : Tendsto (fun k => (N k : ℝ) ^ (-((1 : ℝ) / 3))) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop (by norm_num : (0 : ℝ) < 1 / 3)).comp hN
  filter_upwards [hp.eventually_lt_const (by norm_num : (0 : ℝ) < 1 / 8),
    hN.eventually_gt_atTop 8] with k hk hNk
  have hnpos : 0 < (N k : ℝ) := by linarith
  have he : (N k : ℝ) ^ (-((1 : ℝ) / 3)) =
      (N k : ℝ) ^ ((2 : ℝ) / 3) / N k := by
    rw [show -((1 : ℝ) / 3) = (2 : ℝ) / 3 - 1 by ring,
      Real.rpow_sub hnpos, Real.rpow_one]
  rw [he, div_lt_iff₀ hnpos] at hk
  linarith

example := eventually_density_threshold isRegime_succ.tendsto_N

/-- Membership in the paper's `T'` forces `β ≤ n^(2/3)`.
The larger-bandwidth branch is explicitly empty in the source.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5. -/
theorem mem_intervalT'_bandwidth_le {n : ℕ} {β t : ℝ} (ht : t ∈ intervalT' n β) :
    β ≤ (n : ℝ) ^ ((2 : ℝ) / 3) := by
  by_contra h
  simp [intervalT', h] at ht

example : (0 : ℝ) ∈ intervalT' 1 1 := by simp [intervalT']

/-- The actual normalized sums have a continuous density family
when `n > 4(β + 1)`, on any set of translations. Each density is bounded;
its bound may depend on the translation. This uses the corrected
bandwidth-dependent threshold instead of the false fixed-five claim.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5. -/
theorem exists_bounded_densityFamily_of_bandwidth {β : ℝ} (hβ : 0 < β) {n : ℕ}
    (hn : 4 * (β + 1) < (n : ℝ)) (S : Set ℝ) :
    ∃ q, IsDensityFamily n β S q ∧ ∀ t ∈ S, ∃ C : ℝ, ∀ z, |q t z| ≤ C := by
  have h : ∀ t : ℝ, ∃ q,
      Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ∧
        ∃ C : ℝ, ∀ z, |q z| ≤ C := fun t => exists_continuous_density_scaledSum_lawY hβ t hn
  choose q hc hd hb using h
  exact ⟨q, ⟨fun t _ => hc t, fun t _ => hd t⟩, fun t _ => hb t⟩

example := exists_bounded_densityFamily_of_bandwidth (n := 17)
  (by norm_num : (0 : ℝ) < 3) (by norm_num) Set.univ

/-- The same inversion construction supplies the density-family
condition used by the Kac–Rice error statements.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5. -/
theorem exists_densityFamily_of_bandwidth {β : ℝ} (hβ : 0 < β) {n : ℕ}
    (hn : 4 * (β + 1) < (n : ℝ)) (S : Set ℝ) :
    ∃ q, IsDensityFamily n β S q := by
  obtain ⟨q, hq, _⟩ := exists_bounded_densityFamily_of_bandwidth hβ hn S
  exact ⟨q, hq⟩

example := exists_densityFamily_of_bandwidth (n := 17)
  (by norm_num : (0 : ℝ) < 3) (by norm_num) Set.univ

/-- In every source regime, the actual normalized-sum characteristic
function is eventually integrable at every translation in `T'`.
The bandwidth varies with the sample size; the explicit cap provides the
required threshold uniformly over the translations in the window.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5. -/
theorem eventually_integrable_characteristic_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) :
    ∀ᶠ k in atTop, ∀ t ∈ intervalT' (N k) (B k),
      Integrable (characteristic2 ((Measure.pi fun _ : Fin (N k) => lawY (B k) t).map
        (scaledSum (N k)))) := by
  filter_upwards [eventually_density_threshold hreg.tendsto_N] with k hk t ht
  have hB := mem_intervalT'_bandwidth_le ht
  exact integrable_characteristic_scaledSum_lawY (hreg.B_pos k) t (by linarith)

example := eventually_integrable_characteristic_T' isRegime_succ

/-- The paper's regime eventually has continuous density families
on `T'`, with each density bounded. For a nonempty window, its bandwidth
cap implies the actual inversion threshold. The empty branch uses exactly
the source's definition of `T'`. No Kac–Rice error estimate is a hypothesis.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5. -/
theorem eventually_exists_bounded_densityFamily_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) :
    ∀ᶠ k in atTop, ∃ q,
      IsDensityFamily (N k) (B k) (intervalT' (N k) (B k)) q ∧
        ∀ t ∈ intervalT' (N k) (B k), ∃ C : ℝ, ∀ z, |q t z| ≤ C := by
  filter_upwards [eventually_density_threshold hreg.tendsto_N] with k hk
  by_cases hB : B k ≤ (N k : ℝ) ^ ((2 : ℝ) / 3)
  · exact exists_bounded_densityFamily_of_bandwidth (hreg.B_pos k) (by linarith) _
  · have he : intervalT' (N k) (B k) = ∅ := by simp [intervalT', hB]
    refine ⟨fun _ _ => 0, ?_, ?_⟩
    · rw [he]
      constructor <;> simp
    · rw [he]
      simp

example := eventually_exists_bounded_densityFamily_T' isRegime_succ

/-- The density-family existence part of the two remaining
`cor:error-higher` claims on `T'` follows from the proved Fourier estimates.
Their quantitative error bounds remain separate claims.
Source: arXiv:2412.09080v3, §3.2 `cor:error-higher`, `eq:T'`, and §5.5. -/
theorem eventually_exists_densityFamily_T' {c : ℝ} {N : ℕ → ℕ} {B : ℕ → ℝ}
    (hreg : IsRegime c N B) :
    ∀ᶠ k in atTop, ∃ q, IsDensityFamily (N k) (B k) (intervalT' (N k) (B k)) q := by
  exact (eventually_exists_bounded_densityFamily_T' hreg).mono fun k ⟨q, hq, _⟩ => ⟨q, hq⟩

example := eventually_exists_densityFamily_T' isRegime_succ

end Transformer.Modes
