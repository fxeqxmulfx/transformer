module

public import Transformer.Causal.Section5_ClusteredStability
public import Transformer.Causal.Section5_OneCenterConvergence

/-!
# Clustering to frozen tokens

ArXiv:2411.04990v2, §5, `thm:fixed_centers`, with the documented
source corrections below. Almost every original initialization
converges under every original global solution to an asymptotically
stable critical point clustered near the frozen centers.

The proof treats an empty moving tuple directly. For one frozen
center, the actual limiting interaction gives simple critical
angles and exact circular consensus almost everywhere. For at
least two centers and a nonempty moving tuple, the original
total mass is at least three.
The original parameter inequalities then give a smaller radius
preserving both inequalities and placing the neighborhoods at
most half the interaction peak.

In that radius, the actual scalar force and curvature classify
critical angles. Periodic barriers and signed crossings give
convergence. The Borel derivative detector, countable repelling
fibers, and original-product Fubini exclude repelling limits
almost everywhere. Prefix stability and inward scalar barriers
prove asymptotic stability in the original maximum norm.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Causal

variable (n m : ℕ)

/-- **Theorem (thm:fixed_centers).** *Clustering to frozen tokens for
`K = Q = V = I_2`.*

Let `θ_1, …, θ_m` be `c β^{-1/2}`-separated frozen tokens with weights
`a_j ≥ 1`, `N = n + ∑ a_j`, and let `N, β, ε > 0`, `c > 2 + 2ε` satisfy

  `N h((c - 1 - 2ε) β^{-1/2}) < h(ε β^{-1/2})`,
  `-N g((c - 2ε) β^{-1/2}) < g(ε β^{-1/2})`.

If `φ(0) ∼ μ₀` for an absolutely continuous probability measure `μ₀`, then with
probability one `φ(t)` converges to an asymptotically stable critical point
`φ*` with every `φ*_k` within `ε β^{-1/2}` of some `θ_j`.

**What the source says and what is changed here.**

* *The sign in the condition on `g`.*  The theorem prints
  `N g((c - 2ε) β^{-1/2}) < g(ε β^{-1/2})`. Its left side is negative in the
  regime the theorem is meant for (`g` is negative between the peak
  `τ_β^* < β^{-1/2}` of `h` and `π`), where the printed condition holds for
  free, while the proof ends on
  `g(ε β^{-1/2}) < -N g((c - 2ε) β^{-1/2})` "contradicts the parameters
  choice" — a contradiction only with the minus sign. `lem:interaction`,
  which supplies these conditions, has the minus sign too. It is restored.
* *`m ≥ 1`.*  At `m = 0` and `n ≥ 1` the statement is false: with no frozen
  token the conclusion `∃ j ∈ [m]` fails for every `φ*`, and one particle
  alone does not move. The proof's induction starts from a centre; `0 < m`
  is taken as a hypothesis.
* The angles are lifted to `ℝ`, with `|θ_i - θ_j|` read as the geodesic
  distance on `𝕊^1` (`circDist`); `μ₀` is a probability measure on the
  lifted angles, absolutely continuous with respect to Lebesgue measure.
  Solutions are taken on all of `ℝ`: the field is smooth and `2π`-periodic,
  so every solution is global.

The proof works directly with the smooth frozen-token field. The printed
C¹ version of `lemma:convergence` is false (`Causal.not_sequentialFlow_converges`);
the specialized proof instead uses periodic scalar barriers, countable
repelling fibers, measurable Fubini, and a local stability induction.

Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`. -/
theorem fixed_centers (hm : 0 < m) (β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hh : ((n : ℝ) + ∑ j, a j) * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
      < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(((n : ℝ) + ∑ j, a j) * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))))
      < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (μ₀ : Measure (Idx n → ℝ)) [IsProbabilityMeasure μ₀] (hμ₀ : μ₀ ≪ volume) :
    ∀ᵐ φ₀ ∂μ₀, ∀ φ : ℝ → Idx n → ℝ, φ 0 = φ₀ →
      (∀ t : ℝ, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t) →
      ∃ φstar : Idx n → ℝ,
        Filter.Tendsto φ Filter.atTop (nhds φstar) ∧
        frozenVelocity n m β θ a φstar = 0 ∧
        IsAsymptoticallyStable n (frozenVelocity n m β θ a) φstar ∧
        ∀ k : Idx n, ∃ j : Idx m, circDist (φstar k - θ j) < ε * β ^ (-(1 / 2 : ℝ)) := by
  by_cases hn : n = 0
  · subst n
    apply Filter.Eventually.of_forall
    intro φ₀ φ h0 hφ
    have heq : φ = fun _ => φ₀ := funext (fun _ => Subsingleton.elim _ _)
    refine ⟨φ₀, ?_, Subsingleton.elim _ _, isAsymptoticallyStable_empty _ _, fun k => Fin.elim0 k⟩
    rw [heq]
    exact tendsto_const_nhds
  by_cases hm1 : m = 1
  · subst m
    have hθ : θ = fun _ : Idx 1 => θ 0 := by
      funext j
      have hj : j = 0 := Subsingleton.elim _ _
      rw [hj]
    have ha_const : a = fun _ : Idx 1 => a 0 := by
      funext j
      have hj : j = 0 := Subsingleton.elim _ _
      rw [hj]
    have hmass : 0 < a 0 := lt_of_lt_of_le zero_lt_one (ha 0)
    have hfield : frozenVelocity n 1 β θ a =
        frozenVelocity n 1 β (fun _ => θ 0) (fun _ => a 0) :=
      congrArg₂ (fun centers masses : Idx 1 → ℝ => frozenVelocity n 1 β centers masses) hθ ha_const
    filter_upwards [frozenOneCenterODE_mu_ae_tendsto_stable_consensus β (θ 0) (a 0) hmass n μ₀ hμ₀]
      with φ₀ hφ₀
    intro φ h0 hφ
    have hφ' : ∀ t, HasDerivAt φ (frozenVelocity n 1 β (fun _ => θ 0) (fun _ => a 0) (φ t)) t := by
      simpa only [hfield] using hφ
    obtain ⟨φstar, hlim, hcrit, hstable, hnear⟩ := hφ₀ φ h0 hφ'
    refine ⟨φstar, hlim, ?_, ?_, ?_⟩
    · simpa only [hfield] using hcrit
    · simpa only [hfield] using hstable
    · intro k
      refine ⟨0, ?_⟩
      rw [hnear k]
      exact mul_pos hε (Real.rpow_pos_of_pos hβ _)
  have hm2 : 2 ≤ m := by omega
  have hn1 : 1 ≤ n := by omega
  have hN3 : (3 : ℝ) ≤ (n : ℝ) + ∑ j, a j := by
    have hlower := frozen_total_mass_lower n a ha
    have hnreal : (1 : ℝ) ≤ n := by exact_mod_cast hn1
    have hmreal : (2 : ℝ) ≤ m := by exact_mod_cast hm2
    linarith
  have hpair : ∃ i j : Idx m, i ≠ j := by
    refine ⟨⟨0, by omega⟩, ⟨1, by omega⟩, ?_⟩
    intro heq
    have hv := congrArg Fin.val heq
    norm_num at hv
  exact frozenODE_mu_ae_tendsto_stable_clustered ((n : ℝ) + ∑ j, a j) β ε c hN3 hβ hε hc
    θ a ha hsep hpair hh hg n le_rfl μ₀ hμ₀

/-- The original theorem applies to an actual probability initialization with one moving token, two antipodal frozen centers, and the original concrete parameter values.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`; this instantiates the complete convergence, criticality, stability, and clustering conclusion. -/
example : ∀ᵐ φ₀ ∂(volume : Measure (Idx 1 → ℝ)).restrict (Set.univ.pi (fun _ : Idx 1 => Set.Ico (0 : ℝ) 1)),
    ∀ φ : ℝ → Idx 1 → ℝ, φ 0 = φ₀ →
      (∀ t, HasDerivAt φ (frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) (φ t)) t) →
      ∃ φstar : Idx 1 → ℝ, Filter.Tendsto φ Filter.atTop (nhds φstar) ∧
        frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) φstar = 0 ∧
        IsAsymptoticallyStable 1 (frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1)) φstar ∧
        ∀ k, ∃ j : Idx 2, circDist (φstar k - (![0, Real.pi]) j) <
          (9 / 100 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩ := frozen_sample_parameters
  obtain ⟨η, hβη, hη, hcη, hcap, ha, hsep, hpair, hNη, hhη, hgη, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  let μ : Measure (Idx 1 → ℝ) := volume.restrict (Set.univ.pi (fun _ : Idx 1 => Set.Ico (0 : ℝ) 1))
  have : IsProbabilityMeasure μ := by
    constructor
    change volume.restrict (Set.univ.pi (fun _ : Idx 1 => Set.Ico (0 : ℝ) 1)) Set.univ = 1
    rw [Measure.restrict_apply_univ, Real.volume_pi_Ico]
    simp
  have hmass : ((1 : ℕ) : ℝ) + ∑ j : Idx 2, (fun _ : Idx 2 => (1 : ℝ)) j = 3 := by
    norm_num [Fin.sum_univ_two]
  exact fixed_centers 1 2 (by decide) 11 (9 / 100) (142 / 25) hβ hε hc
    (![0, Real.pi]) (fun _ => 1) ha hsep (by rw [hmass]; exact hh) (by rw [hmass]; exact hg)
    μ (Measure.absolutelyContinuous_of_le Measure.restrict_le_self)

end Transformer.Causal
