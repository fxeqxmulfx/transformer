/-
# Attention's forward pass and Frank-Wolfe — the survival probability

The actual transition law supplies a conditional bound for stopped bad
steps. The finite union estimate then controls their count, including
dependent updates. Almost every trajectory starts at `X₀`, stays in its
closed convex hull, and takes convex-combination steps; the first-exit
lemma bounds the exit event by that count tail.

The diameter input `D` is any bound on distances in the convex set. The
polytope theorem specializes it to `Metric.diam K`. The count threshold is
an integer ceiling, including an exit that reaches the boundary of the
open neighborhood exactly. Measures of complements are converted to real
probabilities only after the probability instance from `IsSAProcess` is
available; no finiteness of an arbitrary measure is assumed.

Source: arXiv:2508.09628v1, §5, `cl: group.cluster.bound`,
`lem: metastab.1`, `eq: union.bound`, `eq: binomial.2`.
-/

import Transformer.FrankWolfe.Section5_FirstExit

open scoped ENNReal BigOperators Pointwise
open MeasureTheory

namespace Transformer.FrankWolfe

variable {d n κ : ℕ}

/-- The binomial union bound for stopped bad steps of the actual attention
process. Source: arXiv:2508.09628v1, §5, `eq: binomial.2`, with the
conditional score estimate `cl: group.cluster.bound`. -/
theorem IsSAProcess.measure_count_stoppedBadSteps_le {v : Idx κ → EucSpace d}
    {σ : Idx n → Idx κ} {c₀ R ρ ε β γ : ℝ}
    (hsep : ∀ a b, a ≠ b →
      inner (𝕜 := ℝ) (v a) (v b) + c₀ ≤ inner (𝕜 := ℝ) (v a) (v a))
    (hR : ∀ a, ‖v a‖ ≤ R)
    (hsmall : 4 * R * (ρ + ε) + 2 * (ρ + ε) ^ 2 ≤ c₀ / 2) (hβ : 0 ≤ β)
    {X₀ : Idx n → EucSpace d} {P : Measure (ℕ → Idx n → EucSpace d)}
    (hP : IsSAProcess β γ X₀ P) (N M : ℕ) :
    P {x | M ≤ (rareStepTimes (stoppedBadStep v σ γ (ρ + ε)) N x).card} ≤
      (N.choose M : ℝ≥0∞) * ENNReal.ofReal (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) ^ M := by
  let := hP.1
  exact measure_count_rare_steps_le P _ _ (pastDet_stoppedBadStep v σ γ (ρ + ε))
    (fun t S hS => hP.measure_stoppedBadStep_le hsep hR hsmall hβ t hS) N M

/-- The preceding estimate expressed as a real probability. Source:
arXiv:2508.09628v1, §5, auxiliary to `eq: binomial.2`. -/
theorem IsSAProcess.count_stoppedBadSteps_real_le {v : Idx κ → EucSpace d}
    {σ : Idx n → Idx κ} {c₀ R ρ ε β γ : ℝ}
    (hsep : ∀ a b, a ≠ b →
      inner (𝕜 := ℝ) (v a) (v b) + c₀ ≤ inner (𝕜 := ℝ) (v a) (v a))
    (hR : ∀ a, ‖v a‖ ≤ R)
    (hsmall : 4 * R * (ρ + ε) + 2 * (ρ + ε) ^ 2 ≤ c₀ / 2) (hβ : 0 ≤ β)
    {X₀ : Idx n → EucSpace d} {P : Measure (ℕ → Idx n → EucSpace d)}
    (hP : IsSAProcess β γ X₀ P) (N M : ℕ) :
    (P {x | M ≤ (rareStepTimes (stoppedBadStep v σ γ (ρ + ε)) N x).card}).toReal ≤
      (N.choose M : ℝ) * (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) ^ M := by
  have hp : 0 ≤ (n : ℝ) ^ 2 * Real.exp (-(β * (c₀ / 2))) := by positivity
  have hb : (N.choose M : ℝ≥0∞) *
      ENNReal.ofReal (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) ^ M ≠ ⊤ := by finiteness
  have h := ENNReal.toReal_mono hb
    (hP.measure_count_stoppedBadSteps_le (σ := σ) hsep hR hsmall hβ N M)
  simpa only [ENNReal.toReal_mul, ENNReal.toReal_pow, ENNReal.toReal_natCast,
    ENNReal.toReal_ofReal hp] using h

/-- Closed convex confinement and the first-exit count give the finite
horizon survival bound. Source: arXiv:2508.09628v1, §5, `lem: metastab.1`,
`eq: binomial.2`; every count is stopped where the score bound remains valid. -/
theorem IsSAProcess.survivalEvent_bound {K : Set (EucSpace d)} (hKc : IsClosed K)
    (hKv : Convex ℝ K) {γ D ρ ε β c₀ R : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (hδ : 0 < γ * D) (hD : ∀ y ∈ K, ∀ y' ∈ K, ‖y - y'‖ ≤ D)
    {v : Idx κ → EucSpace d} {σ : Idx n → Idx κ}
    (hsep : ∀ a b, a ≠ b →
      inner (𝕜 := ℝ) (v a) (v b) + c₀ ≤ inner (𝕜 := ℝ) (v a) (v a))
    (hR : ∀ a, ‖v a‖ ≤ R)
    (hsmall : 4 * R * (ρ + ε) + 2 * (ρ + ε) ^ 2 ≤ c₀ / 2) (hβ : 0 ≤ β)
    {X₀ : Idx n → EucSpace d} {P : Measure (ℕ → Idx n → EucSpace d)}
    (hX₀ : ∀ i, X₀ i ∈ K) (hρX : ∀ i, X₀ i ∈ Metric.ball (v (σ i)) ρ)
    (hP : IsSAProcess β γ X₀ P) (N : ℕ) :
    1 - (N.choose ⌈ε / (γ * D)⌉₊ : ℝ) *
        (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) ^ ⌈ε / (γ * D)⌉₊ ≤
      (P (survivalEvent X₀ σ ε N)).toReal := by
  let := hP.1
  have hsub : (survivalEvent X₀ σ ε N)ᶜ ≤ᵐ[P]
      {x | ⌈ε / (γ * D)⌉₊ ≤
        (rareStepTimes (stoppedBadStep v σ γ (ρ + ε)) N x).card} := by
    filter_upwards [hP.2.1, hP.ae_mem_of_convex hγ0 hγ1 hKc hKv hX₀,
      hP.ae_exists_step] with x h0 hK hstep hx
    exact (survival_or_count_stoppedBadSteps hKv hγ0 hγ1 hδ hD X₀ v σ hX₀ hρX
      x h0 hK hstep N).resolve_left hx
  have hle : (P (survivalEvent X₀ σ ε N)ᶜ).toReal ≤
      (N.choose ⌈ε / (γ * D)⌉₊ : ℝ) *
        (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) ^ ⌈ε / (γ * D)⌉₊ :=
    (ENNReal.toReal_mono (measure_ne_top P _) (measure_mono_ae hsub)).trans
      (hP.count_stoppedBadSteps_real_le hsep hR hsmall hβ N _)
  have he : (P (survivalEvent X₀ σ ε N)ᶜ).toReal =
      1 - (P (survivalEvent X₀ σ ε N)).toReal := by
    change P.real (survivalEvent X₀ σ ε N)ᶜ = 1 - P.real (survivalEvent X₀ σ ε N)
    rw [measureReal_compl (measurableSet_survivalEvent X₀ σ ε N), probReal_univ]
  rw [he] at hle
  linarith

/-- The same binomial estimate holds for the event at every integer step
before a real horizon `t`, using `floor(t)` candidate updates.
Source: arXiv:2508.09628v1, §5, `lem: metastab.1`, `eq: metastability.bound`.
The finite-horizon event includes its last integer time, so its inclusion
in the event of the paper also holds at an integer value of `t`. -/
theorem IsSAProcess.survival_real_horizon_bound {K : Set (EucSpace d)} (hKc : IsClosed K)
    (hKv : Convex ℝ K) {γ D ρ ε β c₀ R : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (hδ : 0 < γ * D) (hD : ∀ y ∈ K, ∀ y' ∈ K, ‖y - y'‖ ≤ D)
    {v : Idx κ → EucSpace d} {σ : Idx n → Idx κ}
    (hsep : ∀ a b, a ≠ b →
      inner (𝕜 := ℝ) (v a) (v b) + c₀ ≤ inner (𝕜 := ℝ) (v a) (v a))
    (hR : ∀ a, ‖v a‖ ≤ R)
    (hsmall : 4 * R * (ρ + ε) + 2 * (ρ + ε) ^ 2 ≤ c₀ / 2) (hβ : 0 ≤ β)
    {X₀ : Idx n → EucSpace d} {P : Measure (ℕ → Idx n → EucSpace d)}
    (hX₀ : ∀ i, X₀ i ∈ K) (hρX : ∀ i, X₀ i ∈ Metric.ball (v (σ i)) ρ)
    (hP : IsSAProcess β γ X₀ P) (t : ℝ) :
    1 - (⌊t⌋₊.choose ⌈ε / (γ * D)⌉₊ : ℝ) *
        (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) ^ ⌈ε / (γ * D)⌉₊ ≤
      (P {x | ∀ s : ℕ, (s : ℝ) < t → ∀ i,
        x s i ∈ groupHull X₀ σ (σ i) + Metric.ball (0 : EucSpace d) ε}).toReal := by
  let := hP.1
  refine (hP.survivalEvent_bound hKc hKv hγ0 hγ1 hδ hD hsep hR hsmall hβ hX₀ hρX ⌊t⌋₊).trans ?_
  exact ENNReal.toReal_mono (measure_ne_top P _)
    (measure_mono (survivalEvent_subset_real_horizon X₀ σ ε t))

/-- All hypotheses are realized by a stationary particle at the origin,
positive step and radii, and a positive bound on its singleton's diameter. -/
example :
    IsClosed ({0} : Set (EucSpace 1)) ∧ Convex ℝ ({0} : Set (EucSpace 1)) ∧
      0 ≤ (1 / 2 : ℝ) ∧ (1 / 2 : ℝ) ≤ 1 ∧ 0 < (1 / 2 : ℝ) * 1 ∧
      (∀ y ∈ ({0} : Set (EucSpace 1)), ∀ y' ∈ ({0} : Set (EucSpace 1)), ‖y - y'‖ ≤ 1) ∧
      (∀ a b : Idx 1, a ≠ b → inner (𝕜 := ℝ) (0 : EucSpace 1) 0 + 1 ≤
        inner (𝕜 := ℝ) (0 : EucSpace 1) 0) ∧
      (∀ a : Idx 1, ‖(fun _ : Idx 1 => (0 : EucSpace 1)) a‖ ≤ (0 : ℝ)) ∧
      4 * (0 : ℝ) * (1 / 20 + 1 / 20) + 2 * (1 / 20 + 1 / 20) ^ 2 ≤ 1 / 2 ∧
      0 ≤ (0 : ℝ) ∧
      (∀ i : Idx 1, (fun _ : Idx 1 => (0 : EucSpace 1)) i ∈ ({0} : Set (EucSpace 1))) ∧
      (∀ i : Idx 1, (fun _ : Idx 1 => (0 : EucSpace 1)) i ∈
        Metric.ball ((fun _ : Idx 1 => (0 : EucSpace 1)) i) (1 / 20)) ∧
      IsSAProcess 0 (1 / 2) (fun _ : Idx 1 => (0 : EucSpace 1))
        (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) := by
  refine ⟨isClosed_singleton, convex_singleton 0, by norm_num, by norm_num, by norm_num,
    ?_, fun a b h => False.elim (h (Subsingleton.elim a b)), by simp, by norm_num,
    le_rfl, by simp, fun i => Metric.mem_ball_self (by norm_num), isSAProcess_single _ _ _⟩
  rintro y hy y' hy'
  simp only [Set.mem_singleton_iff] at hy hy'
  simp [hy, hy']

end Transformer.FrankWolfe
