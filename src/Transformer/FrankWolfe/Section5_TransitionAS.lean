/-
# Attention's forward pass and Frank-Wolfe — what the transition law forces

Consequences of the transition identity `IsSAProcess.measure_inter_prod` that hold along almost
every path: each particle moves to a convex combination of two particles of the previous
configuration, so a convex set containing the initial configuration contains every later one; and
one-step bounds on the probability of leaving a configuration-dependent target set.
-/

import Transformer.FrankWolfe.Section5_Transition

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace FrankWolfe

variable {d n : ℕ}

/-- One-step upper bound: if, at every configuration `z` in a measurable set `G`, the conditional
probability of the target set `𝒜` is at most `c`, then `ℙ[S ∩ {x^t ∈ G} ∩ {(x^t, x_i^{t+1}) ∈ 𝒜}]`
is at most `c ℙ[S ∩ {x^t ∈ G}]`, for every event `S` of the past. -/
theorem IsSAProcess.measure_inter_prod_le {β γ : ℝ} {X₀ : Idx n → EucSpace d}
    {P : Measure (ℕ → Idx n → EucSpace d)} (hP : IsSAProcess β γ X₀ P) (t : ℕ) (i : Idx n)
    {S : Set (ℕ → Idx n → EucSpace d)} (hS : PastDet t S)
    {𝒜 : Set ((Idx n → EucSpace d) × EucSpace d)} (h𝒜 : MeasurableSet 𝒜)
    {G : Set (Idx n → EucSpace d)} (hG : MeasurableSet G) {c : ℝ≥0∞}
    (hc : ∀ z ∈ G, ∑ j, ENNReal.ofReal (attWeight β z i j) *
      𝒜.indicator (fun _ => (1 : ℝ≥0∞)) (z, (1 - γ) • z i + γ • z j) ≤ c) :
    P (S ∩ {x | x t ∈ G} ∩ {x | (x t, x (t + 1) i) ∈ 𝒜}) ≤ c * P (S ∩ {x | x t ∈ G}) := by
  have hSG : PastDet t (S ∩ {x | x t ∈ G}) := hS.inter (pastDet_coord le_rfl hG)
  rw [hP.measure_inter_prod t i hSG h𝒜, ← setLIntegral_const]
  exact setLIntegral_mono' hSG.1 fun x hx => hc _ hx.2

/-- One-step lower bound, the mirror image of `IsSAProcess.measure_inter_prod_le`. -/
theorem IsSAProcess.le_measure_inter_prod {β γ : ℝ} {X₀ : Idx n → EucSpace d}
    {P : Measure (ℕ → Idx n → EucSpace d)} (hP : IsSAProcess β γ X₀ P) (t : ℕ) (i : Idx n)
    {S : Set (ℕ → Idx n → EucSpace d)} (hS : PastDet t S)
    {𝒜 : Set ((Idx n → EucSpace d) × EucSpace d)} (h𝒜 : MeasurableSet 𝒜)
    {G : Set (Idx n → EucSpace d)} (hG : MeasurableSet G) {c : ℝ≥0∞}
    (hc : ∀ z ∈ G, c ≤ ∑ j, ENNReal.ofReal (attWeight β z i j) *
      𝒜.indicator (fun _ => (1 : ℝ≥0∞)) (z, (1 - γ) • z i + γ • z j)) :
    c * P (S ∩ {x | x t ∈ G}) ≤ P (S ∩ {x | x t ∈ G} ∩ {x | (x t, x (t + 1) i) ∈ 𝒜}) := by
  have hSG : PastDet t (S ∩ {x | x t ∈ G}) := hS.inter (pastDet_coord le_rfl hG)
  rw [hP.measure_inter_prod t i hSG h𝒜, ← setLIntegral_const]
  exact setLIntegral_mono' hSG.1 fun x hx => hc _ hx.2

/-- Almost surely, every particle moves to `(1-γ)x_i^t + γ x_j^t` for some particle `j`. -/
theorem IsSAProcess.ae_exists_step {β γ : ℝ} {X₀ : Idx n → EucSpace d}
    {P : Measure (ℕ → Idx n → EucSpace d)} (hP : IsSAProcess β γ X₀ P) :
    ∀ᵐ x ∂P, ∀ (s : ℕ) (i : Idx n), ∃ j, x (s + 1) i = (1 - γ) • x s i + γ • x s j := by
  rw [ae_all_iff]
  intro s
  rw [ae_all_iff]
  intro i
  set 𝒜 : Set ((Idx n → EucSpace d) × EucSpace d) :=
    {p | ∀ j, p.2 ≠ (1 - γ) • p.1 i + γ • p.1 j} with h𝒜
  have hm : MeasurableSet 𝒜 := by
    have : 𝒜 = ⋂ j, {p : (Idx n → EucSpace d) × EucSpace d |
        p.2 = (1 - γ) • p.1 i + γ • p.1 j}ᶜ := by
      ext p
      simp [h𝒜]
    rw [this]
    exact MeasurableSet.iInter fun j => (measurableSet_eq_fun measurable_snd (by fun_prop)).compl
  have hz : ∀ x : ℕ → Idx n → EucSpace d, ∑ j, ENNReal.ofReal (attWeight β (x s) i j) *
      𝒜.indicator (fun _ => (1 : ℝ≥0∞)) (x s, (1 - γ) • x s i + γ • x s j) = 0 := by
    intro x
    refine Finset.sum_eq_zero fun j _ => ?_
    have : (x s, (1 - γ) • x s i + γ • x s j) ∉ 𝒜 := fun h => h j rfl
    simp [Set.indicator_of_notMem this]
  have h0 : P {x : ℕ → Idx n → EucSpace d | (x s, x (s + 1) i) ∈ 𝒜} = 0 := by
    have := hP.measure_inter_prod s i (pastDet_univ s) hm
    simpa only [Set.univ_inter, hz, lintegral_zero] using this
  filter_upwards [measure_eq_zero_iff_ae_notMem.mp h0] with x hx
  by_contra hne
  push Not at hne
  exact hx (by simpa [h𝒜] using hne)

/-- A closed convex set containing the initial configuration contains, almost surely, every later
configuration: each particle moves to a convex combination of two particles, `γ ∈ [0,1]`. -/
theorem IsSAProcess.ae_mem_of_convex {β γ : ℝ} {X₀ : Idx n → EucSpace d}
    {P : Measure (ℕ → Idx n → EucSpace d)} (hP : IsSAProcess β γ X₀ P) (hγ0 : 0 ≤ γ)
    (hγ1 : γ ≤ 1) {K : Set (EucSpace d)} (hKc : IsClosed K) (hKv : Convex ℝ K)
    (hX₀ : ∀ i, X₀ i ∈ K) : ∀ᵐ x ∂P, ∀ (s : ℕ) (i : Idx n), x s i ∈ K := by
  have hstep : ∀ s : ℕ, (∀ᵐ x ∂P, ∀ i, x s i ∈ K) → ∀ᵐ x ∂P, ∀ i, x (s + 1) i ∈ K := by
    intro s hs
    rw [ae_all_iff]
    intro i
    set G : Set (Idx n → EucSpace d) := {z | ∀ k, z k ∈ K} with hG
    have hGm : MeasurableSet G := by
      have : G = ⋂ k, {z : Idx n → EucSpace d | z k ∈ K} := by
        ext z
        simp [hG]
      rw [this]
      exact MeasurableSet.iInter fun k => (measurable_pi_apply k) hKc.measurableSet
    have h𝒜 : MeasurableSet {p : (Idx n → EucSpace d) × EucSpace d | p.2 ∉ K} :=
      (measurable_snd hKc.measurableSet).compl
    have hle := hP.measure_inter_prod_le s i (pastDet_univ s) h𝒜 hGm (c := 0) (fun z hz => by
      refine le_of_eq (Finset.sum_eq_zero fun j _ => ?_)
      have hmem : (1 - γ) • z i + γ • z j ∈ K :=
        hKv (hz i) (hz j) (sub_nonneg.mpr hγ1) hγ0 (by ring)
      simp [Set.indicator_of_notMem (show (z, (1 - γ) • z i + γ • z j) ∉
        {p : (Idx n → EucSpace d) × EucSpace d | p.2 ∉ K} from fun h => h hmem)])
    have hnull : P {x : ℕ → Idx n → EucSpace d | x (s + 1) i ∉ K} = 0 := by
      have h1 : {x : ℕ → Idx n → EucSpace d | x (s + 1) i ∉ K} ⊆
          {x | ¬ ∀ k, x s k ∈ K} ∪
            (Set.univ ∩ {x | x s ∈ G} ∩ {x | (x s, x (s + 1) i) ∈
              {p : (Idx n → EucSpace d) × EucSpace d | p.2 ∉ K}}) := by
        intro x hx
        by_cases hxs : ∀ k, x s k ∈ K
        · exact Or.inr ⟨⟨trivial, hxs⟩, hx⟩
        · exact Or.inl hxs
      refine measure_mono_null h1 (measure_union_null ?_ ?_)
      · exact measure_eq_zero_iff_ae_notMem.mpr (by filter_upwards [hs] with x hx; simpa using hx)
      · simpa using hle
    filter_upwards [measure_eq_zero_iff_ae_notMem.mp hnull] with x hx
    simpa using hx
  have hall : ∀ s : ℕ, ∀ᵐ x ∂P, ∀ i, x s i ∈ K := by
    intro s
    induction s with
    | zero => filter_upwards [hP.2.1] with x hx i using hx ▸ hX₀ i
    | succ s ih => exact hstep s ih
  rw [ae_all_iff]
  intro s
  exact hall s

end FrankWolfe
end Transformer
