/-
# The emergence of clusters in self-attention dynamics — `l:exactasymptotic`

§7 of arXiv:2305.05465v6: in `d = 1`, a token that is not uniformly bounded
has an exact exponential asymptotic `x_i(t) = γ_i e^t + o(e^t)`, `γ_i ≠ 0`.

The proof, on the scalar curves (`exists_tendsto_div_exp`), for a token going
to `+∞` with the largest token `x_N`:

* `x_N e^{-t}` does not increase, since the drift of `x_N` is an average of
  coordinates `≤ x_N`, and it stays above `c > 0`, since `x_N ≥ x_i ≥ c e^t`:
  it converges to some `γ ≥ c`;
* `x_N - x_i` stays bounded, since its derivative is at most
  `x_N - ẋ_i ≤ n/x_i ≤ (n/c) e^{-t}` (`sub_drift_le`), so
  `x_i e^{-t} = x_N e^{-t} - (x_N - x_i) e^{-t} → γ`.

This is the source's argument with its rates dropped: monotone convergence
replaces the estimate `ẋ_n = x_n(1 + o(·))`.  A token going to `-∞` is a token
of `-x` going to `+∞`.

Source: arXiv:2305.05465v6, `l:exactasymptotic`.
-/

import Transformer.Clusters.Section7_DriftAverage
import Mathlib.Topology.Order.MonotoneConvergence

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- **`l:exactasymptotic` for the scalar curves.**  If every `x_k` follows
the drift of `e:Idnonresca`, `x_N` is the largest coordinate for `t ≥ 0`, and
`x_i ≥ c e^t` eventually, then `x_i e^{-t}` converges to some `γ > 0`.

Source: arXiv:2305.05465v6, proof of `l:exactasymptotic`. -/
theorem exists_tendsto_div_exp (x : ℝ → Idx (m + 1) → ℝ) (i N : Idx (m + 1))
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (hmax : ∀ t, 0 ≤ t → ∀ j, x t j ≤ x t N) {c : ℝ} (hc : 0 < c)
    (hgrow : ∀ᶠ t in atTop, c * Real.exp t ≤ x t i) :
    ∃ γ : ℝ, 0 < γ ∧ Tendsto (fun t => x t i / Real.exp t) atTop (𝓝 γ) := by
  obtain ⟨T₀, hT₀⟩ := eventually_atTop.1 hgrow
  set T := max T₀ 0
  have hT : ∀ t, T ≤ t → c * Real.exp t ≤ x t i ∧ ∀ j, x t j ≤ x t N := fun t ht =>
    ⟨hT₀ t (le_of_max_le_left ht), hmax t (le_of_max_le_right ht)⟩
  set D : ℝ → Idx (m + 1) → ℝ := fun t k =>
    ∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j
  have hE : ∀ t, HasDerivAt (fun s => Real.exp (-s)) (Real.exp (-t) * (-1)) t :=
    fun t => (hasDerivAt_neg t).exp
  -- `x_N e^{-t}` does not increase and stays above `c`
  set h : ℝ → ℝ := fun t => x t N * Real.exp (-t)
  have hanti : AntitoneOn h (Set.Ici T) := by
    refine antitoneOn_Ici_of_hasDerivAt
      (f' := fun t => D t N * Real.exp (-t) + x t N * (Real.exp (-t) * -1))
      (fun t _ => (hder t N).mul (hE t)) fun t ht => ?_
    have h1 : D t N ≤ x t N := drift_le (x t) N N (hT t ht).2
    have he := Real.exp_pos (-t)
    nlinarith
  have hlow : ∀ t, T ≤ t → c ≤ h t := by
    intro t ht
    obtain ⟨h1, h2⟩ := hT t ht
    calc c = c * Real.exp t * Real.exp (-t) := by
          rw [mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]
      _ ≤ x t i * Real.exp (-t) := mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
      _ ≤ h t := mul_le_mul_of_nonneg_right (h2 i) (Real.exp_pos _).le
  set φ : ℝ → ℝ := fun s => h (max s T)
  have hφ : Antitone φ := fun a b hab =>
    hanti (Set.mem_Ici.2 (le_max_right a T)) (Set.mem_Ici.2 (le_max_right b T))
      (max_le_max hab le_rfl)
  have hφb : ∀ s, c ≤ φ s := fun s => hlow _ (le_max_right s T)
  have hlimφ := tendsto_atTop_ciInf hφ ⟨c, by rintro _ ⟨s, rfl⟩; exact hφb s⟩
  have hγ : c ≤ ⨅ s, φ s := le_ciInf hφb
  have hlimh : Tendsto h atTop (𝓝 (⨅ s, φ s)) :=
    hlimφ.congr' (eventually_atTop.2 ⟨T, fun s hs => by simp only [φ, max_eq_left hs]⟩)
  -- `x_N - x_i` stays bounded
  set K := ((m : ℝ) + 1) / c
  have hK : 0 ≤ K := div_nonneg (by positivity) hc.le
  set v : ℝ → ℝ := fun t => (x t N - x t i) + K * Real.exp (-t)
  have hvanti : AntitoneOn v (Set.Ici T) := by
    refine antitoneOn_Ici_of_hasDerivAt
      (f' := fun t => (D t N - D t i) + K * (Real.exp (-t) * -1))
      (fun t _ => ((hder t N).sub (hder t i)).add ((hE t).const_mul K)) fun t ht => ?_
    obtain ⟨h1, h2⟩ := hT t ht
    have hxi : 0 < x t i := (mul_pos hc (Real.exp_pos t)).trans_le h1
    have h3 : D t N ≤ x t N := drift_le (x t) N N h2
    have h4 : x t N - D t i ≤ (m + 1) / x t i := sub_drift_le (x t) i N hxi h2
    have h5 : ((m : ℝ) + 1) / x t i ≤ K * Real.exp (-t) := by
      calc ((m : ℝ) + 1) / x t i ≤ (m + 1) / (c * Real.exp t) :=
            div_le_div_of_nonneg_left (by positivity) (mul_pos hc (Real.exp_pos t)) h1
        _ = K * Real.exp (-t) := by
            simp only [K, Real.exp_neg]; field_simp
    show D t N - D t i + K * (Real.exp (-t) * -1) ≤ 0
    linarith
  have hu : ∀ t, T ≤ t → 0 ≤ x t N - x t i ∧ x t N - x t i ≤ v T := by
    intro t ht
    refine ⟨sub_nonneg.2 ((hT t ht).2 i), ?_⟩
    have h1 : v t ≤ v T := hvanti (Set.mem_Ici.2 le_rfl) (Set.mem_Ici.2 ht) ht
    have h2 : 0 ≤ K * Real.exp (-t) := mul_nonneg hK (Real.exp_pos _).le
    simp only [v] at h1
    linarith
  have hlimu : Tendsto (fun t => (x t N - x t i) * Real.exp (-t)) atTop (𝓝 0) := by
    have hup : Tendsto (fun t => v T * Real.exp (-t)) atTop (𝓝 0) := by
      simpa using tendsto_exp_neg_atTop_nhds_zero.const_mul (v T)
    refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hup ?_ ?_
    · filter_upwards [eventually_ge_atTop T] with t ht
      exact mul_nonneg (hu t ht).1 (Real.exp_pos _).le
    · filter_upwards [eventually_ge_atTop T] with t ht
      exact mul_le_mul_of_nonneg_right (hu t ht).2 (Real.exp_pos _).le
  refine ⟨⨅ s, φ s, hc.trans_le hγ, ?_⟩
  have hlim := hlimh.sub hlimu
  rw [sub_zero] at hlim
  refine hlim.congr fun t => ?_
  simp only [h, Real.exp_neg]
  field_simp
  ring

/-- The hypotheses of `exists_tendsto_div_exp` are satisfiable at `m = 0`: the
curve `x(t) = e^t`, with `c = 1`. -/
example : ∃ x : ℝ → Idx 1 → ℝ, (∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t) ∧
    (∀ t, 0 ≤ t → ∀ j, x t j ≤ x t 0) ∧ ∀ᶠ t in atTop, 1 * Real.exp t ≤ x t 0 := by
  refine ⟨fun t _ => Real.exp t, fun t _ => ?_, fun _ _ _ => le_rfl,
    Eventually.of_forall fun _ => by simp⟩
  rw [← Finset.sum_mul, Perspective.sum_softmaxWeight one_pos, one_mul]
  exact Real.hasDerivAt_exp t

/-- **Lemma (l:exactasymptotic).**  A token that is not uniformly bounded has
an exact exponential asymptotic: there is `γ_i ≠ 0` with
`x_i(t) = γ_i e^t + o(e^t)` as `t → +∞`.

Source: arXiv:2305.05465v6, `l:exactasymptotic`. -/
theorem exists_exp_asymptotic (X : ℝ → Idx (m + 1) → EucSpace 1)
    (hX : IdNonrescaledDynamics X) (hord : IsOrderedConfig (X 0)) (i : Idx (m + 1))
    (hub : ¬ IsBoundedToken X i) :
    ∃ γ : ℝ, γ ≠ 0 ∧ Tendsto (fun t => X t i 0 / Real.exp t) atTop (nhds γ) := by
  obtain ⟨A, hA, hAeq⟩ := exists_auxiliary_constant (m + 1) (Nat.succ_pos m)
  simp only [IsBoundedToken, not_exists, not_forall, not_le] at hub
  obtain ⟨t₁, -, ht₁⟩ := hub A
  rcases lt_abs.1 ht₁ with h | h
  · obtain ⟨c, hc, hev⟩ := exists_exp_lower_bound A hA hAeq X hX i t₁ h
    obtain ⟨γ, hγ, hlim⟩ := exists_tendsto_div_exp (fun t j => X t j 0) i (Fin.last m)
      (hasDerivAt_coord X hX)
      (fun t ht j => (Fin.le_last j).lt_or_eq.elim
        (fun hj => (isOrderedConfig_of_nonneg X hX hord t ht j _ hj).le) (fun hj => by rw [hj]))
      hc hev
    exact ⟨γ, hγ.ne', hlim⟩
  · obtain ⟨c, hc, hev⟩ := exists_exp_upper_bound A hA hAeq X hX i t₁ (by linarith)
    have hder : ∀ t k, HasDerivAt (fun s => -X s k 0)
        (∑ j, Perspective.softmaxWeight (fun l => -X t k 0 * -X t l 0) j * -X t j 0) t := by
      intro t k
      convert (hasDerivAt_coord X hX t k).fun_neg using 1
      simp only [mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib]
    obtain ⟨γ, hγ, hlim⟩ := exists_tendsto_div_exp (fun t j => -X t j 0) i 0 hder
      (fun t ht j => (Fin.zero_le j).lt_or_eq.elim
        (fun hj => neg_le_neg (isOrderedConfig_of_nonneg X hX hord t ht _ j hj).le)
        (fun hj => by rw [← hj]))
      hc (hev.mono fun t ht => by linarith)
    refine ⟨-γ, neg_ne_zero.2 hγ.ne', ?_⟩
    simpa only [neg_div, neg_neg] using hlim.neg

/-- The hypotheses of `exists_exp_asymptotic` are satisfiable at `m = 0`: the
one-token solution `x(t) = e^t` is not uniformly bounded. -/
example :
    IdNonrescaledDynamics (n := 1)
        (fun t _ => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) ∧
      IsOrderedConfig (n := 1)
        (fun _ => Real.exp 0 • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) ∧
      ¬ IsBoundedToken
        (fun t (_ : Idx 1) => Real.exp t • (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1)) 0 :=
  ⟨idNonrescaledDynamics_single _, isOrderedConfig_subsingleton _, not_isBoundedToken_exp⟩

end Clusters
end Transformer
