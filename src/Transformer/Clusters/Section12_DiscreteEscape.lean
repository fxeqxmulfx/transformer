module

public import Transformer.Clusters.Section12_MultiHeadPull

/-!
# Discrete scalar escape estimates for the original multi-head proof

ArXiv:2305.05465v6, §7, `l:auxiliary` and `l:unboundedparticles`,
with the discrete time convention of §3, `r:discreterescaling`.
The head-sum pull estimate has the form Λr-B/r for r>0.
For a positive step size and Λ>0, that expression gives geometric
expansion above a fixed threshold. The threshold below is explicit;
it does not depend on a trajectory or assume that escape occurs.

The elementary sequence argument records how an actual step estimate
implies geometric growth after any threshold crossing. Applying the
same argument to the negative sequence gives a three-way alternative:
uniformly bounded, tending to positive infinity, or tending to negative
infinity. Subsequent modules verify the step estimates directly from
the original multi-head equation and its separate head coefficients.
-/

@[expose] public section

open Real Filter Topology

namespace Transformer.Clusters

/-- The original head-sum self-pull gives geometric expansion above a fixed explicit threshold.
Source: arXiv:2305.05465v6, §7, `l:auxiliary`, in the discrete convention of §3. -/
theorem exists_discrete_self_growth_threshold (Δt Λ B : ℝ)
    (hΔt : 0 < Δt) (hΛ : 0 < Λ) (hB : 0 ≤ B) :
    ∃ R : ℝ, 0 < R ∧ ∀ r, R ≤ r →
      (1 + Δt * Λ / 2) * r ≤ r + Δt * (Λ * r - B / r) := by
  let R := 1 + 2 * B / Λ
  have hR1 : 1 ≤ R := by
    have hquot : 0 ≤ 2 * B / Λ := by positivity
    dsimp [R]
    linarith
  refine ⟨R, by linarith, ?_⟩
  intro r hr
  have hr1 : 1 ≤ r := hR1.trans hr
  have hr0 : 0 < r := by linarith
  have hBr : 2 * B / Λ ≤ r := by dsimp [R] at hr; linarith
  have hlarge : 2 * B ≤ r * Λ := (div_le_iff₀ hΛ).mp hBr
  have hrsq : r ≤ r ^ 2 := by nlinarith
  have hsq := mul_le_mul_of_nonneg_left hrsq hΛ.le
  have hdiv : B / r ≤ Λ * r / 2 := (div_le_iff₀ hr0).mpr (by nlinarith)
  have hm := mul_le_mul_of_nonneg_left hdiv hΔt.le
  nlinarith

/-- A guarded actual step estimate gives a geometric lower bound after every threshold crossing.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, with discrete layer indices as in §3. -/
theorem discrete_geometric_lower_bound (f : ℕ → ℝ) {a R : ℝ} (ha : 1 ≤ a) (hR : 0 < R)
    (hstep : ∀ k, R ≤ f k → a * f k ≤ f (k + 1)) (k : ℕ) (hk : R ≤ f k) :
    ∀ l : ℕ, a ^ l * f k ≤ f (k + l) := by
  have hfk : 0 < f k := hR.trans_le hk
  have ha0 : 0 ≤ a := by linarith
  intro l
  induction l with
  | zero => simp only [pow_zero, one_mul, Nat.add_zero]; exact le_rfl
  | succ l ih =>
      have hbase : f k ≤ a ^ l * f k := by
        have hm := mul_le_mul_of_nonneg_right (one_le_pow₀ ha (n := l)) hfk.le
        simpa only [one_mul] using hm
      have hguard : R ≤ f (k + l) := hk.trans (hbase.trans ih)
      calc
        a ^ (l + 1) * f k = a * (a ^ l * f k) := by rw [pow_succ]; ring
        _ ≤ a * f (k + l) := mul_le_mul_of_nonneg_left ih ha0
        _ ≤ f (k + l + 1) := hstep (k + l) hguard
        _ = f (k + (l + 1)) := by congr 1

/-- Crossing an expanding threshold forces convergence to positive infinity along the original layers.
Source: arXiv:2305.05465v6, §7, `l:unboundedparticles`, in discrete time. -/
theorem discrete_tendsto_atTop_of_threshold (f : ℕ → ℝ) {a R : ℝ} (ha : 1 < a) (hR : 0 < R)
    (hstep : ∀ k, R ≤ f k → a * f k ≤ f (k + 1)) (k : ℕ) (hk : R ≤ f k) :
    Tendsto f atTop atTop := by
  have hgeo := discrete_geometric_lower_bound f ha.le hR hstep k hk
  have hp : Tendsto (fun l : ℕ => a ^ l * f k) atTop atTop :=
    (tendsto_pow_atTop_atTop_of_one_lt ha).atTop_mul_const (hR.trans_le hk)
  apply tendsto_atTop.mpr
  intro C
  obtain ⟨l, hl⟩ := eventually_atTop.mp (tendsto_atTop.mp hp C)
  apply eventually_atTop.mpr
  refine ⟨k + l, ?_⟩
  intro N hN
  have hkl : l ≤ N - k := by omega
  have he : k + (N - k) = N := by omega
  have hg := hgeo (N - k)
  rw [he] at hg
  exact (hl (N - k) hkl).trans hg

/-- The precise head-sum self-pull recurrence supplies a threshold and the full escape conclusion.
Source: arXiv:2305.05465v6, §7, `l:auxiliary` and `l:unboundedparticles`, in discrete time. -/
theorem exists_discrete_recurrence_escape_threshold (f : ℕ → ℝ) (Δt Λ B : ℝ)
    (hΔt : 0 < Δt) (hΛ : 0 < Λ) (hB : 0 ≤ B)
    (hrec : ∀ k, 0 < f k → f k + Δt * (Λ * f k - B / f k) ≤ f (k + 1)) :
    ∃ R : ℝ, 0 < R ∧ ∀ k, R ≤ f k →
      (∀ l : ℕ, (1 + Δt * Λ / 2) ^ l * f k ≤ f (k + l)) ∧ Tendsto f atTop atTop := by
  obtain ⟨R, hR, hgrowth⟩ := exists_discrete_self_growth_threshold Δt Λ B hΔt hΛ hB
  have ha : 1 < 1 + Δt * Λ / 2 := by nlinarith [mul_pos hΔt hΛ]
  have hstep : ∀ k, R ≤ f k → (1 + Δt * Λ / 2) * f k ≤ f (k + 1) :=
    fun k hk => (hgrowth (f k) hk).trans (hrec k (hR.trans_le hk))
  refine ⟨R, hR, ?_⟩
  intro k hk
  exact ⟨discrete_geometric_lower_bound f ha.le hR hstep k hk,
    discrete_tendsto_atTop_of_threshold f ha hR hstep k hk⟩

/-- Two reflected guarded step estimates classify a sequence without assuming any escape occurs.
Source: arXiv:2305.05465v6, §7, `l:auxiliary`, in the discrete convention of §3. -/
theorem discrete_bounded_or_tendsto_infinity (f : ℕ → ℝ) {a R : ℝ} (ha : 1 < a) (hR : 0 < R)
    (hpos : ∀ k, R ≤ f k → a * f k ≤ f (k + 1))
    (hneg : ∀ k, R ≤ -f k → a * (-f k) ≤ -f (k + 1)) :
    (∀ k, |f k| ≤ R) ∨ Tendsto f atTop atTop ∨ Tendsto f atTop atBot := by
  by_cases hp : ∃ k, R ≤ f k
  · obtain ⟨k, hk⟩ := hp
    exact Or.inr (Or.inl (discrete_tendsto_atTop_of_threshold f ha hR hpos k hk))
  by_cases hn : ∃ k, R ≤ -f k
  · obtain ⟨k, hk⟩ := hn
    have ht := discrete_tendsto_atTop_of_threshold (fun k => -f k) ha hR hneg k hk
    exact Or.inr (Or.inr (tendsto_neg_atTop_iff.mp ht))
  left
  intro k
  have hp' : f k < R := lt_of_not_ge fun hk => hp ⟨k, hk⟩
  have hn' : -f k < R := lt_of_not_ge fun hk => hn ⟨k, hk⟩
  exact abs_le.mpr ⟨by linarith, hp'.le⟩

/-- The explicit self-growth threshold has strictly positive step, value sum,
and error parameters; the error need not vanish. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- A nonzero geometric sequence realizes the expansion premises, including
an actual threshold crossing, for the lower-bound and positive escape lemmas. -/
example : ∃ f : ℕ → ℝ, (1 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ f 0 ∧
    (∀ k, (1 : ℝ) ≤ f k → (2 : ℝ) * f k ≤ f (k + 1)) ∧
    (∀ k, (1 : ℝ) ≤ -f k → (2 : ℝ) * (-f k) ≤ -f (k + 1)) ∧
    (∀ k, 0 < f k → f k + (1 : ℝ) * ((1 : ℝ) * f k - 1 / f k) ≤ f (k + 1)) := by
  refine ⟨fun k => (2 : ℝ) ^ k, by norm_num, one_pos, by norm_num, ?_, ?_, ?_⟩
  · intro k hk
    change (2 : ℝ) * 2 ^ k ≤ 2 ^ (k + 1)
    rw [pow_succ]
    nlinarith
  · intro k hk
    have hp : 0 ≤ (2 : ℝ) ^ k := by positivity
    linarith
  · intro k hk
    change (2 : ℝ) ^ k + 1 * (1 * 2 ^ k - 1 / 2 ^ k) ≤ 2 ^ (k + 1)
    rw [pow_succ]
    have hd : 0 ≤ 1 / (2 : ℝ) ^ k := by positivity
    nlinarith

/-- The bounded branch is realized by a constant sequence under the same
strictly expanding guard; both guards are actual impossible crossings. -/
example : ∃ f : ℕ → ℝ, (∀ k, |f k| ≤ 1) ∧
    (∀ k, (1 : ℝ) ≤ f k → (2 : ℝ) * f k ≤ f (k + 1)) ∧
    (∀ k, (1 : ℝ) ≤ -f k → (2 : ℝ) * (-f k) ≤ -f (k + 1)) := by
  refine ⟨fun _ => 0, fun _ => by norm_num, ?_, ?_⟩
  · intro k hk; norm_num at hk
  · intro k hk; norm_num at hk

end Transformer.Clusters
