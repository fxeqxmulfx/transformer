module

public import Transformer.Grokking.FiniteRecovery.FlatSpectrum
public import Transformer.Grokking.FlatSpectrum.NuclearBounds
public import Transformer.Grokking.FlatSpectrum.Training.Scope.MarginContinuity

/-! A unique nuclear completion separates the truth from matrices with
a fixed positive error. Finite observed errors and finite nuclear norm
slack are allowed, so this applies at a finite softmax parameter limit. -/

@[expose] public section

open Filter Metric Set
open scoped Topology

namespace Transformer.Grokking.FiniteRecovery

open FlatSpectrum

variable {U V : Type*} [Fintype U] [Fintype V] [DecidableEq V]

/-- Uniform entrywise stability of the unique Flat Spectrum completion. -/
theorem NuclearRecovers.exists_error_threshold {cells : Set (U × V)}
    {truth : Matrix U V ℂ} (hunique : NuclearRecovers cells truth)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ δ > 0, ∀ X : Matrix U V ℂ,
      (∀ c ∈ cells, ‖X c.1 c.2 - truth c.1 c.2‖ ≤ δ) →
      nuclearNorm X ≤ nuclearNorm truth + δ →
      ∀ u v, ‖X u v - truth u v‖ < ε := by
  classical
  by_contra! h
  let slack (n : ℕ) : ℝ := 1 / ((n : ℝ) + 1)
  have hslack : Tendsto slack atTop (𝓝 0) := by
    simpa only [slack, Nat.cast_add, Nat.cast_one] using
      (tendsto_add_atTop_iff_nat 1).2 (tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ))
  choose X hobs hnorm i j herror using fun n => h (slack n) (by unfold slack; positivity)
  let B := closedBall (0 : ℂ) (nuclearNorm truth + 1)
  have hcompact : IsCompact (Set.pi (Set.univ : Set U) fun _ => Set.pi (Set.univ : Set V) fun _ => B) :=
    isCompact_univ_pi fun _ => isCompact_univ_pi fun _ => isCompact_closedBall _ _
  have hB (n : ℕ) : X n ∈ Set.pi (Set.univ : Set U) fun _ => Set.pi (Set.univ : Set V) fun _ => B := by
    intro u _ v _
    have hs : slack n ≤ 1 := by
      unfold slack
      rw [div_le_one (by positivity)]
      linarith [Nat.cast_nonneg (α := ℝ) n]
    exact mem_closedBall_zero_iff.mpr ((norm_apply_le_nuclearNorm (X n) u v).trans
      ((hnorm n).trans (add_le_add (le_refl _) hs)))
  obtain ⟨Y, _, k, hk, hY⟩ := hcompact.tendsto_subseq hB
  have hs := hslack.comp hk.tendsto_atTop
  have hc (u : U) (v : V) : Tendsto (fun n => X (k n) u v) atTop (𝓝 (Y u v)) :=
    tendsto_pi_nhds.mp (tendsto_pi_nhds.mp hY u) v
  have hfit : Y ∈ completions cells truth := by
    intro c hc'
    have h0 := (hc c.1 c.2).sub_const (truth c.1 c.2) |>.norm
    have hle := le_of_tendsto_of_tendsto' h0 hs (fun n => hobs (k n) c hc')
    exact sub_eq_zero.mp (norm_eq_zero.mp (le_antisymm hle (norm_nonneg _)))
  have hmin := (hunique truth (fun _ _ => rfl)).2 rfl
  have hyNorm : nuclearNorm Y ≤ nuclearNorm truth :=
    le_of_tendsto_of_tendsto'
      ((FlatSpectrum.Training.Scope.continuous_nuclearNorm.tendsto Y).comp hY)
      (by simpa only [Function.comp_def, add_zero] using hs.const_add (nuclearNorm truth))
      (fun n => hnorm (k n))
  have heq := (hunique Y hfit).1 fun Z hZ => hyNorm.trans (hmin hZ)
  have hsmall : ∀ᶠ n in atTop, ∀ u v, ‖X (k n) u v - truth u v‖ < ε := by
    apply eventually_all.mpr
    intro u
    apply eventually_all.mpr
    intro v
    have hy := (hc u v).sub_const (truth u v) |>.norm
    rw [heq, sub_self, norm_zero] at hy
    exact hy.eventually (gt_mem_nhds hε)
  obtain ⟨n, hn⟩ := hsmall.exists
  exact (not_lt_of_ge (herror (k n))) (hn (i (k n)) (j (k n)))

example : ∃ δ > 0, ∀ X : Matrix (Fin 2) (Fin 2) ℂ,
    (∀ c ∈ (Set.univ : Set (Fin 2 × Fin 2)), ‖X c.1 c.2 - (1 : Matrix (Fin 2) (Fin 2) ℂ) c.1 c.2‖ ≤ δ) →
    nuclearNorm X ≤ nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) + δ →
    ∀ u v, ‖X u v - (1 : Matrix (Fin 2) (Fin 2) ℂ) u v‖ < 1 / 10 :=
  (nuclearRecovers_univ (1 : Matrix (Fin 2) (Fin 2) ℂ)).exists_error_threshold (by norm_num)

end Transformer.Grokking.FiniteRecovery
