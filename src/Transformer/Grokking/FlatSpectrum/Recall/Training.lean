module

public import Transformer.Grokking.FlatSpectrum.Recall.Basic
public import Transformer.Grokking.FlatSpectrum.Training.AnyShape

/-!
# Training a factorization on recall

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), item AR9:
items R3 and R4 for the recall kernel `K` of a centered readout `φ`, with `m = |κ|` keys and
`c = |ν|` values. The model is the pure factorization `U Vᴴ` of *Training reaches the optimal
fit*: a row of `U` for each queried key and a row of `V` for each dictionary.

* Width. With every pair of a query and a dictionary seen, a factorization of width `k` has
  squared loss at least `c^{m−1} ∑_v |φ(v)|² (m − k) / 2`
  (`mul_sub_card_le_fitLoss_recallKernel`).
* Unseen pairs. For a readout without zeros and a proper set `Ω` of pairs, the trained fits of
  every weight decay and width stay a fixed `δ` below `‖K‖_S1`
  (`exists_nuclearNorm_add_le_recallKernel`), and almost every run of width at least `m` with a
  small step converges to such a fit (`exists_pos_ae_tendsto_decayRun_recallKernel`): training
  does not return recall from part of its table.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Matrix Filter Topology MeasureTheory Training

variable {κ ν k : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν] [DecidableEq ν] [Fintype k]

/-- **A factorization narrower than the number of keys leaves a loss on recall**
(`review/flat_spectrum.md`, AR9): with every pair seen, width `k` has squared loss at least
`c^{m−1} ∑_v |φ(v)|² (m − k) / 2`. -/
theorem mul_sub_card_le_fitLoss_recallKernel {φ : ν → ℂ} (hφ : ∑ v, φ v = 0)
    (U : Matrix κ k ℂ) (V : Matrix (κ → ν) k ℂ) :
    ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2) *
      ((Fintype.card κ : ℝ) - Fintype.card k) / 2 ≤
        fitLoss Set.univ (recallKernel φ) (U * Vᴴ) := by
  have h0 : 0 ≤ (Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * ∑ v, ‖φ v‖ ^ 2 :=
    mul_nonneg (pow_nonneg (Nat.cast_nonneg _) _) (Finset.sum_nonneg fun _ _ => sq_nonneg _)
  rcases h0.eq_or_lt with h | h
  · rw [← h, zero_mul, zero_div]
    exact fitLoss_nonneg _ _ _
  · have h1 := (isFlat_recallKernel (κ := κ) hφ).mul_sub_card_le_fitLoss_univ h U V
    have hle : Fintype.card κ ≤ Fintype.card (κ → ν) := by
      have h2 := rank_le_card_width (recallKernel (κ := κ) φ)
      rwa [rank_eq_card_of_mul_conjTranspose_self h.ne' (recallKernel_mul_conjTranspose hφ)]
        at h2
    rwa [min_eq_left hle] at h1

/-- Three keys with values `±1` and width one: the loss is at least `4 · 2 · (3 − 1) / 2`. -/
example (U : Matrix (Fin 3) (Fin 1) ℂ) (V : Matrix (Fin 3 → Fin 2) (Fin 1) ℂ) :=
  mul_sub_card_le_fitLoss_recallKernel (φ := (![1, -1] : Fin 2 → ℂ))
    (by simp [Fin.sum_univ_two]) U V

/-- **Trained fits stay below recall** (`review/flat_spectrum.md`, AR9 with item WD7): for a
centered readout without zeros and a proper set `Ω` of pairs, some `δ > 0` has
`‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1` for the trained fits of every weight decay `λ > 0` and width. -/
theorem exists_nuclearNorm_add_le_recallKernel {φ : ν → ℂ} (hφ : ∑ v, φ v = 0)
    (h0 : ∀ v, φ v ≠ 0) {Ω : Set (κ × (κ → ν))} (hΩ : Ω ≠ Set.univ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (lam : ℝ), 0 < lam → ∀ (r : ℕ) (U : Matrix κ (Fin r) ℂ)
      (V : Matrix (κ → ν) (Fin r) ℂ), IsDecayedMin (fitLoss Ω (recallKernel φ)) lam U V →
        nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm (recallKernel (κ := κ) φ) :=
  exists_nuclearNorm_add_le_of_not_isMinOn (not_isMinOn_recallKernel hφ h0 hΩ)

/-- **Almost every run of width at least the number of keys trains below recall**
(`review/flat_spectrum.md`, AR9 with items WD7 and WD10), for a centered readout without zeros
and a proper set `Ω` of pairs. -/
theorem exists_pos_ae_tendsto_decayRun_recallKernel {φ : ν → ℂ} (hφ : ∑ v, φ v = 0)
    (h0 : ∀ v, φ v ≠ 0) {Ω : Set (κ × (κ → ν))} (hΩ : Ω ≠ Set.univ)
    (hcard : Fintype.card κ ≤ Fintype.card k) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {lam : ℝ}, 0 < lam → ∀ c : ℝ, ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ →
      ∀ᵐ x : EuclideanSpace ℂ ((κ ⊕ (κ → ν)) × k),
        flatLoss (fitLoss Ω (recallKernel φ)) lam x ≤ c →
          ∃ (U : Matrix κ k ℂ) (V : Matrix (κ → ν) k ℂ),
            Tendsto (decayRun (fun X => mask Ω (X - recallKernel φ)) η lam (factors x).1
              (factors x).2) atTop (𝓝 (U, V)) ∧
                nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm (recallKernel (κ := κ) φ) :=
  exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn
    (not_isMinOn_recallKernel hφ h0 hΩ) (Or.inl hcard)

/-- Three keys with values `±1`, with no pair seen. -/
example := exists_nuclearNorm_add_le_recallKernel (κ := Fin 3) (φ := (![1, -1] : Fin 2 → ℂ))
  (by simp [Fin.sum_univ_two]) (fun v => by fin_cases v <;> simp) Set.empty_ne_univ

/-- Three keys with values `±1`, width three, with no pair seen. -/
example := exists_pos_ae_tendsto_decayRun_recallKernel (κ := Fin 3) (k := Fin 3)
  (φ := (![1, -1] : Fin 2 → ℂ)) (by simp [Fin.sum_univ_two]) (fun v => by fin_cases v <;> simp)
  Set.empty_ne_univ le_rfl

end Transformer.Grokking.FlatSpectrum.Recall
