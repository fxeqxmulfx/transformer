import Transformer.GPTMini.Sparsemax.PeriodicMemoryForward
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Why the structural mask permits fixed width

Derived capacity boundary for arXiv:1602.02068v2, Eq. (1). Any unmasked
QK product equal to an invertible P-slot attention matrix requires width
at least P. This obstructs fixed-width recovery of the former normalized
Gram architecture; its old physical embedding Gram cannot be retained.

The new masked construction has genuine QK score rank at most three while
actual sparsemax attention has rank P. Masking and projection, with proved
threshold zero, supply the full-rank matrix. Thus constant embedding width
does not assume a false low-rank factorization of invertible attention.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- An invertible plain QK product needs at least as many coordinates as memory slots.
Source: the capacity boundary of the normalized chart before arXiv:1602.02068v2, Eq. (1). -/
theorem invertibleQK_width_lower {P W : ℕ}
    (Q : Matrix (Fin P) (Fin W) ℝ) (K : Matrix (Fin W) (Fin P) ℝ)
    (hi : IsUnit (Q * K).det) : P ≤ W := by
  have hr := Matrix.rank_of_isUnit (Q * K) ((Matrix.isUnit_iff_isUnit_det _).mpr hi)
  have hu := (Matrix.rank_mul_le_left Q K).trans (Matrix.rank_le_width Q)
  rw [hr, Fintype.card_fin] at hu
  exact hu

/-- Nonzero full-width Q/K matrices inhabit every invertible-product premise. -/
example : (3 : ℕ) ≤ 3 :=
  invertibleQK_width_lower (1 : Matrix (Fin 3) (Fin 3) ℝ) 1 (by norm_num)

/-- Every ordinary feature realization of the former normalized Gram needs width at least P.
Source: the derived capacity restriction for the inverse chart after arXiv:1602.02068v2, Eq. (1). -/
theorem memoryGram_realization_width_lower {N W : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1))
    (features : Fin W → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ)
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor)
    (he : G = featureGram features) : N + 1 ≤ W := by
  let Q := Matrix.of (fun i d => features d (Sum.inl i))
  let K := Matrix.of (fun d j => features d (Sum.inr j))
  have hs : memoryGramScores G = Q * K := by
    rw [he]
    ext i j
    exact featureGram_apply features (Sum.inl i) (Sum.inr j)
  have hi := memoryGramAttention_det_unit cap floor G hf hG
  rw [memoryGramAttention_normalized cap floor G hG, hs] at hi
  exact invertibleQK_width_lower Q K hi

/-- An actual identity feature Gram inhabits the width-bound hypotheses. -/
example : (2 : ℕ) ≤ 2 := by
  apply memoryGram_realization_width_lower 1 (3 / 4) (memoryIdentityGram 1)
    (memoryIdentityFeatures 1) (by norm_num)
    (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))
  rfl

/-- The old compact incident chart has the same necessary width growth despite linear storage.
Source: the derived memory inverse before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemory_realization_width_lower {N W : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N)
    (features : Fin W → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ)
    (hf : 1 / 2 < floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor)
    (he : localMemoryGram p = featureGram features) : N + 1 ≤ W :=
  memoryGram_realization_width_lower cap floor _ features hf
    (incidentMemoryGram_mem cap floor p (by linarith) hp) he

/-- Genuine nonidentity embeddings from the proved recovery inhabit the incident width premises. -/
example : ∃ features : Fin 8 → Sum (Fin 4) (Fin 4) → ℝ,
    localMemoryGram incidentMemoryExampleParameters = featureGram features ∧ (4 : ℕ) ≤ 8 := by
  obtain ⟨features, he⟩ := incidentMemoryGram_fixedWidth 4 (3 / 4)
    incidentMemoryExampleParameters (by norm_num) incidentMemoryExampleParameters_mem
  exact ⟨features, he, incidentMemory_realization_width_lower 4 (3 / 4) _ features
    (by norm_num) incidentMemoryExampleParameters_mem he⟩

/-- Actual fixed-width masked sparsemax attention has full prototype rank.
Source: the proved inverse in the new chart after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryAttention_rank {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 1 / 2 < floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    (periodicMemoryAttention t).rank = N + 1 := by
  have h := Matrix.rank_of_isUnit (periodicMemoryAttention t)
    ((Matrix.isUnit_iff_isUnit_det _).mpr (periodicMemoryAttention_det_unit floor t hf ht))
  simpa only [Fintype.card_fin] using h

/-- Four prototypes with three learned edges inhabit all full-attention-rank premises. -/
example : (periodicMemoryAttention (fun _ : Fin 3 => (1 / 8 : ℝ))).rank = 4 :=
  periodicMemoryAttention_rank _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Genuine unmasked QK scores have rank at most three for every learned parameter table.
Source: the true constant-width score construction before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryScore_rank_le {N : ℕ} (t : Fin N → ℝ) :
    (Matrix.of (fun i j => periodicMemoryScore t i j)).rank ≤ 3 := by
  let Q := Matrix.of (fun i d => periodicMemoryQuery t i d)
  let K := Matrix.of (fun d j => periodicMemoryKey t j d)
  have he : Matrix.of (fun i j => periodicMemoryScore t i j) = Q * K := by
    ext i j
    rfl
  rw [he]
  exact (Matrix.rank_mul_le_left Q K).trans (Matrix.rank_le_width Q)

/-- For more than three prototypes the actual mask/projection strictly increases the score rank.
Source: the new capacity-escape witness after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemory_mask_rank_gain {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hN : 3 < N + 1) (hf : 1 / 2 < floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    (Matrix.of (fun i j => periodicMemoryScore t i j)).rank < (periodicMemoryAttention t).rank := by
  rw [periodicMemoryAttention_rank floor t hf ht]
  exact lt_of_le_of_lt (periodicMemoryScore_rank_le t) hN

/-- A nonidentity four-prototype example inhabits every strict-rank-gain premise. -/
example : (Matrix.of (fun i j => periodicMemoryScore (fun _ : Fin 3 => (1 / 8 : ℝ)) i j)).rank <
    (periodicMemoryAttention (fun _ : Fin 3 => (1 / 8 : ℝ))).rank :=
  periodicMemory_mask_rank_gain _ _ (by norm_num) (by norm_num) incidentMemoryExampleWeights_mem

/-- Above three prototypes the unmasked dot products cannot be the normalized attention.
Source: the new architecture's explicit mask dependence after sparsemax Eq. (1).
This rules out silently interpreting the physical width-three Gram as the
old normalized affine lift, even though the masked forward is identical. -/
theorem periodicMemoryScore_ne_attention {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hN : 3 < N + 1) (hf : 1 / 2 < floor)
    (ht : t ∈ incidentMemoryWeightDomain N floor) :
    Matrix.of (fun i j => periodicMemoryScore t i j) ≠ periodicMemoryAttention t := by
  intro he
  have hr := periodicMemory_mask_rank_gain floor t hN hf ht
  rw [he] at hr
  exact lt_irrefl _ hr

/-- Four nontrivial prototypes inhabit the actual mask-dependence premises. -/
example : Matrix.of (fun i j => periodicMemoryScore (fun _ : Fin 3 => (1 / 8 : ℝ)) i j) ≠
    periodicMemoryAttention (fun _ : Fin 3 => (1 / 8 : ℝ)) :=
  periodicMemoryScore_ne_attention _ _ (by norm_num) (by norm_num)
    incidentMemoryExampleWeights_mem

/-- The physical width-three Gram cannot equal the old normalized Gram above three slots.
Source: the explicit architectural change required by the capacity boundary after sparsemax Eq. (1). -/
theorem periodicMemoryFeatures_ne_oldGram {N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (t : Fin N → ℝ)
    (hN : 3 < N + 1) (hf : 1 / 2 < floor) (hp : p ∈ incidentMemoryParameterDomain N cap floor) :
    featureGram (periodicMemoryFeatures t) ≠ localMemoryGram p := by
  intro he
  have hw := incidentMemory_realization_width_lower cap floor p (periodicMemoryFeatures t) hf hp he.symm
  omega

/-- Changed nonidentity geometry inhabits the physical-Gram separation hypotheses. -/
example : featureGram (periodicMemoryFeatures (fun _ : Fin 3 => (1 / 8 : ℝ))) ≠
    localMemoryGram incidentMemoryExampleParameters :=
  periodicMemoryFeatures_ne_oldGram _ _ _ _ (by norm_num) (by norm_num)
    incidentMemoryExampleParameters_mem

end Transformer.GPTMini.Sparsemax
