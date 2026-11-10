module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.FlatSupport
public import Transformer.Grokking.FlatSpectrum.Training.AlmostEvery
public import Transformer.Grokking.FlatSpectrum.Training.Recovery
public import Transformer.Grokking.FlatSpectrum.Training.SquaredRun

/-!
# Training does not return a flat kernel of any shape

Source: `review/flat_spectrum.md`, *Flat kernels of any shape* (added), item R4, with Theorem A
and items WD7, WD8 and WD10 of *Training reaches the optimal fit*. Those items are stated for a
square kernel with equal singular values and no zero entry. Their proofs use one fact: `K` is
not a minimum of M from `Ω`. So they hold for every kernel with that property
(`exists_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn`,
`exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn`), and by item R2 for a flat
kernel of any shape when some nonzero cell is unseen:

* no optimal fit multiplies to `K` (`IsFlat.not_exists_isOptimalFit`);
* the trained fits of every weight decay and width stay a fixed `δ` below `‖K‖_S1`
  (`IsFlat.exists_nuclearNorm_add_le`);
* so does the limit of every run of WD8 from a start of full row rank
  (`IsFlat.exists_tendsto_decayRun_nuclearNorm_add_le`), and of almost every run of width at
  least `min(|m|, |n|)` with a small step
  (`IsFlat.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le`).

Item R3 for training: with every cell seen, a factorization of width `k` has squared loss at
least `c (min(|m|, |n|) − k) / 2` (`IsFlat.mul_sub_card_le_fitLoss_univ`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology MeasureTheory

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]

/-- **A run converges below a kernel that is not a minimum of M** (`review/flat_spectrum.md`,
R4 with items WD7 and WD8): some `δ > 0` has: for every `λ > 0`, every run on the squared loss
on `Ω` under the bounds of WD6 and WD8 from a start of full row rank converges, to factors with
`‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1`. -/
theorem exists_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn {K : Matrix m n ℂ}
    {Ω : Set (m × n)}
    (hmin : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {η lam κ ρ : ℝ}, 0 < η → 0 < lam → 0 ≤ κ → frobeniusSq K ≤ κ ^ 2 →
      κ ^ 2 ≤ lam * ρ → 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam →
      η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1 → ∀ {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ},
        frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ → Function.Surjective (stack U₀ V₀).mulVec →
        ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
          Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
            nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K := by
  rw [isMinOn_iff] at hmin
  push Not at hmin
  obtain ⟨X', hX', hlt⟩ := hmin
  refine ⟨nuclearNorm K - nuclearNorm X', sub_pos.mpr hlt, ?_⟩
  intro η lam κ ρ hη hlam hκ0 hκ hρ hstep hdesc U₀ V₀ hw hfull
  obtain ⟨U, V, hlim, hdm⟩ :=
    exists_tendsto_decayRun_fitLoss_isDecayedMin hη hlam hκ0 hκ hρ hstep hdesc hw hfull
  exact ⟨U, V, hlim, by linarith [hdm.nuclearNorm_le hlam hX']⟩

/-- **Almost every run of width `min(|m|, |n|)` converges below a kernel that is not a minimum
of M** (`review/flat_spectrum.md`, R4 with items WD7 and WD10). -/
theorem exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn {K : Matrix m n ℂ}
    {Ω : Set (m × n)}
    (hmin : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {lam : ℝ}, 0 < lam → ∀ c : ℝ, ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ →
      ∀ᵐ x : EuclideanSpace ℂ ((m ⊕ n) × k), flatLoss (fitLoss Ω K) lam x ≤ c →
        ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
          Tendsto (decayRun (fun X => mask Ω (X - K)) η lam (factors x).1 (factors x).2) atTop
            (𝓝 (U, V)) ∧ nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K := by
  rw [isMinOn_iff] at hmin
  push Not at hmin
  obtain ⟨X', hX', hlt⟩ := hmin
  refine ⟨nuclearNorm K - nuclearNorm X', sub_pos.mpr hlt, fun {lam} hlam c => ?_⟩
  obtain ⟨η₀, hη₀, hae⟩ := exists_pos_ae_tendsto_decayRun_isDecayedMin (k := k)
    (fitLoss_first_order Ω K) (continuous_mask_sub Ω K) (analyticAt_fitLoss Ω K) hlam
    (fitLoss_nonneg Ω K) c hcard
  refine ⟨η₀, hη₀, fun η hη hlt' => ?_⟩
  filter_upwards [hae η hη hlt'] with x hx hc
  obtain ⟨U, V, hlim, hdm⟩ := hx hc
  exact ⟨U, V, hlim, by linarith [hdm.nuclearNorm_le hlam hX']⟩

end Transformer.Grokking.FlatSpectrum.Training

namespace Transformer.Grokking.FlatSpectrum

open Matrix Filter Topology MeasureTheory Training

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]

/-- **Theorem A for weight decay, any shape** (`review/flat_spectrum.md`, R4): when a nonzero
cell of a flat `K` is unseen, no optimal fit to the values of `K` on `Ω` multiplies to `K`. -/
theorem IsFlat.not_exists_isOptimalFit {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c) (h : IsFlat K c)
    {Ω : Set (m × n)} {i : m} {j : n} (hij : (i, j) ∉ Ω) (hK : K i j ≠ 0) :
    ¬ ∃ (r : ℕ) (U : Matrix m (Fin r) ℂ) (V : Matrix n (Fin r) ℂ),
      IsOptimalFit {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} U V ∧ U * Vᴴ = K := by
  rintro ⟨r, U, V, hopt, hUV⟩
  exact h.not_isMinOn_of_ne_zero hc hij hK (hUV ▸ (isOptimalFit_iff.mp hopt).2.1)

/-- **Theorem A for training, any shape** (`review/flat_spectrum.md`, R4 with item WD7): when a
nonzero cell of a flat `K` is unseen, some `δ > 0` has `‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1` for the trained
fits of every weight decay `λ > 0` and every width. -/
theorem IsFlat.exists_nuclearNorm_add_le {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : IsFlat K c) {Ω : Set (m × n)} {i : m} {j : n} (hij : (i, j) ∉ Ω) (hK : K i j ≠ 0) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (lam : ℝ), 0 < lam → ∀ (r : ℕ) (U : Matrix m (Fin r) ℂ)
      (V : Matrix n (Fin r) ℂ), IsDecayedMin (fitLoss Ω K) lam U V →
        nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K :=
  exists_nuclearNorm_add_le_of_not_isMinOn (h.not_isMinOn_of_ne_zero hc hij hK)

/-- **Runs do not return a flat kernel of any shape** (`review/flat_spectrum.md`, R4 with items
WD7 and WD8), when a nonzero cell is unseen. -/
theorem IsFlat.exists_tendsto_decayRun_nuclearNorm_add_le {K : Matrix m n ℂ} {c : ℝ}
    (hc : 0 < c) (h : IsFlat K c) {Ω : Set (m × n)} {i : m} {j : n} (hij : (i, j) ∉ Ω)
    (hK : K i j ≠ 0) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {η lam κ ρ : ℝ}, 0 < η → 0 < lam → 0 ≤ κ → frobeniusSq K ≤ κ ^ 2 →
      κ ^ 2 ≤ lam * ρ → 2 * η * (lam ^ 2 + (ρ / 2 + κ) ^ 2) ≤ lam →
      η * (lam + (ρ / 2 + κ) + 2 * ρ) ≤ 1 → ∀ {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ},
        frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ → Function.Surjective (stack U₀ V₀).mulVec →
        ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
          Tendsto (decayRun (fun X => mask Ω (X - K)) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
            nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K :=
  exists_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn
    (h.not_isMinOn_of_ne_zero hc hij hK)

/-- **Almost every run of width `min(|m|, |n|)` does not return a flat kernel**
(`review/flat_spectrum.md`, R4 with items WD7 and WD10), when a nonzero cell is unseen. -/
theorem IsFlat.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le {K : Matrix m n ℂ} {c : ℝ}
    (hc : 0 < c) (h : IsFlat K c) {Ω : Set (m × n)} {i : m} {j : n} (hij : (i, j) ∉ Ω)
    (hK : K i j ≠ 0)
    (hcard : Fintype.card m ≤ Fintype.card k ∨ Fintype.card n ≤ Fintype.card k) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {lam : ℝ}, 0 < lam → ∀ c : ℝ, ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ →
      ∀ᵐ x : EuclideanSpace ℂ ((m ⊕ n) × k), flatLoss (fitLoss Ω K) lam x ≤ c →
        ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
          Tendsto (decayRun (fun X => mask Ω (X - K)) η lam (factors x).1 (factors x).2) atTop
            (𝓝 (U, V)) ∧ nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K :=
  exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn
    (h.not_isMinOn_of_ne_zero hc hij hK) hcard

/-- **The loss of a narrow factorization of a flat kernel** (`review/flat_spectrum.md`, R4 with
R3): with every cell seen, width `k` leaves squared loss at least
`c (min(|m|, |n|) − k) / 2`. -/
theorem IsFlat.mul_sub_card_le_fitLoss_univ {K : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : IsFlat K c) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    c * ((min (Fintype.card m) (Fintype.card n) : ℕ) - Fintype.card k) / 2 ≤
      fitLoss Set.univ K (U * Vᴴ) := by
  have h1 := h.mul_sub_card_le_frobeniusSq_sub hc U Vᴴ
  have h2 : mask Set.univ (U * Vᴴ - K) = U * Vᴴ - K := by
    ext i j
    simp [mask]
  have h3 : frobeniusSq (U * Vᴴ - K) = frobeniusSq (K - U * Vᴴ) := by
    simp only [frobeniusSq, Matrix.sub_apply]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by rw [norm_sub_rev]
  rw [fitLoss, h2, h3]
  linarith

/-! The examples: the `1 × 2` kernel `(1, 1)`, flat with `c = 2`, with no cell seen. -/

example := exists_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn (k := Fin 1 ⊕ Fin 2)
  (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (Ω := ∅) (i := 0) (j := 0) (by simp)
    (by simp))

example := exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn (k := Fin 1)
  (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (Ω := ∅) (i := 0) (j := 0) (by simp)
    (by simp)) (Or.inl le_rfl)

example := isFlat_row_ones.not_exists_isOptimalFit two_pos (Ω := ∅) (i := 0) (j := 0)
  (by simp) (by simp)

example := isFlat_row_ones.exists_nuclearNorm_add_le two_pos (Ω := ∅) (i := 0) (j := 0)
  (by simp) (by simp)

/-- Runs of width `3 = 1 + 2` converge strictly below `(1, 1)`. -/
example := isFlat_row_ones.exists_tendsto_decayRun_nuclearNorm_add_le (k := Fin 1 ⊕ Fin 2)
  two_pos (Ω := ∅) (i := 0) (j := 0) (by simp) (by simp)

/-- Almost every run of width `1 = min(1, 2)` converges strictly below `(1, 1)`. -/
example := isFlat_row_ones.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le (k := Fin 1)
  two_pos (Ω := ∅) (i := 0) (j := 0) (by simp) (by simp) (Or.inl le_rfl)

/-- Width zero leaves the loss `2 (1 − 0) / 2 = 1` on `(1, 1)`. -/
example (U : Matrix (Fin 1) (Fin 0) ℂ) (V : Matrix (Fin 2) (Fin 0) ℂ) :=
  isFlat_row_ones.mul_sub_card_le_fitLoss_univ two_pos U V

end Transformer.Grokking.FlatSpectrum
