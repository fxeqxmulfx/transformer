module

public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss

/-!
# Vanishing weight decay: trained fits become the optimal fits of M

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD7.
The loss is the squared loss on the seen cells `Ω` of a kernel `K`, and a trained fit is a
factorization of least weight-decayed loss among those of every width (`IsDecayedMin`), which
is where a convergent run ends (`isDecayedMin_of_tendsto_decayRun`). Against any matrix `X'`
equal to `K` on `Ω`, a trained fit `U Vᴴ` with weight decay `λ` has

    ½ ‖P_Ω (U Vᴴ − K)‖² + λ ‖U Vᴴ‖_S1 ≤ λ ‖X'‖_S1

(`IsDecayedMin.fitLoss_add_le`): its nuclear norm is at most that of every exact fit and its
loss is at most `λ ‖X'‖_S1`. So, as `λ → 0⁺`,

* a limit of trained fits is an optimal fit of the note, `IsOptimalFit`
  (`isOptimalFit_of_tendsto`);
* if `K` does not have the least nuclear norm among the matrices equal to it on `Ω`, the trained
  products stay a fixed distance below `‖K‖_S1`, for every `λ > 0` and every width
  (`exists_nuclearNorm_add_le_of_not_isMinOn`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq n]
  {Ω : Set (m × n)} {K : Matrix m n ℂ} {lam : ℝ} {U : Matrix m k ℂ} {V : Matrix n k ℂ}

/-- **A trained fit against an exact fit** (`review/flat_spectrum.md`, item WD7): for the
squared loss and `λ ≥ 0`, `L (U Vᴴ) + λ ‖U Vᴴ‖_S1 ≤ λ ‖X'‖_S1` for every `X'` equal to `K` on
`Ω`. -/
theorem IsDecayedMin.fitLoss_add_le (hlam : 0 ≤ lam) (h : IsDecayedMin (fitLoss Ω K) lam U V)
    {X' : Matrix m n ℂ} (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2) :
    fitLoss Ω K (U * Vᴴ) + lam * nuclearNorm (U * Vᴴ) ≤ lam * nuclearNorm X' := by
  have h1 := h.add_nuclearNorm_le hlam X'
  rwa [fitLoss_eq_zero_iff.mpr hX', zero_add] at h1

/-- **A trained fit has no more nuclear norm than any exact fit**
(`review/flat_spectrum.md`, item WD7), for `λ > 0`. -/
theorem IsDecayedMin.nuclearNorm_le (hlam : 0 < lam) (h : IsDecayedMin (fitLoss Ω K) lam U V)
    {X' : Matrix m n ℂ} (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2) :
    nuclearNorm (U * Vᴴ) ≤ nuclearNorm X' := by
  have h1 := h.fitLoss_add_le hlam.le hX'
  have h2 := fitLoss_nonneg Ω K (U * Vᴴ)
  exact le_of_mul_le_mul_left (by linarith) hlam

/-- **The loss of a trained fit is at most `λ ‖X'‖_S1`** (`review/flat_spectrum.md`, item
WD7), for every `X'` equal to `K` on `Ω`. -/
theorem IsDecayedMin.fitLoss_le (hlam : 0 ≤ lam) (h : IsDecayedMin (fitLoss Ω K) lam U V)
    {X' : Matrix m n ℂ} (hX' : ∀ ij ∈ Ω, X' ij.1 ij.2 = K ij.1 ij.2) :
    fitLoss Ω K (U * Vᴴ) ≤ lam * nuclearNorm X' := by
  have h1 := h.fitLoss_add_le hlam hX'
  have h2 := mul_nonneg hlam (nuclearNorm_nonneg (U * Vᴴ))
  linarith

/-- The `1 × 1` kernel `1` at `λ = ¾`: the trained product `¼` against the exact fit `1`. -/
example := (isDecayedMin_one_sub (s := 1 / 2) (by norm_num) (by norm_num)).fitLoss_add_le
  (by norm_num) (X' := 1) fun _ _ => rfl

example := (isDecayedMin_one_sub (s := 1 / 2) (by norm_num) (by norm_num)).nuclearNorm_le
  (by norm_num) (X' := 1) fun _ _ => rfl

example := (isDecayedMin_one_sub (s := 1 / 2) (by norm_num) (by norm_num)).fitLoss_le
  (by norm_num) (X' := 1) fun _ _ => rfl

/-- **Limits of trained fits are optimal fits** (`review/flat_spectrum.md`, item WD7): if the
weight decays `λ_j > 0` tend to `0` and trained fits `(U_j, V_j)` for the squared loss on `Ω`
converge, then the limit is an optimal fit to the values of `K` on `Ω`: it fits exactly and has
the least weight decay among the exact fits of every width. -/
theorem isOptimalFit_of_tendsto {lams : ℕ → ℝ} (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0)) {r : ℕ} {Us : ℕ → Matrix m (Fin r) ℂ}
    {Vs : ℕ → Matrix n (Fin r) ℂ} (hmin : ∀ j, IsDecayedMin (fitLoss Ω K) (lams j) (Us j) (Vs j))
    {Uinf : Matrix m (Fin r) ℂ} {Vinf : Matrix n (Fin r) ℂ} (hU : Tendsto Us atTop (𝓝 Uinf))
    (hV : Tendsto Vs atTop (𝓝 Vinf)) :
    IsOptimalFit {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} Uinf Vinf := by
  have hprod : Tendsto (fun j => Us j * (Vs j)ᴴ) atTop (𝓝 (Uinf * Vinfᴴ)) :=
    (continuous_mul_conjTranspose.tendsto (Uinf, Vinf)).comp (hU.prodMk_nhds hV)
  refine ⟨?_, fun r' U' V' h' => ?_⟩
  · have hloss : Tendsto (fun j => fitLoss Ω K (Us j * (Vs j)ᴴ)) atTop
        (𝓝 (fitLoss Ω K (Uinf * Vinfᴴ))) := ((continuous_fitLoss Ω K).tendsto _).comp hprod
    have hbound : Tendsto (fun j => lams j * nuclearNorm K) atTop (𝓝 0) := by
      simpa using hlam.mul_const (nuclearNorm K)
    exact fitLoss_eq_zero_iff.mp (le_antisymm (le_of_tendsto_of_tendsto' hloss hbound fun j =>
      (hmin j).fitLoss_le (hpos j).le (X' := K) fun _ _ => rfl) (fitLoss_nonneg _ _ _))
  · have hwd : Tendsto (fun j => (frobeniusSq (Us j) + frobeniusSq (Vs j)) / 2) atTop
        (𝓝 ((frobeniusSq Uinf + frobeniusSq Vinf) / 2)) :=
      (((continuous_frobeniusSq.tendsto Uinf).comp hU).add
        ((continuous_frobeniusSq.tendsto Vinf).comp hV)).div_const 2
    refine le_of_tendsto' hwd fun j => ?_
    have h1 := hmin j r' U' V'
    unfold decayedLoss at h1
    rw [fitLoss_eq_zero_iff.mpr h', zero_add] at h1
    have h2 := fitLoss_nonneg Ω K (Us j * (Vs j)ᴴ)
    exact le_of_mul_le_mul_left (by linarith) (hpos j)

/-- **Trained fits stay below a kernel that M does not return** (`review/flat_spectrum.md`,
item WD7): if `K` does not have the least nuclear norm among the matrices equal to it on `Ω`,
some `δ > 0` has `‖U Vᴴ‖_S1 + δ ≤ ‖K‖_S1` for the trained fits of every weight decay `λ > 0`
and every width. -/
theorem exists_nuclearNorm_add_le_of_not_isMinOn
    (h : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (lam : ℝ), 0 < lam → ∀ (r : ℕ) (U : Matrix m (Fin r) ℂ)
      (V : Matrix n (Fin r) ℂ), IsDecayedMin (fitLoss Ω K) lam U V →
        nuclearNorm (U * Vᴴ) + δ ≤ nuclearNorm K := by
  rw [isMinOn_iff] at h
  push Not at h
  obtain ⟨X', hX', hlt⟩ := h
  exact ⟨nuclearNorm K - nuclearNorm X', sub_pos.mpr hlt, fun lam hlam r U V hmin => by
    have := hmin.nuclearNorm_le hlam hX'
    linarith⟩

/-- The shrinking family of the `1 × 1` kernel `1`: its limit `1 · 1` is an optimal fit. -/
example := isOptimalFit_of_tendsto shrink_pos tendsto_shrink isDecayedMin_shrink
  tendsto_shrink_factor tendsto_shrink_factor

/-- With no cell seen, the trained fits of the `a³ + a b` kernel for `p = 3` stay below it. -/
example := exists_nuclearNorm_add_le_of_not_isMinOn
  (not_isMinOn_of_singularValues_eq (singularValues_cubicKernel 3)
    (fun a b => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
    (Set.empty_ne_univ (α := ZMod 3 × ZMod 3)))

end Transformer.Grokking.FlatSpectrum.Training
