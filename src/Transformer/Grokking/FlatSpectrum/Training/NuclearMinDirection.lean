module

public import Transformer.Grokking.FlatSpectrum.Training.NuclearMin
public import Transformer.Grokking.FlatSpectrum.Training.SeenDirection

/-!
# The direction of trained products of any width

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE10 and XE11.
`MarginDirection`, `LeastNorm` and `SeenDirection` state items XE10 and XE11 for trained fits
`(U_j, V_j)` of one column type. They hold for every sequence of trained products `X_j`, the
minimizers of `L + λ_j ‖·‖_S1` (`IsNuclearMin`), by `forall_seq_isNuclearMin`. As `λ_j → 0⁺`:

* a limit of `X_j / ‖X_j‖_S1` has nuclear norm one and the largest margin
  (`margin_eq_maxMargin_of_tendsto_inv_smul_of_isNuclearMin`), and then `X_j / μ(X_j)`
  converges to that limit divided by `μ*` (`tendsto_inv_margin_smul_of_isNuclearMin`);
* if one product of the unit ball has the largest margin, `X_j / ‖X_j‖_S1` converges to it
  (`tendsto_inv_smul_of_unique_of_isNuclearMin`);
* for a code mirrored at a gap `0 < g < 2` and a flat kernel without zero entries that shows
  the labels, `X_j / ‖X_j‖_S1 → K / ‖K‖_S1` exactly when every cell is seen
  (`tendsto_inv_smul_codeProbe_iff_of_isNuclearMin`), and then `X_j / μ(X_j) → K / g`
  (`tendsto_inv_margin_smul_codeProbe_of_isNuclearMin`); so for the kernel `ψ(a³ + a b)` with
  the code `ψ` and `p ≥ 3` (`tendsto_inv_smul_cubicKernel_iff_of_isNuclearMin`).

Limit points of `X_j / ‖X_j‖_S1` exist by `exists_subseq_tendsto_inv_smul`, which is about any
sequence of matrices.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n S C : Type*} [Fintype m] [Fintype n] [Fintype S] [Fintype C] [DecidableEq n]
  [Nonempty S] [Nontrivial C] {A : S → C → Matrix m n ℂ} {y : S → C} {X₀ Z₀ : Matrix m n ℂ}
  {lams : ℕ → ℝ} {X : ℕ → Matrix m n ℂ}

/-- **A limit of the normalized trained products has the largest margin**
(`review/flat_spectrum.md`, item XE10, statement 2). -/
theorem margin_eq_maxMargin_of_tendsto_inv_smul_of_isNuclearMin (hsep : 0 < margin A y X₀)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j))
    (hZ : Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z₀)) :
    nuclearNorm Z₀ = 1 ∧ margin A y Z₀ = maxMargin A y :=
  forall_seq_isNuclearMin (P := fun X =>
      Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z₀) →
        nuclearNorm Z₀ = 1 ∧ margin A y Z₀ = maxMargin A y)
    (fun j => (hpos j).le)
    (fun _ _ h hZ => margin_eq_maxMargin_of_tendsto_inv_smul hsep hpos hlam h hZ) hX hZ

/-- **Scaled to margin one, the trained products converge where their direction does**
(`review/flat_spectrum.md`, item XE10, statement 4). -/
theorem tendsto_inv_margin_smul_of_isNuclearMin (hsep : 0 < margin A y X₀)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j))
    (hZ : Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z₀)) :
    Tendsto (fun j => (margin A y (X j))⁻¹ • X j) atTop (𝓝 ((maxMargin A y)⁻¹ • Z₀)) :=
  forall_seq_isNuclearMin (P := fun X =>
      Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z₀) →
        Tendsto (fun j => (margin A y (X j))⁻¹ • X j) atTop (𝓝 ((maxMargin A y)⁻¹ • Z₀)))
    (fun j => (hpos j).le) (fun _ _ h hZ => tendsto_inv_margin_smul hsep hpos hlam h hZ) hX hZ

variable [DecidableEq m]

/-- **A unique product of largest margin is the direction of the trained products**
(`review/flat_spectrum.md`, item XE10, statement 3). -/
theorem tendsto_inv_smul_of_unique_of_isNuclearMin (hsep : 0 < margin A y X₀)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j))
    (huniq : ∀ Z, nuclearNorm Z ≤ 1 → margin A y Z = maxMargin A y → Z = Z₀) :
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z₀) :=
  forall_seq_isNuclearMin
    (P := fun X => Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 Z₀))
    (fun j => (hpos j).le) (fun _ _ h => tendsto_inv_smul_of_unique hsep hpos hlam h huniq) hX

variable {φ : C → ℂ} {cell : S → m × n} {K : Matrix m n ℂ} {g c₀ : ℝ}

/-- **Trained products converge in direction to a flat kernel exactly when every cell is
seen** (`review/flat_spectrum.md`, item XE11 with XE9): for a code mirrored at a gap
`0 < g < 2` and a flat `K` without zero entries that shows the labels of the seen cells. -/
theorem tendsto_inv_smul_codeProbe_iff_of_isNuclearMin (h : IsMirroredCode φ g) (hg0 : 0 < g)
    (hg2 : g < 2) (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hK0 : ∀ i j, K i j ≠ 0)
    (hc : 0 < c₀) (hflat : IsFlat K c₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss (codeProbe φ cell) y) (lams j) (X j)) :
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)) ↔
      Function.Surjective cell :=
  forall_seq_isNuclearMin (P := fun X =>
      Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)) ↔
        Function.Surjective cell)
    (fun j => (hpos j).le)
    (fun _ _ hmin => tendsto_inv_smul_codeProbe_iff h hg0 hg2 hK hK0 hc hflat hpos hlam hmin) hX

/-- **With every cell seen, the trained products scaled to margin one converge to `K / g`**
(`review/flat_spectrum.md`, item XE11). -/
theorem tendsto_inv_margin_smul_codeProbe_of_isNuclearMin (h : IsMirroredCode φ g)
    (hg0 : 0 < g) (hg2 : g < 2) (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s))
    (hsurj : Function.Surjective cell) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss (codeProbe φ cell) y) (lams j) (X j)) :
    Tendsto (fun j => (margin (codeProbe φ cell) y (X j))⁻¹ • X j) atTop (𝓝 (g⁻¹ • K)) :=
  forall_seq_isNuclearMin (P := fun X =>
      Tendsto (fun j => (margin (codeProbe φ cell) y (X j))⁻¹ • X j) atTop (𝓝 (g⁻¹ • K)))
    (fun j => (hpos j).le)
    (fun _ _ hmin => tendsto_inv_margin_smul_codeProbe h hg0 hg2 hK hsurj hc hflat hpos hlam
      hmin) hX

omit [Fintype C] [Nontrivial C] [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- **The trained products of `a³ + a b` converge in direction to its kernel exactly when
every cell is seen** (`review/flat_spectrum.md`, item XE11): the kernel `ψ(a³ + a b)` on
`ℤ/p`, `p ≥ 3`, with the code `ψ`. -/
theorem tendsto_inv_smul_cubicKernel_iff_of_isNuclearMin {p : ℕ} [NeZero p] (hp : 3 ≤ p)
    (cell : S → ZMod p × ZMod p) {X : ℕ → Matrix (ZMod p) (ZMod p) ℂ}
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss (codeProbe (ZMod.stdAddChar : ZMod p → ℂ) cell)
      fun s => (cell s).1 ^ 3 + (cell s).1 * (cell s).2) (lams j) (X j)) :
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop
      (𝓝 ((nuclearNorm (cubicKernel p))⁻¹ • cubicKernel p)) ↔ Function.Surjective cell :=
  forall_seq_isNuclearMin (P := fun X =>
      Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop
        (𝓝 ((nuclearNorm (cubicKernel p))⁻¹ • cubicKernel p)) ↔ Function.Surjective cell)
    (fun j => (hpos j).le)
    (fun _ _ hmin => tendsto_inv_smul_cubicKernel_iff hp cell hpos hlam hmin) hX

/-! The witnesses: trained products of `signProbe`, where `1` is the only product of the unit
ball with the largest margin, and of the kernel of `a³ + a b` for `p = 3` with all nine cells
seen, along `λ_j = 1 / (j + 1)`. -/

example : ∃ X : ℕ → Matrix (Fin 1) (Fin 1) ℂ,
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 1) ∧
      (nuclearNorm (1 : Matrix (Fin 1) (Fin 1) ℂ) = 1 ∧
        margin signProbe (fun _ => true) (1 : Matrix (Fin 1) (Fin 1) ℂ)
          = maxMargin signProbe fun _ => true) ∧
      Tendsto (fun j => (margin signProbe (fun _ => true) (X j))⁻¹ • X j) atTop
        (𝓝 ((maxMargin signProbe fun _ => true)⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℂ))) := by
  obtain ⟨X, h⟩ := exists_seq_isNuclearMin_ceLoss signProbe (fun _ => true)
    (lams := fun j => 1 / ((j : ℝ) + 1)) fun j => by positivity
  have h1 := tendsto_inv_smul_of_unique_of_isNuclearMin margin_signProbe_one_pos
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h
    fun Z => eq_one_of_margin_signProbe
  exact ⟨X, h1, margin_eq_maxMargin_of_tendsto_inv_smul_of_isNuclearMin
    margin_signProbe_one_pos (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h h1, tendsto_inv_margin_smul_of_isNuclearMin
    margin_signProbe_one_pos (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h h1⟩

/-- Trained products of the kernel of `a³ + a b` for `p = 3` with every cell seen converge in
direction to the kernel, and scaled to margin one to the kernel over `1 − cos (2π/3)`. -/
example : ∃ X : ℕ → Matrix (ZMod 3) (ZMod 3) ℂ,
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop
        (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) ∧
      Tendsto (fun j => (margin (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) id)
        (fun s => s.1 ^ 3 + s.1 * s.2) (X j))⁻¹ • X j) atTop
        (𝓝 ((1 - Real.cos (2 * Real.pi / (3 : ℕ)))⁻¹ • cubicKernel 3)) := by
  obtain ⟨X, h⟩ := exists_seq_isNuclearMin_ceLoss
    (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) (id : ZMod 3 × ZMod 3 → ZMod 3 × ZMod 3))
    (fun s => s.1 ^ 3 + s.1 * s.2) (lams := fun j => 1 / ((j : ℝ) + 1)) fun j => by positivity
  exact ⟨X, (tendsto_inv_smul_cubicKernel_iff_of_isNuclearMin (by norm_num) id
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h).2
      Function.surjective_id,
    tendsto_inv_margin_smul_codeProbe_of_isNuclearMin (isMirroredCode_stdAddChar (by norm_num))
      (one_sub_cos_two_pi_div_pos (by norm_num)) (one_sub_cos_two_pi_div_lt_two (by norm_num))
      (fun _ => rfl) Function.surjective_id (by norm_num) (isFlat_cubicKernel 3)
      (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h⟩

/-- The same direction by the equivalence for a mirrored code. -/
example : ∃ X : ℕ → Matrix (ZMod 3) (ZMod 3) ℂ,
    Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop
      (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) := by
  obtain ⟨X, h⟩ := exists_seq_isNuclearMin_ceLoss
    (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) (id : ZMod 3 × ZMod 3 → ZMod 3 × ZMod 3))
    (fun s => s.1 ^ 3 + s.1 * s.2) (lams := fun j => 1 / ((j : ℝ) + 1)) fun j => by positivity
  exact ⟨X, (tendsto_inv_smul_codeProbe_iff_of_isNuclearMin
    (isMirroredCode_stdAddChar (by norm_num)) (one_sub_cos_two_pi_div_pos (by norm_num))
    (one_sub_cos_two_pi_div_lt_two (by norm_num)) (fun _ => rfl)
    (fun i j => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel])) (by norm_num)
    (isFlat_cubicKernel 3) (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h).2 Function.surjective_id⟩

end Transformer.Grokking.FlatSpectrum.Training
