module

public import Transformer.Grokking.FlatSpectrum.Training.NuclearMin
public import Transformer.Grokking.FlatSpectrum.Training.LeastNorm
public import Transformer.Grokking.FlatSpectrum.Training.SeenMargin

/-!
# Vanishing weight decay for trained products of any width

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE6 and XE9.
`MaxMarginLimit`, `LeastNorm` and `SeenMargin` state the limits of items XE6 and XE9 for
trained fits `(U_j, V_j)` of one column type. They hold for every sequence of trained products
`X_j`, the minimizers of `L + λ_j ‖·‖_S1` (`IsNuclearMin`), by `forall_seq_isNuclearMin`; so
the widths of the fits may change along the sequence
(`tendsto_margin_div_nuclearNorm_of_widths`). As `λ_j → 0⁺`:

* `‖X_j‖_S1 → ∞` (`tendsto_nuclearNorm_atTop_of_isNuclearMin`), and `θ < μ(X_j) / ‖X_j‖_S1` in
  the end for `θ` below the margin of a competitor (`eventually_lt_margin_div_of_isNuclearMin`);
* `μ(X_j) / ‖X_j‖_S1 → μ*` and `‖X_j‖_S1 / μ(X_j) → 1 / μ*`
  (`tendsto_margin_div_nuclearNorm_of_isNuclearMin`,
  `tendsto_nuclearNorm_div_margin_of_isNuclearMin`);
* the trained products pass the margin of a kernel that M does not return by a fixed `δ`
  (`exists_pos_eventually_add_le_margin_div_of_isNuclearMin`) and do not converge in direction
  to it (`not_tendsto_inv_nuclearNorm_smul_of_isNuclearMin`), as for a flat kernel that shows
  the labels and has a nonzero unseen cell
  (`not_tendsto_inv_nuclearNorm_smul_codeProbe_of_isNuclearMin`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n S C : Type*} [Fintype m] [Fintype n] [Fintype S] [Fintype C] [DecidableEq n]
  [Nonempty S] [Nontrivial C] {Ω : Set (m × n)} {A : S → C → Matrix m n ℂ} {y : S → C}
  {K X' X₀ : Matrix m n ℂ} {μ' : ℝ} {lams : ℕ → ℝ} {X : ℕ → Matrix m n ℂ}

/-- **The norm of trained products grows without bound as the weight decay vanishes**
(`review/flat_spectrum.md`, item XE6, statement 1). -/
theorem tendsto_nuclearNorm_atTop_of_isNuclearMin (hX' : nuclearNorm X' ≤ 1) (hμ' : 0 < μ')
    (hgap : ∀ s c, c ≠ y s → μ' ≤ gap A y X' s c) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0)) (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j)) :
    Tendsto (fun j => nuclearNorm (X j)) atTop atTop :=
  forall_seq_isNuclearMin (P := fun X => Tendsto (fun j => nuclearNorm (X j)) atTop atTop)
    (fun j => (hpos j).le) (fun _ _ h => tendsto_nuclearNorm_atTop hX' hμ' hgap hpos hlam h) hX

/-- **The margin of trained products passes that of every competitor**
(`review/flat_spectrum.md`, item XE6, statement 2). -/
theorem eventually_lt_margin_div_of_isNuclearMin (hX' : nuclearNorm X' ≤ 1) {θ : ℝ}
    (hθ0 : 0 ≤ θ) (hθ : θ < margin A y X') (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0)) (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j)) :
    ∀ᶠ j in atTop, θ < margin A y (X j) / nuclearNorm (X j) :=
  forall_seq_isNuclearMin (P := fun X => ∀ᶠ j in atTop, θ < margin A y (X j) / nuclearNorm (X j))
    (fun j => (hpos j).le) (fun _ _ h => eventually_lt_margin_div hX' hθ0 hθ hpos hlam h) hX

/-- **The normalized margin of trained products converges to the largest margin**
(`review/flat_spectrum.md`, item XE6, statement 3). -/
theorem tendsto_margin_div_nuclearNorm_of_isNuclearMin (hsep : 0 < margin A y X₀)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j)) :
    Tendsto (fun j => margin A y (X j) / nuclearNorm (X j)) atTop (𝓝 (maxMargin A y)) :=
  forall_seq_isNuclearMin (P := fun X =>
      Tendsto (fun j => margin A y (X j) / nuclearNorm (X j)) atTop (𝓝 (maxMargin A y)))
    (fun j => (hpos j).le) (fun _ _ h => tendsto_margin_div_nuclearNorm hsep hpos hlam h) hX

/-- **The norm of trained products for their margin tends to the least norm at margin one**
(`review/flat_spectrum.md`, item XE6, statement 4). -/
theorem tendsto_nuclearNorm_div_margin_of_isNuclearMin (hsep : 0 < margin A y X₀)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j)) :
    Tendsto (fun j => nuclearNorm (X j) / margin A y (X j)) atTop (𝓝 (maxMargin A y)⁻¹) :=
  forall_seq_isNuclearMin (P := fun X =>
      Tendsto (fun j => nuclearNorm (X j) / margin A y (X j)) atTop (𝓝 (maxMargin A y)⁻¹))
    (fun j => (hpos j).le) (fun _ _ h => tendsto_nuclearNorm_div_margin hsep hpos hlam h) hX

/-- **The widths of the trained fits may change along the sequence**
(`review/flat_spectrum.md`, item XE6, statement 3): for trained fits `(U_j, V_j)` of the
widths `r_j`, `μ(X_j) / ‖X_j‖_S1 → μ*`. -/
theorem tendsto_margin_div_nuclearNorm_of_widths {r : ℕ → ℕ}
    {Us : ∀ j, Matrix m (Fin (r j)) ℂ} {Vs : ∀ j, Matrix n (Fin (r j)) ℂ}
    (hsep : 0 < margin A y X₀) (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => margin A y (Us j * (Vs j)ᴴ) / nuclearNorm (Us j * (Vs j)ᴴ)) atTop
      (𝓝 (maxMargin A y)) :=
  tendsto_margin_div_nuclearNorm_of_isNuclearMin hsep hpos hlam fun j =>
    (hmin j).isNuclearMin (hpos j).le

/-- **The trained products pass the margin of a kernel that M does not return**
(`review/flat_spectrum.md`, item XE9), by a `δ > 0` chosen before the sequence. -/
theorem exists_pos_eventually_add_le_margin_div_of_isNuclearMin
    (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0) (hK : 0 < margin A y K)
    (hnot : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {lams : ℕ → ℝ} {X : ℕ → Matrix m n ℂ}, (∀ j, 0 < lams j) →
      Tendsto lams atTop (𝓝 0) → (∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j)) →
        ∀ᶠ j in atTop, margin A y K / nuclearNorm K + δ ≤
          margin A y (X j) / nuclearNorm (X j) := by
  obtain ⟨δ, hδ, h⟩ := exists_pos_eventually_add_le_margin_div (k := n) hA hK hnot
  exact ⟨δ, hδ, fun {lams X} hpos hlam hX => forall_seq_isNuclearMin
    (P := fun X => ∀ᶠ j in atTop, margin A y K / nuclearNorm K + δ ≤
      margin A y (X j) / nuclearNorm (X j))
    (fun j => (hpos j).le) (fun _ _ hmin => h hpos hlam hmin) hX⟩

/-- **The trained products do not converge in direction to a kernel that M does not return**
(`review/flat_spectrum.md`, item XE9). -/
theorem not_tendsto_inv_nuclearNorm_smul_of_isNuclearMin
    (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0) (hK : 0 < margin A y K)
    (hnot : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss A y) (lams j) (X j)) :
    ¬ Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)) :=
  forall_seq_isNuclearMin (P := fun X =>
      ¬ Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)))
    (fun j => (hpos j).le)
    (fun _ _ h => not_tendsto_inv_nuclearNorm_smul hA hK hnot hpos hlam h) hX

/-- **The trained products do not converge in direction to a flat kernel that shows the
labels** (`review/flat_spectrum.md`, item XE9 with R2 and XE8), when a nonzero cell of the
kernel is unseen: for an injective unimodular answer code. -/
theorem not_tendsto_inv_nuclearNorm_smul_codeProbe_of_isNuclearMin [DecidableEq m] {φ : C → ℂ}
    {cell : S → m × n} {c₀ : ℝ} {i₀ : m} {j₀ : n} (hφ : ∀ c, ‖φ c‖ = 1)
    (hinj : Function.Injective φ) (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hc : 0 < c₀)
    (hflat : IsFlat K c₀) (hij : (i₀, j₀) ∉ Set.range cell) (hKij : K i₀ j₀ ≠ 0)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hX : ∀ j, IsNuclearMin (ceLoss (codeProbe φ cell) y) (lams j) (X j)) :
    ¬ Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)) :=
  not_tendsto_inv_nuclearNorm_smul_of_isNuclearMin
    (fun s c _ _ h => codeProbe_apply_of_notMem h s c) (margin_codeProbe_pos hφ hinj hK)
    (hflat.not_isMinOn_of_ne_zero hc hij hKij) hpos hlam hX

/-! The witnesses: trained products of `signProbe` along `λ_j = 1 / (j + 1)`, trained fits of
it of the widths `j + 1`, and the kernels `(1, 1)` and `ψ(a³ + a b)`, `p = 3`, of item XE9. -/

example : ∃ X : ℕ → Matrix (Fin 1) (Fin 1) ℂ,
    Tendsto (fun j => nuclearNorm (X j)) atTop atTop ∧
      (∀ᶠ j in atTop, 1 < margin signProbe (fun _ => true) (X j) / nuclearNorm (X j)) ∧
      Tendsto (fun j => margin signProbe (fun _ => true) (X j) / nuclearNorm (X j)) atTop
        (𝓝 2) ∧
      Tendsto (fun j => nuclearNorm (X j) / margin signProbe (fun _ => true) (X j)) atTop
        (𝓝 2⁻¹) := by
  obtain ⟨X, h⟩ := exists_seq_isNuclearMin_ceLoss signProbe (fun _ => true)
    (lams := fun j => 1 / ((j : ℝ) + 1)) fun j => by positivity
  have h3 := tendsto_margin_div_nuclearNorm_of_isNuclearMin margin_signProbe_one_pos
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h
  have h4 := tendsto_nuclearNorm_div_margin_of_isNuclearMin margin_signProbe_one_pos
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h
  rw [maxMargin_signProbe] at h3 h4
  exact ⟨X, tendsto_nuclearNorm_atTop_of_isNuclearMin nuclearNorm_one_fin_one.le two_pos
    le_gap_signProbe_one (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h,
    eventually_lt_margin_div_of_isNuclearMin nuclearNorm_one_fin_one.le zero_le_one
      (one_lt_two.trans_le (le_margin le_gap_signProbe_one)) (fun j => by positivity)
      tendsto_one_div_add_atTop_nhds_zero_nat h, h3, h4⟩

/-- Trained fits of the widths `j + 1`. -/
example : ∃ (Us Vs : ∀ j : ℕ, Matrix (Fin 1) (Fin (j + 1)) ℂ),
    Tendsto (fun j => margin signProbe (fun _ => true) (Us j * (Vs j)ᴴ)
      / nuclearNorm (Us j * (Vs j)ᴴ)) atTop (𝓝 (maxMargin signProbe fun _ => true)) := by
  choose Us Vs h using fun j : ℕ => exists_isDecayedMin_ceLoss (k := Fin (j + 1)) signProbe
    (fun _ => true) (lam := 1 / ((j : ℝ) + 1)) (by positivity) (Or.inl (by simp))
  exact ⟨Us, Vs, tendsto_margin_div_nuclearNorm_of_widths (r := fun j => j + 1)
    margin_signProbe_one_pos (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h⟩

example := exists_pos_eventually_add_le_margin_div_of_isNuclearMin
  (A := codeProbe signCode fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2))) (y := fun _ => true)
  (fun s c _ _ h => codeProbe_apply_of_notMem h s c)
  (margin_codeProbe_pos (K := Matrix.of ![![(1 : ℂ), 1]]) norm_signCode injective_signCode
    fun _ => by simp [signCode])
  (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (i := 0) (j := 1) (by simp) (by simp))

/-- Trained products of `(1, 1)` from its first cell do not converge in direction to it. -/
example : ∃ X : ℕ → Matrix (Fin 1) (Fin 2) ℂ,
    ¬ Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop
      (𝓝 ((nuclearNorm (Matrix.of ![![(1 : ℂ), 1]]))⁻¹ • Matrix.of ![![(1 : ℂ), 1]])) := by
  obtain ⟨X, h⟩ := exists_seq_isNuclearMin_ceLoss
    (codeProbe signCode fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2))) (fun _ => true)
    (lams := fun j => 1 / ((j : ℝ) + 1)) fun j => by positivity
  exact ⟨X, not_tendsto_inv_nuclearNorm_smul_of_isNuclearMin
    (fun s c _ _ h => codeProbe_apply_of_notMem h s c)
    (margin_codeProbe_pos norm_signCode injective_signCode fun _ => by simp [signCode])
    (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (i := 0) (j := 1) (by simp) (by simp))
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h⟩

/-- Trained products of the kernel of `a³ + a b` for `p = 3` from the cell `(0, 0)`, with the
code `ψ`, do not converge in direction to it. -/
example : ∃ X : ℕ → Matrix (ZMod 3) (ZMod 3) ℂ,
    ¬ Tendsto (fun j => (nuclearNorm (X j))⁻¹ • X j) atTop
      (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) := by
  obtain ⟨X, h⟩ := exists_seq_isNuclearMin_ceLoss
    (codeProbe (ZMod.stdAddChar : AddChar (ZMod 3) ℂ) fun _ : Fin 1 => ((0 : ZMod 3),
      (0 : ZMod 3))) (fun _ => 0) (lams := fun j => 1 / ((j : ℝ) + 1)) fun j => by positivity
  exact ⟨X, not_tendsto_inv_nuclearNorm_smul_codeProbe_of_isNuclearMin
    (fun c => AddChar.norm_apply _ c) ZMod.injective_stdAddChar
    (fun _ => by simp [cubicKernel, taskKernel]) (by norm_num) (isFlat_cubicKernel 3)
    (i₀ := 1) (j₀ := 0) (by simp) (norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h⟩

end Transformer.Grokking.FlatSpectrum.Training
