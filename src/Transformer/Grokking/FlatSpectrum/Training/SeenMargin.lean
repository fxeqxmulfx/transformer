module

public import Transformer.Grokking.FlatSpectrum.Training.AnswerCode
public import Transformer.Grokking.FlatSpectrum.Training.SeenCells

/-!
# Vanishing weight decay under the cross-entropy does not return a flat kernel

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE9.
Let the probes vanish off the seen cells `Ω`, so that the gaps and the margin read only `Ω`
(`gap_eq_of_seen`, `margin_eq_of_seen`). Let `K` have a positive margin and not be a minimum of
M from `Ω`: some `Y` equal to `K` on `Ω` has `‖Y‖_S1 < ‖K‖_S1`. It has the margin of `K`, so
its normalized margin is larger, and by item XE6 the trained fits pass it. There is `δ > 0`
with, for all trained fits `X_j` of weight decays `λ_j → 0⁺`,

    μ(K) / ‖K‖_S1 + δ ≤ μ(X_j) / ‖X_j‖_S1    in the end

(`exists_pos_eventually_add_le_margin_div`). One gap of `K` is its margin, and that gap of
`X_j / ‖X_j‖_S1` stays `δ` above it, so the normalized products do not converge to
`K / ‖K‖_S1` (`not_tendsto_inv_nuclearNorm_smul`).

By items R2 and XE8 this holds for the kernel of the note: a flat `K` that shows the labels
with an injective unimodular answer code, when a nonzero cell is unseen
(`exists_pos_eventually_add_le_margin_div_of_codeProbe`,
`not_tendsto_inv_nuclearNorm_smul_of_codeProbe`). The witnesses are the `1 × 2` kernel `(1, 1)`
with the code `±1` on its first cell and the kernel of `a³ + a b` for `p = 3` with the code `ψ`
on the cell `(0, 0)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] {Ω : Set (m × n)}
  {A : S → C → Matrix m n ℂ} {y : S → C} {K X Y : Matrix m n ℂ} {lams : ℕ → ℝ}
  {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ}

/-- The gaps of probes that vanish off `Ω` read only the cells of `Ω`. -/
lemma gap_eq_of_seen (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0)
    (h : ∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) (s : S) (c : C) :
    gap A y X s c = gap A y Y s c := by
  rw [gap, gap, pairing_eq_of_seen (hA s (y s)) h, pairing_eq_of_seen (hA s c) h]

/-- The margin of probes that vanish off `Ω` reads only the cells of `Ω`. -/
lemma margin_eq_of_seen (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0)
    (h : ∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) : margin A y X = margin A y Y := by
  simp only [margin, gap_eq_of_seen hA h]

variable [Fintype S] [Fintype C] [DecidableEq n]

/-- **The trained fits pass the margin of a kernel that M does not return**
(`review/flat_spectrum.md`, item XE9): if the probes vanish off `Ω`, `K` has a positive margin
and `K` is not a minimum of M from `Ω`, some `δ > 0` has: for all trained fits `X_j` of the
cross-entropy with `λ_j → 0⁺`, `μ(K) / ‖K‖_S1 + δ ≤ μ(X_j) / ‖X_j‖_S1` in the end. -/
theorem exists_pos_eventually_add_le_margin_div [Nonempty S] [Nontrivial C]
    (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0) (hK : 0 < margin A y K)
    (hnot : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {lams : ℕ → ℝ} {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ},
      (∀ j, 0 < lams j) → Tendsto lams atTop (𝓝 0) →
      (∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) →
        ∀ᶠ j in atTop, margin A y K / nuclearNorm K + δ ≤
          margin A y (Us j * (Vs j)ᴴ) / nuclearNorm (Us j * (Vs j)ᴴ) := by
  rw [isMinOn_iff] at hnot
  push Not at hnot
  obtain ⟨Y, hY, hlt⟩ := hnot
  have hmY : margin A y Y = margin A y K := margin_eq_of_seen hA hY
  have hY0 : 0 < nuclearNorm Y := nuclearNorm_pos_of_margin_pos (hmY.symm ▸ hK)
  have h1 := div_lt_div_of_pos_left hK hY0 hlt
  have h2 := div_pos hK (hY0.trans hlt)
  refine ⟨(margin A y K / nuclearNorm Y - margin A y K / nuclearNorm K) / 2, by linarith,
    fun hpos hlam hmin => ?_⟩
  have hX' : nuclearNorm ((nuclearNorm Y)⁻¹ • Y) ≤ 1 := by
    rw [nuclearNorm_real_smul (inv_nonneg.mpr hY0.le), inv_mul_cancel₀ hY0.ne']
  have hm : margin A y ((nuclearNorm Y)⁻¹ • Y) = margin A y K / nuclearNorm Y := by
    rw [margin_smul (inv_nonneg.mpr hY0.le), hmY, inv_mul_eq_div]
  filter_upwards [eventually_lt_margin_div hX'
    (θ := (margin A y K / nuclearNorm K + margin A y K / nuclearNorm Y) / 2) (by linarith)
    (by rw [hm]; linarith) hpos hlam hmin] with j hj
  linarith

/-- **The trained fits do not converge in direction to a kernel that M does not return**
(`review/flat_spectrum.md`, item XE9): under the hypotheses of
`exists_pos_eventually_add_le_margin_div`, `X_j / ‖X_j‖_S1` does not tend to `K / ‖K‖_S1`. -/
theorem not_tendsto_inv_nuclearNorm_smul [Nonempty S] [Nontrivial C]
    (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0) (hK : 0 < margin A y K)
    (hnot : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K)
    (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss A y) (lams j) (Us j) (Vs j)) :
    ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm K)⁻¹ • K)) := by
  obtain ⟨δ, hδ, hev⟩ := exists_pos_eventually_add_le_margin_div (k := k) hA hK hnot
  obtain ⟨s, c, hc, hg⟩ := exists_gap_eq_margin A y K
  intro hlim
  have hcont : Continuous fun X : Matrix m n ℂ => gap A y X s c :=
    (continuous_pairing continuous_const continuous_id).sub
      (continuous_pairing continuous_const continuous_id)
  have hev' : ∀ᶠ j in atTop, margin A y K / nuclearNorm K + δ ≤
      gap A y ((nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) s c := by
    filter_upwards [hev hpos hlam hmin] with j hj
    rw [gap_smul, inv_mul_eq_div]
    exact hj.trans (div_le_div_of_nonneg_right (margin_le_gap _ hc) (nuclearNorm_nonneg _))
  have h1 := ge_of_tendsto ((hcont.tendsto _).comp hlim) hev'
  rw [gap_smul, hg, inv_mul_eq_div] at h1
  linarith

variable [DecidableEq m] {φ : C → ℂ} {cell : S → m × n} {c₀ : ℝ} {i : m} {j : n}

/-- **Cross-entropy training passes the margin of a flat kernel that shows the labels**
(`review/flat_spectrum.md`, item XE9 with R2 and XE8): for an injective unimodular answer code
and a flat `K` that shows the labels on the seen cells and has a nonzero unseen cell. -/
theorem exists_pos_eventually_add_le_margin_div_of_codeProbe [Nonempty S] [Nontrivial C]
    (hφ : ∀ c, ‖φ c‖ = 1) (hinj : Function.Injective φ)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hij : (i, j) ∉ Set.range cell) (hKij : K i j ≠ 0) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ {lams : ℕ → ℝ} {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ},
      (∀ j, 0 < lams j) → Tendsto lams atTop (𝓝 0) →
      (∀ j, IsDecayedMin (ceLoss (codeProbe φ cell) y) (lams j) (Us j) (Vs j)) →
        ∀ᶠ j in atTop, margin (codeProbe φ cell) y K / nuclearNorm K + δ ≤
          margin (codeProbe φ cell) y (Us j * (Vs j)ᴴ) / nuclearNorm (Us j * (Vs j)ᴴ) :=
  exists_pos_eventually_add_le_margin_div (fun s c _ _ h => codeProbe_apply_of_notMem h s c)
    (margin_codeProbe_pos hφ hinj hK) (hflat.not_isMinOn_of_ne_zero hc hij hKij)

/-- **Cross-entropy training does not converge in direction to a flat kernel that shows the
labels** (`review/flat_spectrum.md`, item XE9 with R2 and XE8), when a nonzero cell of the
kernel is unseen and the weight decay vanishes. -/
theorem not_tendsto_inv_nuclearNorm_smul_of_codeProbe [Nonempty S] [Nontrivial C]
    (hφ : ∀ c, ‖φ c‖ = 1) (hinj : Function.Injective φ)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hij : (i, j) ∉ Set.range cell) (hKij : K i j ≠ 0) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe φ cell) y) (lams j) (Us j) (Vs j)) :
    ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm K)⁻¹ • K)) :=
  not_tendsto_inv_nuclearNorm_smul (fun s c _ _ h => codeProbe_apply_of_notMem h s c)
    (margin_codeProbe_pos hφ hinj hK) (hflat.not_isMinOn_of_ne_zero hc hij hKij) hpos hlam hmin

/-! The witnesses. For `(1, 1)` the first cell is seen and labeled `true`; for the kernel of
`a³ + a b` with `p = 3` the cell `(0, 0)` is seen and labeled `0`, and `(1, 0)` is unseen. -/

example := gap_eq_of_seen (Ω := Set.univ) (A := signProbe) (y := fun _ => true)
  (fun _ _ _ _ h => absurd (Set.mem_univ _) h) (X := 1) (Y := 1) (fun _ _ => rfl) 0 false

example := margin_eq_of_seen (Ω := Set.univ) (A := signProbe) (y := fun _ => true)
  (fun _ _ _ _ h => absurd (Set.mem_univ _) h) (X := 1) (Y := 1) fun _ _ => rfl

example := exists_pos_eventually_add_le_margin_div (k := Fin 1)
  (A := codeProbe signCode fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2))) (y := fun _ => true)
  (fun s c _ _ h => codeProbe_apply_of_notMem h s c)
  (margin_codeProbe_pos (K := Matrix.of ![![(1 : ℂ), 1]]) norm_signCode injective_signCode
    fun _ => by simp [signCode])
  (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (i := 0) (j := 1) (by simp) (by simp))

example := exists_pos_eventually_add_le_margin_div_of_codeProbe (k := Fin 1)
  (cell := fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2))) (y := fun _ => true) norm_signCode
  injective_signCode (K := Matrix.of ![![(1 : ℂ), 1]]) (fun _ => by simp [signCode]) two_pos
  isFlat_row_ones (i := 0) (j := 1) (by simp) (by simp)

/-- Trained fits of `(1, 1)` from its first cell do not converge in direction to it. -/
example : ∃ (Us : ℕ → Matrix (Fin 1) (Fin 1) ℂ) (Vs : ℕ → Matrix (Fin 2) (Fin 1) ℂ),
    ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm (Matrix.of ![![(1 : ℂ), 1]]))⁻¹ • Matrix.of ![![(1 : ℂ), 1]])) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_ceLoss (k := Fin 1)
    (codeProbe signCode fun _ : Fin 1 => ((0 : Fin 1), (0 : Fin 2))) (fun _ => true)
    (lams := fun j => 1 / ((j : ℝ) + 1)) (fun j => by positivity) (Or.inl le_rfl)
  exact ⟨Us, Vs, not_tendsto_inv_nuclearNorm_smul
    (fun s c _ _ h => codeProbe_apply_of_notMem h s c)
    (margin_codeProbe_pos norm_signCode injective_signCode fun _ => by simp [signCode])
    (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (i := 0) (j := 1) (by simp) (by simp))
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h⟩

/-- Trained fits of the kernel of `a³ + a b` for `p = 3` from the cell `(0, 0)`, with the code
`ψ`, do not converge in direction to it. -/
example : ∃ Us Vs : ℕ → Matrix (ZMod 3) (ZMod 3) ℂ,
    ¬ Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_ceLoss (k := ZMod 3)
    (codeProbe (ZMod.stdAddChar : AddChar (ZMod 3) ℂ) fun _ : Fin 1 => ((0 : ZMod 3),
      (0 : ZMod 3))) (fun _ => 0) (lams := fun j => 1 / ((j : ℝ) + 1)) (fun j => by positivity)
    (Or.inl le_rfl)
  exact ⟨Us, Vs, not_tendsto_inv_nuclearNorm_smul_of_codeProbe
    (fun c => AddChar.norm_apply _ c) ZMod.injective_stdAddChar
    (fun _ => by simp [cubicKernel, taskKernel]) (by norm_num) (isFlat_cubicKernel 3)
    (i := 1) (j := 0) (by simp) (norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h⟩

end Transformer.Grokking.FlatSpectrum.Training
