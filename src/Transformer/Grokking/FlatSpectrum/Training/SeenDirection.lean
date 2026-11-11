module

public import Transformer.Grokking.FlatSpectrum.Training.MirroredUnique

/-!
# Cross-entropy training returns a flat kernel in direction exactly when every cell is seen

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE11.
Let the answer code be mirrored at a gap `0 < g < 2`, and let the flat kernel `K` show the
labels of the seen cells. As the weight decay vanishes:

* with every cell seen, the trained fits converge in direction to `K`,
  `X_j / ‖X_j‖_S1 → K / ‖K‖_S1` (`tendsto_inv_smul_codeProbe`): by `MirroredUnique` the
  normalized kernel is the only product of largest margin, and item XE10 applies;
* if `K` has no zero entry, they converge in direction to `K` exactly when every cell is seen
  (`tendsto_inv_smul_codeProbe_iff`): an unseen cell excludes the limit by item XE9.

For the kernel `ψ(a³ + a b)` of the note with the code `ψ` and `p ≥ 3`, the trained fits
converge in direction to the kernel exactly when the samples see every cell
(`tendsto_inv_smul_cubicKernel_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] [Fintype S] [Fintype C] [Nonempty S] [Nontrivial C] {φ : C → ℂ}
  {cell : S → m × n} {y : S → C} {K : Matrix m n ℂ} {g c₀ : ℝ} {lams : ℕ → ℝ}
  {Us : ℕ → Matrix m k ℂ} {Vs : ℕ → Matrix n k ℂ}

/-- **With every cell seen, cross-entropy training converges in direction to the flat kernel**
(`review/flat_spectrum.md`, item XE11): for a code mirrored at a gap `0 < g < 2` and a flat
`K` that shows the labels, `X_j / ‖X_j‖_S1 → K / ‖K‖_S1` as `λ_j → 0⁺`. -/
theorem tendsto_inv_smul_codeProbe (h : IsMirroredCode φ g) (hg0 : 0 < g) (hg2 : g < 2)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hsurj : Function.Surjective cell)
    (hc : 0 < c₀) (hflat : IsFlat K c₀) (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe φ cell) y) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm K)⁻¹ • K)) :=
  tendsto_inv_smul_of_unique (X₀ := K) ((margin_codeProbe_eq h hK).symm ▸ hg0) hpos hlam hmin
    fun _ h1 h2 => eq_inv_smul_of_margin_codeProbe_eq h hg0 hK hsurj hc hflat h1 hg2 h2

/-- **Cross-entropy training converges in direction to a flat kernel exactly when every cell
is seen** (`review/flat_spectrum.md`, item XE11 with XE9): for a code mirrored at a gap
`0 < g < 2` and a flat `K` without zero entries that shows the labels of the seen cells. -/
theorem tendsto_inv_smul_codeProbe_iff (h : IsMirroredCode φ g) (hg0 : 0 < g) (hg2 : g < 2)
    (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s)) (hK0 : ∀ i j, K i j ≠ 0) (hc : 0 < c₀)
    (hflat : IsFlat K c₀) (hpos : ∀ j, 0 < lams j) (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe φ cell) y) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm K)⁻¹ • K)) ↔ Function.Surjective cell := by
  refine ⟨fun hlim => ?_, fun hsurj =>
    tendsto_inv_smul_codeProbe h hg0 hg2 hK hsurj hc hflat hpos hlam hmin⟩
  by_contra hns
  obtain ⟨⟨i, j⟩, hij⟩ := not_forall.1 hns
  exact not_tendsto_inv_nuclearNorm_smul_of_codeProbe h.1 (h.injective hg0) hK hc hflat
    (i := i) (j := j) (fun ⟨s, hs⟩ => hij ⟨s, hs⟩) (hK0 i j) hpos hlam hmin hlim

omit [Fintype C] [Nontrivial C] [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- **The trained fits of `a³ + a b` converge in direction to its kernel exactly when every
cell is seen** (`review/flat_spectrum.md`, item XE11): the kernel `ψ(a³ + a b)` on `ℤ/p`,
`p ≥ 3`, with the code `ψ`, samples that see the cells `cell s`, and `λ_j → 0⁺`. -/
theorem tendsto_inv_smul_cubicKernel_iff {p : ℕ} [NeZero p] (hp : 3 ≤ p)
    (cell : S → ZMod p × ZMod p) {Us Vs : ℕ → Matrix (ZMod p) k ℂ} (hpos : ∀ j, 0 < lams j)
    (hlam : Tendsto lams atTop (𝓝 0))
    (hmin : ∀ j, IsDecayedMin (ceLoss (codeProbe (ZMod.stdAddChar : ZMod p → ℂ) cell)
      fun s => (cell s).1 ^ 3 + (cell s).1 * (cell s).2) (lams j) (Us j) (Vs j)) :
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm (cubicKernel p))⁻¹ • cubicKernel p)) ↔ Function.Surjective cell := by
  have : Fact (1 < p) := ⟨by omega⟩
  exact tendsto_inv_smul_codeProbe_iff (isMirroredCode_stdAddChar (by omega))
    (one_sub_cos_two_pi_div_pos (by omega)) (one_sub_cos_two_pi_div_lt_two hp) (fun _ => rfl)
    (fun i j => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
    (Nat.cast_pos.2 (by omega)) (isFlat_cubicKernel p) hpos hlam hmin

/-! The witness: the kernel of `a³ + a b` for `p = 3` with all nine cells seen and the code
`ψ`. Trained fits exist for every `λ_j = 1 / (j + 1)`, and they converge in direction to the
kernel. -/

/-- By the statement for a mirrored code. -/
example : ∃ Us Vs : ℕ → Matrix (ZMod 3) (ZMod 3) ℂ,
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_ceLoss (k := ZMod 3)
    (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) (id : ZMod 3 × ZMod 3 → ZMod 3 × ZMod 3))
    (fun s => s.1 ^ 3 + s.1 * s.2) (lams := fun j => 1 / ((j : ℝ) + 1))
    (fun j => by positivity) (Or.inl le_rfl)
  exact ⟨Us, Vs, tendsto_inv_smul_codeProbe (isMirroredCode_stdAddChar (by norm_num))
    (one_sub_cos_two_pi_div_pos (by norm_num)) (one_sub_cos_two_pi_div_lt_two (by norm_num))
    (fun _ => rfl) Function.surjective_id (by norm_num) (isFlat_cubicKernel 3)
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat h⟩

/-- By the equivalence for a mirrored code. -/
example : ∃ Us Vs : ℕ → Matrix (ZMod 3) (ZMod 3) ℂ,
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_ceLoss (k := ZMod 3)
    (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) (id : ZMod 3 × ZMod 3 → ZMod 3 × ZMod 3))
    (fun s => s.1 ^ 3 + s.1 * s.2) (lams := fun j => 1 / ((j : ℝ) + 1))
    (fun j => by positivity) (Or.inl le_rfl)
  exact ⟨Us, Vs, (tendsto_inv_smul_codeProbe_iff (isMirroredCode_stdAddChar (by norm_num))
    (one_sub_cos_two_pi_div_pos (by norm_num)) (one_sub_cos_two_pi_div_lt_two (by norm_num))
    (fun _ => rfl) (fun i j => norm_ne_zero_iff.mp (by simp [cubicKernel, taskKernel]))
    (by norm_num) (isFlat_cubicKernel 3) (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h).2 Function.surjective_id⟩

/-- By the equivalence for the kernel of `a³ + a b`. -/
example : ∃ Us Vs : ℕ → Matrix (ZMod 3) (ZMod 3) ℂ,
    Tendsto (fun j => (nuclearNorm (Us j * (Vs j)ᴴ))⁻¹ • (Us j * (Vs j)ᴴ)) atTop
      (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) := by
  obtain ⟨Us, Vs, h⟩ := exists_seq_isDecayedMin_ceLoss (k := ZMod 3)
    (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) (id : ZMod 3 × ZMod 3 → ZMod 3 × ZMod 3))
    (fun s => s.1 ^ 3 + s.1 * s.2) (lams := fun j => 1 / ((j : ℝ) + 1))
    (fun j => by positivity) (Or.inl le_rfl)
  exact ⟨Us, Vs, (tendsto_inv_smul_cubicKernel_iff (by norm_num) id (fun j => by positivity)
    tendsto_one_div_add_atTop_nhds_zero_nat h).2 Function.surjective_id⟩

end Transformer.Grokking.FlatSpectrum.Training
