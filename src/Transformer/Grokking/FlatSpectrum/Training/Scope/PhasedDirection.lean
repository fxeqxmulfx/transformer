module

public import Transformer.Grokking.FlatSpectrum.Training.SeenDirection
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhasedRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.UnseenCellLimit

/-!
# One run with a decreasing weight decay: the direction of its product

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE18, with the
items XE10, XE11 and XE15.

For the run in phases of `PhasedRun`, whose weight decay passes through a positive sequence
`λ_j → 0`, on samples that some product classifies with a positive margin and from stacked
factors of full row rank:

* if one product `Z₀` of the unit ball of the nuclear norm has the largest margin, there are
  lengths of the phases for which the products at the phase ends converge in direction to
  it, `X_j / ‖X_j‖_S1 → Z₀` (`exists_tendsto_inv_smul_phaseProduct`);
* for an answer code mirrored at a gap `0 < g < 2` and a flat kernel `K` that shows the
  labels, with every cell seen, that direction is the kernel's, `X_j / ‖X_j‖_S1 → K / ‖K‖_S1`
  (`exists_tendsto_inv_smul_phaseProduct_codeProbe`);
* in particular for the kernel `ψ(a³ + a b)` on `ℤ/p`, `p ≥ 3`, with the code `ψ` and every
  cell seen (`exists_tendsto_inv_smul_phaseProduct_cubicKernel`);
* for the kernel of signs `[[1, 1], [−1, 1]]` with the code `±1` and every cell seen but
  `(1, 1)`, the products at all late phase ends decode the unseen cell to the class of `−1`,
  where the kernel shows `+1` (`exists_eventually_phaseProduct_unseen_wrong`).

So with every cell seen one run of gradient descent, whose weight decay decreases to `0`,
returns the flat kernel in direction, and with one unseen cell of the kernel of signs it
answers that cell wrongly: the statements of XE11 and XE15 for a single run, at its phase
ends.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m]
  [DecidableEq n] [Fintype S] [Fintype C] [Nonempty S] [Nontrivial C]
  {A : S → C → Matrix m n ℂ} {y : S → C} {X₀ : Matrix m n ℂ} {a : ℝ} {lam : ℕ → ℝ}
  {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ}

/-- **One run with a decreasing weight decay converges in direction to the product of largest
margin, when that is unique** (`review/flat_spectrum.md`, item XE18 with XE10): if `Z₀` is the
only product of the unit ball with the margin `μ*`, the run in phases has, for some lengths of
the phases, `X_j / ‖X_j‖_S1 → Z₀` at the phase ends. -/
theorem exists_tendsto_inv_smul_phaseProduct (ha0 : 0 ≤ a)
    (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2) (hsep : 0 < margin A y X₀)
    (hpos : ∀ j, 0 < lam j) (hlam : Tendsto lam atTop (𝓝 0))
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) {Z₀ : Matrix m n ℂ}
    (huniq : ∀ Z : Matrix m n ℂ, nuclearNorm Z ≤ 1 → margin A y Z = maxMargin A y → Z = Z₀) :
    ∃ T : ℕ → ℕ, Tendsto (fun j => (nuclearNorm (phaseProduct A y a lam T U₀ V₀ j))⁻¹ •
      phaseProduct A y a lam T U₀ V₀ j) atTop (𝓝 Z₀) := by
  obtain ⟨T, -, h⟩ := exists_tendsto_margin_div_phaseProduct ha0 ha hsep hpos hlam hfull
  exact ⟨T, tendsto_inv_smul_of_tendsto_margin_div h huniq⟩

variable {φ : C → ℂ} {cell : S → m × n} {K : Matrix m n ℂ} {g c₀ : ℝ}

/-- **With every cell seen, one run with a decreasing weight decay converges in direction to
the flat kernel** (`review/flat_spectrum.md`, item XE18 with XE11): for a code mirrored at a
gap `0 < g < 2` and a flat `K` that shows the labels, the run in phases has, for some lengths
of the phases, `X_j / ‖X_j‖_S1 → K / ‖K‖_S1` at the phase ends. The probes of a unimodular
code have norm `1`. -/
theorem exists_tendsto_inv_smul_phaseProduct_codeProbe (h : IsMirroredCode φ g) (hg0 : 0 < g)
    (hg2 : g < 2) (hK : ∀ s, K (cell s).1 (cell s).2 = φ (y s))
    (hsurj : Function.Surjective cell) (hc : 0 < c₀) (hflat : IsFlat K c₀)
    (hpos : ∀ j, 0 < lam j) (hlam : Tendsto lam atTop (𝓝 0))
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ T : ℕ → ℕ, Tendsto (fun j =>
      (nuclearNorm (phaseProduct (codeProbe φ cell) y 1 lam T U₀ V₀ j))⁻¹ •
        phaseProduct (codeProbe φ cell) y 1 lam T U₀ V₀ j) atTop (𝓝 ((nuclearNorm K)⁻¹ • K)) :=
  exists_tendsto_inv_smul_phaseProduct (X₀ := K) zero_le_one
    (fun s c => by rw [frobeniusSq_codeProbe, h.1 c])
    ((margin_codeProbe_eq h hK).symm ▸ hg0) hpos hlam hfull
    fun _ h1 h2 => eq_inv_smul_of_margin_codeProbe_eq h hg0 hK hsurj hc hflat h1 hg2 h2

omit [Fintype C] [Nontrivial C] [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- **One run with a decreasing weight decay returns the kernel of `a³ + a b` in direction
when every cell is seen** (`review/flat_spectrum.md`, item XE18 with XE11): the kernel
`ψ(a³ + a b)` on `ℤ/p`, `p ≥ 3`, with the code `ψ`, samples that see every cell, and the run in
phases from stacked factors of full row rank. -/
theorem exists_tendsto_inv_smul_phaseProduct_cubicKernel {p : ℕ} [NeZero p] (hp : 3 ≤ p)
    {cell : S → ZMod p × ZMod p} (hsurj : Function.Surjective cell) (hpos : ∀ j, 0 < lam j)
    (hlam : Tendsto lam atTop (𝓝 0)) {U₀ V₀ : Matrix (ZMod p) k ℂ}
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ T : ℕ → ℕ, Tendsto (fun j =>
      (nuclearNorm (phaseProduct (codeProbe (ZMod.stdAddChar : ZMod p → ℂ) cell)
        (fun s => (cell s).1 ^ 3 + (cell s).1 * (cell s).2) 1 lam T U₀ V₀ j))⁻¹ •
        phaseProduct (codeProbe (ZMod.stdAddChar : ZMod p → ℂ) cell)
          (fun s => (cell s).1 ^ 3 + (cell s).1 * (cell s).2) 1 lam T U₀ V₀ j) atTop
      (𝓝 ((nuclearNorm (cubicKernel p))⁻¹ • cubicKernel p)) := by
  have : Fact (1 < p) := ⟨by omega⟩
  exact exists_tendsto_inv_smul_phaseProduct_codeProbe (isMirroredCode_stdAddChar (by omega))
    (one_sub_cos_two_pi_div_pos (by omega)) (one_sub_cos_two_pi_div_lt_two hp) (fun _ => rfl)
    hsurj (Nat.cast_pos.2 (by omega)) (isFlat_cubicKernel p) hpos hlam hfull

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [Fintype S] [Fintype C]
  [Nonempty S] [Nontrivial C] in
/-- **With one unseen cell of the kernel of signs, one run with a decreasing weight decay
answers it with the wrong class** (`review/flat_spectrum.md`, item XE18 with XE15): for the
code `±1` and every cell of `[[1, 1], [−1, 1]]` seen but `(1, 1)`, the run in phases has, for
some lengths of the phases, at every late phase end a product whose unseen cell is nearer to
the codeword of `−1` than to that of the label `+1` that the kernel shows. -/
theorem exists_eventually_phaseProduct_unseen_wrong (hpos : ∀ j, 0 < lam j)
    (hlam : Tendsto lam atTop (𝓝 0)) {U₀ V₀ : Matrix (Fin 2) k ℂ}
    (hfull : Function.Surjective (stack U₀ V₀).mulVec) :
    ∃ T : ℕ → ℕ, ∀ᶠ j in atTop,
      ‖phaseProduct (codeProbe signCode threeCells) threeLabels 1 lam T U₀ V₀ j 1 1 -
          signCode false‖ <
        ‖phaseProduct (codeProbe signCode threeCells) threeLabels 1 lam T U₀ V₀ j 1 1 -
          signCode (signLabel (1, 1))‖ := by
  obtain ⟨T, h⟩ := exists_tendsto_inv_smul_phaseProduct
    (A := codeProbe signCode threeCells) (y := threeLabels) (a := 1) (X₀ := signFill)
    zero_le_one (fun s c => by rw [frobeniusSq_codeProbe, norm_signCode])
    (by rw [margin_signFill]; exact one_pos) hpos hlam hfull
    fun _ hZ hμ => eq_signFill_of_margin hZ hμ
  refine ⟨T, ?_⟩
  have h11 := (Complex.continuous_re.tendsto _).comp
    (((continuous_id.matrix_elem 1 1).tendsto _).comp h)
  have hneg : (signFill 1 1).re < 0 := by norm_num [signFill]
  filter_upwards [(tendsto_order.1 h11).2 0 hneg] with j hj
  rw [show signLabel (1, 1) = true by decide]
  refine norm_sub_signCode_lt ?_
  simp only [Function.comp_apply, id, Matrix.smul_apply, Complex.real_smul,
    Complex.re_ofReal_mul] at hj
  by_contra hge
  exact absurd hj (not_lt.2 (mul_nonneg (inv_nonneg.2 (nuclearNorm_nonneg _)) (not_lt.1 hge)))

/-! The witnesses, along the weight decays `1 / (j + 1)` and from the start `U₀ = [1 0]`,
`V₀ = [0 1]` of full row rank: one sample of a `1 × 1` product read with the code `±1`, where
`1` is the only product of largest margin; the kernel of `a³ + a b` for `p = 3` with all nine
cells seen and the code `ψ`; and the three seen cells of the kernel of signs. -/

/-- The run in phases of the `1 × 1` witness converges in direction to `1`. -/
example := exists_tendsto_inv_smul_phaseProduct (y := fun _ => true) zero_le_one
  frobeniusSq_signProbe_le margin_signProbe_one_pos (lam := fun j => 1 / ((j : ℝ) + 1))
  (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat
  (U₀ := fromCols (1 : Matrix (Fin 1) (Fin 1) ℂ) (0 : Matrix (Fin 1) (Fin 1) ℂ))
  (V₀ := fromCols (0 : Matrix (Fin 1) (Fin 1) ℂ) (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) fun _ => eq_one_of_margin_signProbe

/-- By the statement for a mirrored code. -/
example := exists_tendsto_inv_smul_phaseProduct_codeProbe
  (cell := (id : ZMod 3 × ZMod 3 → ZMod 3 × ZMod 3)) (y := fun s => s.1 ^ 3 + s.1 * s.2)
  (isMirroredCode_stdAddChar (by norm_num)) (one_sub_cos_two_pi_div_pos (by norm_num))
  (one_sub_cos_two_pi_div_lt_two (by norm_num)) (fun _ => rfl) Function.surjective_id
  (by norm_num) (isFlat_cubicKernel 3) (lam := fun j => 1 / ((j : ℝ) + 1))
  (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat
  (U₀ := fromCols (1 : Matrix (ZMod 3) (ZMod 3) ℂ) (0 : Matrix (ZMod 3) (ZMod 3) ℂ))
  (V₀ := fromCols (0 : Matrix (ZMod 3) (ZMod 3) ℂ) (1 : Matrix (ZMod 3) (ZMod 3) ℂ))
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

/-- The run in phases of the kernel of `a³ + a b` for `p = 3` with all nine cells seen. -/
example : ∃ T : ℕ → ℕ, Tendsto (fun j =>
    (nuclearNorm (phaseProduct (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) id)
      (fun s => s.1 ^ 3 + s.1 * s.2) 1 (fun j => 1 / ((j : ℝ) + 1)) T
        (fromCols (1 : Matrix (ZMod 3) (ZMod 3) ℂ) (0 : Matrix (ZMod 3) (ZMod 3) ℂ))
        (fromCols (0 : Matrix (ZMod 3) (ZMod 3) ℂ) (1 : Matrix (ZMod 3) (ZMod 3) ℂ)) j))⁻¹ •
      phaseProduct (codeProbe (ZMod.stdAddChar : ZMod 3 → ℂ) id)
        (fun s => s.1 ^ 3 + s.1 * s.2) 1 (fun j => 1 / ((j : ℝ) + 1)) T
        (fromCols (1 : Matrix (ZMod 3) (ZMod 3) ℂ) (0 : Matrix (ZMod 3) (ZMod 3) ℂ))
        (fromCols (0 : Matrix (ZMod 3) (ZMod 3) ℂ) (1 : Matrix (ZMod 3) (ZMod 3) ℂ)) j) atTop
    (𝓝 ((nuclearNorm (cubicKernel 3))⁻¹ • cubicKernel 3)) :=
  exists_tendsto_inv_smul_phaseProduct_cubicKernel le_rfl Function.surjective_id
    (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat
    fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

/-- The run in phases on the three seen cells of the kernel of signs. -/
example := exists_eventually_phaseProduct_unseen_wrong (lam := fun j => 1 / ((j : ℝ) + 1))
  (fun j => by positivity) tendsto_one_div_add_atTop_nhds_zero_nat
  (U₀ := fromCols (1 : Matrix (Fin 2) (Fin 2) ℂ) (0 : Matrix (Fin 2) (Fin 2) ℂ))
  (V₀ := fromCols (0 : Matrix (Fin 2) (Fin 2) ℂ) (1 : Matrix (Fin 2) (Fin 2) ℂ))
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope
