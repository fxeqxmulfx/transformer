module

public import Transformer.Grokking.FlatSpectrum.ProductLoopPeri
public import Transformer.Grokking.FlatSpectrum.ProductLoopGateParity

/-!
# One Peri-normalized pass where products need two

Source: `review/flat_spectrum.md`, *Loops*, item LP6. LP5's three copies of `x y` over `ZMod 2`,
with one more scratch slot (`PeriState`). The prelude is one leaf layer: leaf `i` gets
`cᵢ = [xᵢ yᵢ = 1]`, the root and the extra slot get `0` (`periPrelude`). The core is one Peri
layer with the ideal RMS and two units (`periCore`): `s (s − 3)` into the extra slot with weight
`2`, and `1 · (2s − 3)` into the root with weight `3`, where `s = ∑ᵢ cᵢ`. The branch
`(2s(s − 3), 3(2s − 3))` is a Pythagorean pair of norm `2s² − 6s + 9` (`peri_pythagoras`), so the
core adds `c √(2n) · 3(2s − 3)/(2s² − 6s + 9)` to the root, `n` its width. The readout
`6 · 1 − 4s + 5 root/(c √(2n))` then holds `(−1)^s = ψ(∑ᵢ xᵢ yᵢ)` after one pass
(`periReadout_loopState`), by `(−1)^s = 6 − 4s + 15 (2s − 3)/(2s² − 6s + 9)` for `s ≤ 3`
(`neg_one_pow_eq_peri`).

No product core does it (`not_productLoop_parity`), so the core is no product layer of any width
(`not_isProductNet_periCore`). The root `3(2s − 3)` of the branch has order `2`, and the output
RMS multiplies it by `1/‖B z‖ = 1/(2s² − 6s + 9)`, symmetric under `s ↦ 3 − s`, so without a
third difference and of order at most `4`; their product reads all six operands.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

/-- **A normalized direction and a line give the parity**:
`(−1)^s = 6 − 4s + 15 (2s − 3)/(2s² − 6s + 9)` for `s ≤ 3`. -/
theorem neg_one_pow_eq_peri {s : ℕ} (hs : s ≤ 3) :
    ((-1 : ℂ) ^ s) = 6 - 4 * (s : ℂ) + 15 * (2 * s - 3) / (2 * s ^ 2 - 6 * s + 9) := by
  interval_cases s <;> norm_num

example := neg_one_pow_eq_peri (s := 3) le_rfl

/-- The branch `(2s(s − 3), 3(2s − 3))` has norm `2s² − 6s + 9`. -/
theorem peri_pythagoras (s : ℝ) :
    (2 * s * (s - 3)) ^ 2 + (3 * (2 * s - 3)) ^ 2 = (2 * s ^ 2 - 6 * s + 9) ^ 2 := by ring

/-- Two distinct unit slots: the energy of `a e₁ + b e₂` is `|a|² + |b|²`. -/
lemma stateEnergy_single_add {κ : Type*} [Fintype κ] [DecidableEq κ] {c₁ c₂ : κ} (h : c₁ ≠ c₂)
    (a b : ℂ) :
    stateEnergy (a • Pi.single c₁ 1 + b • Pi.single c₂ 1 : κ → ℂ) = ‖a‖ ^ 2 + ‖b‖ ^ 2 := by
  rw [stateEnergy, Fintype.sum_eq_add c₁ c₂ h fun c hc => by simp [hc.1, hc.2]]
  simp [h, h.symm]

example := stateEnergy_single_add (κ := Bool) (by decide : true ≠ false) 1 1

/-- A zero message adds no energy. -/
lemma stateEnergy_elim_zero {κ μ : Type*} [Fintype κ] [Fintype μ] (v : κ → ℂ) :
    stateEnergy (Sum.elim v (0 : μ → ℂ)) = stateEnergy v := by
  simp [stateEnergy, Fintype.sum_sum_type]

/-- LP5's leaves `some i` and root `none`, and one more slot `()`. -/
abbrev PeriScratch : Type := Option (Fin 3) ⊕ Unit

/-- The loop's state: the features and the scratch. -/
abbrev PeriState : Type := ParityFeature ⊕ PeriScratch

/-- The root, which the readout reads. -/
abbrev periRoot : PeriState := Sum.inr (Sum.inl none)

/-- The extra slot, which only carries part of the branch's norm. -/
abbrev periExtra : PeriState := Sum.inr (Sum.inr ())

/-- The pair that scratch slot `t` reads: LP5's, and pair `0` for the extra slot. -/
def periPair : PeriScratch → Fin 3 := Sum.elim parityPair fun _ => 0

/-- What the prelude writes: LP5's values, and `0` into the extra slot. -/
def periValue : PeriScratch → ZMod 2 → ZMod 2 → ℂ := Sum.elim parityValue fun _ _ _ => 0

/-- **The prelude**: one leaf layer, with `10` units. -/
def periPrelude : (PeriState → ℂ) → PeriState → ℂ := leafLayer periPair periValue

theorem isProductNet_periPrelude : IsProductNet 10 1 periPrelude :=
  IsProductNet.of_layer (by simpa [periPrelude] using isProductLayer_leafLayer periPair periValue)

lemma periPrelude_leaf (x y : Fin 3 → ZMod 2) (i : Fin 3) :
    periPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inr (Sum.inl (some i))) =
      ((x i * y i).val : ℂ) :=
  leafLayer_features periPair periValue x y (Sum.inl (some i))

lemma periPrelude_root (x y : Fin 3 → ZMod 2) :
    periPrelude (Sum.elim (pairFeatures x y) 0) periRoot = 0 :=
  leafLayer_features periPair periValue x y (Sum.inl none)

lemma periPrelude_feature (z : PeriState → ℂ) (f : ParityFeature) :
    periPrelude z (Sum.inl f) = z (Sum.inl f) := by
  simp [periPrelude, leafLayer, Finset.sum_apply]

variable (ν : Type)

/-- `s = ∑ᵢ cᵢ`, the sum of the leaves. -/
def periSum : (PeriState ⊕ ν → ℂ) →ₗ[ℂ] ℂ :=
  ∑ i : Fin 3, LinearMap.proj (Sum.inl (Sum.inr (Sum.inl (some i))))

/-- The constant `1`: the one-hot features of `x₀` sum to one. -/
def periOne : (PeriState ⊕ ν → ℂ) →ₗ[ℂ] ℂ :=
  ∑ a : ZMod 2, LinearMap.proj (Sum.inl (Sum.inl (Sum.inl (0, a))))

/-- The first readings of the core's units: `s` and `1`. -/
def periGate : Fin 2 → (PeriState ⊕ ν → ℂ) →ₗ[ℂ] ℂ := ![periSum ν, periOne ν]

/-- The second readings of the core's units: `s − 3` and `2s − 3`. -/
def periUp : Fin 2 → (PeriState ⊕ ν → ℂ) →ₗ[ℂ] ℂ :=
  ![periSum ν - (3 : ℂ) • periOne ν, (2 : ℂ) • periSum ν - (3 : ℂ) • periOne ν]

/-- The outputs of the core's units: weight `2` into the extra slot, `3` into the root. -/
def periOut : Fin 2 → PeriState ⊕ ν → ℂ :=
  ![(2 : ℂ) • Sum.elim (Pi.single periExtra 1) 0, (3 : ℂ) • Sum.elim (Pi.single periRoot 1) 0]

/-- **The core**: one Peri layer with the ideal RMS and two units, `s (s − 3)` into the extra slot
with weight `2` and `1 · (2s − 3)` into the root with weight `3`. -/
noncomputable def periCore [Fintype ν] (scale : ℝ) : (PeriState ⊕ ν → ℂ) → PeriState ⊕ ν → ℂ :=
  periLayer 0 scale (periGate ν) (periUp ν) (periOut ν)

/-- The core's branch on `u` and any message: `2S(S − 3O)` on the extra slot, `3O(2S − 3O)` on
the root; `S` sums the leaves of `u`, `O` its features of `x₀`. -/
lemma periBranch_apply {u : PeriState → ℂ} {S O : ℂ}
    (hS : ∑ i : Fin 3, u (Sum.inr (Sum.inl (some i))) = S)
    (hO : ∑ a : ZMod 2, u (Sum.inl (Sum.inl (0, a))) = O) (m : ν → ℂ) :
    productBranch (periGate ν) (periUp ν) (periOut ν) (Sum.elim u m) =
      Sum.elim ((2 * (S * (S - 3 * O))) • Pi.single periExtra 1 +
        (3 * (O * (2 * S - 3 * O))) • Pi.single periRoot 1) (0 : ν → ℂ) := by
  have hs : periSum ν (Sum.elim u m) = S := by simp [periSum, LinearMap.sum_apply, hS]
  have ho : periOne ν (Sum.elim u m) = O := by simp [periOne, LinearMap.sum_apply, hO]
  funext c
  rcases c with c | j
  · simp [productBranch, periGate, periUp, periOut, Fin.sum_univ_two, Finset.sum_apply, hs, ho]
    ring
  · simp [productBranch, periOut, Fin.sum_univ_two, Finset.sum_apply]

example (u : PeriState → ℂ) := periBranch_apply Unit (u := u) rfl rfl 0

/-- **The readout**: `w · root + 6 · 1 − 4 s`. -/
noncomputable def periReadout (w : ℂ) : (PeriState → ℂ) →ₗ[ℂ] ℂ :=
  w • LinearMap.proj periRoot + (6 : ℂ) • ∑ a : ZMod 2, LinearMap.proj (Sum.inl (Sum.inl (0, a)))
    - (4 : ℂ) • ∑ i : Fin 3, LinearMap.proj (Sum.inr (Sum.inl (some i)))

/-- **One Peri-normalized pass holds `ψ(∑ᵢ xᵢ yᵢ)` for three pairs over `ZMod 2`**, at every
nonzero depth scale `c` and whatever the message, with the root read with weight `5/(c √(2n))`. -/
theorem periReadout_loopState [Fintype ν] (scale : ℝ) (hscale : scale ≠ 0)
    (R : (PeriState → ℂ) →ₗ[ℂ] ν → ℂ) (x y : Fin 3 → ZMod 2) :
    periReadout ((5 / (scale * Real.sqrt (2 * Fintype.card (PeriState ⊕ ν))) : ℝ) : ℂ)
      (loopState (periCore ν scale) R (periPrelude (Sum.elim (pairFeatures x y) 0)) 0) =
        ZMod.stdAddChar (∑ i, x i * y i) := by
  set u := periPrelude (Sum.elim (pairFeatures x y) 0) with hu
  set s : ℕ := ∑ i, (x i * y i).val with hs
  have hN : s ≤ 3 := by
    rw [hs, Fin.sum_univ_three]
    linarith [ZMod.val_lt (x 0 * y 0), ZMod.val_lt (x 1 * y 1), ZMod.val_lt (x 2 * y 2)]
  have hleaf : ∑ i : Fin 3, u (Sum.inr (Sum.inl (some i))) = (s : ℂ) := by
    simp only [hu, periPrelude_leaf, hs, Nat.cast_sum]
  have hone : ∑ a : ZMod 2, u (Sum.inl (Sum.inl (0, a))) = 1 := by
    simp [hu, periPrelude_feature, pairFeatures, oneHot]
  have hroot : u periRoot = 0 := periPrelude_root x y
  have hB := periBranch_apply ν hleaf hone 0
  rw [show (2 * ((s : ℂ) * (s - 3 * 1)) : ℂ) = ((2 * s * (s - 3) : ℝ) : ℂ) by push_cast; ring,
    show (3 * (1 * (2 * (s : ℂ) - 3 * 1)) : ℂ) = ((3 * (2 * s - 3) : ℝ) : ℂ) by
      push_cast; ring] at hB
  have hpos : (0 : ℝ) < 2 * s ^ 2 - 6 * s + 9 := by nlinarith [sq_nonneg ((s : ℝ) - 3 / 2)]
  have hE : stateEnergy (productBranch (periGate ν) (periUp ν) (periOut ν) (Sum.elim u 0)) =
      (2 * s ^ 2 - 6 * s + 9) ^ 2 := by
    rw [hB, stateEnergy_elim_zero (μ := ν), stateEnergy_single_add (by simp), Complex.norm_real,
      Complex.norm_real, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, sq_abs, peri_pythagoras]
  have hψ : ZMod.stdAddChar (∑ i, x i * y i) = (-1 : ℂ) ^ s := by
    rw [stdAddChar_sum, hs, ← Finset.prod_pow_eq_pow_sum]
    exact Finset.prod_congr rfl fun i _ => stdAddChar_zmod_two _
  have h2 : (0 : ℝ) < Real.sqrt (2 * Fintype.card (PeriState ⊕ ν)) :=
    Real.sqrt_pos.2 (by positivity)
  change periReadout _ (periCore ν scale (Sum.elim u 0) ∘ Sum.inl) = _
  rw [periCore, periLayer_eps_zero, hE, Real.sqrt_sq hpos.le, hB, hψ, neg_one_pow_eq_peri hN]
  generalize Real.sqrt (2 * (Fintype.card (PeriState ⊕ ν) : ℝ)) = K at h2 ⊢
  simp [periReadout, LinearMap.sum_apply, hleaf, hone, hroot]
  have hc : (scale : ℂ) ≠ 0 := by exact_mod_cast hscale
  have hr : (K : ℂ) ≠ 0 := by exact_mod_cast h2.ne'
  have hd : (2 * (s : ℂ) ^ 2 - 6 * s + 9) ≠ 0 := by exact_mod_cast hpos.ne'
  field_simp
  ring

/-- Three products equal to one: the readout is `ψ(1) = −1`. -/
example : periReadout ((5 / (1 * Real.sqrt (2 * Fintype.card (PeriState ⊕ Unit))) : ℝ) : ℂ)
    (loopState (periCore Unit 1) 0 (periPrelude (Sum.elim (pairFeatures 1 1) 0)) 0) = -1 := by
  rw [periReadout_loopState Unit 1 one_ne_zero, stdAddChar_zmod_two,
    show (∑ i : Fin 3, (1 : Fin 3 → ZMod 2) i * (1 : Fin 3 → ZMod 2) i).val = 1 by decide, pow_one]

/-- **The Peri core is no product layer**, of any width: with it, one pass of a product core
would hold the parity, against `not_productLoop_parity`. -/
theorem not_isProductNet_periCore [Fintype ν] {m : ℕ} (scale : ℝ) (hscale : scale ≠ 0) :
    ¬ IsProductNet m 1 (periCore ν scale) := fun hL =>
  not_productLoop_parity isProductNet_periPrelude hL 0 (periReadout _)
    fun x y => periReadout_loopState ν scale hscale 0 x y

example : ¬ IsProductNet 5 1 (periCore Unit 1) := not_isProductNet_periCore Unit 1 one_ne_zero

end Transformer.Grokking.FlatSpectrum
