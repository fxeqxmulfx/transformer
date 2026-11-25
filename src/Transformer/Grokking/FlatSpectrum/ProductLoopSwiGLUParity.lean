module

public import Transformer.Grokking.FlatSpectrum.ProductLoopSwiGLU
public import Transformer.Grokking.FlatSpectrum.ProductLoopGateParity

/-!
# One SwiGLU pass where products need two

Source: `review/flat_spectrum.md`, *Loops*, item LP7. LP5's prelude and stream: three copies of
`x y` over `ZMod 2`, leaf `i` holding `cᵢ = [xᵢ yᵢ = 1]` and the root `0`. The core is one SwiGLU
layer of two units writing into the root (`swigluCore`), both behind `SiLU(−s ln 2)`, where
`s = ∑ᵢ cᵢ`: one with the value `1`, the sum of the features of `x₀`, and weight `6/ln 2`, one with
the value `s` and weight `−18/ln 2`. The readout `root + 1 − 6s` then holds
`(−1)^s = ψ(∑ᵢ xᵢ yᵢ)` after one pass (`swigluReadout_loopState`), by `neg_one_pow_eq_silu`.

No product core does it (`not_productLoop_parity`), so the core is no product layer of any width
(`not_isProductNet_swigluCore`). The gate's input `−s ln 2` has order `2`, and
`SiLU(−s ln 2) = −s ln 2/(1 + 2^s)`, whose third difference is `−(2/15) ln 2`, reads all six
operands.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset
open Transformer.Landscape.Neural (silu)

variable (ν : Type)

/-- The values of the core's units: `Re 1` and `Re s`. -/
noncomputable def swigluValue : Fin 2 → (ParityState ⊕ ν → ℂ) →ₗ[ℝ] ℝ :=
  ![Complex.reLm ∘ₗ (parityOne ν).restrictScalars ℝ,
    Complex.reLm ∘ₗ (paritySum ν).restrictScalars ℝ]

/-- The outputs of the core's units: weights `6/ln 2` and `−18/ln 2` into the root. -/
noncomputable def swigluOut : Fin 2 → ParityState ⊕ ν → ℂ :=
  ![(6 / Real.log 2 : ℝ) • Sum.elim (Pi.single parityRoot 1) 0,
    (-18 / Real.log 2 : ℝ) • Sum.elim (Pi.single parityRoot 1) 0]

/-- **The core**: one SwiGLU layer of two units writing into the root, both behind
`SiLU(−s ln 2)`: the value `1` with weight `6/ln 2`, and the value `s` with weight `−18/ln 2`. -/
noncomputable def swigluCore : (ParityState ⊕ ν → ℂ) → ParityState ⊕ ν → ℂ :=
  swigluLayer ![parityLogit ν, parityLogit ν] (swigluValue ν) (swigluOut ν)

/-- **The readout**: `root + 1 − 6s`. -/
noncomputable def swigluReadout : (ParityState → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj parityRoot + ∑ a : ZMod 2, LinearMap.proj (Sum.inl (Sum.inl (0, a)))
    - (6 : ℂ) • ∑ i : Fin 3, LinearMap.proj (Sum.inr (some i))

/-- The readout after one pass of the core on a state `u` and a message `m`. -/
lemma swigluReadout_swigluCore (u : ParityState → ℂ) (m : ν → ℂ) :
    swigluReadout (swigluCore ν (Sum.elim u m) ∘ Sum.inl) =
      u parityRoot + ∑ a : ZMod 2, u (Sum.inl (Sum.inl (0, a))) - 6 * ∑ i, u (Sum.inr (some i)) +
        (6 / Real.log 2 * silu (-Real.log 2 * (∑ i, u (Sum.inr (some i))).re) *
          (∑ a : ZMod 2, u (Sum.inl (Sum.inl (0, a)))).re : ℝ) -
        (18 / Real.log 2 * silu (-Real.log 2 * (∑ i, u (Sum.inr (some i))).re) *
          (∑ i, u (Sum.inr (some i))).re : ℝ) := by
  simp [swigluReadout, swigluCore, swigluLayer, ComplexFFN.swigluFFN, swigluValue, swigluOut,
    paritySum, parityOne, parityLogit, Fin.sum_univ_two, Finset.sum_apply, LinearMap.sum_apply]
  ring

/-- **One SwiGLU pass holds `ψ(∑ᵢ xᵢ yᵢ)` for three pairs over `ZMod 2`**, whatever the
message. -/
theorem swigluReadout_loopState (R : (ParityState → ℂ) →ₗ[ℂ] ν → ℂ) (x y : Fin 3 → ZMod 2) :
    swigluReadout (loopState (swigluCore ν) R (parityPrelude (Sum.elim (pairFeatures x y) 0)) 0) =
      ZMod.stdAddChar (∑ i, x i * y i) := by
  set s : ℕ := ∑ i, (x i * y i).val with hs
  have hN : s ≤ 3 := by
    rw [hs, Fin.sum_univ_three]
    linarith [ZMod.val_lt (x 0 * y 0), ZMod.val_lt (x 1 * y 1), ZMod.val_lt (x 2 * y 2)]
  have hleaf :
      ∑ i, parityPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inr (some i)) = (s : ℂ) := by
    simp only [parityPrelude_leaf, hs, Nat.cast_sum]
  have hone :
      ∑ a : ZMod 2, parityPrelude (Sum.elim (pairFeatures x y) 0) (Sum.inl (Sum.inl (0, a))) = 1 :=
    by simp [parityPrelude_feature, pairFeatures, oneHot]
  have hψ : ZMod.stdAddChar (∑ i, x i * y i) = (-1 : ℂ) ^ s := by
    rw [stdAddChar_sum, hs, ← Finset.prod_pow_eq_pow_sum]
    exact Finset.prod_congr rfl fun i _ => stdAddChar_zmod_two _
  change swigluReadout (swigluCore ν (Sum.elim _ 0) ∘ Sum.inl) = _
  rw [swigluReadout_swigluCore, parityPrelude_root, hleaf, hone, Complex.natCast_re,
    Complex.one_re, hψ, neg_one_pow_eq_silu hN,
    show -Real.log 2 * (s : ℝ) = -((s : ℝ) * Real.log 2) by ring]
  push_cast
  ring

/-- Three products equal to one: the readout is `ψ(1) = −1`. -/
example : swigluReadout (loopState (swigluCore Unit) 0
    (parityPrelude (Sum.elim (pairFeatures 1 1) 0)) 0) = -1 := by
  rw [swigluReadout_loopState, stdAddChar_zmod_two,
    show (∑ i : Fin 3, (1 : Fin 3 → ZMod 2) i * (1 : Fin 3 → ZMod 2) i).val = 1 by decide, pow_one]

/-- **The SwiGLU core is no product layer**, of any width: with it, one pass of a product core
would hold the parity, against `not_productLoop_parity`. -/
theorem not_isProductNet_swigluCore [Fintype ν] {m : ℕ} :
    ¬ IsProductNet m 1 (swigluCore ν) := fun hL =>
  not_productLoop_parity isProductNet_parityPrelude hL 0 swigluReadout
    fun x y => swigluReadout_loopState ν 0 x y

example : ¬ IsProductNet 5 1 (swigluCore Unit) := not_isProductNet_swigluCore Unit

end Transformer.Grokking.FlatSpectrum
