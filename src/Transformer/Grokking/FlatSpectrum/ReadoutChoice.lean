module

public import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar
public import Transformer.Grokking.FlatSpectrum.CharacterRecovery
public import Transformer.Grokking.FlatSpectrum.CubicInterpolant

/-!
# The readout decides: `a b` through `ψ` and through `χ`

Source: `review/flat_spectrum.md`, Lean status, Theorem B "in different bases": "For `r ≠ 0`
every singular value of `[ψ_r(a b)]` is `√p`"; *Character check*: "`χ` a faithful character,
`χ(g^k) = e^{2πik/(p−1)}` for a primitive root `g`".

Added:

* Read through `ψ_r`, `r ≠ 0`, the table of `a b` is flat with entries of modulus one, so on no
  proper `Ω` does it have the least nuclear norm among the matrices equal to it there
  (`not_isMinOn_taskKernel_mul`): M never returns it from an incomplete set.
* Read through a multiplicative character `χ`, the same table is `χ(a) χ(b)`, which M recovers
  exactly from the sets of `forall_isMinOn_taskKernel_mul_mulChar_iff_connected`.
* Some `χ` is injective on `ZMod p` (`exists_injective_mulChar`): `χ(g^k) = e^{2πik/(p − 1)}`
  for a generator `g` of `(ZMod p)ˣ`, and `χ(0) = 0`. Through an injective readout the table
  gives back the task (`eq_of_taskKernel_eq`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

/-- Through an injective readout the table gives back the task. -/
lemma eq_of_taskKernel_eq {G : Type*} {f g : G → G → G} {φ : G → ℂ}
    (hφ : Function.Injective φ) (h : taskKernel f φ = taskKernel g φ) : f = g :=
  funext fun a => funext fun b =>
    hφ (by simpa [taskKernel] using congrArg (fun K : Matrix G G ℂ => K a b) h)

example := eq_of_taskKernel_eq (f := fun a b : Fin 2 => a + b) (g := fun a b => a + b)
  (φ := fun x => ((x : ℕ) : ℂ)) (fun _ _ h => Fin.ext (Nat.cast_injective h)) rfl

variable {p : ℕ} [Fact p.Prime]

/-- **`ψ_r(a b)` is never M's answer on an incomplete set** (added): for `r ≠ 0` and every
proper `Ω`, some matrix equal to `[ψ_r(a b)]` on `Ω` has a smaller nuclear norm. -/
theorem not_isMinOn_taskKernel_mul {r : ZMod p} (hr : r ≠ 0) {Ω : Set (ZMod p × ZMod p)}
    (hΩ : Ω ≠ Set.univ) :
    ¬ IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω,
        X' ij.1 ij.2 = taskKernel (· * ·) (fun c => ZMod.stdAddChar (r * c)) ij.1 ij.2}
      (taskKernel (· * ·) fun c => ZMod.stdAddChar (r * c)) :=
  not_isMinOn_of_singularValues_eq (singularValues_taskKernel_mul hr)
    (fun a b => norm_ne_zero_iff.mp (by simp [taskKernel])) hΩ

example := @not_isMinOn_taskKernel_mul 3 ⟨Nat.prime_three⟩ 1 (by decide) ∅ Set.empty_ne_univ

/-- **An injective multiplicative character** (`review/flat_spectrum.md`, Character check: "a
faithful character, `χ(g^k) = e^{2πik/(p−1)}` for a primitive root `g`"): some multiplicative
character of `ZMod p` is injective. -/
theorem exists_injective_mulChar : ∃ χ : MulChar (ZMod p) ℂ, Function.Injective χ := by
  let e := (zmodCyclicMulEquiv (inferInstance : IsCyclic (ZMod p)ˣ)).symm
  let f : (ZMod p)ˣ →* ℂˣ :=
    Circle.toUnits.comp ((ZMod.toCircle (N := Nat.card (ZMod p)ˣ)).toMonoidHom.comp
      e.toMonoidHom)
  have h1 : Function.Injective Circle.toUnits := fun x y h =>
    Circle.coe_injective (by simpa [Circle.toUnits_apply] using congrArg Units.val h)
  have h2 : Function.Injective (ZMod.toCircle (N := Nat.card (ZMod p)ˣ)).toMonoidHom :=
    fun a b h => Multiplicative.toAdd.injective (ZMod.injective_toCircle (by simpa using h))
  have hf : Function.Injective f := by
    simp only [f, MonoidHom.coe_comp, MulEquiv.coe_toMonoidHom]
    exact h1.comp (h2.comp e.injective)
  refine ⟨MulChar.ofUnitHom f, fun x y hxy => ?_⟩
  by_cases hx : x = 0
  · rcases eq_or_ne y 0 with hy | hy
    · rw [hx, hy]
    · have h0 : MulChar.ofUnitHom f y = 0 := by rw [← hxy, hx, MulChar.map_zero]
      exact absurd h0 ((mulChar_ne_zero_iff _ y).mpr hy)
  · have hy : y ≠ 0 := by
      rintro rfl
      exact (mulChar_ne_zero_iff _ x).mpr hx (by rw [hxy, MulChar.map_zero])
    have hu := MulChar.ofUnitHom_coe f (Units.mk0 x hx)
    have hw := MulChar.ofUnitHom_coe f (Units.mk0 y hy)
    rw [Units.val_mk0] at hu hw
    have hfe : f (Units.mk0 x hx) = f (Units.mk0 y hy) :=
      Units.ext (by rw [← hu, ← hw, hxy])
    simpa using congrArg Units.val (hf hfe)

example := @exists_injective_mulChar 3 ⟨Nat.prime_three⟩

end Transformer.Grokking.FlatSpectrum
