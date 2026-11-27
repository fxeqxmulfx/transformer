module

public import Transformer.Grokking.FiniteRecovery.DivisionMask
public import Transformer.Grokking.FiniteRecovery.ArgandParityCircuit
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.PoweredDivision

/-! A tied 239-token vocabulary for the native Argand division capacity
witness. The unit operands use a faithful character and its conjugate.
The residue zero uses two indicator coordinates of the same total squared
norm. Every token shares the first coordinate, including the separators. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2 FlatSpectrum
open scoped ComplexConjugate InnerProductSpace Classical

abbrev Stream := Parity.Stream
abbrev Character := MulChar (ZMod 97) ℂ

noncomputable def residue (t : Fin 239) : Option (ZMod 97) :=
  if h : ∃ r, residueToken r = t then some h.choose else none

theorem residue_some {t : Fin 239} {r : ZMod 97} : residue t = some r ↔ residueToken r = t := by
  unfold residue
  split
  · rename_i h
    rw [Option.some.injEq]
    exact ⟨fun he => he ▸ h.choose_spec, fun he => residueToken_injective (h.choose_spec.trans he.symm)⟩
  · rename_i h
    constructor
    · intro he
      cases he
    · exact fun he => (h ⟨r,he⟩).elim

theorem residue_small {t : Fin 239} (ht : t.val < 22) : residue t = none := by
  unfold residue
  apply dite_eq_right
  rintro ⟨r,hr⟩
  have he := congrArg Fin.val hr
  change 22 + r.val = t.val at he
  omega

noncomputable def phase (χ : Character) (t : Fin 239) : ℂ := (residue t).elim 0 χ
noncomputable def flag (t : Fin 239) : ℝ := if t = residueToken 0 then 1 else 0

theorem phase_residue (χ : Character) (r : ZMod 97) : phase χ (residueToken r) = χ r := by
  rw [phase, residue_some.mpr rfl]
  rfl

theorem small_features (χ : Character) {t : Fin 239} (ht : t.val < 22) :
    phase χ t = 0 ∧ flag t = 0 := by
  have hn : t ≠ residueToken 0 := fun he => by subst t; norm_num [residueToken] at ht
  simp [phase,residue_small ht,flag,hn]

noncomputable def embedding (χ : Character) (t : Fin 239) : Stream :=
  EuclideanSpace.single 0 1 + EuclideanSpace.single 1 (phase χ t) +
  EuclideanSpace.single 2 (conj (phase χ t)) + EuclideanSpace.single 3 (flag t : ℂ) +
  EuclideanSpace.single 4 (flag t : ℂ)

theorem embedding_inner (χ : Character) (t : Fin 239) (v : Stream) :
    ⟪embedding χ t,v⟫_ℂ = v 0 + conj (phase χ t)*v 1 + phase χ t*v 2 +
      (flag t : ℂ)*v 3 + (flag t : ℂ)*v 4 := by
  simp [embedding,inner_add_left,EuclideanSpace.inner_single_left]

theorem embedding_coords (χ : Character) (t : Fin 239) :
    let v := embedding χ t
    v 0 = 1 ∧ v 1 = phase χ t ∧ v 2 = conj (phase χ t) ∧ v 3 = (flag t : ℂ) ∧ v 4 = (flag t : ℂ) := by
  norm_num [embedding,EuclideanSpace.single,PiLp.single_apply]

private theorem norm_formula (χ : Character) (t : Fin 239) :
    ‖embedding χ t‖^2 = 1 + 2*Complex.normSq (phase χ t) + 2*(flag t)^2 := by
  obtain ⟨h0,h1,h2,h3,h4⟩ := embedding_coords χ t
  rw [← inner_self_eq_norm_sq (𝕜 := ℂ)]
  change (⟪embedding χ t,embedding χ t⟫_ℂ).re = _
  rw [embedding_inner,h0,h1,h2,h3,h4,mul_comm (conj _) (phase χ t),Complex.mul_conj]
  simp only [Complex.add_re,Complex.ofReal_re,Complex.one_re,← Complex.ofReal_mul]
  ring

theorem embedding_norm_residue (χ : Character) (r : ZMod 97) : ‖embedding χ (residueToken r)‖ = Real.sqrt 3 := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  have h := norm_formula χ (residueToken r)
  rw [phase_residue] at h
  have hs : ‖embedding χ (residueToken r)‖ ^ 2 = 3 := by
    by_cases hr : r = 0
    · subst r
      norm_num [flag,χ.map_zero] at h
      exact h
    · have ht : residueToken r ≠ residueToken 0 := fun he => hr (residueToken_injective he)
      norm_num [flag,ht,Complex.normSq_eq_norm_sq,norm_mulChar_eq_one χ hr] at h
      exact h
  nlinarith [norm_nonneg (embedding χ (residueToken r)),Real.sqrt_nonneg 3,Real.sq_sqrt (by norm_num : 0 ≤ (3:ℝ))]

theorem embedding_norm_small (χ : Character) {t : Fin 239} (ht : t.val < 22) :
    ‖embedding χ t‖ = 1 := by
  obtain ⟨hp,hf⟩ := small_features χ ht
  have h := norm_formula χ t
  rw [hp,hf] at h
  norm_num at h
  exact h.resolve_right (by linarith [norm_nonneg (embedding χ t)])

example (χ : Character) : ‖embedding χ 10‖ = 1 := embedding_norm_small χ (by decide)
example (χ : Character) : ‖embedding χ (residueToken 0)‖ = Real.sqrt 3 := embedding_norm_residue χ 0

end Transformer.Grokking.FiniteRecovery.Argand.Division
