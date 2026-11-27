module

public import Transformer.Grokking.FiniteRecovery.OperandMixer

/-! A tied vocabulary for *arbitrary* unknown unit row and column factors
and any of the 239 possible constant-row outputs. Input-only coordinates
carry the two factors. The classifying coordinates carry the output
character and an indicator centered at the equals sign. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division.Factors

open Transformer.PhasorV2 FlatSpectrum
open scoped Classical InnerProductSpace ComplexConjugate

abbrev UnitFactor := DivisionUnit → Additive DivisionUnit

noncomputable def unitAt (t : Fin 239) : Option DivisionUnit :=
  if h : ∃ u : DivisionUnit, residueToken u.val = t then some h.choose else none

theorem unitAt_some {t : Fin 239} {u : DivisionUnit} : unitAt t = some u ↔ residueToken u.val = t := by
  unfold unitAt
  split
  · rename_i h
    rw [Option.some.injEq]
    exact ⟨fun he => he ▸ h.choose_spec,fun he => Units.val_injective (residueToken_injective (h.choose_spec.trans he.symm))⟩
  · rename_i h
    constructor
    · intro he
      cases he
    · exact fun he => (h ⟨u,he⟩).elim

theorem unitAt_zero : unitAt (residueToken 0) = none := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  unfold unitAt
  exact dite_eq_right fun ⟨u,h⟩ => u.ne_zero (residueToken_injective h)

theorem unitAt_small {t : Fin 239} (ht : t.val < 22) : unitAt t = none := by
  unfold unitAt
  apply dite_eq_right
  rintro ⟨u,hu⟩
  have he := residue_some.mpr hu
  rw [residue_small ht] at he
  contradiction

/-- Centering makes every possible output choice compatible with the native
equals-sign skip. The target indicator exceeds every other tag by one. -/
def tag (z t : Fin 239) : ℝ := (if t = z then 1 else 0) - (if (1 : Fin 239) = z then 1 else 0)

theorem tag_values (z t : Fin 239) : tag z t = 0 ∨ tag z t = 1 ∨ tag z t = -1 := by
  by_cases ht : t = z <;> by_cases hz : (1 : Fin 239) = z <;> simp [tag,ht,hz]

theorem tag_gap (z t : Fin 239) (ht : t ≠ z) : tag z t < tag z z := by simp [tag,ht]

noncomputable def number (t : Fin 239) : ℝ := if (residue t).isSome then 1 else 0

structure Channels where
  symbol : ℂ
  row : ℂ
  column : ℂ
  targetTag : ℝ
  zeroFlag : ℝ
  specialKey : ℝ
  numberKey : ℝ
  padding : ℝ

noncomputable def channels (χ : Character) (r c : UnitFactor) (z t : Fin 239) : Channels :=
  let feature (map : UnitFactor) := (unitAt t).elim 0 (fun u => χ (map u).toMul.val)
  let k := tag z t
  let n := number t
  ⟨phase χ t,feature r,feature c,k,flag t,(1-n)*k^2,n,n*(1-k^2)⟩

noncomputable def Channels.vector (p : Channels) : Stream :=
  EuclideanSpace.single 0 1 + EuclideanSpace.single 1 p.symbol + EuclideanSpace.single 2 p.row +
  EuclideanSpace.single 3 (p.targetTag : ℂ) + EuclideanSpace.single 4 p.column +
  EuclideanSpace.single 8 (p.zeroFlag : ℂ) + EuclideanSpace.single 9 (p.zeroFlag : ℂ) +
  EuclideanSpace.single 10 (p.zeroFlag : ℂ) + EuclideanSpace.single 11 (p.specialKey : ℂ) +
  EuclideanSpace.single 12 (p.numberKey : ℂ) + EuclideanSpace.single 13 (p.padding : ℂ)

noncomputable def vocab (χ : Character) (r c : UnitFactor) (z t : Fin 239) : Stream :=
  (channels χ r c z t).vector

theorem Channels.inner (p : Channels) (v : Stream) : ⟪p.vector,v⟫_ℂ =
    v 0 + conj p.symbol*v 1 + conj p.row*v 2 + (p.targetTag:ℂ)*v 3 + conj p.column*v 4 +
      (p.zeroFlag:ℂ)*v 8 + (p.zeroFlag:ℂ)*v 9 + (p.zeroFlag:ℂ)*v 10 + (p.specialKey:ℂ)*v 11 +
      (p.numberKey:ℂ)*v 12 + (p.padding:ℂ)*v 13 := by
  simp [vector,inner_add_left,EuclideanSpace.inner_single_left]

theorem Channels.energy (p : Channels) : ‖p.vector‖^2 =
    1 + Complex.normSq p.symbol + Complex.normSq p.row + Complex.normSq p.column +
    p.targetTag^2 + 3*p.zeroFlag^2 + p.specialKey^2 + p.numberKey^2 + p.padding^2 := by
  rw [← inner_self_eq_norm_sq (𝕜 := ℂ)]
  change (⟪p.vector,p.vector⟫_ℂ).re = _
  rw [p.inner]
  norm_num [vector,EuclideanSpace.single,PiLp.single_apply,Complex.conj_mul',← Complex.ofReal_mul,
    ← Complex.ofReal_pow,Complex.normSq_eq_norm_sq]
  ring

private theorem norm_of_energy (v : Stream) (e : ℝ) (he : 0 ≤ e) (h : ‖v‖^2 = e) : ‖v‖ = Real.sqrt e := by
  nlinarith [norm_nonneg v,Real.sqrt_nonneg e,Real.sq_sqrt he]

theorem norm_unit (χ : Character) (r c : UnitFactor) (z : Fin 239) (a : DivisionUnit) :
    ‖vocab χ r c z (residueToken a.val)‖ = Real.sqrt 6 := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  have hu : unitAt (residueToken a.val) = some a := unitAt_some.mpr rfl
  have hn : residue (residueToken a.val) = some a.val := residue_some.mpr rfl
  have h0 : residueToken a.val ≠ residueToken 0 := fun he => a.ne_zero (residueToken_injective he)
  apply norm_of_energy _ _ (by norm_num)
  rw [vocab,Channels.energy]
  rcases tag_values z (residueToken a.val) with h | h | h <;>
    norm_num [channels,number,hn,hu,h,flag,h0,phase_residue,Complex.normSq_eq_norm_sq,
      norm_mulChar_eq_one χ a.ne_zero,norm_mulChar_eq_one χ (r a).toMul.ne_zero,
      norm_mulChar_eq_one χ (c a).toMul.ne_zero]

theorem norm_zero (χ : Character) (r c : UnitFactor) (z : Fin 239) :
    ‖vocab χ r c z (residueToken 0)‖ = Real.sqrt 6 := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  have hn : residue (residueToken 0) = some 0 := residue_some.mpr rfl
  apply norm_of_energy _ _ (by norm_num)
  rw [vocab,Channels.energy]
  rcases tag_values z (residueToken 0) with h | h | h <;>
    norm_num [channels,number,hn,unitAt_zero,h,flag,phase_residue,χ.map_zero]

theorem norm_small (χ : Character) (r c : UnitFactor) (z : Fin 239) {t : Fin 239} (ht : t.val < 22) :
    ‖vocab χ r c z t‖ = if tag z t = 0 then 1 else Real.sqrt 3 := by
  obtain ⟨hp,hf⟩ := small_features χ ht
  have he := (channels χ r c z t).energy
  rcases tag_values z t with h | h | h
  · rw [ite_eq_left h]
    have hn : ‖vocab χ r c z t‖^2 = 1 := by
      simpa [vocab,channels,hp,hf,number,residue_small ht,unitAt_small ht,h] using he
    simpa using norm_of_energy _ _ (by norm_num) hn
  all_goals
    rw [ite_eq_right (by rw [h]; norm_num)]
    apply norm_of_energy _ _ (by norm_num)
    rw [vocab,Channels.energy]
    norm_num [channels,hp,hf,number,residue_small ht,unitAt_small ht,h]

end Transformer.Grokking.FiniteRecovery.Argand.Division.Factors
