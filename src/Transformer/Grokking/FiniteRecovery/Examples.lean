module

public import Transformer.Grokking.FiniteRecovery.FlatSpectrum
public import Transformer.Grokking.FlatSpectrum.CutRank
public import Transformer.Grokking.FlatSpectrum.FewestCells

/-! Examples from review/flat_spectrum.md, Functions of n variables, NV1 and
NV5. We apply them to the same word and observation definitions as Argand. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Matrix Transformer.Grokking.FlatSpectrum
open scoped BigOperators

variable {p left right T : ℕ} [NeZero p]

/-- A target with a fixed bilinear interaction between two groups of inputs. -/
def bilinearTask (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p)) : Word (ZMod p) T → ZMod p :=
  fun word =>
    let (u, v) := cut.symm word
    g u + h v + u ⬝ᵥ M *ᵥ v

theorem targetKernel_bilinearTask
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p)) :
    targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M) = bilinearKernel p g h M := by
  ext u v
  simp [targetKernel, bilinearTask, bilinearKernel]

/-- NV5 applied to words of arbitrary length and an arbitrary cut of their positions. -/
theorem rank_bilinearTask [Fact p.Prime]
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    (M : Matrix (Fin left) (Fin right) (ZMod p)) :
    (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)).rank = p ^ M.rank := by
  rw [targetKernel_bilinearTask, rank_bilinearKernel_eq_pow]

example := rank_bilinearTask (appendCut (ZMod 3) 2 2) (fun _ => 0) (fun _ => 0)
  (Matrix.single 0 0 1)

/-- Enumerate the words in one half of a balanced cut. -/
noncomputable def wordEnumeration (p length : ℕ) [NeZero p] :
    Fin (p ^ length) ≃ Word (ZMod p) length :=
  (Fintype.equivFinOfCardEq (by rw [card_word, ZMod.card])).symm

def enumeratedCut {length size : ℕ} (e : Fin (size + 1) ≃ Word (ZMod p) length) :
    Fin (size + 1) × Fin (size + 1) ≃ Word (ZMod p) (length + length) :=
  (Equiv.prodCongr e e).trans (appendCut (ZMod p) length length)

/-- Sum all tokens, with the two equal-length groups indexed by the enumeration. -/
def modularSum {length size : ℕ} (e : Fin (size + 1) ≃ Word (ZMod p) length) :
    Word (ZMod p) (length + length) → ZMod p :=
  fun word =>
    let (u, v) := (enumeratedCut e).symm word
    (∑ i, e u i) + ∑ j, e v j

theorem targetKernel_modularSum {length size : ℕ}
    (e : Fin (size + 1) ≃ Word (ZMod p) length) :
    let phase : Fin (size + 1) → ℂ := fun u => ZMod.stdAddChar (∑ i, e u i)
    targetKernel (enumeratedCut e) ZMod.stdAddChar (modularSum e) = vecMulVec phase phase := by
  ext u v
  simp [targetKernel, modularSum, vecMulVec_apply, AddChar.map_add_eq_mul]

/-- Theorem E for modular addition of any even number of tokens. The
staircase has twice the number of row indices minus one observed cells. -/
theorem modularSum_nuclear_recovers {length size : ℕ}
    (e : Fin (size + 1) ≃ Word (ZMod p) length) :
    NuclearRecovers (staircase size)
      (targetKernel (enumeratedCut e) ZMod.stdAddChar (modularSum e)) := by
  rw [targetKernel_modularSum]
  exact forall_isMinOn_vecMulVec_staircase (fun i j => by simp [vecMulVec_apply])

example := modularSum_nuclear_recovers (size := 8) (wordEnumeration 3 2)

example : (staircase 8).ncard = 17 := by
  rw [ncard_staircase]

/-- An injective code and staircase observations recover the class of all
targets whose phase table factors into unit-modulus rank-one entries. -/
theorem rankOne_recovers_staircase {A : Type*} [Fintype A] [Nonempty A] {size : ℕ}
    (cut : Fin (size + 1) × Fin (size + 1) ≃ Word A T) (code : A → ℂ)
    (hcode : Function.Injective code) :
    ∃ learner, Recovers (inputsAtCells cut (staircase size))
      {f | ∃ u v, targetKernel cut code f = vecMulVec u v ∧
        ∀ i j, ‖vecMulVec u v i j‖ = 1} learner := by
  apply exists_learner_of_nuclear_recovers cut code hcode
  rintro f ⟨u, v, hkernel, hunit⟩
  rw [observedCells_inputsAtCells, hkernel]
  exact forall_isMinOn_vecMulVec_staircase hunit

example := rankOne_recovers_staircase (enumeratedCut (size := 8) (wordEnumeration 3 2))
  ZMod.stdAddChar ZMod.injective_stdAddChar

/-- NV1 applied to any proper part of the observed graph. -/
theorem bilinearTask_not_minimum
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    {M : Matrix (Fin left) (Fin right) (ZMod p)} (hM : Function.Injective M.vecMul)
    (inputs : Finset (Word (ZMod p) T)) (hmissing : ∃ word, word ∉ inputs) :
    ¬ IsMinOn nuclearNorm
      (completions (observedCells cut inputs) (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)))
      (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)) := by
  rw [targetKernel_bilinearTask]
  exact not_isMinOn_bilinearKernel p g h hM (observedCells_ne_univ _ _ hmissing)

/-- The sum of copies of the cubic-product target across a balanced cut. -/
def cubicPairs (p length : ℕ) [NeZero p] : Word (ZMod p) (length + length) → ZMod p :=
  bilinearTask (appendCut (ZMod p) length length) (fun u => ∑ i, u i ^ 3) (fun _ => 0) 1

/-- NV1 applied to a proper part of the observed graph for any number of
cubic-product pairs. -/
theorem cubicPairs_not_minimum (p length : ℕ) [NeZero p]
    (inputs : Finset (Word (ZMod p) (length + length)))
    (hmissing : ∃ word, word ∉ inputs) :
    ¬ IsMinOn nuclearNorm
      (completions (observedCells (appendCut (ZMod p) length length) inputs)
        (targetKernel (appendCut (ZMod p) length length) ZMod.stdAddChar (cubicPairs p length)))
      (targetKernel (appendCut (ZMod p) length length) ZMod.stdAddChar (cubicPairs p length)) := by
  exact bilinearTask_not_minimum _ _ _ (vecMul_injective_of_isUnit isUnit_one) inputs hmissing

/-- Nuclear completion of these flat tables requires every labeled word (NV1). -/
theorem nuclearRecovers_bilinear_iff_full
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (g : Word (ZMod p) left → ZMod p) (h : Word (ZMod p) right → ZMod p)
    {M : Matrix (Fin left) (Fin right) (ZMod p)} (hM : Function.Injective M.vecMul)
    (inputs : Finset (Word (ZMod p) T)) :
    NuclearRecovers (observedCells cut inputs)
      (targetKernel cut ZMod.stdAddChar (bilinearTask cut g h M)) ↔ inputs = Finset.univ := by
  constructor
  · intro hrecovers
    apply Finset.eq_univ_of_forall
    intro word
    by_contra hmissing
    exact bilinearTask_not_minimum cut g h hM inputs ⟨word, hmissing⟩
      ((hrecovers _ (fun _ _ => rfl)).mpr rfl)
  · rintro rfl
    rw [observedCells_univ]
    exact nuclearRecovers_univ _

example (inputs : Finset (Word (ZMod 2) 8)) :=
  nuclearRecovers_bilinear_iff_full (appendCut (ZMod 2) 4 4) 0 0
    (vecMul_injective_of_isUnit isUnit_one) inputs
example (inputs : Finset (Word (ZMod 2) 16)) :=
  nuclearRecovers_bilinear_iff_full (appendCut (ZMod 2) 8 8) 0 0
    (vecMul_injective_of_isUnit isUnit_one) inputs
example (inputs : Finset (Word (ZMod 2) 32)) :=
  nuclearRecovers_bilinear_iff_full (appendCut (ZMod 2) 16 16) 0 0
    (vecMul_injective_of_isUnit isUnit_one) inputs

/-- The admissible targets share their interaction while each group may
contribute an arbitrary function of its own inputs. -/
def knownCouplingFamily (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (M : Matrix (Fin left) (Fin right) (ZMod p)) : Set (Word (ZMod p) T → ZMod p) :=
  {target | ∃ g h, target = bilinearTask cut g h M}

omit [NeZero p] in
/-- A full row and a full column determine the function in the known-coupling
class. Subtracting the known interaction makes opposite corners add up. -/
theorem knownCoupling_recovers_cross
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (M : Matrix (Fin left) (Fin right) (ZMod p)) (inputs : Finset (Word (ZMod p) T))
    (u0 : Word (ZMod p) left) (v0 : Word (ZMod p) right)
    (hrow : ∀ v, cut (u0, v) ∈ inputs) (hcol : ∀ u, cut (u, v0) ∈ inputs) :
    ∃ learner, Recovers inputs (knownCouplingFamily cut M) learner := by
  apply (exists_learner_iff_injOn inputs _).mpr
  rintro f ⟨g, h, rfl⟩ f' ⟨g', h', rfl⟩ heq
  have hagree := observe_eq_iff_eqOn.mp heq
  funext word
  obtain ⟨⟨u, v⟩, rfl⟩ := cut.surjective word
  have h₁ := hagree (hrow v)
  have h₂ := hagree (hcol u)
  have h₀ := hagree (hrow v0)
  simp only [bilinearTask, Equiv.symm_apply_apply] at h₁ h₂ h₀ ⊢
  linear_combination h₁ + h₂ - h₀

/-- A cross on the balanced cut of four ternary tokens. -/
noncomputable def fourTokenCross : Finset (Word (ZMod 3) 4) := by
  classical
  exact Finset.univ.filter fun word =>
    let (u, v) := (appendCut (ZMod 3) 2 2).symm word
    u = 0 ∨ v = 0

private theorem fourTokenCross_has_hidden_word : ∃ word, word ∉ fourTokenCross := by
  have hne : (fun _ : Fin 2 => (1 : ZMod 3)) ≠ 0 := by
    intro h
    exact one_ne_zero (congrFun h 0)
  refine ⟨appendCut (ZMod 3) 2 2 (fun _ => 1, fun _ => 1), ?_⟩
  simp [fourTokenCross, hne]

/-- On these observations the raw cubic phase table fails nuclear completion. -/
example := cubicPairs_not_minimum 3 2 fourTokenCross fourTokenCross_has_hidden_word

/-- The same observed cross suffices when the bilinear coupling is specified. -/
example : ∃ learner, Recovers fourTokenCross (knownCouplingFamily (appendCut (ZMod 3) 2 2) 1)
    learner :=
  knownCoupling_recovers_cross _ _ _ 0 0 (by intro v; simp [fourTokenCross])
    (by intro u; simp [fourTokenCross])

end Transformer.Grokking.FiniteRecovery
