module

public import Transformer.Grokking.FiniteRecovery.Examples
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatBest

/-! Exact observation counts for Flat Spectrum in the word formulation.
The mask is chosen in advance, and the bound is uniform over the specified
class. See Flat Spectrum, Theorem E and PT1–PT4. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Matrix Transformer.Grokking.FlatSpectrum
open Transformer.Grokking.FlatSpectrum.TablePart

variable {A U V : Type*} {T : ℕ}

/-- Observing a cell and observing its corresponding word count as the
same labeled pair. -/
@[simp] theorem ncard_observedCells (cut : U × V ≃ Word A T) (inputs : Finset (Word A T)) :
    (observedCells cut inputs).ncard = inputs.card := by
  have himage : observedCells cut inputs = cut.symm '' (inputs : Set (Word A T)) := by
    ext cell
    simp [observedCells]
  rw [himage, Set.ncard_image_of_injective _ cut.symm.injective, Set.ncard_coe_finset]

@[simp] theorem card_inputsAtCells [Fintype A] (cut : U × V ≃ Word A T) (cells : Set (U × V)) :
    (inputsAtCells cut cells).card = cells.ncard := by
  rw [← ncard_observedCells cut, observedCells_inputsAtCells]

/-- Passing to a cut preserves identification by a given observation pattern. -/
theorem exists_learner_iff_determines [Nonempty A] (cut : U × V ≃ Word A T)
    (inputs : Finset (Word A T)) (family : Set (U × V → A)) :
    (∃ learner, Recovers inputs {f | f ∘ cut ∈ family} learner) ↔
      Determines family (observedCells cut inputs) := by
  rw [exists_learner_iff_injOn]
  constructor
  · intro h f hf g hg hagree
    have hfg : f ∘ cut.symm = g ∘ cut.symm := h
      (by simpa [Function.comp_def] using hf) (by simpa [Function.comp_def] using hg)
      (observe_eq_iff_eqOn.mpr fun word hw => by
        simpa [Function.comp_def] using hagree (cut.symm word) (by simpa [observedCells] using hw))
    exact funext fun cell => by simpa using congrFun hfg (cut cell)
  · intro h f hf g hg hobs
    have hfg : f ∘ cut = g ∘ cut := h _ hf _ hg fun cell hmem =>
      observe_eq_iff_eqOn.mp hobs hmem
    exact funext fun word => by simpa using congrFun hfg (cut.symm word)

/-- For each rank-one unit-modulus table on a balanced cut, the least
number of observations for unique nuclear completion is twice the number
of rows minus one. A staircase attains it. -/
theorem isLeast_rankOne_nuclear_observations [Fintype A] {size : ℕ}
    (cut : Fin (size + 1) × Fin (size + 1) ≃ Word A T)
    {u v : Fin (size + 1) → ℂ} (hunit : ∀ i j, ‖vecMulVec u v i j‖ = 1) :
    IsLeast {n : ℕ | ∃ inputs : Finset (Word A T), inputs.card = n ∧
      NuclearRecovers (observedCells cut inputs) (vecMulVec u v)} (2 * size + 1) := by
  refine ⟨⟨inputsAtCells cut (staircase size), by simp [ncard_staircase], ?_⟩, ?_⟩
  · rw [observedCells_inputsAtCells]
    exact forall_isMinOn_vecMulVec_staircase hunit
  · rintro _ ⟨inputs, rfl, hrecovers⟩
    have h := two_mul_card_le_ncard_add_one_of_forall_isMinOn hunit hrecovers
    rw [Fintype.card_fin, ncard_observedCells] at h
    omega

example := isLeast_rankOne_nuclear_observations (enumeratedCut (size := 3) (wordEnumeration 2 2))
  (u := fun _ => 1) (v := fun _ => 1) fun i j => by simp [vecMulVec_apply]

/-- Modular sums attain the rank-one nuclear completion threshold. -/
theorem isLeast_modularSum_nuclear_observations {p length size : ℕ} [NeZero p]
    (e : Fin (size + 1) ≃ Word (ZMod p) length) :
    IsLeast {n : ℕ | ∃ inputs : Finset (Word (ZMod p) (length + length)), inputs.card = n ∧
      NuclearRecovers (observedCells (enumeratedCut e) inputs)
        (targetKernel (enumeratedCut e) ZMod.stdAddChar (modularSum e))} (2 * size + 1) := by
  simp only [targetKernel_modularSum]
  exact isLeast_rankOne_nuclear_observations (enumeratedCut e)
    (fun i j => by simp [vecMulVec_apply])

example := isLeast_modularSum_nuclear_observations (size := 15) (wordEnumeration 2 4)
example := isLeast_modularSum_nuclear_observations (size := 255) (wordEnumeration 2 8)
example := isLeast_modularSum_nuclear_observations (size := 65535) (wordEnumeration 2 16)

example : (2 * 15 + 1 : ℕ) = 31 ∧ (2 * 255 + 1 : ℕ) = 511 ∧
    (2 * 65535 + 1 : ℕ) = 131071 := by decide

variable {p left right : ℕ} [Fact p.Prime]

/-- A known coupling leaves only the row and column functions to determine.
A full row and column attain the information bound `p^left + p^right - 1`. -/
theorem isLeast_knownCoupling_observations
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T)
    (M : Matrix (Fin left) (Fin right) (ZMod p)) :
    IsLeast {n : ℕ | ∃ inputs : Finset (Word (ZMod p) T), inputs.card = n ∧
      ∃ learner, Recovers inputs (knownCouplingFamily cut M) learner}
      (p ^ left + p ^ right - 1) := by
  classical
  let cells := cross (ZMod p) (Fin left) (Fin right)
  have hc : cells.card + 1 = p ^ left + p ^ right := by
    simpa [cells, ZMod.card] using card_cross (F := ZMod p) (ι := Fin left) (κ := Fin right)
  refine ⟨⟨inputsAtCells cut cells, ?_, ?_⟩, ?_⟩
  · rw [card_inputsAtCells, Set.ncard_coe_finset]
    omega
  · apply knownCoupling_recovers_cross cut M _ 0 0
    · intro v
      simpa [inputsAtCells] using col_mem_cross (ι := Fin left) v
    · intro u
      simpa [inputsAtCells] using row_mem_cross (κ := Fin right) u
  · rintro _ ⟨inputs, rfl, learner, hlearn⟩
    let V := {v : Word (ZMod p) right // v ≠ 0}
    let Params := (Word (ZMod p) left → ZMod p) × (V → ZMod p)
    let extend (h : V → ZMod p) (v : Word (ZMod p) right) : ZMod p :=
      if hv : v ≠ 0 then h ⟨v, hv⟩ else 0
    let target (q : Params) := bilinearTask cut q.1 (extend q.2) M
    have hmem (q : Params) : target q ∈ knownCouplingFamily cut M := ⟨q.1, extend q.2, rfl⟩
    have hinj : Function.Injective (fun (q : Params) (w : inputs) => target q w) := by
      intro q q' heq
      have he := (exists_learner_iff_injOn _ _).mp ⟨learner, hlearn⟩ (hmem q) (hmem q')
        (observe_eq_iff_eqOn.mpr fun w hw => congrFun heq ⟨w, hw⟩)
      have hrow : q.1 = q'.1 := by
        funext u
        simpa [target, extend, bilinearTask] using congrFun he (cut (u, 0))
      refine Prod.ext hrow (funext fun v => ?_)
      simpa [target, extend, bilinearTask, hrow, v.property] using congrFun he (cut (0, v))
    have hone : Fintype.card {v : Word (ZMod p) right // v = 0} = 1 :=
      Fintype.card_eq_one_iff.mpr ⟨⟨0, rfl⟩, fun v => Subtype.ext v.property⟩
    have hV : Fintype.card V = p ^ right - 1 := by
      rw [Fintype.card_subtype_compl (fun v : Word (ZMod p) right => v = 0),
        hone, card_word, ZMod.card]
    have hcard := Fintype.card_le_of_injective _ hinj
    have hpow : p ^ (p ^ left + (p ^ right - 1)) ≤ p ^ inputs.card := by
      simpa [Params, Fintype.card_fun, Fintype.card_coe, hV, card_word, ZMod.card, pow_add] using hcard
    have hle := (Nat.pow_le_pow_iff_right (Fact.out : p.Prime).one_lt).mp hpow
    have hp : 0 < p ^ right := pow_pos (Fact.out : p.Prime).pos _
    omega

example := isLeast_knownCoupling_observations (appendCut (ZMod 2) 4 4) 1

/-- The flat class from PT3 allows an unknown bilinear coupling as well as
arbitrary functions on the two groups. -/
def unknownCouplingFamily
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T) :
    Set (Word (ZMod p) T → ZMod p) :=
  {f | f ∘ cut ∈ flatClass (ZMod p) (Fin left) (Fin right)}

/-- If the coupling is unknown, the exact minimum number of fixed observations is
`p^left + p^right - 1 + left * right` (PT4). A full row and column together with
one cell for each pair of unit vectors attain the lower bound. -/
theorem isLeast_unknownCoupling_observations
    (cut : Word (ZMod p) left × Word (ZMod p) right ≃ Word (ZMod p) T) :
    IsLeast {n : ℕ | ∃ inputs : Finset (Word (ZMod p) T), inputs.card = n ∧
      ∃ learner, Recovers inputs (unknownCouplingFamily cut) learner}
      (p ^ left + p ^ right + left * right - 1) := by
  classical
  let cells := crossPart (ZMod p) (Fin left) (Fin right)
  have hcard : cells.card + 1 = p ^ left + p ^ right + left * right := by
    simpa [cells, ZMod.card] using card_crossPart (F := ZMod p) (ι := Fin left) (κ := Fin right)
  refine ⟨⟨inputsAtCells cut cells, ?_, ?_⟩, ?_⟩
  · rw [card_inputsAtCells, Set.ncard_coe_finset]
    omega
  · unfold unknownCouplingFamily
    apply (exists_learner_iff_determines cut _ _).mpr
    rw [observedCells_inputsAtCells]
    exact determines_crossPart
  · rintro _ ⟨inputs, rfl, hlearn⟩
    let observed := inputs.map cut.symm.toEmbedding
    have heq : (observed : Set (Word (ZMod p) left × Word (ZMod p) right)) =
        observedCells cut inputs := by
      ext cell
      simp [observed, observedCells]
    have hdet := (exists_learner_iff_determines cut inputs _).mp hlearn
    rw [← heq] at hdet
    have hbound := add_le_card_of_determines hdet
    rw [Fintype.card_fin, Fintype.card_fin, ZMod.card, show observed.card = inputs.card by
      simp [observed]] at hbound
    omega

/-- The least fixed mask sizes at eight, sixteen, and thirty-two binary tokens,
respectively, for the class with an unknown bilinear coupling. -/
example := isLeast_unknownCoupling_observations (appendCut (ZMod 2) 4 4)
example := isLeast_unknownCoupling_observations (appendCut (ZMod 2) 8 8)
example := isLeast_unknownCoupling_observations (appendCut (ZMod 2) 16 16)

example : (2 ^ 4 + 2 ^ 4 + 4 * 4 - 1 : ℕ) = 47 ∧
    (2 ^ 8 + 2 ^ 8 + 8 * 8 - 1 : ℕ) = 575 ∧
    (2 ^ 16 + 2 ^ 16 + 16 * 16 - 1 : ℕ) = 131327 := by decide

end Transformer.Grokking.FiniteRecovery
