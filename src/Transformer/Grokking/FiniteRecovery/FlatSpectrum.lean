module

public import Mathlib.Logic.Equiv.Fin.Basic
public import Transformer.Grokking.FiniteRecovery.Basic
public import Transformer.Grokking.FlatSpectrum.FlatCapacity

/-! Apply the phase-kernel recovery statements of review/flat_spectrum.md to
words and to the observed graph in FiniteRecovery.Basic. A cut may include
any permutation of the input positions. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Transformer.Grokking.FlatSpectrum

variable {A U V : Type*} {T : ℕ}

/-- Place the left and right groups in contiguous positions. -/
def appendCut (A : Type*) (left right : ℕ) :
    Word A left × Word A right ≃ Word A (left + right) :=
  (Equiv.sumArrowEquivProdArrow (Fin left) (Fin right) A).symm.trans
    (Equiv.arrowCongr finSumFinEquiv (Equiv.refl A))

/-- Each cell of a cut is the phase of the target on the corresponding word. -/
def targetKernel (cut : U × V ≃ Word A T) (code : A → ℂ) (target : Word A T → A) :
    Matrix U V ℂ :=
  Matrix.of fun u v => code (target (cut (u, v)))

/-- The same observed inputs, indexed as cells of the cut. -/
def observedCells (cut : U × V ≃ Word A T) (inputs : Finset (Word A T)) :
    Set (U × V) :=
  {cell | cut cell ∈ inputs}

@[simp] theorem observedCells_univ [Fintype A] (cut : U × V ≃ Word A T) :
    observedCells cut Finset.univ = Set.univ := by
  ext cell
  exact ⟨fun _ => Set.mem_univ _, fun _ => Finset.mem_univ _⟩

/-- Pull a matrix observation pattern back to the finite set of input words. -/
noncomputable def inputsAtCells [Fintype A] (cut : U × V ≃ Word A T)
    (cells : Set (U × V)) : Finset (Word A T) := by
  classical
  exact Finset.univ.filter fun word => cut.symm word ∈ cells

@[simp] theorem observedCells_inputsAtCells [Fintype A] (cut : U × V ≃ Word A T)
    (cells : Set (U × V)) : observedCells cut (inputsAtCells cut cells) = cells := by
  ext cell
  simp [observedCells, inputsAtCells]

/-- A hidden input gives a hidden cell for any cut. -/
theorem observedCells_ne_univ (cut : U × V ≃ Word A T) (inputs : Finset (Word A T))
    (hmissing : ∃ word, word ∉ inputs) : observedCells cut inputs ≠ Set.univ := by
  obtain ⟨word, hword⟩ := hmissing
  intro hall
  have hcell : cut.symm word ∈ observedCells cut inputs :=
    hall.symm ▸ Set.mem_univ (cut.symm word)
  exact hword (by simpa [observedCells] using hcell)

/-- The matrices fitting a prescribed phase table on the observed cells. -/
def completions (cells : Set (U × V)) (target : Matrix U V ℂ) : Set (Matrix U V ℂ) :=
  {candidate | ∀ cell ∈ cells, candidate cell.1 cell.2 = target cell.1 cell.2}

variable [Fintype U] [Fintype V] [DecidableEq V]

/-- The unique minimum nuclear norm completion equals the target. -/
def NuclearRecovers (cells : Set (U × V)) (target : Matrix U V ℂ) : Prop :=
  ∀ candidate ∈ completions cells target,
    IsMinOn nuclearNorm (completions cells target) candidate ↔ candidate = target

/-- Observing every cell leaves a singleton feasible set. -/
theorem nuclearRecovers_univ (target : Matrix U V ℂ) :
    NuclearRecovers Set.univ target := by
  have heq : ∀ X ∈ completions Set.univ target, X = target := fun X hX =>
    Matrix.ext fun u v => hX (u, v) (Set.mem_univ _)
  intro X hX
  refine ⟨fun _ => heq X hX, fun h Z hZ => ?_⟩
  rw [h, heq Z hZ]
  exact le_rfl

/-- When an injective phase code and nuclear completion recover the phase table
of every admissible target, the observed graph determines the token-valued target. -/
theorem exists_learner_of_nuclear_recovers [Nonempty A] (cut : U × V ≃ Word A T)
    (code : A → ℂ) (hcode : Function.Injective code) (inputs : Finset (Word A T))
    (family : Set (Word A T → A))
    (hfamily : ∀ f ∈ family, NuclearRecovers (observedCells cut inputs) (targetKernel cut code f)) :
    ∃ learner, Recovers inputs family learner := by
  apply (exists_learner_iff_injOn inputs family).mpr
  intro f hf g hg heq
  have hagree := observe_eq_iff_eqOn.mp heq
  have hcell (cell : U × V) (h : cell ∈ observedCells cut inputs) :
      targetKernel cut code f cell.1 cell.2 = targetKernel cut code g cell.1 cell.2 :=
    congrArg code (hagree h)
  have hfit : targetKernel cut code g ∈ completions (observedCells cut inputs)
      (targetKernel cut code f) := fun cell h => (hcell cell h).symm
  have hall : completions (observedCells cut inputs) (targetKernel cut code f) =
      completions (observedCells cut inputs) (targetKernel cut code g) := by
    ext candidate
    constructor
    · intro h cell hmem
      exact (h cell hmem).trans (hcell cell hmem)
    · intro h cell hmem
      exact (h cell hmem).trans (hcell cell hmem).symm
  have hmin := (hfamily g hg _ (fun _ _ => rfl)).mpr rfl
  rw [← hall] at hmin
  have hkernel := (hfamily f hf _ hfit).mp hmin
  funext word
  obtain ⟨cell, rfl⟩ := cut.surjective word
  exact hcode (congrArg (fun M : Matrix U V ℂ => M cell.1 cell.2) hkernel).symm

/-- A norm gap gives a quantitative error bound for any model's completed table.
When the model and truth agree on the observed cells, the error is all hidden. -/
theorem completion_error_of_nuclear_gap (truth model competitor : Matrix U V ℂ) (slack : ℝ)
    (hnorm : nuclearNorm model ≤ nuclearNorm competitor + slack)
    (hgap : 0 ≤ nuclearNorm truth - nuclearNorm competitor - slack) :
    (nuclearNorm truth - nuclearNorm competitor - slack) ^ 2 ≤
      (Fintype.card V : ℝ) * frobeniusSq (truth - model) := by
  have htriangle := nuclearNorm_add_le model (truth - model)
  have hsum : model + (truth - model) = truth := by abel
  rw [hsum] at htriangle
  have hle : nuclearNorm truth - nuclearNorm competitor - slack ≤
      nuclearNorm (truth - model) := by linarith
  have hrank : ((truth - model).rank : ℝ) ≤ Fintype.card V := by
    exact_mod_cast (truth - model).rank_le_card_width
  calc
    (nuclearNorm truth - nuclearNorm competitor - slack) ^ 2
        ≤ nuclearNorm (truth - model) ^ 2 := pow_le_pow_left₀ hgap hle 2
    _ ≤ ((truth - model).rank : ℝ) * frobeniusSq (truth - model) :=
          nuclearNorm_sq_le_rank_mul_frobeniusSq _
    _ ≤ (Fintype.card V : ℝ) * frobeniusSq (truth - model) :=
          mul_le_mul_of_nonneg_right hrank (frobeniusSq_nonneg _)

example : observedCells (appendCut (Fin 2) 1 1) ∅ ≠ Set.univ :=
  observedCells_ne_univ _ _ ⟨fun _ => 0, by simp⟩

example : (nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) -
    nuclearNorm (1 : Matrix (Fin 2) (Fin 2) ℂ) - 0) ^ 2 ≤
    (Fintype.card (Fin 2) : ℝ) * frobeniusSq ((1 : Matrix (Fin 2) (Fin 2) ℂ) - 1) :=
  completion_error_of_nuclear_gap 1 1 1 0 (by rw [add_zero]) (by simp)

end Transformer.Grokking.FiniteRecovery
