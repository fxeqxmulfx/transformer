module

public import Transformer.Grokking.FiniteRecovery.Examples

/-! Recovering additive row and column labels along the staircase of Flat
Spectrum Theorem E. Only its diagonal and superdiagonal are read. Prefix
sums give a label for any requested cell without computing a nuclear norm. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Finset FlatSpectrum

variable {G : Type*} [AddCommGroup G] {s : ℕ}

/-- A column label relative to column zero, propagated along the observed path. -/
def staircaseColumn (diagonal : Fin (s+1) → G) (edge : Fin s → G) (v : Fin (s+1)) : G :=
  ∑ j ∈ range v.val, if hj : j < s then edge ⟨j,hj⟩ - diagonal (Fin.castSucc ⟨j,hj⟩) else 0

/-- The recovered table takes only the 2s+1 supplied labels as inputs. -/
def staircaseDecode (diagonal : Fin (s+1) → G) (edge : Fin s → G) (u v : Fin (s+1)) : G :=
  diagonal u - staircaseColumn diagonal edge u + staircaseColumn diagonal edge v

/-- The superdiagonal minus the diagonal is precisely the increment between
adjacent column labels. The column offset cancels in every decoded cell. -/
theorem staircaseDecode_additive (row col : Fin (s+1) → G) (u v : Fin (s+1)) :
    staircaseDecode (fun i => row i + col i) (fun j => row j.castSucc + col j.succ) u v =
      row u + col v := by
  have hc (i : Fin (s+1)) : staircaseColumn (fun j => row j + col j)
      (fun j => row j.castSucc + col j.succ) i = col i - col 0 := by
    induction i using Fin.induction with
    | zero => simp [staircaseColumn]
    | succ j ih =>
      unfold staircaseColumn at ih ⊢
      rw [Fin.val_succ, sum_range_succ, dite_eq_left j.isLt]
      change _ + (row j.castSucc + col j.succ - (row j.castSucc + col j.castSucc)) = _
      rw [show (∑ a ∈ range j.val, _) = _ from ih]
      abel
  unfold staircaseDecode
  rw [hc u, hc v]
  simp [sub_eq_add_neg, add_comm, add_left_comm, add_assoc]

variable {k : ℕ}

/-- The disjoint index of the two kinds of observed words, used for PAGE sampling. -/
def staircaseWord (e : Fin (s+1) ≃ Word (ZMod 2) k) :
    Fin (s+1) ⊕ Fin s → Word (ZMod 2) (k+k)
  | .inl i => enumeratedCut e (i,i)
  | .inr j => enumeratedCut e (j.castSucc,j.succ)

theorem range_staircaseWord (e : Fin (s+1) ≃ Word (ZMod 2) k) :
    Set.range (staircaseWord e) = (enumeratedCut e) '' staircase s := by
  ext x
  constructor
  · rintro ⟨i,rfl⟩
    cases i with
    | inl j => exact ⟨_, Or.inl ⟨j,rfl⟩, rfl⟩
    | inr j => exact ⟨_, Or.inr ⟨j,rfl⟩, rfl⟩
  · rintro ⟨_,(⟨i,rfl⟩ | ⟨i,rfl⟩),rfl⟩
    · exact ⟨.inl i,rfl⟩
    · exact ⟨.inr i,rfl⟩

/-- Decode at an arbitrary word using only the observed components. -/
def staircasePrediction (e : Fin (s+1) ≃ Word (ZMod 2) k)
    (labels : Fin (s+1) ⊕ Fin s → ZMod 2) (x : Word (ZMod 2) (k+k)) : ZMod 2 :=
  let ij := (enumeratedCut e).symm x
  staircaseDecode (fun i => labels (.inl i)) (fun j => labels (.inr j)) ij.1 ij.2

theorem staircasePrediction_modularSum (e : Fin (s+1) ≃ Word (ZMod 2) k) :
    staircasePrediction e (modularSum e ∘ staircaseWord e) = modularSum e := by
  funext x
  obtain ⟨⟨u,v⟩,rfl⟩ := (enumeratedCut e).surjective x
  simpa [staircasePrediction, modularSum, staircaseWord] using
    staircaseDecode_additive (fun i => ∑ j, e i j) (fun i => ∑ j, e i j) u v

example := staircasePrediction_modularSum (wordEnumeration 2 4)

end Transformer.Grokking.FiniteRecovery
