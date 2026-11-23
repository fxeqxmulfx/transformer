module
public import Transformer.Phasor.TripleTransport.PathCoefficients

/-! The signed path table associates a learned scalar to each hash bucket.
An exact address is the tuple of cue, repeatedly followed relation and
final relation. A finer addressing scheme can preserve every old sum, and
can independently suppress a different tuple that collides in the old table. -/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.RefinedPathAddress
variable {K B C : Type*}

/-- Sum signed dense contributions to one chosen classifier margin.
The sign and the projection of the emitted vector are included in each coefficient. -/
noncomputable def total (weights : B → ℝ) (bucket : K → B) (rows : List (K × ℝ)) : ℝ :=
  (rows.map (fun p => weights (bucket p.1) * p.2)).sum

/-- Coarse weights lift to the refined table by copying them to each of
their subaddresses. All finite causal prefixes then have the same margin. -/
theorem lift (old : B → ℝ) (weights : C → ℝ) (fine : K → C) (parent : C → B)
    (rows : List (K × ℝ))
    (agree : ∀ p ∈ rows, weights (fine p.1) = old (parent (fine p.1))) :
    total weights fine rows = total old (parent ∘ fine) rows := by
  induction rows with
  | nil => rfl
  | cons p ps ih =>
    change weights (fine p.1) * p.2 + total weights fine ps =
      old (parent (fine p.1)) * p.2 + total old (parent ∘ fine) ps
    rw [agree p List.mem_cons_self, ih (fun q hq => agree q (List.mem_cons_of_mem _ hq))]

example : total (fun n : ℕ => (n%2 : ℕ)) id [(3,(2:ℝ)),(4,5)] =
    total (fun n : ℕ => (n:ℝ)) ((·%2) ∘ id) [(3,(2:ℝ)),(4,5)] := by
  apply lift
  intro p hp
  rfl

/-- A colliding query and distractor cannot receive weights one and zero
from the same scalar, regardless of the training algorithm. -/
theorem conflict (bucket : K → B) (good other : K) (same : bucket good = bucket other) :
    ¬ ∃ w : B → ℝ, w (bucket good) = 1 ∧ w (bucket other) = 0 := by
  rintro ⟨w, a, b⟩
  rw [same, b] at a
  norm_num at a

/-- A second coordinate that separates the addresses suffices to represent
these two independent coefficients after refinement. -/
theorem separates [DecidableEq C] (bucket : K → C) (good other : K)
    (apart : bucket good ≠ bucket other) :
    ∃ w : C → ℝ, w (bucket good) = 1 ∧ w (bucket other) = 0 := by
  exact ⟨fun k => if k = bucket good then 1 else 0, by simp, by simp [Ne.symm apart]⟩

example : ¬ ∃ w : ℕ → ℝ, w (1%2) = 1 ∧ w (3%2) = 0 :=
  conflict (fun n : ℕ => n%2) 1 3 (by decide)
example : ∃ w : ℕ → ℝ, w (1%8) = 1 ∧ w (3%8) = 0 :=
  separates (fun n : ℕ => n%8) 1 3 (by decide)
end Transformer.Phasor.TripleTransport.RefinedPathAddress
