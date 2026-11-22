module

public import Transformer.Phasor.SuccessorDisplacement

/-!
# Two consecutive reads of a relation observed in a causal prefix

Source: the relation-path trial in `experiments/phasor_value_transport/README.md`.
Every completed triple gives its middle token as a relation. For each relation
the read follows two stored edges and applies a learned scalar to the resulting
tied embedding displacement. Every token can serve as a relation or an endpoint.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {α β : Type*} {p D : ℕ}

/-- Last destination of a source and relation in the first `n` triples.
The triple starting at `j` uses positions `j`, `j+1` and `j+2`. -/
def latestRelation [DecidableEq α] (xs : ℕ → α) : ℕ → α → α → Option α
  | 0, _, _ => none
  | n+1, k, h => if xs n = k ∧ xs (n+1) = h then some (xs (n+2))
    else latestRelation xs n k h

/-- The last matching completed triple supplies the destination regardless of
earlier writes or intervening triples for other sources or relations. -/
theorem latestRelation_lookup [DecidableEq α] (xs : ℕ → α) (k h : α) (j n : ℕ)
    (hj : j < n) (hw : xs j = k ∧ xs (j+1) = h)
    (hlast : ∀ m, j < m → m < n → ¬(xs m = k ∧ xs (m+1) = h)) :
    latestRelation xs n k h = some (xs (j+2)) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases t : j = n
    · subst n
      simp [latestRelation, hw]
    · have earlier : j < n := by omega
      rw [latestRelation, ite_eq_right (hlast n earlier (by omega))]
      exact ih earlier (fun m hm hn => hlast m hm (by omega))

/-- A path follows the same relation twice in one prefix. Missing either edge
returns the source, making the tied displacement zero. At position `i` the
causal model uses `n = i-2`, so every read ends before the current token. -/
def relationEnd [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k h : α) : α :=
  ((latestRelation xs n k h).bind fun mid => latestRelation xs n mid h).getD k

/-- Two latest writes join even if their triples occur in reverse order or
other writes separate them. No ordering between `j` and `l` is assumed. -/
theorem relationEnd_of_writes [DecidableEq α] (xs : ℕ → α) (k h : α) (j l n : ℕ)
    (hj : j < n) (hl : l < n) (hw : xs j = k ∧ xs (j+1) = h)
    (hw' : xs l = xs (j+2) ∧ xs (l+1) = h)
    (hlast : ∀ m, j < m → m < n → ¬(xs m = k ∧ xs (m+1) = h))
    (hlast' : ∀ m, l < m → m < n → ¬(xs m = xs (j+2) ∧ xs (m+1) = h)) :
    relationEnd xs n k h = xs (l+2) := by
  simp only [relationEnd, latestRelation_lookup xs k h j n hj hw hlast, Option.bind_some,
    latestRelation_lookup xs (xs (j+2)) h l n hl hw' hlast', Option.getD_some]

/-- Every injective renaming of the alphabet preserves the two reads. -/
theorem relationEnd_map [DecidableEq α] [DecidableEq β] (f : α → β) (hf : Function.Injective f)
    (xs : ℕ → α) (n : ℕ) (k h : α) :
    relationEnd (fun j => f (xs j)) n (f k) (f h) = f (relationEnd xs n k h) := by
  have table (m : ℕ) (a : α) :
      latestRelation (fun j => f (xs j)) m (f a) (f h) = (latestRelation xs m a h).map f := by
    induction m with
    | zero => rfl
    | succ m ih =>
      by_cases hit : xs m = a ∧ xs (m+1) = h <;> simp [latestRelation, hf.eq_iff, hit, ih]
  unfold relationEnd
  rw [table]
  cases hfirst : latestRelation xs n k h with
  | none => rfl
  | some mid =>
    simp only [Option.map_some, Option.bind_some, table n mid]
    cases latestRelation xs n mid h <;> rfl

/-- Combined vocabulary coefficients for one shared scalar per relation.
`dest h` is its two-step read. The rest of the model supplies `q`. -/
noncomputable def relationWeights (q amplitudes : Fin p → ℝ) (k : Fin p) (dest : Fin p → Fin p)
    (c : Fin p) : ℝ :=
  q c + ∑ h, displacementWeight (fun _ => 0) (amplitudes h) k (dest h) c

/-- The coefficient implementation equals a sum of hidden displacements for
arbitrary learned embeddings. Distinct relations can share an endpoint. -/
theorem relationWeights_read (q amplitudes : Fin p → ℝ) (k : Fin p) (dest : Fin p → Fin p)
    (E : Fin p → EucSpace D) :
    ∑ c, relationWeights q amplitudes k dest c • E c =
      (∑ c, q c • E c) + ∑ h, amplitudes h • (E (dest h) - E k) := by
  simp only [relationWeights, add_smul, Finset.sum_smul, Finset.sum_add_distrib]
  congr 1
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro h _
  simpa using displacementWeight_read E (fun _ => 0) (amplitudes h) k (dest h)

/-- Adjoint of the relation scalars. Each receives the difference of the two
upstream basis derivatives even when other relations reach the same token. -/
theorem relationAmplitude_adjoint (E : Fin p → EucSpace D) (k : Fin p) (dest : Fin p → Fin p)
    (change : Fin p → ℝ) (v : EucSpace D) :
    ⟪v, ∑ h, change h • (E (dest h) - E k)⟫ =
      ∑ h, change h * (⟪v, E (dest h)⟫ - ⟪v, E k⟫) := by
  simp only [inner_sum, real_inner_smul_right, inner_sub_right]

/-- The edge 2→3 is written before 1→2. Both use relation 0. -/
example :
    let xs := fun j => (![2,0,3,4,1,0,2] : Fin 7 → Fin 5) (Fin.ofNat 7 j)
    relationEnd xs 5 1 0 = 3 := by decide

example (v : α) [DecidableEq α] :
    latestRelation (fun _ => v) 4 v v = some v := by
  exact latestRelation_lookup (fun _ => v) v v 3 4 (by decide) ⟨rfl, rfl⟩ (by intros; omega)

example (v : α) [DecidableEq α] : relationEnd (fun _ => v) 4 v v = v := by
  exact relationEnd_of_writes (fun _ => v) v v 3 3 4 (by decide) (by decide)
    ⟨rfl, rfl⟩ ⟨rfl, rfl⟩ (by intros; omega) (by intros; omega)

example (xs : ℕ → ℕ) (k h n : ℕ) :
    relationEnd (fun j => xs j + 9) n (k+9) (h+9) = relationEnd xs n k h + 9 :=
  relationEnd_map (fun x : ℕ => x+9) (by intro a b hab; dsimp at hab; omega) xs n k h

end Transformer.Phasor
