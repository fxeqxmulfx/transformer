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

/-- Relations whose two-step read ends at a token distinct from the query. -/
def movingRelations (k : Fin p) (dest : Fin p → Fin p) : Finset (Fin p) :=
  Finset.univ.filter fun h => dest h ≠ k

/-- Mean weighted displacement over moving reads. An empty set contributes
zero. Source: the normalized relation-path trial in the experiment README. -/
noncomputable def relationMeanOffset (E : Fin p → EucSpace D) (k : Fin p) (dest : Fin p → Fin p)
    (amplitudes : Fin p → ℝ) : EucSpace D :=
  ((movingRelations k dest).card : ℝ)⁻¹ •
    ∑ h ∈ movingRelations k dest, amplitudes h • (E (dest h) - E k)

/-- Dividing each vocabulary coefficient by the number of moving reads
implements the mean hidden displacement, including empty and repeated endpoints. -/
theorem relationMean_read (q amplitudes : Fin p → ℝ) (k : Fin p) (dest : Fin p → Fin p)
    (E : Fin p → EucSpace D) :
    ∑ c, relationWeights q (fun h => ((movingRelations k dest).card : ℝ)⁻¹ * amplitudes h)
      k dest c • E c = (∑ c, q c • E c) + relationMeanOffset E k dest amplitudes := by
  rw [relationWeights_read]
  congr 1
  simp only [relationMeanOffset, Finset.smul_sum, movingRelations, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro h _
  by_cases hit : dest h = k
  · simp [hit]
  · simp [hit, smul_smul]

/-- Adding arbitrarily many active relations cannot exceed a common bound on
their individual weighted displacements, including the case of no active read. -/
theorem norm_relationMeanOffset_le (E : Fin p → EucSpace D) (k : Fin p) (dest : Fin p → Fin p)
    (amplitudes : Fin p → ℝ) (B : ℝ) (hB : 0 ≤ B)
    (bound : ∀ h ∈ movingRelations k dest, ‖amplitudes h • (E (dest h) - E k)‖ ≤ B) :
    ‖relationMeanOffset E k dest amplitudes‖ ≤ B := by
  let s := movingRelations k dest
  change ‖(s.card : ℝ)⁻¹ • ∑ h ∈ s, amplitudes h • (E (dest h) - E k)‖ ≤ B
  by_cases hs : s.Nonempty
  · have hc : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.mpr hs
    have total : ‖∑ h ∈ s, amplitudes h • (E (dest h) - E k)‖ ≤ (s.card : ℝ) * B := by
      simpa using norm_sum_le_of_le s bound
    rw [norm_smul, Real.norm_of_nonneg (by positivity)]
    calc
      _ ≤ (s.card : ℝ)⁻¹ * ((s.card : ℝ) * B) :=
        mul_le_mul_of_nonneg_left total (by positivity)
      _ = B := by field_simp
  · have he : s = ∅ := Finset.not_nonempty_iff_eq_empty.mp hs
    simpa [he] using hB

/-- When every moving relation reaches one token, unit amplitudes give one
full displacement to it, independently of the number of agreeing relations. -/
theorem relationMeanOffset_unanimous (E : Fin p → EucSpace D) (k r : Fin p) (dest : Fin p → Fin p)
    (hn : (movingRelations k dest).Nonempty) (agree : ∀ h ∈ movingRelations k dest, dest h = r) :
    relationMeanOffset E k dest (fun _ => 1) = E r - E k := by
  let s := movingRelations k dest
  have hc : (s.card : ℝ) ≠ 0 := by exact_mod_cast (Finset.card_pos.mpr hn).ne'
  have terms : ∑ h ∈ s, (1 : ℝ) • (E (dest h) - E k) = ∑ _h ∈ s, (E r - E k) := by
    exact Finset.sum_congr rfl (fun h hh => by rw [agree h hh, one_smul])
  change (s.card : ℝ)⁻¹ • _ = _
  rw [terms, Finset.sum_const, ← Nat.cast_smul_eq_nsmul ℝ, smul_smul, inv_mul_cancel₀ hc, one_smul]

example (E : Fin 3 → EucSpace D) :
    ‖relationMeanOffset E 0 (fun _ => 1) (fun _ => 1)‖ ≤ ‖E 1 - E 0‖ := by
  apply norm_relationMeanOffset_le _ _ _ _ _ (norm_nonneg _)
  intro h _
  simp

example (E : Fin 3 → EucSpace D) : relationMeanOffset E 0 (fun _ => 1) (fun _ => 1) = E 1 - E 0 :=
  relationMeanOffset_unanimous _ _ _ _ (by decide) (by intros; rfl)

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
