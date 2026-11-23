module

public import Transformer.Phasor.TripleTransport.DenseInduction
public import Mathlib.Data.Finset.Lattice.Fold

/-!
# Indexing a causal dense induction read

Source: the Phasor dense induction experiment, section "Dense induction on
the causal embedding prefix". Keys that recur can share a timestamp and
neighbors, with arbitrary overlapping angular neighborhoods.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.DenseInduction.ClassIndex

variable {K : Type*}

/-- Zero denotes an absent donor. Every completed donor has positive index. -/
def stamp (accept : K → K → Bool) (xs : ℕ → K) (q : K) (n : ℕ) : ℕ :=
  (findDonor accept xs q n).getD 0

/-- Any matching key in the prefix precedes or supplies the returned successor. -/
theorem hit_le (a : K → K → Bool) (xs : ℕ → K) (q : K) (i n : ℕ)
    (early : i < n) (hit : a q (xs i) = true) : i+1 ≤ stamp a xs q n := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases h : a q (xs n) = true
    · simp [stamp,findDonor,h]; omega
    · have hi : i < n := by
        by_contra! ge
        obtain rfl : i=n := by omega
        contradiction
      simpa [stamp,findDonor,h] using ih hi

/-- Every positive lookup result has a matching key in the current prefix. -/
theorem attained (a : K → K → Bool) (xs : ℕ → K) (q : K) (n : ℕ)
    (pos : 0 < stamp a xs q n) :
    ∃ i, i < n ∧ a q (xs i) = true ∧ stamp a xs q n=i+1 := by
  induction n with
  | zero => simp [stamp,findDonor] at pos
  | succ n ih =>
    by_cases h : a q (xs n) = true
    · exact ⟨n,by omega,h,by simp [stamp,findDonor,h]⟩
    · have old : stamp a xs q (n+1) = stamp a xs q n := by simp [stamp,findDonor,h]
      obtain ⟨i,lo,hit,eq⟩ := ih (by omega)
      exact ⟨i,by omega,hit,old.trans eq⟩

/-- For a newly encountered query, search every distinct preceding key
once. Its latest matching successor is the largest qualifying stamp. -/
theorem grouped [DecidableEq K] (a : K → K → Bool) (xs : ℕ → K) (q : K) (n : ℕ)
    (keys : Finset K) (contains : ∀ i, i<n → xs i ∈ keys) :
    stamp a xs q n = keys.sup (fun k => if a q k then stamp (fun u v => decide (u=v)) xs k n else 0) := by
  apply Nat.le_antisymm
  · by_cases pos : 0 < stamp a xs q n
    · obtain ⟨i,lo,hit,eq⟩ := attained _ _ _ _ pos
      rw [eq]
      have lower := hit_le (fun (u v : K) => decide (u=v)) xs (xs i) i n lo (by simp)
      have upper := Finset.le_sup (f := fun k => if a q k then stamp (fun u v => decide (u=v)) xs k n else 0) (contains i lo)
      rw [hit] at upper
      exact lower.trans upper
    · omega
  · apply Finset.sup_le
    intro k _
    split
    · rename_i hit
      by_cases pos : 0 < stamp (fun u v : K => decide (u=v)) xs k n
      · obtain ⟨i,lo,eq,h⟩ := attained _ _ _ _ pos
        have same : k=xs i := by simpa using eq
        rw [h]
        exact hit_le _ _ _ _ _ lo (same ▸ hit)
      · omega
    · omega

/-- One angular comparison records both directions of a neighborhood edge. -/
theorem angular_symmetric {d : ℕ} (x y : Fin d → ℝ) : accepts x y = accepts y x := by
  have dot : product x y = product y x :=
    Finset.sum_congr rfl (fun _ _ => mul_comm ..)
  have eq : eligible x y ↔ eligible y x := by
    unfold eligible
    rw [dot,mul_comm (product x x) (product y y)]
  classical
  simp [accepts,eq]

example : stamp (fun u v : ℕ => decide (u=v)) (fun i => i%3) 1 7 =
    ({0,1,2} : Finset ℕ).sup (fun k => if decide (1=k) then
      stamp (fun u v : ℕ => decide (u=v)) (fun i => i%3) k 7 else 0) := by
  apply grouped
  intro i hi
  have h := Nat.mod_lt i (by decide : 0<3)
  simp only [Finset.mem_insert,Finset.mem_singleton]
  omega


/-- Once the next successor is available, exactly queries accepting its
preceding key receive the new timestamp. All other timestamps carry over. -/
theorem advance (accept : K → K → Bool) (xs : ℕ → K) (q : K) (n : ℕ) :
    stamp accept xs q (n+1) = if accept q (xs n) then n+1 else stamp accept xs q n := by
  cases h : accept q (xs n) <;> simp [stamp,findDonor,h]

/-- An index may keep separate identities for inputs with identical
geometry. Coincident directions and overlapping neighborhoods both use
the latest matching timestamp. Injectivity of geometry is unnecessary. -/
theorem grouped_geometry {V : Type*} [DecidableEq K] (geometry : K → V) (a : V → V → Bool)
    (xs : ℕ → K) (q : K) (n : ℕ) (keys : Finset K) (cover : ∀ i, i<n → xs i ∈ keys) :
    stamp a (geometry ∘ xs) (geometry q) n =
      keys.sup (fun k => if a (geometry q) (geometry k) then
        stamp (fun u v => decide (u=v)) xs k n else 0) := by
  have compose (i : ℕ) :
      stamp a (geometry ∘ xs) (geometry q) i =
        stamp (fun u v => a (geometry u) (geometry v)) xs q i := by
    induction i with
    | zero => rfl
    | succ i ih => simp only [advance,Function.comp_apply,ih]
  exact (compose n).trans (grouped _ xs q n keys cover)

example : 5 ≤ stamp (fun a b : ℕ => decide (a=b)) (fun i => i%3) 1 6 :=
  hit_le _ _ _ 4 6 (by decide) (by decide)

example : ∃ i, i < 6 ∧ (decide (1=i%3) : Bool) = true ∧
    stamp (fun a b : ℕ => decide (a=b)) (fun i => i%3) 1 6 = i+1 :=
  attained _ _ _ _ (by decide)

example : stamp (fun a b : ℕ => decide (a=b)) (fun i => (i%4)%2) 0 8 =
    ({0,1,2,3} : Finset ℕ).sup (fun k => if decide (0=k%2) then
      stamp (fun a b : ℕ => decide (a=b)) (fun i => i%4) k 8 else 0) := by
  apply grouped_geometry (fun i => i%2) (fun a b => decide (a=b)) (fun i => i%4) 0
  intro i hi
  have := Nat.mod_lt i (by decide : 0<4)
  simp only [Finset.mem_insert,Finset.mem_singleton]
  omega

variable [DecidableEq K]

/-- Cached classes from other sequences have no timestamp until the current
sequence supplies a completed predecessor from that class. -/
theorem absent_stamp (xs : ℕ → K) (k : K) (n : ℕ) (absent : ∀ i, i<n → xs i ≠ k) :
    stamp (fun u v => decide (u=v)) xs k n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simpa [advance, Ne.symm (absent n (by omega))] using ih (fun i hi => absent i (by omega))

/-- Every cached superset of the current prefix's classes gives exactly the
local backwards read when its per-sequence timestamps are reset. -/
theorem shared_geometry (a : K → K → Bool) (xs : ℕ → K) (q : K) (n : ℕ)
    (localKeys cache : Finset K) (cover : ∀ i, i<n → xs i ∈ localKeys) (incl : localKeys⊆cache) :
    localKeys.sup (fun k => if a q k then stamp (fun u v => decide (u=v)) xs k n else 0) =
      cache.sup (fun k => if a q k then stamp (fun u v => decide (u=v)) xs k n else 0) := by
  exact (grouped a xs q n localKeys cover).symm.trans
    (grouped a xs q n cache (fun i hi => incl (cover i hi)))

example : stamp (fun u v : ℕ => decide (u=v)) (fun i => i % 3) 4 8 = 0 :=
  absent_stamp _ _ _ (fun i _ => by have := Nat.mod_lt i (by decide : 0<3); omega)

example :
    ({0,1} : Finset ℕ).sup (fun k => if decide (1=k) then stamp (fun u v => decide (u=v)) (fun i => i%2) k 5 else 0) =
    ({0,1,2,3,4} : Finset ℕ).sup (fun k => if decide (1=k) then stamp (fun u v => decide (u=v)) (fun i => i%2) k 5 else 0) := by
  exact shared_geometry (fun u v : ℕ => decide (u=v)) (fun i => i%2) 1 5 {0,1} {0,1,2,3,4}
    (fun i _ => by
      have := Nat.mod_lt i (by decide : 0<2)
      simp only [Finset.mem_insert, Finset.mem_singleton]
      omega)
    (by decide)

end Transformer.Phasor.TripleTransport.DenseInduction.ClassIndex
