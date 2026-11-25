module

public import Transformer.Grokking.FlatSpectrum.ProductLoopScan

/-!
# The scan loop: one link per pass

Source: `review/flat_spectrum.md`, *Loops*, item LP3. A table of links on the keys `V` is a matrix
`A`. A map `f` gives the matrix of zeros and ones `[f i = j]` (`funMatrix`), and its powers are the
tables of the iterates of `f` (`funMatrix_pow`).

The scan keeps `A` and a start `c` in the prelude state and writes a key into scratch slots `S`.
Unit `(i, j)` of its core multiplies `A i j` by the reading of the key slot `i` plus the message
at `i`, and writes the product into `S j`: `|V|²` units whose gates read only the table, a scan
layer (`isScanLayer_scanCore`). The message is `S − P`, as in LP2, so each pass reads in its key
slots what the last one wrote, and pass `n + 1` writes `c A^{n+1}` (`loopState_scanCore`).

For the table of a map and a one-hot start `v`, pass `n + 1` writes the one-hot key `f^[n+1] v`
(`loopState_scanCore_funMatrix`). If every key reaches a fixed point of `f` within `N` steps, as
with absorbing terminals, every pass from the `N`-th on writes `f^[N] v`
(`loopState_scanCore_of_le`): the scan answers after `N` passes and then stands still.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Matrix

/-- Once every point has reached a fixed point of `f` by step `N`, the iterates stand still. -/
lemma iterate_add_eq {X : Type*} {f : X → X} {N : ℕ} (hf : ∀ v, f (f^[N] v) = f^[N] v) :
    ∀ j, f^[N + j] = f^[N]
  | 0 => rfl
  | j + 1 => by
    funext v
    rw [← add_assoc, Function.iterate_succ_apply', iterate_add_eq hf j, hf]

variable {V : Type} [DecidableEq V]

/-- The table of a map as a matrix of zeros and ones, `funMatrix f i j = [f i = j]`. -/
def funMatrix (f : V → V) : Matrix V V ℂ := fun i j => if f i = j then 1 else 0

variable [Fintype V]

/-- Following `f` and then `g` multiplies their tables. -/
lemma funMatrix_mul (f g : V → V) : funMatrix f * funMatrix g = funMatrix (g ∘ f) := by
  ext i k
  rw [Matrix.mul_apply, Fintype.sum_eq_single (f i) fun j hj => by simp [funMatrix, Ne.symm hj]]
  simp [funMatrix]

/-- **The powers of a table are the tables of the iterates.** -/
lemma funMatrix_pow (f : V → V) (n : ℕ) : funMatrix f ^ n = funMatrix f^[n] := by
  induction n with
  | zero =>
    ext i j
    by_cases h : i = j <;> simp [funMatrix, h]
  | succ n ih => rw [pow_succ, ih, funMatrix_mul, ← Function.iterate_succ']

/-- The one-hot key `v` times the table of `f` is the one-hot key `f v`. -/
lemma single_vecMul_funMatrix (f : V → V) (v : V) :
    Pi.single v 1 ᵥ* funMatrix f = Pi.single (f v) 1 := by
  ext j
  rw [single_one_vecMul]
  by_cases h : f v = j
  · subst h
    simp [funMatrix]
  · simp [funMatrix, h, Ne.symm h]

/-- The scan's stream: the table `A`, the key slots `P` that a pass reads and `S` that it
writes. -/
abbrev ScanSlot (V : Type) := (V × V) ⊕ (V ⊕ V)

/-- The scan's state: the table `A`, the read key `c` and the written key `s`. -/
def scanState (A : Matrix V V ℂ) (c s : V → ℂ) : ScanSlot V → ℂ :=
  Sum.elim (fun t => A t.1 t.2) (Sum.elim c s)

/-- The reading of the table entry `t`. -/
def scanTable (t : V × V) : (ScanSlot V → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inl t)

/-- The reading of the key slot `i` plus the message at `i`. -/
def scanKey (i : V) : (ScanSlot V ⊕ V → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inl (Sum.inr (Sum.inl i))) +
    LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inr i)

/-- **The scan core**: unit `(i, j)` multiplies the entry `A i j` by the reading of the key slot
`i` plus the message, and writes the product into `S j`. -/
def scanCore (w : ScanSlot V ⊕ V → ℂ) : ScanSlot V ⊕ V → ℂ :=
  w + ∑ t : V × V, (scanTable t (w ∘ Sum.inl) * scanKey t.1 w) •
    Pi.single (Sum.inl (Sum.inr (Sum.inr t.2))) 1

/-- The scan core is a scan layer of `|V|²` units: its gates read only the table. -/
lemma isScanLayer_scanCore : IsScanLayer (Fintype.card V ^ 2) (scanCore (V := V)) :=
  ⟨V × V, inferInstance, le_of_eq (by rw [Fintype.card_prod]; ring), scanTable,
    fun t => scanKey t.1, fun t => Pi.single (Sum.inl (Sum.inr (Sum.inr t.2))) 1, fun _ => rfl⟩

/-- **The message**: the written key minus the read one, `m = S − P`. -/
def scanRead : (ScanSlot V → ℂ) →ₗ[ℂ] V → ℂ :=
  LinearMap.pi fun j => LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inr (Sum.inr j)) -
    LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inr (Sum.inl j))

omit [DecidableEq V] [Fintype V] in
lemma scanRead_scanState (A : Matrix V V ℂ) (c s : V → ℂ) : scanRead (scanState A c s) = s - c :=
  rfl

/-- One pass with the message `m` adds `(c + m) A` to the written key and keeps the rest. -/
lemma scanCore_pass (A : Matrix V V ℂ) (c s m : V → ℂ) :
    scanCore (Sum.elim (scanState A c s) m) ∘ Sum.inl = scanState A c (s + (c + m) ᵥ* A) := by
  funext t
  rcases t with t | i | j
  · simp [scanCore, scanState]
  · simp [scanCore, scanState]
  · simp only [Function.comp_apply, scanCore, Pi.add_apply, Finset.sum_apply, Pi.smul_apply,
      smul_eq_mul, Pi.single_apply, Sum.inl.injEq, Sum.inr.injEq, mul_ite, mul_one, mul_zero]
    rw [Fintype.sum_prod_type]
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    simp [scanTable, scanKey, scanState, vecMul_apply_eq_sum, mul_comm]

/-- **The scan follows one link per pass**: pass `n + 1` from the read key `c` writes
`c A^{n+1}`. -/
theorem loopState_scanCore (A : Matrix V V ℂ) (c : V → ℂ) (n : ℕ) :
    loopState scanCore scanRead (scanState A c 0) n = scanState A c (c ᵥ* A ^ (n + 1)) := by
  induction n with
  | zero =>
    show scanCore (Sum.elim (scanState A c 0) 0) ∘ Sum.inl = _
    rw [scanCore_pass]
    simp
  | succ n ih =>
    rw [loopState, ih, scanRead_scanState, scanCore_pass, zero_add, add_sub_cancel, vecMul_vecMul,
      ← pow_succ]

/-- For the table of a map and a one-hot start `v`, pass `n + 1` writes the one-hot key
`f^[n+1] v`. -/
theorem loopState_scanCore_funMatrix (f : V → V) (v : V) (n : ℕ) :
    loopState scanCore scanRead (scanState (funMatrix f) (Pi.single v 1) 0) n =
      scanState (funMatrix f) (Pi.single v 1) (Pi.single (f^[n + 1] v) 1) := by
  rw [loopState_scanCore, funMatrix_pow, single_vecMul_funMatrix]

/-- **The scan answers after `N` passes and stands still**: if every key reaches a fixed point of
`f` within `N` steps, every pass from the `N`-th on writes the one-hot key `f^[N] v`. -/
theorem loopState_scanCore_of_le {f : V → V} {N : ℕ} (hf : ∀ v, f (f^[N] v) = f^[N] v) (v : V)
    {n : ℕ} (hn : N ≤ n + 1) :
    loopState scanCore scanRead (scanState (funMatrix f) (Pi.single v 1) 0) n =
      scanState (funMatrix f) (Pi.single v 1) (Pi.single (f^[N] v) 1) := by
  obtain ⟨j, hj⟩ := Nat.exists_eq_add_of_le hn
  rw [loopState_scanCore_funMatrix, hj, iterate_add_eq hf]

/-- Three links on four keys, the last key absorbing: from pass `3` on, the scan from key `0`
writes key `3`. -/
example (n : ℕ) (hn : 2 ≤ n) :
    loopState scanCore scanRead
        (scanState (funMatrix fun i : Fin 4 => if i = 3 then 3 else i + 1) (Pi.single 0 1) 0) n =
      scanState (funMatrix fun i : Fin 4 => if i = 3 then 3 else i + 1) (Pi.single 0 1)
        (Pi.single 3 1) :=
  loopState_scanCore_of_le (N := 3) (by decide) 0 (by omega)

end Transformer.Grokking.FlatSpectrum
