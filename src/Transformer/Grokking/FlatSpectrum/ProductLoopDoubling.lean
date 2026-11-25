module

public import Transformer.Grokking.FlatSpectrum.ProductLoopScanCore

/-!
# The doubling loop: the table squares itself every pass

Source: `review/flat_spectrum.md`, *Loops*, item LP3. The doubling loop keeps the table `A` in the
prelude state and writes a table into scratch slots `S`. Unit `(l, i, j)` of its core multiplies
the reading of entry `(i, l)` by the reading of entry `(l, j)`, each the table slot plus the
message, and writes the product into `S (i, j)`: `|V|³` units (`isProductLayer_doublingCore`).
Both factors read the message, so it is no scan layer. The message is `S − A`, so each pass
squares the table the last one wrote, and pass `n + 1` writes `A^{2^{n+1}}`
(`loopState_doublingCore`).

For the table of a map, pass `n + 1` writes the table of `f^[2^{n+1}]`
(`loopState_doublingCore_funMatrix`). If every key reaches a fixed point of `f` within `N` steps,
every pass with `N ≤ 2^{n+1}` writes the table of `f^[N]` (`loopState_doublingCore_of_le`):
`⌈log₂ N⌉` passes, where a scan layer needs about `N`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Matrix

variable {V : Type}

/-- The doubling loop's stream: the table `A` and the written table `S`. -/
abbrev DoublingSlot (V : Type) := (V × V) ⊕ (V × V)

/-- The doubling loop's state: the table `A` and the written table `B`. -/
def doublingState (A B : Matrix V V ℂ) : DoublingSlot V → ℂ :=
  Sum.elim (fun t => A t.1 t.2) fun t => B t.1 t.2

/-- The reading of entry `t`: the table slot plus the message. -/
def doublingEntry (t : V × V) : (DoublingSlot V ⊕ V × V → ℂ) →ₗ[ℂ] ℂ :=
  LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inl (Sum.inl t)) +
    LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inr t)

/-- **The message**: the written table minus the table, `m = S − A`. -/
def doublingRead : (DoublingSlot V → ℂ) →ₗ[ℂ] V × V → ℂ :=
  LinearMap.pi fun t => LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inr t) -
    LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (Sum.inl t)

lemma doublingRead_doublingState (A B : Matrix V V ℂ) :
    doublingRead (doublingState A B) = fun t => (B - A) t.1 t.2 :=
  rfl

variable [Fintype V] [DecidableEq V]

/-- **The doubling core**: unit `(l, i, j)` multiplies the readings of entries `(i, l)` and
`(l, j)` and writes the product into `S (i, j)`. -/
def doublingCore (w : DoublingSlot V ⊕ V × V → ℂ) : DoublingSlot V ⊕ V × V → ℂ :=
  w + ∑ t : V × (V × V), (doublingEntry (t.2.1, t.1) w * doublingEntry (t.1, t.2.2) w) •
    Pi.single (Sum.inl (Sum.inr t.2)) 1

/-- The doubling core is a product layer of `|V|³` units. -/
lemma isProductLayer_doublingCore :
    IsProductLayer (Fintype.card V ^ 3) (doublingCore (V := V)) :=
  ⟨V × (V × V), inferInstance, le_of_eq (by rw [Fintype.card_prod, Fintype.card_prod]; ring),
    fun t => doublingEntry (t.2.1, t.1), fun t => doublingEntry (t.1, t.2.2),
    fun t => Pi.single (Sum.inl (Sum.inr t.2)) 1, fun _ => rfl⟩

/-- One pass with the message `M` adds `(A + M)²` to the written table and keeps the table. -/
lemma doublingCore_pass (A B M : Matrix V V ℂ) :
    doublingCore (Sum.elim (doublingState A B) fun t => M t.1 t.2) ∘ Sum.inl =
      doublingState A (B + (A + M) * (A + M)) := by
  funext t
  rcases t with t | ⟨i, j⟩
  · simp [doublingCore, doublingState]
  · simp only [Function.comp_apply, doublingCore, Pi.add_apply, Finset.sum_apply, Pi.smul_apply,
      smul_eq_mul, Pi.single_apply, Sum.inl.injEq, Sum.inr.injEq, mul_ite, mul_one, mul_zero]
    rw [Fintype.sum_prod_type]
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    simp [doublingEntry, doublingState, Matrix.mul_apply]

/-- **The table squares itself every pass**: pass `n + 1` writes `A^{2^{n+1}}`. -/
theorem loopState_doublingCore (A : Matrix V V ℂ) (n : ℕ) :
    loopState doublingCore doublingRead (doublingState A 0) n =
      doublingState A (A ^ 2 ^ (n + 1)) := by
  induction n with
  | zero =>
    show doublingCore (Sum.elim (doublingState A 0) fun t => (0 : Matrix V V ℂ) t.1 t.2) ∘
      Sum.inl = _
    rw [doublingCore_pass]
    simp [sq]
  | succ n ih =>
    rw [loopState, ih, doublingRead_doublingState, doublingCore_pass, zero_add, add_sub_cancel,
      ← pow_add, ← two_mul, ← pow_succ']

/-- For the table of a map, pass `n + 1` writes the table of `f^[2^{n+1}]`. -/
theorem loopState_doublingCore_funMatrix (f : V → V) (n : ℕ) :
    loopState doublingCore doublingRead (doublingState (funMatrix f) 0) n =
      doublingState (funMatrix f) (funMatrix f^[2 ^ (n + 1)]) := by
  rw [loopState_doublingCore, funMatrix_pow]

/-- **Doubling answers after `⌈log₂ N⌉` passes and stands still**: if every key reaches a fixed
point of `f` within `N` steps, every pass with `N ≤ 2^{n+1}` writes the table of `f^[N]`. -/
theorem loopState_doublingCore_of_le {f : V → V} {N : ℕ} (hf : ∀ v, f (f^[N] v) = f^[N] v)
    {n : ℕ} (hn : N ≤ 2 ^ (n + 1)) :
    loopState doublingCore doublingRead (doublingState (funMatrix f) 0) n =
      doublingState (funMatrix f) (funMatrix f^[N]) := by
  obtain ⟨j, hj⟩ := Nat.exists_eq_add_of_le hn
  rw [loopState_doublingCore_funMatrix, hj, iterate_add_eq hf]

/-- Seven links on eight keys, the last key absorbing: from pass `3` on, the doubling loop
writes the table of `f^[7]`, which sends every key to `7`; the scan would take seven passes. -/
example (n : ℕ) (hn : 2 ≤ n) :
    loopState doublingCore doublingRead
        (doublingState (funMatrix fun i : Fin 8 => if i = 7 then 7 else i + 1) 0) n =
      doublingState (funMatrix fun i : Fin 8 => if i = 7 then 7 else i + 1)
        (funMatrix fun _ => 7) := by
  rw [loopState_doublingCore_of_le (N := 7) (by decide) (by
    calc 7 ≤ 2 ^ 3 := by norm_num
      _ ≤ 2 ^ (n + 1) := Nat.pow_le_pow_right (by norm_num) (by omega))]
  congr 2
  funext i
  revert i
  decide

end Transformer.Grokking.FlatSpectrum
