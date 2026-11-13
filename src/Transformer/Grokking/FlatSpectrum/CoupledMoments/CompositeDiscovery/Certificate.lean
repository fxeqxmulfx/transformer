module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.ModularKernel
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Completion

/-!
# From the original categorical matrix to the identification theorem

Source: `review/flat_spectrum.md`, CM28, the observed matrix certificate.
The matrix below contains every original observation, the two shift gauges,
and one class-coordinate row. Replaying row operations to coordinate rows
proves that its kernel is zero. A checked normalized joint vector then proves
cyclicity of the original joint module. The matrix contains no pin for a
missing axis and no unseen label. The certificate is a finite collection of
equations on actual data, rather than a claimed rank or optimizer status.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

variable {R n C S : Type*} [CommRing R]
variable [Fintype n] [Fintype C] [DecidableEq n] [DecidableEq C]

/-- A coordinate row in the actual coefficient matrix, CM28. -/
def coordinate {V : Type*} [DecidableEq V] (v : V) : V → R :=
  fun j => if v = j then 1 else 0

/-- The original observed matrix plus two gauges and one checked anchor
row, CM28. Row/column/class names are compared only for equality. -/
def anchoredMatrix (cell : S → n × n) (labels : S → C) (i₀ : n) (c₀ c₁ : C) :
    S ⊕ Fin 3 → n ⊕ (n ⊕ C) → R
  | .inl t => fun v => coordinate (.inl (cell t).1) v +
      coordinate (.inr (.inl (cell t).2)) v - coordinate (.inr (.inr (labels t))) v
  | .inr k => if k = 0 then coordinate (.inl i₀)
      else if k = 1 then coordinate (.inr (.inr c₀)) else coordinate (.inr (.inr c₁))

/-- Membership in the real joint module and a zero anchor imply every
actual matrix equation, CM28. Arbitrary extra constraints cannot enter here. -/
theorem anchoredMatrix_zero {cell : S → n × n} {labels : S → C}
    {i₀ : n} {c₀ c₁ : C} {x : n ⊕ (n ⊕ C) → R}
    (hx : x ∈ jointKernel R cell labels i₀ c₀) (h1 : x (.inr (.inr c₁)) = 0) :
    ∀ i, rowAction (anchoredMatrix cell labels i₀ c₀ c₁) x i = 0 := by
  intro i
  rcases i with t | k
  · simpa [rowAction, anchoredMatrix, coordinate, sub_mul, add_mul,
      Finset.sum_sub_distrib, Finset.sum_add_distrib, ite_mul] using sub_eq_zero.mpr (hx.1 t)
  · fin_cases k <;> simp [rowAction, anchoredMatrix, coordinate, ite_mul, hx.2.1, hx.2.2, h1]

variable [DecidableEq S]

/-- The actual observed matrix and its replayed instructions certify the
joint cyclicity used by the completion theorem, CM28. No unknown true code,
unseen answer or semantic rank claim is among the certificate inputs. -/
theorem cyclic_of_row_certificate {cell : S → n × n} {labels : S → C}
    {i₀ : n} {c₀ c₁ : C} {g : n ⊕ (n ⊕ C) → R}
    (hg : g ∈ jointKernel R cell labels i₀ c₀) (h1 : g (.inr (.inr c₁)) = 1)
    (ops : List (RowOp (S ⊕ Fin 3) R)) (pivot : n ⊕ (n ⊕ C) → S ⊕ Fin 3)
    (hrows : ∀ j k, reduceRows ops (anchoredMatrix cell labels i₀ c₀ c₁) (pivot j) k =
      if j = k then 1 else 0) : Cyclic (jointKernel R cell labels i₀ c₀) := by
  apply cyclic_of_anchored_kernel hg h1
  intro x hx hx1
  exact kernel_zero_of_rows ops _ pivot hrows (anchoredMatrix_zero hx hx1)

/-- The same replayed certificate makes the actual normalized observed
vector unique, CM28. This checks a shared anchor value, not an unknown code. -/
theorem normalized_vector_unique {cell : S → n × n} {labels : S → C}
    {i₀ : n} {c₀ c₁ : C} {x y : n ⊕ (n ⊕ C) → R}
    (hx : x ∈ jointKernel R cell labels i₀ c₀)
    (hy : y ∈ jointKernel R cell labels i₀ c₀)
    (ha : x (.inr (.inr c₁)) = y (.inr (.inr c₁)))
    (ops : List (RowOp (S ⊕ Fin 3) R)) (pivot : n ⊕ (n ⊕ C) → S ⊕ Fin 3)
    (hrows : ∀ j k, reduceRows ops (anchoredMatrix cell labels i₀ c₀ c₁) (pivot j) k =
      if j = k then 1 else 0) : x = y := by
  have hd := (jointKernel R cell labels i₀ c₀).sub_mem hx hy
  have h0 : (x - y) (.inr (.inr c₁)) = 0 := by
    change x (.inr (.inr c₁)) - y (.inr (.inr c₁)) = 0
    rw [ha, sub_self]
  exact sub_eq_zero.mp (kernel_zero_of_rows ops _ pivot hrows (anchoredMatrix_zero hd h0))

/-- A small exact observation set used to exercise the complete matrix
certificate, with its lower-right cell unobserved. -/
def binaryCells : Fin 3 → Fin 2 × Fin 2 := ![(0, 0), (0, 1), (1, 0)]

/-- Labels of the three cells in a two-class cyclic table. -/
def binaryLabels : Fin 3 → ZMod 2 := ![0, 1, 1]

/-- Row and column phases behind the certificate witness. -/
def binaryPhase (i : Fin 2) : ZMod 2 := i.val

/-- The actual normalized joint vector of the witness. -/
def binaryVector : Fin 2 ⊕ (Fin 2 ⊕ ZMod 2) → ZMod 2 :=
  pack binaryPhase binaryPhase id

/-- Six explicit row additions; no symbolic oracle reports their result. -/
def binaryProgram : List (RowOp (Fin 3 ⊕ Fin 3) (ZMod 2)) :=
  [.add (.inl 0) (.inr 0) (-1), .add (.inl 0) (.inr 1) 1,
    .add (.inl 1) (.inr 0) (-1), .add (.inl 1) (.inr 2) 1,
    .add (.inl 2) (.inl 0) (-1), .add (.inl 2) (.inr 2) 1]

/-- Locate each coordinate row after the actual six instructions. -/
def binaryPivot : Fin 2 ⊕ (Fin 2 ⊕ ZMod 2) → Fin 3 ⊕ Fin 3
  | .inl i => if i = 0 then .inr 0 else .inl 2
  | .inr (.inl j) => if j = 0 then .inl 0 else .inl 1
  | .inr (.inr c) => if c = 0 then .inr 1 else .inr 2

/-- Finite replay proves the entire original anchored matrix certificate,
CM28. This is not an assumption about a solver's answer. -/
theorem binary_rows : ∀ j k,
    reduceRows binaryProgram (anchoredMatrix binaryCells binaryLabels 0 0 1)
      (binaryPivot j) k = if j = k then 1 else 0 := by
  decide

/-- The replayed original matrix certifies the observable joint condition. -/
theorem binary_cyclic : Cyclic (jointKernel (ZMod 2) binaryCells binaryLabels 0 0) := by
  apply cyclic_of_row_certificate (g := binaryVector) ?_ (by decide)
    binaryProgram binaryPivot binary_rows
  refine ⟨?_, ?_, ?_⟩ <;> decide

example : ∀ i, rowAction (anchoredMatrix binaryCells binaryLabels 0 0 1)
    (fun _ => (0 : ZMod 2)) i = 0 :=
  anchoredMatrix_zero (by refine ⟨?_, ?_, ?_⟩ <;> decide) (by decide)

example : Cyclic (jointKernel (ZMod 2) binaryCells binaryLabels 0 0) :=
  cyclic_of_row_certificate (g := binaryVector)
    (by refine ⟨?_, ?_, ?_⟩ <;> decide) (by decide)
    binaryProgram binaryPivot binary_rows

example : binaryVector = binaryVector :=
  normalized_vector_unique
    (by refine ⟨?_, ?_, ?_⟩ <;> decide) (by refine ⟨?_, ?_, ?_⟩ <;> decide)
    (c₁ := 1) rfl binaryProgram binaryPivot binary_rows

/-- The finite matrix certificate identifies a truly held-out answer under
unknown structural realizability, through the observed-only learner, CM28. -/
theorem binary_recovers : (discover 2 binaryCells binaryLabels 0 0).map (fun y => y 1 1) =
    some 0 := by
  rw [discover_eq_some (Equiv.refl (ZMod 2)) binaryPhase binaryPhase
    (by decide) 0 0 binary_cyclic]
  decide

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery
