module

public import Transformer.Grokking.Convex.ConvexDepth.RepeatedRows.Model
public import Transformer.Grokking.Convex.ConvexDepth.UniformRows.Projection

/-!
# Complete complex coefficients, products and storage bounds

Source: `review/convex/moments.md`, CM70. The same row representation stores
every real and imaginary head coefficient. A sparse source column permits
shared class values outside its actual exceptional classes. The capacity
budget still charges the complete logical head, irrespective of stored size.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.RepeatedRows

open scoped BigOperators
open ObservedRows
open LatentPrograms (features head retained)
open Landscape.Neural (crossEntropy)

variable {F C : Type*} [Fintype C]

/-- Store each complete realified complex coefficient row, CM70. -/
noncomputable def packHead (baseline : F → ℝ) (W : F → C → ℝ) : F → Row C ℝ :=
  fun f => encode (baseline f) (W f)

/-- Recover every original class coefficient of a stored head, CM70. -/
noncomputable def unpackHead (W : F → Row C ℝ) : F → C → ℝ :=
  fun f => read (W f)

/-- No coefficient disappears when equal values share storage, CM70. -/
theorem unpack_packHead (baseline : F → ℝ) (W : F → C → ℝ) :
    unpackHead (packHead baseline W) = W := by
  funext f
  exact read_encode (baseline f) (W f)

variable {J X S : Type*} [Fintype J] [Fintype S] [DecidableEq S] [DecidableEq C]

omit [Fintype J] [DecidableEq S] in
/-- Decoder and head compression together preserve the entire complete
complex head, CM70, for every real parameter tensor and arbitrary source. -/
theorem complete_head (U : Matrix S S ℂ) (labels : J → S → Option C)
    (baseline : (j : J) → Unknown labels j → ℝ) (r : Reduced labels)
    (headBaseline : J × (S × Bool) → ℝ) :
    unpackHead (packHead headBaseline
      (head U 1 (expand labels (unpack labels (pack labels baseline r))))) =
      head U 1 (expand labels r) := by
  rw [unpack_packHead]
  exact congrArg (head U 1) (expand_unpack_pack labels baseline r)

omit [DecidableEq S] in
/-- Every actual realified complex product is unchanged at every query,
CM70, including states never observed during fitting. -/
theorem complex_logits (U : Matrix S S ℂ) (program : J → X → S)
    (labels : J → S → Option C) (baseline : (j : J) → Unknown labels j → ℝ)
    (r : Reduced labels) (headBaseline : J × (S × Bool) → ℝ) (x : X) :
    headLogits (features U program)
      (unpackHead (packHead headBaseline
        (head U 1 (expand labels (unpack labels (pack labels baseline r)))))) x =
      headLogits (features U program) (head U 1 (expand labels r)) x := by
  have he := complete_head U labels baseline r headBaseline
  exact congrArg (fun W => headLogits (features U program) W x) he

omit [DecidableEq S] in
/-- Ordinary indexed CE uses the same logits, record IDs and actual batch
denominator after both encodings, CM70. No grouping or deduplication occurs. -/
theorem complex_batch_ce {I : Type*} (U : Matrix S S ℂ) (program : J → X → S)
    (labels : J → S → Option C) (baseline : (j : J) → Unknown labels j → ℝ)
    (r : Reduced labels) (headBaseline : J × (S × Bool) → ℝ)
    (inputs : I → X) (targets : I → C) (batch : Finset I) :
    (∑ i ∈ batch, crossEntropy (headLogits (features U program)
      (unpackHead (packHead headBaseline
        (head U 1 (expand labels (unpack labels (pack labels baseline r))))))
      (inputs i)) (targets i)) / (batch.card : ℝ) =
      BoundedLatent.batchLoss U program inputs targets batch (expand labels r) := by
  rw [complete_head]
  rfl

/-- The complete coefficient budget remains charged to every logical class,
CM70. Storing repeated numbers once does not reduce the mathematical norm. -/
theorem complex_budget (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    (baseline : (j : J) → Unknown (dictionary program samples) j → ℝ)
    (r : Reduced (dictionary program samples))
    (hr : r ∈ domain (dictionary program samples) (retained program order samples) δ R)
    (headBaseline : J × (S × Bool) → ℝ) :
    unpackHead (packHead headBaseline (head U 1 (expand (dictionary program samples)
      (unpack _ (pack _ baseline r))))) ∈ headBudget (2 * R * Fintype.card S) := by
  rw [complete_head]
  exact complete_budget U hU program order samples hr

omit [Fintype J] [Fintype C] [DecidableEq S] [DecidableEq C] in
/-- Two raw columns agreeing on a source column's support give exactly the
same conjugate coefficient, CM70. Zero source entries outside the support are
proved irrelevant to this coefficient, not to other columns or head budgets. -/
theorem table_column_agreement (U : Matrix S S ℂ) (T : Matrix S C ℝ)
    (f : S) (c d : C) (active : Finset S)
    (hU : ∀ s, s ∉ active → U s f = 0)
    (hT : ∀ s, s ∈ active → T s c = T s d) :
    tableHead U T f c = tableHead U T f d := by
  classical
  simp only [tableHead, Matrix.mul_apply, Matrix.conjTranspose_apply, complexTable]
  apply Finset.sum_congr rfl
  intro s _
  by_cases hs : s ∈ active
  · rw [hT s hs]
  · rw [hU s hs]
    simp

omit [Fintype J] [DecidableEq S] [DecidableEq C] in
/-- A real or imaginary complete head row stores at most one shared value
plus its actual exceptional classes, CM70. Arbitrary nonuniform classes remain
explicit; this bound imposes no new constraint on their values. -/
theorem head_slots_bound (U : Matrix S S ℂ) (T : Matrix S C ℝ)
    (f : S) (imaginary : Bool) (c₀ : C) (active : Finset S) (allowed : Finset C)
    (hU : ∀ s, s ∉ active → U s f = 0)
    (hT : ∀ s, s ∈ active → ∀ c, c ∉ allowed → T s c = T s c₀) :
    slots (encode (complexCoefficients (tableHead U T) (f, imaginary) c₀)
      (complexCoefficients (tableHead U T) (f, imaginary))) ≤ 1 + allowed.card := by
  apply slots_le_exceptions
  intro c hc
  have he := table_column_agreement U T f c c₀ active hU (fun s hs => hT s hs c hc)
  unfold complexCoefficients
  rw [he]

omit [Fintype J] [DecidableEq S] in
/-- For class-uniform unknown rows, only the actual known labels within a
source column's support need exceptional head values, CM70. This is a
conditional storage bound over the full model; nonuniform rows still use the
unrestricted lossless representation. The shared class must be outside that set. -/
theorem uniform_unknown_head_slots (U : Matrix S S ℂ) (labels : J → S → Option C)
    (r : Reduced labels) (j : J) (f : S) (imaginary : Bool) (c₀ : C)
    (active : Finset S) (allowed : Finset C)
    (hU : ∀ s, s ∉ active → U s f = 0)
    (hlabels : ∀ s, s ∈ active → ∀ y, labels j s = some y → y ∈ allowed)
    (hc₀ : c₀ ∉ allowed) (hu : UniformRows.ClassUniform labels r) :
    slots (encode (head U 1 (expand labels r) (j, (f, imaginary)) c₀)
      (head U 1 (expand labels r) (j, (f, imaginary)))) ≤ 1 + allowed.card := by
  apply head_slots_bound U (fun s c => 1 * (expand labels r).2 j s c)
    f imaginary c₀ active allowed hU
  intro s hs c hc
  cases hl : labels j s with
  | none =>
    rw [expand_unknown labels r j ⟨s, hl⟩ c, expand_unknown labels r j ⟨s, hl⟩ c₀]
    exact congrArg (fun v : ℝ => 1 * v) (hu j ⟨s, hl⟩ c c₀)
  | some y =>
    have hy := hlabels s hs y hl
    have hcy : c ≠ y := by intro he; apply hc; simpa [he] using hy
    have hby : c₀ ≠ y := by intro he; apply hc₀; simpa [he] using hy
    rw [expand_known labels r j s y hl c, expand_known labels r j s y hl c₀]
    simp [hcy, hby]

end Transformer.Grokking.Convex.ConvexDepth.RepeatedRows
