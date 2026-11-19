module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Model
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedHinge.Projection.Raw

/-!
# Immutable objects address learned query, position and adjacent-pair weights

Source: `review/convex/moments.md`, CM86. The last visible object is the query;
the remaining prefix reads learned position/object weights and learned adjacent
pair weights shared across positions. All overlapping pairs and repetitions
contribute. No key/value roles, arithmetic operations or answers are built into
this reader. The diagnostic address rows certify the actual weight lookups.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear

open scoped BigOperators Matrix
open LearnedHinge (Prefix)

variable {O : Type*} [Fintype O] [DecidableEq O] {T : ℕ}

/-- Remove only the final visible object, which supplies the query, CM86. -/
def past (tokens : Prefix O (T + 1)) : Prefix O T :=
  ⟨tokens.val.dropLast, by have h := tokens.property; simp only [List.length_dropLast]; omega⟩

/-- Query bias and original object addresses, CM86. -/
abbrev QueryCoordinate (O : Type*) := Option O

/-- One shared address per ordered object pair, when a pair can occur, CM86.
The zero-sized type for T < 2 is justified by `no_pair_position`. -/
abbrev PairCoordinate (O : Type*) (T : ℕ) := Fin (if 2 ≤ T then 1 else 0) × O × O

/-- Bias, all position/object entries and all possible shared pair entries,
CM86. The coordinates are addresses of learned weights. -/
abbrev ContextCoordinate (O : Type*) (T : ℕ) :=
  Option ((Fin T × O) ⊕ PairCoordinate O T)

/-- Every possible visible adjacent position uses the same pair table, CM86. -/
def pairFlag (k : Fin (T - 1)) : Fin (if 2 ≤ T then 1 else 0) :=
  ⟨0, by
    have h : 2 ≤ T := by have hk := k.isLt; omega
    simp [h]⟩

/-- Ordered adjacent objects at a position; absent positions contribute zero,
CM86. Repeated occurrences retain separate summands in the reader. -/
def pairAt (tokens : Prefix O T) (k : Fin (T - 1)) : Option (PairCoordinate O T) :=
  match tokens.val[k.val]?, tokens.val[k.val + 1]? with
  | some a, some b => some (pairFlag k, a, b)
  | _, _ => none

/-- Actual query lookup of learned weights, including the learned bias, CM86. -/
def readQuery (tokens : Prefix O (T + 1)) (w : QueryCoordinate O → ℂ) : ℂ :=
  w none + match tokens.val.getLast? with
    | none => 0
    | some o => w (some o)

/-- Actual learned context lookup, retaining every position and every visible
adjacent occurrence, CM86. Pair weights are shared, not occurrence-specific. -/
def readContext (tokens : Prefix O T) (w : ContextCoordinate O T → ℂ) : ℂ :=
  w none + (∑ i : Fin T, match tokens.val[i.val]? with
    | none => 0
    | some o => w (some (.inl (i, o)))) +
  ∑ k : Fin (T - 1), match pairAt tokens k with
    | none => 0
    | some p => w (some (.inr p))

/-- Diagnostic query row for the learned lookup identity, CM86. -/
def queryRow (tokens : Prefix O (T + 1)) : QueryCoordinate O → ℂ
  | none => 1
  | some o => if tokens.val.getLast? = some o then 1 else 0

/-- Diagnostic context row; a pair entry records its full multiplicity,
CM86. The pair flag is part of the tested address. -/
def contextRow (tokens : Prefix O T) : ContextCoordinate O T → ℂ
  | none => 1
  | some (.inl (i, o)) => if tokens.val[i.val]? = some o then 1 else 0
  | some (.inr p) => ∑ k : Fin (T - 1), if pairAt tokens k = some p then 1 else 0

/-- Finite address sums recover an optional learned lookup exactly, CM86. -/
theorem optional_lookup {A : Type*} [Fintype A] [DecidableEq A]
    (v : Option A) (w : A → ℂ) :
    (∑ a, w a * (if v = some a then 1 else 0)) =
      match v with | none => 0 | some a => w a := by
  cases v with
  | none => simp
  | some a => simpa only [Option.some.injEq, mul_ite, mul_one, mul_zero] using
      Fintype.sum_ite_eq a w

/-- The query's diagnostic row equals the actual learned object lookup,
CM86, including the empty-prefix bias. -/
theorem queryRow_read (tokens : Prefix O (T + 1)) (w : QueryCoordinate O → ℂ) :
    w ⬝ᵥ queryRow tokens = readQuery tokens w := by
  simp only [dotProduct, Fintype.sum_option, queryRow, mul_one, readQuery]
  rw [optional_lookup]

/-- Every learned position and adjacent-pair weight is read with its exact
multiplicity, CM86. No object class, occurrence or overlap is dropped. -/
theorem contextRow_read (tokens : Prefix O T) (w : ContextCoordinate O T → ℂ) :
    w ⬝ᵥ contextRow tokens = readContext tokens w := by
  simp only [dotProduct, Fintype.sum_option, contextRow, mul_one,
    Fintype.sum_sum_type, readContext]
  rw [← add_assoc]
  congr 1
  · congr 1
    rw [Fintype.sum_prod_type]
    apply Finset.sum_congr rfl
    intro i hi
    simpa using optional_lookup (A := O) tokens.val[i.val]?
      (fun o => w (some (.inl (i, o))))
  · simp only [Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro k hk
    cases pairAt tokens k <;> simp

/-- Short context caps admit no adjacent positions, CM86. Removing the
zero-sized pair table at these caps loses no possible occurrence. -/
theorem no_pair_position (hT : T < 2) : IsEmpty (Fin (T - 1)) := by
  have h : T - 1 = 0 := by omega
  rw [h]
  infer_instance

example : IsEmpty (Fin (1 - 1)) := no_pair_position (by decide : 1 < 2)

omit [DecidableEq O] in
/-- The query-sized factor width retains every original object plus bias,
CM86. -/
theorem query_coordinate_count : Fintype.card (QueryCoordinate O) = 1 + Fintype.card O := by
  simp only [Fintype.card_option]
  omega

omit [DecidableEq O] in
/-- Complete context count, including all object pairs when any pair is
possible, CM86. This is not a sparse observed-pair count. -/
theorem context_coordinate_count :
    Fintype.card (ContextCoordinate O T) =
      1 + T * Fintype.card O + (if 2 ≤ T then 1 else 0) * Fintype.card O ^ 2 := by
  simp only [Fintype.card_option, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  ring

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear
