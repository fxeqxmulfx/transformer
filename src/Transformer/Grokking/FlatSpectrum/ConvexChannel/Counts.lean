module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Adaptive

/-!
# Causal count features preserve and enlarge the convex channel family

Source: `review/flat_spectrum.md`, CFt1-CFt3. Adding input-only features does
not multiply learned factors: the channel still learns one joint tensor.
Zero extension embeds every old channel with exactly the same predictions.
A fixed query with two separate Boolean slots cannot strictly classify XOR;
one count slot represents every rule of the count, including this witness.
This is capacity, not identification of counts absent from training.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K L V C : Type*} [Fintype K] [Fintype L] [Fintype C]

/-- Append keys with zero routing mass; all old coordinates remain intact. -/
def appendKeys (W : Channel Q K V C) : Channel Q (K ⊕ L) V C :=
  fun q k v c => Sum.elim (fun j => W q j v c) (fun _ => 0) k

/-- The old feasible domain embeds into the extended one (CFt1). -/
theorem feasible_appendKeys {W : Channel Q K V C} (hW : Feasible W) :
    Feasible (appendKeys (L := L) W) := by
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    cases k with
    | inl k => exact hW.1 q k v c
    | inr k => exact le_rfl
  · intro q k v w
    cases k with
    | inl k => exact hW.2.1 q k v w
    | inr k => simp [appendKeys]
  · intro q v
    simpa [appendKeys, Fintype.sum_sum_type] using hW.2.2 q v

omit [Fintype C] in
/-- Every old prediction is retained for arbitrary values of the new keys. -/
theorem logits_appendKeys (W : Channel Q K V C) (q : Q) (D : K → V)
    (extra : L → V) :
    logits (appendKeys W) q (Sum.elim D extra) = logits W q D := by
  funext c
  simp [logits, appendKeys, Fintype.sum_sum_type]

/-- The old loss is retained exactly, on any finite labeled sample (CFt1). -/
theorem answerLoss_appendKeys {S : Type*} [Fintype S]
    (query : S → Q) (data : S → K → V) (extra : S → L → V)
    (label : S → C) (W : Channel Q K V C) :
    answerLoss query (fun s => Sum.elim (data s) (extra s)) label (appendKeys W) =
      answerLoss query data label W := by
  unfold answerLoss
  apply Finset.sum_congr rfl
  intro s _
  rw [logits_appendKeys]

/-- Appending a fixed count observation retains jointly convex ordinary CE;
the count map can be any input-only function, not a learned product (CFt1). -/
theorem extended_answerLoss_convex {S : Type*} [Fintype S]
    (query : S → Q) (data : S → K → V) (extra : S → L → V) (label : S → C) :
    ConvexOn ℝ Set.univ
      (answerLoss query (fun s => Sum.elim (data s) (extra s)) label) :=
  answerLoss_convex query (fun s => Sum.elim (data s) (extra s)) label

/-- Query-specific parameters cannot repair XOR when the query is fixed
and the two features remain separate (CFt2). No feasibility assumption helps. -/
theorem no_fixed_query_xor (W : Channel Q Bool Bool Bool) (q : Q) :
    ¬ ∀ a b : Bool, logits W q (pairData a b) (!(Bool.xor a b)) <
      logits W q (pairData a b) (Bool.xor a b) := by
  have h := no_shared_xor (fun (_ : Unit) => W q)
  exact h

/-- The extra feature is a count of one token in the visible prefix only.
It contains neither a class label nor a decoder. -/
def countData {T : Type*} [DecidableEq T] (D : K → ℕ) (token : T)
    (xs : List T) : K ⊕ Unit → ℕ :=
  Sum.elim D (fun _ => xs.count token)

omit [Fintype K] in
/-- A finite context bound also bounds the new value alphabet (CFt3). -/
theorem countData_le {T : Type*} [DecidableEq T] (D : K → ℕ) (token : T)
    (xs : List T) (N : ℕ) (h : xs.length ≤ N) : countData D token xs (.inr ()) ≤ N := by
  exact (List.count_le_length (a := token) (l := xs)).trans h

/-- A possible learned count rule, used only as a parameter witness. -/
def countChannel [DecidableEq K] [DecidableEq C] (rule : Q → ℕ → C) :
    Channel Q (K ⊕ Unit) ℕ C :=
  vertex (fun _ => Sum.inr ()) (fun q _ n => rule q n)

/-- Any rule of the count is feasible, without constraining other learned
channels to use this feature (CFt3). -/
theorem feasible_countChannel [DecidableEq K] [DecidableEq C]
    (rule : Q → ℕ → C) : Feasible (countChannel (K := K) rule) := by
  apply feasible_vertex

omit [Fintype C] in
/-- The count-rule witness produces exactly its declared class (CFt3).
The rule is a model parameter, never an argument of the feature extractor. -/
theorem logits_countChannel [DecidableEq K] [DecidableEq C]
    {T : Type*} [DecidableEq T] (rule : Q → ℕ → C) (q : Q)
    (D : K → ℕ) (token : T) (xs : List T) (c : C) :
    logits (countChannel rule) q (countData D token xs) c =
      if c = rule q (xs.count token) then 1 else 0 := by
  simp [logits, countChannel, countData, vertex]
  split_ifs <;> simp_all

/-- Adding the count feature strictly repairs the two-bit capacity failure
with a unit margin (CFt3). Parity is specified only in the witness weights. -/
theorem count_xor_margin (a b : Bool) :
    let W := countChannel (K := Bool) (fun (_ : Unit) n => decide (n % 2 = 1))
    let D := countData (fun _ : Bool => 0) true [a, b]
    logits W () D (Bool.xor a b) - logits W () D (!(Bool.xor a b)) = 1 := by
  cases a <;> cases b <;>
    norm_num [logits_countChannel, List.count_cons]

example := feasible_appendKeys (L := Bool)
  (feasible_vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))

example := feasible_countChannel (K := Bool)
  (fun (_ : Unit) n => decide (n % 2 = 1))

example := countData_le (fun _ : Bool => 0) true [false, true] 2 (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
