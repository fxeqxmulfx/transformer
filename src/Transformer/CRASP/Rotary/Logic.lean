module

public import Transformer.CRASP.FixedFinite
public import Transformer.CRASP.FixedSign

/-!
# Counting logic with position-dependent integer coefficients

Extension of arXiv:2506.16055v3, Appendix B.2/F, specified in
`review/flat_spectrum.md`, *Uniform nonperiodic RoPE*, U1.
A kernel is fixed independently of the input word and gives a coefficient
for each query/source position pair. Counts bind the source position; nested
counts use their own query position. No transformer forward or word predicate
is admitted as a primitive. Source kernels are constructed in `Source`.
-/

@[expose] public section

namespace Transformer.CRASP.Rotary

universe u v

mutual
/-- Boolean tests and integer comparisons, with a first-position predicate. -/
inductive Form (σ : Type u) (κ : Type v) : Type (max u v)
  | truth : Bool → Form σ κ
  | sym : σ → Form σ κ
  | first : Form σ κ
  | and : Form σ κ → Form σ κ → Form σ κ
  | or : Form σ κ → Form σ κ → Form σ κ
  | neg : Form σ κ → Form σ κ
  | lt : Term σ κ → Term σ κ → Form σ κ
/-- Integer arithmetic and causal weighted counts over position pairs. -/
inductive Term (σ : Type u) (κ : Type v) : Type (max u v)
  | const : ℤ → Term σ κ
  | add : Term σ κ → Term σ κ → Term σ κ
  | scale : ℤ → Term σ κ → Term σ κ
  | count : κ → Form σ κ → Term σ κ
end

variable {σ : Type u} {κ : Type v}

mutual
/-- Only counts increase depth; the coefficients themselves read no tokens. -/
def Form.depth : Form σ κ → ℕ
  | .truth _ | .sym _ | .first => 0
  | .and φ ψ | .or φ ψ => max φ.depth ψ.depth
  | .neg φ => φ.depth
  | .lt A B => max A.depth B.depth
def Term.depth : Term σ κ → ℕ
  | .const _ => 0
  | .add A B => max A.depth B.depth
  | .scale _ A => A.depth
  | .count _ φ => φ.depth + 1
end

variable [DecidableEq σ]

mutual
/-- Interpret a fixed kernel family on a token word and a positive position. -/
def Form.sat (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) : Form σ κ → Bool
  | .truth b => b
  | .sym a => decide (w[i - 1]? = some a)
  | .first => decide (i = 1)
  | .and φ ψ => φ.sat K w i && ψ.sat K w i
  | .or φ ψ => φ.sat K w i || ψ.sat K w i
  | .neg φ => !(φ.sat K w i)
  | .lt A B => decide (A.val K w i < B.val K w i)
/-- The query index stays fixed while the counted predicate reads each source. -/
def Term.val (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) : Term σ κ → ℤ
  | .const z => z
  | .add A B => A.val K w i + B.val K w i
  | .scale z A => z * A.val K w i
  | .count k φ => ∑ j ∈ Finset.Icc 1 i, if φ.sat K w j then K k i j else 0
end

/-- Finite disjunction in the explicit syntax. -/
def Form.any : List (Form σ κ) → Form σ κ
  | [] => .truth false
  | φ :: L => .or φ (any L)
/-- Finite conjunction in the explicit syntax. -/
def Form.all : List (Form σ κ) → Form σ κ
  | [] => .truth true
  | φ :: L => .and φ (all L)
/-- A finite sum uses only the integer addition constructor. -/
def Term.sum : List (Term σ κ) → Term σ κ
  | [] => .const 0
  | A :: L => .add A (sum L)

omit [DecidableEq σ] in
/-- A disjunction introduces no additional counting (U1). -/
theorem Form.depth_any (L : List (Form σ κ)) (d : ℕ)
    (h : ∀ φ ∈ L, φ.depth ≤ d) : (any L).depth ≤ d := by
  induction L with
  | nil => exact Nat.zero_le _
  | cons φ L ih =>
      exact max_le (h _ (List.mem_cons_self ..))
        (ih fun ψ hψ => h _ (List.mem_cons_of_mem _ hψ))

omit [DecidableEq σ] in
/-- Parallel coordinate tests introduce no additional counting (U1). -/
theorem Form.depth_all (L : List (Form σ κ)) (d : ℕ)
    (h : ∀ φ ∈ L, φ.depth ≤ d) : (all L).depth ≤ d := by
  induction L with
  | nil => exact Nat.zero_le _
  | cons φ L ih =>
      exact max_le (h _ (List.mem_cons_self ..))
        (ih fun ψ hψ => h _ (List.mem_cons_of_mem _ hψ))

omit [DecidableEq σ] in
/-- A finite sum retains the maximum depth of its summands (U1). -/
theorem Term.depth_sum (L : List (Term σ κ)) (d : ℕ)
    (h : ∀ A ∈ L, A.depth ≤ d) : (sum L).depth ≤ d := by
  induction L with
  | nil => exact Nat.zero_le _
  | cons A L ih =>
      exact max_le (h _ (List.mem_cons_self ..))
        (ih fun B hB => h _ (List.mem_cons_of_mem _ hB))

/-- Finite disjunction has existential semantics (U1). -/
theorem Form.sat_any (L : List (Form σ κ)) (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) :
    (any L).sat K w i = true ↔ ∃ φ ∈ L, φ.sat K w i = true := by
  induction L with
  | nil => simp [any, sat]
  | cons φ L ih => simp [any, sat, ih]

/-- Finite conjunction has universal semantics (U1). -/
theorem Form.sat_all (L : List (Form σ κ)) (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) :
    (all L).sat K w i = true ↔ ∀ φ ∈ L, φ.sat K w i = true := by
  induction L with
  | nil => simp [all, sat]
  | cons φ L ih => simp [all, sat, ih]

/-- The syntactic sum agrees with integer addition (U1). -/
theorem Term.val_sum (L : List (Term σ κ)) (K : κ → ℕ → ℕ → ℤ) (w : List σ) (i : ℕ) :
    (sum L).val K w i = (L.map (fun A => A.val K w i)).sum := by
  induction L with
  | nil => rfl
  | cons A L ih => simp [sum, val, ih]

/-- Ordinary causal counting is the constant-one kernel, so the extension
retains the original counting operation rather than replacing its semantics. -/
theorem Term.ordinary_count (φ : Form σ Unit) (w : List σ) (i : ℕ) :
    (Term.count () φ).val (fun _ _ _ => 1) w i =
      ((Finset.Icc 1 i).filter (fun j => φ.sat (fun _ _ _ => 1) w j)).card := by
  change (∑ j ∈ Finset.Icc 1 i, if φ.sat (fun _ _ _ => 1) w j then (1 : ℤ) else 0) = _
  exact Finset.sum_boole _ _

/-- Concrete nonempty formula and term lists satisfy the depth assumptions. -/
example : (∀ φ ∈ ([.sym true] : List (Form Bool Unit)), φ.depth ≤ 0) ∧
    (∀ A ∈ ([.const 3] : List (Term Bool Unit)), A.depth ≤ 0) := by
  simp [Form.depth, Term.depth]

end Transformer.CRASP.Rotary
