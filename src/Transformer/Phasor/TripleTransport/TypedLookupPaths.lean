module

public import Transformer.Phasor.TripleTransport.LocalReadLoop
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Repeating causal keyed reads and learning their endpoint coefficients

The Phasor synthetic lookup study uses the latest write for each directed
key and relation. This adapts Percepta's Append-only Lookup Machine,
“LookUp gates”, and the shared passes in Thinking Machines Lab's
*Introducing Inkling 2 Mini*, “Looping design”. Each read sees the same
prefix. Relations and endpoint votes are addressed by dense geometry.
The supervised loss is ordinary full-vocabulary cross-entropy.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.TypedLookupPaths

/-- A store holds an optional successor for each directed key and relation.
Applying a new write at that address replaces its earlier value. -/
def write {A B : Type*} [DecidableEq A] (m : A → Option B) (a : A) (v : B) :
    A → Option B := fun k => if k = a then some v else m k

/-- Later writes to different addresses preserve the most recent matching
value. An address can itself be a pair of source and directed relation. -/
theorem latest {A B : Type*} [DecidableEq A] (m : A → Option B) (a : A) (v : B)
    (later : List (A × B)) (apart : ∀ p ∈ later, p.1 ≠ a) :
    (later.foldl (fun acc p => write acc p.1 p.2) (write m a v)) a = some v := by
  have keep (n : A → Option B) (xs : List (A × B)) (h : ∀ p ∈ xs, p.1 ≠ a) :
      (xs.foldl (fun acc p => write acc p.1 p.2) n) a = n a := by
    induction xs generalizing n with
    | nil => rfl
    | cons p xs ih =>
      rw [List.foldl_cons, ih _ (fun q hq => h q (.tail _ hq))]
      simp [write, Ne.symm (h p (.head _))]
  rw [keep _ later apart]
  simp [write]

example : ([(2,4),(1,5)].foldl (fun m p => write m p.1 p.2)
    (write (fun _ : ℕ => some 8) 0 7)) 0 = some 7 := by
  apply latest
  simp

variable {K V : Type*}

def follow (read : K → Option K) (key : K) : ℕ → K
  | 0 => key
  | n+1 => (read (follow read key n)).getD (follow read key n)

/-- A genuine path can end before the fixed loop count. After the last
defined link its endpoint remains the query of every subsequent read. -/
theorem saturates (read : K → Option K) (x : ℕ → K) (n extra : ℕ)
    (edges : ∀ i < n, read (x i) = some (x (i+1)))
    (terminal : read (x n) = none) :
    follow read (x 0) (n+extra) = x n := by
  have along : ∀ i ≤ n, follow read (x 0) i = x i := by
    intro i hi
    induction i with
    | zero => rfl
    | succ i ih =>
      rw [follow, ih (by omega), edges i (by omega)]
      rfl
  induction extra with
  | zero => exact along n (by omega)
  | succ extra ih =>
    rw [Nat.add_succ, follow, ih, terminal]
    rfl

/-- A final read may use a different relation and return a different
value type. The same repeat count serves all shorter terminating paths. -/
theorem finish (read : K → Option K) (tail : K → Option V)
    (x : ℕ → K) (n extra : ℕ) (v : V)
    (edges : ∀ i < n, read (x i) = some (x (i+1)))
    (terminal : read (x n) = none) (out : tail (x n) = some v) :
    tail (follow read (x 0) (n+extra)) = some v := by
  rw [saturates read x n extra edges terminal, out]

private def chain (k : ℕ) : Option ℕ :=
  if k < 2 then some (k+1) else none

example : follow chain 0 (2+5) = 2 :=
  saturates chain id 2 5 (by intro i hi; simp [chain,hi]) (by decide)

example : (fun k => if k=2 then some "v" else none) (follow chain 0 (2+5)) = some "v" :=
  finish chain (fun k => if k=2 then some "v" else none) id 2 5 "v" (by intro i hi; simp [chain,hi]) (by decide) (by decide)

open scoped BigOperators

def score {d : ℕ} (e z : Fin d → ℝ) : ℝ := ∑ i, e i * z i

/-- All path witnesses addressing one scalar parameter contribute their
signed endpoint vectors to a single direction v. For any other residual z,
a positive increase of that parameter improves a tied-logit margin exactly
when the direction favors the correct class. The statement covers collisions
between different directed paths. -/
theorem gain_iff {d : ℕ} (good rival z v : Fin d → ℝ) (k : ℝ) (positive : 0 < k) :
    score good z - score rival z <
        score good (fun i => z i+k*v i) - score rival (fun i => z i+k*v i) ↔
    score rival v < score good v := by
  have linear (e : Fin d → ℝ) :
      score e (fun i => z i+k*v i) = score e z + k*score e v := by
    simp only [score, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [linear, linear]
  constructor
  · intro h
    by_contra! bad
    nlinarith [mul_nonneg positive.le (sub_nonneg.mpr bad)]
  · intro h
    nlinarith [mul_pos positive (sub_pos.mpr h)]

example : score (d:=1) (fun _ : Fin 1 => 3) (fun _ => 1) - score (d:=1) (fun _ => 1) (fun _ => 1) <
    score (d:=1) (fun _ => 3) (fun _ => 1+2*4) - score (d:=1) (fun _ => 1) (fun _ => 1+2*4) := by
  apply (gain_iff (d:=1) (fun _=>3) (fun _=>1) (fun _=>1) (fun _=>4) 2 (by norm_num)).2
  norm_num [score]

variable {C : Type*} [Fintype C]

noncomputable def loss (base v : C → ℝ) (truth : C) (w : ℝ) : ℝ :=
  Real.log (∑ i, Real.exp (base i + w*v i)) - (base truth + w*v truth)

noncomputable def slope (base v : C → ℝ) (truth : C) (w : ℝ) : ℝ :=
  (∑ i, Real.exp (base i + w*v i) * v i) / (∑ i, Real.exp (base i+w*v i)) - v truth

private theorem partition_pos (base v : C → ℝ) (y : C) (w : ℝ) :
    0 < ∑ i, Real.exp (base i+w*v i) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨y, Finset.mem_univ _⟩

/-- The response direction of a single reusable path coefficient is its full
softmax-weighted mean logit direction minus the target logit's direction. -/
theorem differentiates (base v : C → ℝ) (y : C) (w : ℝ) :
    HasDerivAt (loss base v y) (slope base v y w) w := by
  have linear (i : C) : HasDerivAt (fun x : ℝ => base i+x*v i) (v i) w := by
    simpa using (((hasDerivAt_id w).mul_const (v i)).const_add (base i))
  have partition : HasDerivAt (fun x : ℝ => ∑ i, Real.exp (base i+x*v i))
      (∑ i, Real.exp (base i+w*v i)*v i) w :=
    HasDerivAt.fun_sum fun i _ => (linear i).exp
  exact (partition.log (ne_of_gt (partition_pos base v y w))).sub (linear y)

/-- A path need only tie some competing classes. If it weakly favors the
true class against all classes and strictly favors it over one, ordinary
answer loss supplies a strictly negative derivative at every finite weight. -/
theorem descent (base v : C → ℝ) (y : C) (w : ℝ) (top : ∀ i, v i ≤ v y)
    (strict : ∃ i, v i < v y) : slope base v y w < 0 := by
  have comp : (∑ i, Real.exp (base i+w*v i)*v i) <
      ∑ i, Real.exp (base i+w*v i)*v y := by
    apply Finset.sum_lt_sum
    · intro i _
      exact mul_le_mul_of_nonneg_left (top i) (Real.exp_pos _).le
    · obtain ⟨i,hi⟩ := strict
      exact ⟨i, Finset.mem_univ _, mul_lt_mul_of_pos_left hi (Real.exp_pos _)⟩
  rw [← Finset.sum_mul] at comp
  unfold slope
  apply sub_neg.mpr
  apply (div_lt_iff₀ (partition_pos base v y w)).2
  simpa [mul_comm] using comp

example : HasDerivAt (loss (fun i : Bool => if i then 2 else 4)
    (fun i => if i then 5 else 1) true) (slope (fun i => if i then 2 else 4)
    (fun i => if i then 5 else 1) true 0) 0 := differentiates _ _ _ _

example : slope (fun i : Bool => if i then 2 else 4) (fun i => if i then 5 else 1) true 0 < 0 := by
  apply descent
  · intro i; cases i <;> norm_num
  · exact ⟨false, by norm_num⟩
end Transformer.Phasor.TripleTransport.TypedLookupPaths
