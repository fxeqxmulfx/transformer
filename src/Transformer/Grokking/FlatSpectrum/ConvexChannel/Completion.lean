module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Adaptive

/-!
# Complete unobserved decoder rows without changing the linear objective

Source: `review/flat_spectrum.md`, MF5. Replacing a decoder on zero-gradient
rows preserves every vertex's linear cost, hence preserves the exact oracle.
The replacement may use pooled training information. Its correctness on unseen
inputs is a separate assumption, not a consequence of the zero-gradient fact.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C S : Type*} [Fintype Q] [Fintype K] [Fintype V] [Fintype C]
  [DecidableEq K] [DecidableEq C]

/-- Fill only rows whose supplied mask is true, retaining all other choices. -/
def completeDecode (missing : Q → K → V → Bool)
    (decode fallback : Q → K → V → C) : Q → K → V → C :=
  fun q k v => if missing q k v then fallback q k v else decode q k v

omit [Fintype Q] [Fintype K] [Fintype V] [Fintype C] [DecidableEq K] [DecidableEq C] in
/-- Zero-gradient rows may use arbitrary fallback choices, MF5. -/
theorem gradient_completeDecode (G : Channel Q K V C)
    (missing : Q → K → V → Bool) (decode fallback : Q → K → V → C)
    (hzero : ∀ q k v, missing q k v = true → ∀ c, G q k v c = 0)
    (q : Q) (k : K) (v : V) :
    G q k v (completeDecode missing decode fallback q k v) = G q k v (decode q k v) := by
  unfold completeDecode
  split_ifs with h
  · rw [hzero q k v h, hzero q k v h]
  · rfl

/-- **Completion changes no vertex cost**, MF5, including a selected key whose
unobserved values have been filled from a shared decoder. -/
theorem linearCost_completeDecode (G : Channel Q K V C)
    (missing : Q → K → V → Bool) (decode fallback : Q → K → V → C)
    (hzero : ∀ q k v, missing q k v = true → ∀ c, G q k v c = 0)
    (key : Q → K) :
    linearCost G (vertex key (completeDecode missing decode fallback)) =
      linearCost G (vertex key decode) := by
  have hv (d : Q → K → V → C) : linearCost G (vertex key d) =
      ∑ q, ∑ v, G q (key q) v (d q (key q) v) := by
    simp [linearCost, vertex, mul_ite]
  rw [hv, hv]
  apply Finset.sum_congr rfl
  intro q _
  apply Finset.sum_congr rfl
  intro v _
  exact gradient_completeDecode G missing decode fallback hzero q (key q) v

/-- **An exact unrestricted oracle remains exact after completion**, MF5.
The original class/key minimization conditions are unchanged. -/
theorem linear_oracle_completeDecode [Nonempty V] (G : Channel Q K V C)
    (missing : Q → K → V → Bool) (key : Q → K)
    (decode fallback : Q → K → V → C)
    (hzero : ∀ q k v, missing q k v = true → ∀ c, G q k v c = 0)
    (hclass : ∀ q k v c, G q k v (decode q k v) ≤ G q k v c)
    (hkey : ∀ q k, (∑ v, G q (key q) v (decode q (key q) v)) ≤
      ∑ v, G q k v (decode q k v))
    {W : Channel Q K V C} (hW : Feasible W) :
    linearCost G (vertex key (completeDecode missing decode fallback)) ≤ linearCost G W := by
  rw [linearCost_completeDecode G missing decode fallback hzero]
  exact linear_oracle G key decode hclass hkey hW

omit [Fintype Q] [Fintype V] in
/-- Completed choices still define a feasible joint tensor, MF5; they do not
multiply a learned routing distribution by a separately learned decoder. -/
theorem feasible_completed_vertex (missing : Q → K → V → Bool) (key : Q → K)
    (decode fallback : Q → K → V → C) :
    Feasible (vertex key (completeDecode missing decode fallback)) :=
  feasible_vertex _ _

omit [Fintype Q] [Fintype K] [Fintype V] [Fintype C] [DecidableEq K] [DecidableEq C] in
/-- A retained sum of sample residuals is zero on an unobserved query/value
row, regardless of the states at which those residuals were evaluated (MF5). -/
theorem residual_sum_unobserved [Fintype S] [DecidableEq Q] [DecidableEq V]
    (query : S → Q) (data : S → K → V) (residual : S → C → ℝ)
    (q : Q) (k : K) (v : V) (c : C)
    (h : ∀ s, query s ≠ q ∨ data s k ≠ v) :
    (∑ s, if query s = q ∧ data s k = v then residual s c else 0) = 0 := by
  apply Finset.sum_eq_zero
  intro s _
  rcases h s with hq | hv
  · simp [hq]
  · simp [hv]

/-! Nonempty examples with an actually missing row and a nonzero observed cost. -/
private def missingExample (q k v : Bool) : Bool := q && k && v
private def gradientExample (q k v c : Bool) : ℝ :=
  if missingExample q k v then 0 else if c then 1 else 0

private theorem example_zero (q k v : Bool) (h : missingExample q k v = true)
    (c : Bool) : gradientExample q k v c = 0 := by
  simp [gradientExample, h]

example := example_zero true true true rfl true

example := gradient_completeDecode gradientExample missingExample
  (fun _ _ _ => false) (fun _ _ _ => true) example_zero true true true

example := linearCost_completeDecode gradientExample missingExample
  (fun _ _ _ => false) (fun _ _ _ => true) example_zero (fun _ => true)

example := linear_oracle_completeDecode gradientExample missingExample (fun _ => true)
  (fun _ _ _ => false) (fun _ _ _ => true) example_zero
  (by intro q k v c; cases q <;> cases k <;> cases v <;> cases c <;>
    norm_num [gradientExample, missingExample])
  (by intro q k; cases q <;> cases k <;>
    norm_num [gradientExample, missingExample, Fintype.sum_bool])
  (feasible_vertex (fun _ => false) (fun _ _ => (id : Bool → Bool)))

example := residual_sum_unobserved (fun _ : Unit => false) (fun (_ : Unit) (k : Bool) => k)
  (fun _ : Unit => fun c : Bool => if c then (1 : ℝ) else -1) true true true true
  (by intro s; exact Or.inl (by decide))

end Transformer.Grokking.FlatSpectrum.ConvexChannel
