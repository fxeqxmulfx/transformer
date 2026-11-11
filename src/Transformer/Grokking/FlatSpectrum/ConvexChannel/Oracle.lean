module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Representation

/-!
# An exact linear oracle for the convex channel domain

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC8.
For each query, first minimize each key/value's class cost. Then select
the key with the least summed value cost. This finite enumeration gives
a global linear minimizer, usable in Frank-Wolfe with ordinary answer CE.
There is no nonconvex inner fitting problem or task-specific operation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C : Type*} [Fintype Q] [Fintype K] [Fintype V] [Fintype C]
  [DecidableEq K] [DecidableEq C]

/-- A deterministic routing/output vertex, specified by categorical
choices. The oracle chooses these from the gradient, not from task labels (CC8). -/
def vertex (key : Q → K) (decode : Q → K → V → C) : Channel Q K V C :=
  fun q k v c => if k = key q then (if c = decode q k v then 1 else 0) else 0

omit [Fintype Q] [Fintype V] in
/-- Every oracle vertex lies in the joint convex channel domain (CC8). -/
theorem feasible_vertex (key : Q → K) (decode : Q → K → V → C) :
    Feasible (vertex key decode) := by
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    simp only [vertex]
    split_ifs <;> norm_num
  · intro q k v w
    by_cases h : k = key q <;> simp [vertex, h]
  · intro q v
    simp [vertex]

/-- The full finite-dimensional linear objective on the parameter tensor (CC8). -/
noncomputable def linearCost (G W : Channel Q K V C) : ℝ :=
  ∑ q, ∑ k, ∑ v, ∑ c, G q k v c * W q k v c

/-- Choosing least-cost classes and then the least-cost key gives a
global linear minimizer over all feasible joint channels (CC8). -/
theorem linear_oracle [Nonempty V] (G : Channel Q K V C)
    (key : Q → K) (decode : Q → K → V → C)
    (hclass : ∀ q k v c, G q k v (decode q k v) ≤ G q k v c)
    (hkey : ∀ q k, (∑ v, G q (key q) v (decode q (key q) v)) ≤
      ∑ v, G q k v (decode q k v))
    {W : Channel Q K V C} (hW : Feasible W) :
    linearCost G (vertex key decode) ≤ linearCost G W := by
  obtain ⟨v₀⟩ := ‹Nonempty V›
  have hpoint (q : Q) (k : K) (v : V) :
      G q k v (decode q k v) * (∑ c, W q k v c) ≤
        ∑ c, G q k v c * W q k v c := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun c _ =>
      mul_le_mul_of_nonneg_right (hclass q k v c) (hW.1 q k v c)
  have hpart (q : Q) (k : K) :
      (∑ v, G q k v (decode q k v)) * (∑ c, W q k v₀ c) ≤
        ∑ v, ∑ c, G q k v c * W q k v c := by
    rw [Finset.sum_mul]
    apply Finset.sum_le_sum
    intro v _
    rw [hW.2.1 q k v₀ v]
    exact hpoint q k v
  have hrow (q : Q) : (∑ v, G q (key q) v (decode q (key q) v)) ≤
      ∑ k, ∑ v, ∑ c, G q k v c * W q k v c := by
    calc
      _ = (∑ v, G q (key q) v (decode q (key q) v)) *
          (∑ k, ∑ c, W q k v₀ c) := by rw [hW.2.2 q v₀, mul_one]
      _ = ∑ k, (∑ v, G q (key q) v (decode q (key q) v)) *
          (∑ c, W q k v₀ c) := by rw [Finset.mul_sum]
      _ ≤ ∑ k, (∑ v, G q k v (decode q k v)) * (∑ c, W q k v₀ c) :=
        Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_right (hkey q k)
          (Finset.sum_nonneg fun c _ => hW.1 q k v₀ c)
      _ ≤ _ := Finset.sum_le_sum fun k _ => hpart q k
  have hv : linearCost G (vertex key decode) =
      ∑ q, ∑ v, G q (key q) v (decode q (key q) v) := by
    simp [linearCost, vertex, mul_ite]
  rw [hv]
  exact Finset.sum_le_sum fun q _ => hrow q

omit [Fintype Q] [Fintype V] in
/-- Convex combinations of an iterate and the oracle vertex stay feasible;
this is the update used by the native trainer (CC8). -/
theorem feasible_oracle_step {W : Channel Q K V C} (hW : Feasible W)
    (key : Q → K) (decode : Q → K → V → C) (η : ℝ) (hη : 0 ≤ η) (hη₁ : η ≤ 1) :
    Feasible ((1 - η) • W + η • vertex key decode) :=
  convex_feasible hW (feasible_vertex key decode) (sub_nonneg.mpr hη₁) hη (by ring)

example := feasible_vertex (id : Bool → Bool) (fun _ _ : Bool => Bool.not)

example := linear_oracle
  (fun q k v c : Bool => if k = q ∧ c = !v then (-1 : ℝ) else 0)
  id (fun _ _ : Bool => Bool.not)
  (by intro q k v c; cases q <;> cases k <;> cases v <;> cases c <;> norm_num)
  (by intro q k; cases q <;> cases k <;> norm_num [Fintype.sum_bool])
  (feasible_vertex (id : Bool → Bool) (fun _ _ : Bool => Bool.not))

example := feasible_oracle_step (feasible_generator (fun _ : Bool => Bool.not))
  id (fun _ _ : Bool => Bool.not) (1 / 2) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
