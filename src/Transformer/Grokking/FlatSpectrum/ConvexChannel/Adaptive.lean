module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Shared

/-!
# Preserve query-specific capacity while preferring shared oracle solutions

Source: `review/flat_spectrum.md`, MF4. Mandatory sharing can remove a realizable
task: two Boolean slots have additive logits, so no shared tensor strictly
classifies XOR on all four inputs. A query-indexed feasible vertex does.

The repair retains the full convex query-indexed domain. A shared candidate is
preferred only when its linear cost is no larger than the exact unrestricted
oracle. This preference is itself an exact oracle; it is not a restriction of
the feasible set or a learned product of routing and decoder parameters.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

/-- Two visible categorical inputs for the MF4 capacity witness. -/
def pairData (a b : Bool) (k : Bool) : Bool := if k then b else a

/-- Shared additive logits obey a rectangle identity, independently of
feasibility or scale (MF4). -/
theorem shared_rectangle (W : Channel Unit Bool Bool Bool) (c : Bool) :
    logits W () (pairData false false) c + logits W () (pairData true true) c =
      logits W () (pairData false true) c + logits W () (pairData true false) c := by
  simp only [logits, Fintype.sum_bool, pairData, Bool.false_eq_true, ite_false, ite_true]
  ring

/-- **Mandatory sharing loses capacity**, MF4: not even an unconstrained shared
tensor strictly classifies XOR on these four inputs. -/
theorem no_shared_xor (W : Channel Unit Bool Bool Bool) :
    ¬ ∀ a b : Bool, logits W () (pairData a b) (!(Bool.xor a b)) <
      logits W () (pairData a b) (Bool.xor a b) := by
  intro h
  have h00 := h false false
  have h01 := h false true
  have h10 := h true false
  have h11 := h true true
  have h0 := shared_rectangle W false
  have h1 := shared_rectangle W true
  simp only [Bool.xor_false, Bool.false_xor, Bool.xor_self, Bool.not_false,
    Bool.not_true] at h00 h01 h10 h11
  linarith

/-- The full query-indexed domain represents XOR with a unit correct margin;
the visible first input is the query (MF4). -/
theorem untied_xor_margin (a b : Bool) :
    let W := vertex (fun _ : Bool => true) (fun q _ v : Bool => Bool.xor q v)
    logits W a (pairData a b) (Bool.xor a b) -
      logits W a (pairData a b) (!(Bool.xor a b)) = 1 := by
  cases a <;> cases b <;>
    norm_num [logits, vertex, pairData, Fintype.sum_bool]

variable {Q K V C : Type*} [Fintype Q] [Fintype K] [Fintype V] [Fintype C]
  [DecidableEq K] [DecidableEq C]

/-- Choose a shared candidate only when its exact linear cost is at least as
good as the unrestricted vertex. All parameters remain joint coefficients. -/
noncomputable def preferShared (G : Channel Q K V C) (align : Q → K ≃ K)
    (A : Channel Unit K V C) (key : Q → K) (decode : Q → K → V → C) :
    Channel Q K V C := by
  classical
  exact if linearCost G (tiedChannel align A) ≤ linearCost G (vertex key decode)
    then tiedChannel align A else vertex key decode

/-- The optional sharing rule always produces a feasible tensor, MF4. -/
theorem feasible_preferShared (G : Channel Q K V C) (align : Q → K ≃ K)
    {A : Channel Unit K V C} (hA : Feasible A) (key : Q → K)
    (decode : Q → K → V → C) : Feasible (preferShared G align A key decode) := by
  classical
  unfold preferShared
  split_ifs
  · exact feasible_tiedChannel align hA
  · exact feasible_vertex key decode

/-- **An exact unrestricted oracle with a sharing preference**, MF4. The usual
class and key minimization conditions are retained, so no optimization capacity
is traded for sharing. No held-out labels enter the comparison. -/
theorem linear_oracle_preferShared [Nonempty V] (G : Channel Q K V C)
    (align : Q → K ≃ K) (A : Channel Unit K V C)
    (key : Q → K) (decode : Q → K → V → C)
    (hclass : ∀ q k v c, G q k v (decode q k v) ≤ G q k v c)
    (hkey : ∀ q k, (∑ v, G q (key q) v (decode q (key q) v)) ≤
      ∑ v, G q k v (decode q k v))
    {W : Channel Q K V C} (hW : Feasible W) :
    linearCost G (preferShared G align A key decode) ≤ linearCost G W := by
  classical
  have hmin := linear_oracle G key decode hclass hkey hW
  unfold preferShared
  split_ifs with h
  · exact h.trans hmin
  · exact hmin

/-- Every FW step with optional sharing stays in the original convex domain. -/
theorem feasible_preferShared_step (G : Channel Q K V C) (align : Q → K ≃ K)
    {A : Channel Unit K V C} (hA : Feasible A) (key : Q → K)
    (decode : Q → K → V → C) {W : Channel Q K V C} (hW : Feasible W)
    (η : ℝ) (hη : 0 ≤ η) (hη₁ : η ≤ 1) :
    Feasible ((1 - η) • W + η • preferShared G align A key decode) :=
  convex_feasible hW (feasible_preferShared G align hA key decode)
    (sub_nonneg.mpr hη₁) hη (by ring)

example := feasible_vertex (fun _ : Bool => true) (fun q _ v : Bool => Bool.xor q v)

example := feasible_preferShared (fun _ _ _ _ : Bool => 0)
  (fun _ => Equiv.refl Bool) (feasible_vertex (fun _ : Unit => false)
    (fun _ _ => (id : Bool → Bool))) id (fun _ _ => Bool.not)

example := linear_oracle_preferShared (fun _ _ _ _ : Bool => 0)
  (fun _ => Equiv.refl Bool) (vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))
  id (fun _ _ => Bool.not) (by intros; exact le_rfl) (by intros; simp)
  (feasible_vertex (fun _ => false) (fun _ _ => (id : Bool → Bool)))

example := feasible_preferShared_step (fun _ _ _ _ : Bool => 0)
  (fun _ => Equiv.refl Bool) (feasible_vertex (fun _ : Unit => false)
    (fun _ _ => (id : Bool → Bool))) id (fun _ _ => Bool.not)
  (feasible_vertex (fun _ => false) (fun _ _ => (id : Bool → Bool)))
  (1 / 2) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
