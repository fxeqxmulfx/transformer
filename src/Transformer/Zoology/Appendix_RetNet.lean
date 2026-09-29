/-
# RetNet recurrence and finite-memory recall

Arora et al., arXiv:2312.04927v1, Appendix `app:retnet-proof`, equations
`eq: retnet-recurrence` and `eq: retnet-poly`, and Corollary `cor: space-ar`.
The paper's communication argument concerns *bits*.  A matrix of real
numbers has no fixed bit capacity without a representation or precision
bound, so the lower bound below states its finite-state premise explicitly.
-/

import Transformer.Zoology.Section3_MQAR

namespace Transformer.Zoology

/-- One projected key/value pair, used as an outer-product update.
Source: Appendix equation `eq: retnet-recurrence`. -/
abbrev ProjectedPair (d : ℕ) := (Fin d → ℝ) × (Fin d → ℝ)

/-- The RetNet state is a `d × d` real matrix.
Source: Appendix equation `eq: retnet-recurrence`. -/
abbrev RetNetState (d : ℕ) := Fin d → Fin d → ℝ

/-- A single RetNet state transition `Z ↦ γZ + AᵀV`.
Source: Appendix equation `eq: retnet-recurrence`. -/
def retNetStep {d : ℕ} (γ : ℝ) (Z : RetNetState d)
    (pair : ProjectedPair d) : RetNetState d :=
  fun r c => γ * Z r c + pair.1 r * pair.2 c

/-- Run the RetNet state recurrence through a finite prefix.
Source: Appendix `algo: retnet`, state update. -/
def retNetRun {d : ℕ} (γ : ℝ) (pairs : List (ProjectedPair d)) :
    RetNetState d :=
  pairs.foldl (retNetStep γ) (fun _ _ => 0)

/-- The state after appending a pair obeys exactly the paper's recurrence.
Source: Appendix equation `eq: retnet-recurrence`. -/
theorem retNetRun_append {d : ℕ} (γ : ℝ)
    (pairs : List (ProjectedPair d)) (pair : ProjectedPair d) :
    retNetRun γ (pairs ++ [pair]) = retNetStep γ (retNetRun γ pairs) pair := by
  simp [retNetRun, List.foldl_append]

/-- Any finite memory state that answers every one-way index query exactly
must distinguish all input bit strings.  This is the deterministic exact
case underlying the information argument in Appendix `cor: space-ar`.
The randomized `2/3`-success result cited there is stronger and requires
its own communication-complexity proof. -/
theorem exact_index_encoder_injective {n s : ℕ}
    (encode : (Fin n → Bool) → Fin s) (decode : Fin s → Fin n → Bool)
    (hcorrect : ∀ bits i, decode (encode bits) i = bits i) :
    Function.Injective encode := by
  intro x y hxy
  funext i
  calc x i = decode (encode x) i := (hcorrect x i).symm
    _ = decode (encode y) i := by rw [hxy]
    _ = y i := hcorrect y i

/-- If the retained state has `b` bits, exact recall of one of `n`
independently chosen bits requires `n ≤ b`.  This correct finite-state
specialization of Appendix Corollary `cor: space-ar` does not assume that a
real-valued RetNet state automatically uses only `O(d²)` bits. -/
theorem exact_index_requires_bits (n b : ℕ)
    (encode : (Fin n → Bool) → Fin (2 ^ b))
    (decode : Fin (2 ^ b) → Fin n → Bool)
    (hcorrect : ∀ bits i, decode (encode bits) i = bits i) :
    n ≤ b := by
  have hinj := exact_index_encoder_injective encode decode hcorrect
  have hcard := Fintype.card_le_of_injective encode hinj
  simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin] at hcard
  exact (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).mp hcard

/-- The premise of the finite-state lower bound is satisfiable: one bit of
memory stores one input bit and answers the only possible query. -/
example : ∃ encode : (Fin 1 → Bool) → Fin (2 ^ 1),
    ∃ decode : Fin (2 ^ 1) → Fin 1 → Bool,
      ∀ bits i, decode (encode bits) i = bits i := by
  refine ⟨(fun bits => if bits 0 then 1 else 0),
    (fun state _ => state = 1), ?_⟩
  intro bits i
  fin_cases i
  cases h : bits 0 <;> simp [h]

end Transformer.Zoology
