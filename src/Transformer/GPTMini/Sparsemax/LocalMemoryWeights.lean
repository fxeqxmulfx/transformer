import Transformer.GPTMini.Sparsemax.PermutationMemoryGram

/-!
# Linear-size learned edge weights with a self-weight budget

Derived restriction before arXiv:1602.02068v2, Eq. (1). A dictionary of
N+1 slots learns N nonnegative path-edge weights. Their total is at most
one minus the diagonal floor. The remaining mass weights the identity
Gram atom. The resulting atom weights are probabilities, while the identity
mass is at least the requested floor. All constraints are linear.

This global edge budget is stronger than requiring a floor separately in
every row. It is an explicit architectural tradeoff that permits a compact
convex Gram parametrization and changing sparse supports. It does not freeze
the edge weights, assume target routing labels, or assert that their order
is learned. Both a zero-edge and a nonzero-edge witness inhabit the domain.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Nonnegative learned path edges with a total off-diagonal budget.
Source: the derived compact structural restriction for arXiv:1602.02068v2, Eq. (1). -/
def localWeightDomain (N : ℕ) (floor : ℝ) : Set (Fin N → ℝ) :=
  {t | (∀ e, 0 ≤ t e) ∧ (∑ e, t e) ≤ 1 - floor}

/-- Identity receives the residual mass; every path edge receives its learned weight.
Source: the convex mixture of genuine atoms preceding arXiv:1602.02068v2, Eq. (1). -/
def localMemoryWeights {N : ℕ} (t : Fin N → ℝ) : Option (Fin N) → ℝ
  | none => 1 - ∑ e, t e
  | some e => t e

/-- The atom weights always have total mass one, even before imposing nonnegativity.
Source: the residual identity coefficient in the derived arXiv:1602.02068v2, Eq. (1) chart. -/
theorem localMemoryWeights_sum {N : ℕ} (t : Fin N → ℝ) :
    (∑ e, localMemoryWeights t e) = 1 := by
  rw [Fintype.sum_option]
  change (1 - ∑ e, t e) + (∑ e, t e) = 1
  ring

/-- The edge domain is convex with no fixed support assumption.
Source: the linear structural budget before arXiv:1602.02068v2, Eq. (1). -/
theorem localWeightDomain_convex (N : ℕ) (floor : ℝ) : Convex ℝ (localWeightDomain N floor) := by
  intro t ht s hs a b ha hb hab
  refine ⟨?_, ?_⟩
  · intro e
    change 0 ≤ a * t e + b * s e
    exact add_nonneg (mul_nonneg ha (ht.1 e)) (mul_nonneg hb (hs.1 e))
  · change (∑ e, (a * t e + b * s e)) ≤ 1 - floor
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    have ht' := mul_le_mul_of_nonneg_left ht.2 ha
    have hs' := mul_le_mul_of_nonneg_left hs.2 hb
    calc
      _ ≤ a * (1 - floor) + b * (1 - floor) := add_le_add ht' hs'
      _ = 1 - floor := by rw [← add_mul, hab, one_mul]

/-- Every feasible edge coordinate is bounded by the total available budget.
Source: nonnegative path weights before arXiv:1602.02068v2, Eq. (1). -/
theorem localWeightDomain_coordinateBound {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (ht : t ∈ localWeightDomain N floor) (e : Fin N) :
    0 ≤ t e ∧ t e ≤ 1 - floor := by
  refine ⟨ht.1 e, ?_⟩
  exact (Finset.single_le_sum (fun j hj => ht.1 j) (Finset.mem_univ e)).trans ht.2

/-- A positive learned edge inhabits every coordinate-bound hypothesis. -/
example : 0 ≤ (fun _ : Fin 1 => (1 / 4 : ℝ)) 0 ∧
    (fun _ : Fin 1 => (1 / 4 : ℝ)) 0 ≤ 1 - (3 / 4 : ℝ) := by
  apply localWeightDomain_coordinateBound (3 / 4) (fun _ : Fin 1 => (1 / 4 : ℝ)) ?_ 0
  constructor
  · intro e
    norm_num
  · norm_num [Fin.sum_univ_one]

/-- The residual identity mass is at least the required self-weight floor.
Source: the derived strict-dominance budget for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryWeights_identity_floor {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (ht : t ∈ localWeightDomain N floor) : floor ≤ localMemoryWeights t none := by
  change floor ≤ 1 - ∑ e, t e
  linarith [ht.2]

/-- The floor is attained by a genuine nonzero edge witness. -/
example : (3 / 4 : ℝ) ≤ localMemoryWeights (fun _ : Fin 1 => (1 / 4 : ℝ)) none := by
  apply localMemoryWeights_identity_floor
  constructor
  · intro e
    norm_num
  · norm_num [Fin.sum_univ_one]

/-- A nonnegative floor makes every feasible atom coefficient nonnegative.
Source: the compact convex Gram mixture before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryWeights_nonneg {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ localWeightDomain N floor) (e : Option (Fin N)) :
    0 ≤ localMemoryWeights t e := by
  rcases e with _ | e
  · exact hf.trans (localMemoryWeights_identity_floor floor t ht)
  · exact ht.1 e

/-- The nonnegativity premises hold jointly with a positive off-diagonal weight. -/
example : 0 ≤ localMemoryWeights (fun _ : Fin 1 => (1 / 4 : ℝ)) (some 0) := by
  apply localMemoryWeights_nonneg (3 / 4) _ (by norm_num)
  constructor
  · intro e
    norm_num
  · norm_num [Fin.sum_univ_one]

/-- Zero edges inhabit every floor at most one.
Source: the identity endpoint of the derived arXiv:1602.02068v2, Eq. (1) family. -/
theorem zero_mem_localWeightDomain (N : ℕ) (floor : ℝ) (hf : floor ≤ 1) :
    (0 : Fin N → ℝ) ∈ localWeightDomain N floor := by
  refine ⟨fun e => le_refl 0, ?_⟩
  simp only [Pi.zero_apply, Finset.sum_const_zero]
  linarith

/-- A strict inverse floor genuinely permits the identity endpoint. -/
example : (0 : Fin 3 → ℝ) ∈ localWeightDomain 3 (3 / 4) :=
  zero_mem_localWeightDomain _ _ (by norm_num)

/-- The edge domain is inhabited exactly when the requested floor does not exceed one.
Source: the explicit linear budget for arXiv:1602.02068v2, Eq. (1). -/
theorem localWeightDomain_nonempty_iff (N : ℕ) (floor : ℝ) :
    (localWeightDomain N floor).Nonempty ↔ floor ≤ 1 := by
  constructor
  · rintro ⟨t, ht⟩
    have hn : 0 ≤ ∑ e, t e := Finset.sum_nonneg (fun e he => ht.1 e)
    linarith [ht.2]
  · intro hf
    exact ⟨0, zero_mem_localWeightDomain N floor hf⟩

/-- Atom weights depend affinely on the N learned edge coordinates.
Source: the compact parametrization before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryWeights_affine {N : ℕ} (t s : Fin N → ℝ) (a b : ℝ)
    (hab : a + b = 1) :
    localMemoryWeights (a • t + b • s) = a • localMemoryWeights t + b • localMemoryWeights s := by
  funext e
  rcases e with _ | e
  · change 1 - (∑ j, (a * t j + b * s j)) =
      a * (1 - ∑ j, t j) + b * (1 - ∑ j, s j)
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    nlinarith
  · rfl

/-- Two distinct endpoints and their midpoint inhabit the affine-weight premise. -/
example : localMemoryWeights ((1 / 2 : ℝ) • (0 : Fin 1 → ℝ) +
    (1 / 2 : ℝ) • (fun _ : Fin 1 => (1 / 4 : ℝ))) =
    (1 / 2 : ℝ) • localMemoryWeights (0 : Fin 1 → ℝ) +
      (1 / 2 : ℝ) • localMemoryWeights (fun _ : Fin 1 => (1 / 4 : ℝ)) :=
  localMemoryWeights_affine _ _ _ _ (by norm_num)

end Transformer.GPTMini.Sparsemax
