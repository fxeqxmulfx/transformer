import Transformer.GPTMini.Sparsemax.LocalMemoryCore

/-!
# Linear-size trainable path weights and independent query/key norms

Derived compact architecture for arXiv:1602.02068v2, Eq. (1). Learn N path
weights and 2(N+1) nonnegative diagonal additions, stored as 3(N+1)-1 scalar
coordinates. The full Gram is the affine path core plus this diagonal PSD
addition. Cross scores are unaffected by the additions; squared norms of
both Q and K become independently trainable as one plus the corresponding
addition. The parameter domain uses only linear inequalities and is convex.

Same-family off-diagonal entries remain zero. This is an explicit restricted
architecture, not a compact parametrization of every PSD Gram. The number
of stored coordinates is linear in dictionary size; dense materialization
of the full derived Gram is unnecessary and would still be quadratic.
Feasibility and actual sparse attention are proved in subsequent modules.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- N learned path weights and one independent norm addition for each Q/K column.
Source: the compact embedding restriction before arXiv:1602.02068v2, Eq. (1). -/
abbrev LocalMemoryParameters (N : ℕ) :=
  (Fin N → ℝ) × (Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ)

/-- Linear structural bounds for all stored learned coordinates.
Source: the compact convex memory domain before arXiv:1602.02068v2, Eq. (1). -/
def localMemoryParameterDomain (N : ℕ) (cap floor : ℝ) : Set (LocalMemoryParameters N) :=
  {p | p.1 ∈ localWeightDomain N floor ∧ ∀ x, 0 ≤ p.2 x ∧ p.2 x ≤ cap - 1}

/-- A compact learned Gram, including independent Q and K squared-norm additions.
Source: the affine Gram construction preceding arXiv:1602.02068v2, Eq. (1). -/
def localMemoryGram {N : ℕ} (p : LocalMemoryParameters N) : EmbeddingGram (N + 1) :=
  localMemoryCore p.1 + Matrix.diagonal p.2

/-- Every Gram entry is determined by the stored path and diagonal coordinates.
Source: the derived compact parametrization for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryGram_apply {N : ℕ} (p : LocalMemoryParameters N)
    (x y : Sum (Fin (N + 1)) (Fin (N + 1))) :
    localMemoryGram p x y = localMemoryCore p.1 x y + Matrix.diagonal p.2 x y := by
  change (localMemoryCore p.1 + Matrix.diagonal p.2) x y = _
  exact Matrix.add_apply _ _ _ _

/-- The genuine squared norm of every learned query/key column is one plus its own addition.
Source: the compact embedding construction before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryGram_diagonal {N : ℕ} (p : LocalMemoryParameters N)
    (x : Sum (Fin (N + 1)) (Fin (N + 1))) : localMemoryGram p x x = 1 + p.2 x := by
  rw [localMemoryGram_apply, localMemoryCore_diagonal, Matrix.diagonal_apply_eq]

/-- Independent diagonal additions do not change the actual Q/K cross scores.
Source: the two disjoint embedding copies before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryGram_scores {N : ℕ} (p : LocalMemoryParameters N) :
    memoryGramScores (localMemoryGram p) = memoryGramScores (localMemoryCore p.1) := by
  ext i j
  change localMemoryGram p (Sum.inl i) (Sum.inr j) = localMemoryCore p.1 (Sum.inl i) (Sum.inr j)
  rw [localMemoryGram_apply, Matrix.diagonal_apply_ne _ (by intro h; cases h), add_zero]

/-- The whole compact parameter domain is convex, including both embedding norm families.
Source: the linear path and squared-norm bounds for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryParameterDomain_convex (N : ℕ) (cap floor : ℝ) :
    Convex ℝ (localMemoryParameterDomain N cap floor) := by
  intro p hp q hq a b ha hb hab
  refine ⟨localWeightDomain_convex N floor hp.1 hq.1 ha hb hab, ?_⟩
  intro x
  change 0 ≤ a * p.2 x + b * q.2 x ∧ a * p.2 x + b * q.2 x ≤ cap - 1
  have hpl := mul_nonneg ha (hp.2 x).1
  have hql := mul_nonneg hb (hq.2 x).1
  have hpu := mul_le_mul_of_nonneg_left (hp.2 x).2 ha
  have hqu := mul_le_mul_of_nonneg_left (hq.2 x).2 hb
  constructor
  · exact add_nonneg hpl hql
  · calc
      _ ≤ a * (cap - 1) + b * (cap - 1) := add_le_add hpu hqu
      _ = cap - 1 := by rw [← add_mul, hab, one_mul]

/-- The identity parameter point inhabits every cap at least one and floor at most one.
Source: the zero-edge, zero-addition witness for arXiv:1602.02068v2, Eq. (1). -/
theorem zero_mem_localMemoryParameterDomain (N : ℕ) (cap floor : ℝ)
    (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    (0 : LocalMemoryParameters N) ∈ localMemoryParameterDomain N cap floor := by
  refine ⟨zero_mem_localWeightDomain N floor hf, ?_⟩
  intro x
  change 0 ≤ (0 : ℝ) ∧ (0 : ℝ) ≤ cap - 1
  constructor
  · exact le_refl 0
  · linarith

/-- A multi-slot memory satisfies all compact-domain bound premises. -/
example : (0 : LocalMemoryParameters 3) ∈ localMemoryParameterDomain 3 4 (3 / 4) :=
  zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)

/-- Nonemptiness characterizes the actual admissible cap and floor ranges.
Source: the explicit compact structural domain for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryParameterDomain_nonempty_iff (N : ℕ) (cap floor : ℝ) :
    (localMemoryParameterDomain N cap floor).Nonempty ↔ 1 ≤ cap ∧ floor ≤ 1 := by
  constructor
  · rintro ⟨p, hp⟩
    have hn := hp.2 (Sum.inl 0)
    have hf := (localWeightDomain_nonempty_iff N floor).1 ⟨p.1, hp.1⟩
    exact ⟨by linarith, hf⟩
  · rintro ⟨hc, hf⟩
    exact ⟨0, zero_mem_localMemoryParameterDomain N cap floor hc hf⟩

/-- The compact Gram is affine jointly in every stored learned coordinate.
Source: the derived embedding chart before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryGram_affine {N : ℕ} (p q : LocalMemoryParameters N) (a b : ℝ)
    (hab : a + b = 1) :
    localMemoryGram (a • p + b • q) = a • localMemoryGram p + b • localMemoryGram q := by
  change localMemoryCore (a • p.1 + b • q.1) + Matrix.diagonal (a • p.2 + b • q.2) = _
  have hd : Matrix.diagonal (a • p.2 + b • q.2) =
      Matrix.diagonal (a • p.2) + Matrix.diagonal (b • q.2) :=
    (Matrix.diagonal_add (a • p.2) (b • q.2)).symm
  rw [localMemoryCore_affine p.1 q.1 a b hab, hd,
    Matrix.diagonal_smul, Matrix.diagonal_smul]
  change _ = a • (localMemoryCore p.1 + Matrix.diagonal p.2) +
    b • (localMemoryCore q.1 + Matrix.diagonal q.2)
  module

/-- Jointly changed edges and norms inhabit the compact affine-Gram premise. -/
example : localMemoryGram ((1 / 2 : ℝ) • (0 : LocalMemoryParameters 1) +
    (1 / 2 : ℝ) • ((fun _ : Fin 1 => (1 / 4 : ℝ)),
      (fun _ : Sum (Fin 2) (Fin 2) => (1 : ℝ)))) =
    (1 / 2 : ℝ) • localMemoryGram (0 : LocalMemoryParameters 1) +
      (1 / 2 : ℝ) • localMemoryGram ((fun _ : Fin 1 => (1 / 4 : ℝ)),
        (fun _ : Sum (Fin 2) (Fin 2) => (1 : ℝ))) :=
  localMemoryGram_affine _ _ _ _ (by norm_num)

/-- The exact finite index set of stored scalar parameters.
Source: the compact path and Q/K-norm chart for arXiv:1602.02068v2, Eq. (1). -/
abbrev LocalMemoryParameterIndex (N : ℕ) := Fin N ⊕ (Fin (N + 1) ⊕ Fin (N + 1))

/-- Flatten the two parameter tables without dropping any learned coordinate.
Source: the complete compact storage for arXiv:1602.02068v2, Eq. (1). -/
def localMemoryParameterCoordinates {N : ℕ} (p : LocalMemoryParameters N) :
    LocalMemoryParameterIndex N → ℝ := Sum.elim p.1 p.2

/-- The stored coordinate vector uniquely determines every path weight and both norm tables.
Source: the actual finite compact parametrization for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryParameterCoordinates_injective (N : ℕ) :
    Function.Injective (localMemoryParameterCoordinates (N := N)) := by
  intro p q h
  apply Prod.ext
  · funext e
    exact congrFun h (Sum.inl e)
  · funext x
    exact congrFun h (Sum.inr x)

/-- P=N+1 memory slots store exactly 3P-1 scalar coordinates, rather than a free dense Gram.
Source: the derived compact storage count for arXiv:1602.02068v2, Eq. (1).
This counts coordinates, not the dimension of a degenerate boundary domain. -/
theorem localMemoryParameterIndex_card (N : ℕ) :
    Fintype.card (LocalMemoryParameterIndex N) = 3 * (N + 1) - 1 := by
  simp only [LocalMemoryParameterIndex, Fintype.card_sum, Fintype.card_fin]
  omega

/-- Changed two-slot edges and independently stored query/key norm additions.
Source: the nonidentity compact witness for arXiv:1602.02068v2, Eq. (1). -/
def localMemoryExampleParameters : LocalMemoryParameters 1 :=
  ((fun _ => 1 / 4), (fun _ => 1))

/-- The nonidentity witness satisfies every compact-domain constraint.
Source: the concrete compact embedding construction for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryExampleParameters_mem :
    localMemoryExampleParameters ∈ localMemoryParameterDomain 1 4 (3 / 4) := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro e
    norm_num [localMemoryExampleParameters]
  · norm_num [localMemoryExampleParameters, Fin.sum_univ_one]
  · intro x
    norm_num [localMemoryExampleParameters]

end Transformer.GPTMini.Sparsemax
