import Transformer.GPTMini.Sparsemax.IncidentMemoryCore
import Transformer.GPTMini.Sparsemax.LocalMemorySelection

/-!
# Compact convex learned Q/K parameters with separate row budgets

Derived extension before arXiv:1602.02068v2, Eq. (1). Retain the N path
coordinates and the 2(N+1) independent query/key norm additions. Replace
only the global edge budget by separate incident budgets. The domain is
convex and compact, with exactly the same 3P-1 stored coordinates.

Every squared complete-coordinate criterion has a proved unique constrained
minimum for cap at least one and floor at most one. Its reference may lie
outside the domain. Later modules derive such a reference from observed
data and transfer the actual sparsemax and common-value guarantees.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Separate incident budgets and independent bounded query/key norm additions.
Source: the relaxed compact embedding domain before arXiv:1602.02068v2, Eq. (1). -/
def incidentMemoryParameterDomain (N : ℕ) (cap floor : ℝ) : Set (LocalMemoryParameters N) :=
  {p | p.1 ∈ incidentMemoryWeightDomain N floor ∧ ∀ x, 0 ≤ p.2 x ∧ p.2 x ≤ cap - 1}

/-- All compact learned coordinates retain a convex feasible domain.
Source: the separate linear path and squared-norm bounds before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryParameterDomain_convex (N : ℕ) (cap floor : ℝ) :
    Convex ℝ (incidentMemoryParameterDomain N cap floor) := by
  intro p hp q hq a b ha hb hab
  refine ⟨incidentMemoryWeightDomain_convex N floor hp.1 hq.1 ha hb hab, ?_⟩
  intro x
  change 0 ≤ a * p.2 x + b * q.2 x ∧ a * p.2 x + b * q.2 x ≤ cap - 1
  constructor
  · exact add_nonneg (mul_nonneg ha (hp.2 x).1) (mul_nonneg hb (hq.2 x).1)
  · calc
      _ ≤ a * (cap - 1) + b * (cap - 1) :=
        add_le_add (mul_le_mul_of_nonneg_left (hp.2 x).2 ha)
          (mul_le_mul_of_nonneg_left (hq.2 x).2 hb)
      _ = cap - 1 := by rw [← add_mul, hab, one_mul]

/-- Every formerly feasible compact point is feasible with the separate budgets.
Source: the derived domain enlargement before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryParameterDomain_subset_incident (N : ℕ) (cap floor : ℝ) :
    localMemoryParameterDomain N cap floor ⊆ incidentMemoryParameterDomain N cap floor := by
  intro p hp
  exact ⟨localWeightDomain_subset_incident N floor hp.1, hp.2⟩

/-- Zero parameters inhabit every cap at least one and floor at most one.
Source: the genuine identity witness before arXiv:1602.02068v2, Eq. (1). -/
theorem zero_mem_incidentMemoryParameterDomain (N : ℕ) (cap floor : ℝ)
    (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    (0 : LocalMemoryParameters N) ∈ incidentMemoryParameterDomain N cap floor :=
  localMemoryParameterDomain_subset_incident N cap floor
    (zero_mem_localMemoryParameterDomain N cap floor hc hf)

/-- The nonempty-domain premises have a genuine multi-slot instance. -/
example : (0 : LocalMemoryParameters 3) ∈ incidentMemoryParameterDomain 3 4 (3 / 4) :=
  zero_mem_incidentMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)

/-- Nonemptiness has the same exact cap and floor range as before the enlargement.
Source: the independent norm and local mass bounds for arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryParameterDomain_nonempty_iff (N : ℕ) (cap floor : ℝ) :
    (incidentMemoryParameterDomain N cap floor).Nonempty ↔ 1 ≤ cap ∧ floor ≤ 1 := by
  constructor
  · rintro ⟨p, hp⟩
    have hn := hp.2 (Sum.inl 0)
    exact ⟨by linarith, (incidentMemoryWeightDomain_nonempty_iff N floor).1 ⟨p.1, hp.1⟩⟩
  · rintro ⟨hc, hf⟩
    exact ⟨0, zero_mem_incidentMemoryParameterDomain N cap floor hc hf⟩

/-- Three learned edges and independently stored query/key additions.
Source: an enlarged-domain embedding witness for arXiv:1602.02068v2, Eq. (1). -/
def incidentMemoryExampleParameters : LocalMemoryParameters 3 :=
  ((fun _ => 1 / 8), (fun _ => 1))

/-- The new witness satisfies the entire relaxed compact domain.
Source: the concrete separate-budget construction before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryExampleParameters_mem :
    incidentMemoryExampleParameters ∈ incidentMemoryParameterDomain 3 4 (3 / 4) := by
  refine ⟨incidentMemoryExampleWeights_mem, ?_⟩
  intro x
  norm_num [incidentMemoryExampleParameters]

/-- The witness remains excluded by the former global parameter domain.
Source: strict domain enlargement for the derived arXiv:1602.02068v2, Eq. (1) architecture. -/
theorem incidentMemoryExampleParameters_not_global :
    incidentMemoryExampleParameters ∉ localMemoryParameterDomain 3 4 (3 / 4) := by
  intro h
  exact incidentMemoryExampleWeights_not_global h.1

/-- Each incident budget is a continuous function of the actual stored parameters.
Source: the finite linear inequalities preceding arXiv:1602.02068v2, Eq. (1). -/
theorem localIncidentWeight_continuous {N : ℕ} (i : Fin (N + 1)) :
    Continuous (fun p : LocalMemoryParameters N => localIncidentWeight p.1 i) := by
  apply continuous_finsetSum
  intro e he
  by_cases h : i = e.castSucc ∨ i = e.succ
  · simp only [h, ite_true]
    exact (continuous_apply e).comp continuous_fst
  · simp only [h, ite_false]
    exact continuous_const

/-- Separate-budget parameters form a compact finite intersection of linear coordinate bounds.
Source: the relaxed compact domain before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryParameterDomain_compact (N : ℕ) (cap floor : ℝ) :
    IsCompact (incidentMemoryParameterDomain N cap floor) := by
  let upper : LocalMemoryParameters N := ((fun _ => 1 - floor), (fun _ => cap - 1))
  have hd : incidentMemoryParameterDomain N cap floor =
      Set.Icc (0 : LocalMemoryParameters N) upper ∩
        {p : LocalMemoryParameters N | ∀ i, localIncidentWeight p.1 i ≤ 1 - floor} := by
    ext p
    constructor
    · intro hp
      refine ⟨⟨⟨hp.1.1, fun x => (hp.2 x).1⟩, ⟨?_, fun x => (hp.2 x).2⟩⟩, hp.1.2⟩
      intro e
      exact (incidentMemoryWeightDomain_coordinateBound floor p.1 hp.1 e).2
    · rintro ⟨⟨hl, hu⟩, hs⟩
      exact ⟨⟨hl.1, hs⟩, fun x => ⟨hl.2 x, hu.2 x⟩⟩
  have hs : IsClosed {p : LocalMemoryParameters N | ∀ i, localIncidentWeight p.1 i ≤ 1 - floor} := by
    rw [Set.ofPred_forall]
    exact isClosed_iInter (fun i => isClosed_le (localIncidentWeight_continuous i) continuous_const)
  rw [hd]
  exact CompactIccSpace.isCompact_Icc.inter_right hs

/-- Every reference has exactly one minimizer on the enlarged compact domain.
Source: the proved compact domain and strictly convex extra criterion for
arXiv:1602.02068v2, Eq. (1). Existence is a conclusion, not an assumed optimization input. -/
theorem incidentMemoryQuadratic_existsUnique {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    ∃! p, p ∈ incidentMemoryParameterDomain N cap floor ∧
      IsMinOn (localMemoryQuadratic reference) (incidentMemoryParameterDomain N cap floor) p := by
  have hn := (incidentMemoryParameterDomain_nonempty_iff N cap floor).2 ⟨hc, hf⟩
  obtain ⟨p, hp, hm⟩ := (incidentMemoryParameterDomain_compact N cap floor).exists_isMinOn hn
    (localMemoryQuadratic_continuous reference).continuousOn
  refine ⟨p, ⟨hp, hm⟩, ?_⟩
  intro q hq
  exact localMemoryQuadratic_unique_min reference _ (incidentMemoryParameterDomain_convex N cap floor)
    q p hq.1 hp hq.2 hm

/-- An infeasible reference still has a unique actual constrained minimum. -/
example : ∃! p, p ∈ incidentMemoryParameterDomain 1 4 (3 / 4) ∧
    IsMinOn (localMemoryQuadratic localMemoryOutsideReference)
      (incidentMemoryParameterDomain 1 4 (3 / 4)) p :=
  incidentMemoryQuadratic_existsUnique _ _ _ (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
