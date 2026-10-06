import Transformer.GPTMini.Sparsemax.LocalMemoryQuadratic
import Transformer.GPTMini.Sparsemax.LocalMemoryFeasibility
import Mathlib.Topology.Order.Compact

/-!
# Existence and uniqueness of an additional compact-memory selection

Derived selection for arXiv:1602.02068v2, Eq. (1). The linear compact
parameter domain is a closed total-edge-budget restriction of a finite
coordinate box. It is compact and nonempty at cap at least one and floor
at most one. The complete squared-coordinate criterion is continuous and
strictly convex, so it attains exactly one constrained minimum.

The reference need not be feasible. It supplies additional information;
the selected parameters are its constrained projection, not an equality
constraint fixing the attention. With a floor above one half, the selected
Gram has genuine bounded-width embeddings and invertible actual attention.
This resolves choice under the extra criterion, not identification from
the output-only forward. No model task loss or FFN is selected here.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Every compact path/norm coordinate is a continuous function of the learned parameters.
Source: the complete compact embedding chart for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryParameterCoordinates_continuous {N : ℕ} (i : LocalMemoryParameterIndex N) :
    Continuous (fun p : LocalMemoryParameters N => localMemoryParameterCoordinates p i) := by
  rcases i with e | x
  · exact (continuous_apply e).comp continuous_fst
  · exact (continuous_apply x).comp continuous_snd

/-- The additional squared-coordinate selection criterion is continuous for every reference.
Source: the derived compact-memory criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryQuadratic_continuous {N : ℕ} (reference : LocalMemoryParameters N) :
    Continuous (localMemoryQuadratic reference) := by
  exact continuous_finsetSum Finset.univ (fun i hi =>
    ((localMemoryParameterCoordinates_continuous i).sub continuous_const).pow 2)

/-- The actual linear compact parameter domain is compact at every cap and floor.
Source: the finite coordinate bounds and edge budget for arXiv:1602.02068v2, Eq. (1).
Invalid cap/floor choices may give an empty compact set; nonemptiness is proved separately. -/
theorem localMemoryParameterDomain_compact (N : ℕ) (cap floor : ℝ) :
    IsCompact (localMemoryParameterDomain N cap floor) := by
  let upper : LocalMemoryParameters N := ((fun _ => 1 - floor), (fun _ => cap - 1))
  have hd : localMemoryParameterDomain N cap floor =
      Set.Icc (0 : LocalMemoryParameters N) upper ∩
        {p : LocalMemoryParameters N | (∑ e, p.1 e) ≤ 1 - floor} := by
    ext p
    constructor
    · intro hp
      refine ⟨⟨⟨hp.1.1, fun x => (hp.2 x).1⟩, ⟨?_, fun x => (hp.2 x).2⟩⟩, hp.1.2⟩
      intro e
      exact (localWeightDomain_coordinateBound floor p.1 hp.1 e).2
    · rintro ⟨⟨hl, hu⟩, hs⟩
      exact ⟨⟨hl.1, hs⟩, fun x => ⟨hl.2 x, hu.2 x⟩⟩
  have hc : Continuous (fun p : LocalMemoryParameters N => ∑ e, p.1 e) :=
    continuous_finsetSum Finset.univ (fun e he => (continuous_apply e).comp continuous_fst)
  rw [hd]
  exact CompactIccSpace.isCompact_Icc.inter_right (isClosed_le hc continuous_const)

/-- Every reference, feasible or otherwise, has exactly one selected compact parameter point.
Source: the continuous strictly convex criterion and proved compact domain for
arXiv:1602.02068v2, Eq. (1). Existence is concluded, not assumed as an optimization input. -/
theorem localMemoryQuadratic_existsUnique {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    ∃! p, p ∈ localMemoryParameterDomain N cap floor ∧
      IsMinOn (localMemoryQuadratic reference) (localMemoryParameterDomain N cap floor) p := by
  have hn := (localMemoryParameterDomain_nonempty_iff N cap floor).2 ⟨hc, hf⟩
  obtain ⟨p, hp, hm⟩ := (localMemoryParameterDomain_compact N cap floor).exists_isMinOn hn
    (localMemoryQuadratic_continuous reference).continuousOn
  refine ⟨p, ⟨hp, hm⟩, ?_⟩
  intro q hq
  exact localMemoryQuadratic_unique_min reference _ (localMemoryParameterDomain_convex N cap floor)
    q p hq.1 hp hq.2 hm

/-- A reference deliberately outside both the edge and norm bounds.
Source: the constrained selection witness for arXiv:1602.02068v2, Eq. (1). -/
def localMemoryOutsideReference : LocalMemoryParameters 1 := ((fun _ => 1), (fun _ => 5))

/-- The supplied reference is not itself an admissible attention parameter point.
Source: the explicit extra-criterion witness for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryOutsideReference_not_mem :
    localMemoryOutsideReference ∉ localMemoryParameterDomain 1 4 (3 / 4) := by
  intro h
  have hb := h.1.2
  norm_num [localMemoryOutsideReference, Fin.sum_univ_one] at hb

/-- An actual infeasible reference still has one genuine constrained minimum. -/
example : ∃! p, p ∈ localMemoryParameterDomain 1 4 (3 / 4) ∧
    IsMinOn (localMemoryQuadratic localMemoryOutsideReference)
      (localMemoryParameterDomain 1 4 (3 / 4)) p :=
  localMemoryQuadratic_existsUnique _ _ _ (by norm_num) (by norm_num)

/-- The selected parameters are chosen from a proved unique constrained minimum.
Source: the additional compact-memory criterion for arXiv:1602.02068v2, Eq. (1).
This noncomputable definition records an optimization result, not a solver algorithm. -/
def localMemorySelectedParameters {N : ℕ} (cap floor : ℝ) (reference : LocalMemoryParameters N)
    (hc : 1 ≤ cap) (hf : floor ≤ 1) : LocalMemoryParameters N :=
  Classical.choose (localMemoryQuadratic_existsUnique cap floor reference hc hf).exists

/-- The selected point satisfies every actual linear parameter-domain constraint.
Source: the proved constrained minimum for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemorySelectedParameters_mem {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    localMemorySelectedParameters cap floor reference hc hf ∈ localMemoryParameterDomain N cap floor :=
  (Classical.choose_spec (localMemoryQuadratic_existsUnique cap floor reference hc hf).exists).1

/-- A reference outside the domain still gives an actually admissible selected point. -/
example : localMemorySelectedParameters 4 (3 / 4) localMemoryOutsideReference (by norm_num) (by norm_num) ∈
    localMemoryParameterDomain 1 4 (3 / 4) :=
  localMemorySelectedParameters_mem _ _ _ (by norm_num) (by norm_num)

/-- The selected point minimizes the criterion on the complete convex parameter domain.
Source: the proved optimization result for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemorySelectedParameters_min {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    IsMinOn (localMemoryQuadratic reference) (localMemoryParameterDomain N cap floor)
      (localMemorySelectedParameters cap floor reference hc hf) :=
  (Classical.choose_spec (localMemoryQuadratic_existsUnique cap floor reference hc hf).exists).2

/-- Both existence-bound premises are inhabited with an infeasible reference. -/
example : IsMinOn (localMemoryQuadratic localMemoryOutsideReference)
    (localMemoryParameterDomain 1 4 (3 / 4))
    (localMemorySelectedParameters 4 (3 / 4) localMemoryOutsideReference (by norm_num) (by norm_num)) :=
  localMemorySelectedParameters_min _ _ _ (by norm_num) (by norm_num)

/-- The uniquely selected parameters supply genuine embeddings and an actual attention inverse.
Source: the additional criterion and structural memory guarantees for
arXiv:1602.02068v2, Eq. (1); no embedding factorization or minimizer is assumed. -/
theorem localMemorySelected_embeddings_inverse {N : ℕ} (cap floor : ℝ)
    (reference : LocalMemoryParameters N) (hc : 1 ≤ cap) (hf : 1 / 2 < floor) (hf1 : floor ≤ 1) :
    ∃ p ∈ localMemoryParameterDomain N cap floor,
      IsMinOn (localMemoryQuadratic reference) (localMemoryParameterDomain N cap floor) p ∧
      (∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
        localMemoryGram p = featureGram features) ∧
      IsUnit (memoryGramAttention (localMemoryGram p)).det := by
  obtain ⟨p, ⟨hp, hm⟩, hu⟩ := localMemoryQuadratic_existsUnique cap floor reference hc hf1
  exact ⟨p, hp, hm, localMemoryGram_fixedWidth cap floor p (by linarith) hp,
    localMemoryAttention_det_unit cap floor p hf hp⟩

/-- The selected point has ordinary embedding coordinates and an inverse even for an infeasible reference. -/
example : ∃ p ∈ localMemoryParameterDomain 1 4 (3 / 4),
    IsMinOn (localMemoryQuadratic localMemoryOutsideReference)
      (localMemoryParameterDomain 1 4 (3 / 4)) p ∧
    (∃ features : Fin 4 → Sum (Fin 2) (Fin 2) → ℝ, localMemoryGram p = featureGram features) ∧
    IsUnit (memoryGramAttention (localMemoryGram p)).det :=
  localMemorySelected_embeddings_inverse _ _ _ (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
