import Transformer.GPTMini.Sparsemax.PairedMatchingHead
import Transformer.GPTMini.Sparsemax.AtomicMatchingOptimality
import Mathlib.Topology.Order.Compact

/-!
# Convex training and global prices for contextual matching heads

New binding-preserving atomic model following arXiv:2211.11052v1, Appendix
A.4. The eligible atoms retain free original V-by-2H Q/K and V-by-D value
tables. Only their causal input encoder changes; no route labels, selected
interaction dictionary or independent pair parameters are introduced.

The actual head response still uses sparsemax Eq. (1) of arXiv:1602.02068v2.
Its finite probability mixture is linear in the distributional state, so
convex output criteria remain convex. Continuity gives attained global head
pricing. A certified price bound transfers to every feasible mixture. None
of these statements supplies an efficient globally optimal Q/K search.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Observed outputs of the actual contextual head, before any convexification.
Source: the paired §3.1 encoder using original sparsemax Eq. (1). -/
def pairedHeadSample {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (h : MatchingHead V (2 * H) D) : Fin R → ℝ :=
  fun r => pairedHeadOutput h (tokens r) (rows r) (channels r)

/-- Exact finite mixture of freely selected contextual physical heads.
Source: the genuine binding variant of Appendix A.4's atomic mixture. -/
def pairedMixtureSample {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    MatchingMixture V (2 * H) D →ₗ[ℝ] (Fin R → ℝ) :=
  Finsupp.linearCombination ℝ (pairedHeadSample tokens rows channels)

/-- The entire prediction class ranges over every eligible original physical table.
Source: the bounded Appendix A.4 domain with the binding-preserving encoder. -/
def pairedPredictionSet {V D R T : ℕ} (H : ℕ) (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    Set (Fin R → ℝ) :=
  pairedMixtureSample tokens rows channels '' matchingMixtureDomain V (2 * H) D cap

/-- Price a new physical contextual head by the supporting output functional.
Source: genuine-head pricing after Appendix A.4; atoms need not be previously stored. -/
def pairedHeadPrice {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ)
    (h : MatchingHead V (2 * H) D) : ℝ :=
  matchingOutputPairing g (pairedHeadSample tokens rows channels h)

/-- Finite contextual observations are continuous across actual sparsemax support changes.
Source: sparsemax Eq. (1) composed with the new factorized learned-table encoder. -/
theorem pairedHeadSample_continuous {V H D R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    Continuous (pairedHeadSample (H := H) tokens rows channels) := by
  apply continuous_pi
  intro r
  exact pairedHeadOutput_continuous (tokens r) (rows r) (channels r)

/-- A one-head mixture evaluates the genuine contextual forward, without response cancellation.
Source: the repaired encoder inside Appendix A.4's finite atomic state. -/
theorem pairedMixture_single_output {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (h : MatchingHead V (2 * H) D) :
    pairedMixtureSample tokens rows channels (Finsupp.single h 1) =
      pairedHeadSample tokens rows channels h := by
  rw [pairedMixtureSample, Finsupp.linearCombination_single, one_smul]

/-- Stored predictions are literal sums of contextual sparsemax/value products.
Source: sparsemax Eq. (1) in the binding-preserving Appendix A.4 mixture. -/
theorem pairedMixtureSample_apply {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (μ : MatchingMixture V (2 * H) D) (r : Fin R) :
    pairedMixtureSample tokens rows channels μ r =
      ∑ h ∈ μ.support, μ h * pairedHeadOutput h (tokens r) (rows r) (channels r) := by
  simp only [pairedMixtureSample, Finsupp.linearCombination_apply, Finsupp.sum,
    Finset.sum_apply, Pi.smul_apply, smul_eq_mul, pairedHeadSample]

/-- Every convex output criterion remains convex with freely selectable contextual heads.
Source: the distributional convexity of Appendix A.4, now with preserved neighboring-token binding. -/
theorem pairedMixture_criterion_convex {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (criterion : (Fin R → ℝ) → ℝ) (hc : ConvexOn ℝ Set.univ criterion) :
    ConvexOn ℝ (matchingMixtureDomain V (2 * H) D cap)
      (criterion ∘ pairedMixtureSample tokens rows channels) :=
  (hc.comp_linearMap (pairedMixtureSample tokens rows channels)).subset
    (fun _ _ => Set.mem_univ _) (matchingMixtureDomain_convex V (2 * H) D cap)

/-- A nonconstant observed criterion inhabits the contextual convexity premise. -/
example : ConvexOn ℝ (matchingMixtureDomain 2 2 1 1)
    ((fun y : Fin 1 → ℝ => y 0) ∘ pairedMixtureSample (H := 1)
      (fun _ : Fin 1 => (id : Fin 2 → Fin 2)) (fun _ => 1) (fun _ => (0 : Fin 1))) :=
  pairedMixture_criterion_convex (H := 1) _ _ _ _ _
    ((LinearMap.proj (R := ℝ) (φ := fun _ : Fin 1 => ℝ) 0).convexOn convex_univ)

/-- Contextual mixture prices are the coefficient-weighted physical head prices.
Source: the original value product in the new Appendix A.4 mixture. -/
theorem pairedMixture_price {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ)
    (μ : MatchingMixture V (2 * H) D) :
    matchingOutputPairing g (pairedMixtureSample tokens rows channels μ) =
      ∑ h ∈ μ.support, μ h * pairedHeadPrice tokens rows channels g h := by
  rw [pairedMixtureSample, Finsupp.linearCombination_apply]
  simp only [Finsupp.sum, map_sum, map_smul, smul_eq_mul, pairedHeadPrice]

/-- A price lower bound for all contextual heads extends to every feasible mixture.
Source: unit-mass atomic training after Appendix A.4; global pricing stays an explicit premise. -/
theorem pairedMixture_price_lower {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (g : Fin R → ℝ) (c : ℝ)
    (hp : ∀ h ∈ matchingHeadBox V (2 * H) D cap, c ≤ pairedHeadPrice tokens rows channels g h)
    (μ : MatchingMixture V (2 * H) D) (hμ : μ ∈ matchingMixtureDomain V (2 * H) D cap) :
    c ≤ matchingOutputPairing g (pairedMixtureSample tokens rows channels μ) := by
  classical
  have hs := hμ.2.2
  rw [matchingMixtureMass_eq_sum] at hs
  rw [pairedMixture_price]
  calc
    c = (∑ h ∈ μ.support, μ h) * c := by rw [hs, one_mul]
    _ = ∑ h ∈ μ.support, μ h * c := Finset.sum_mul _ _ _
    _ ≤ ∑ h ∈ μ.support, μ h * pairedHeadPrice tokens rows channels g h := by
      apply Finset.sum_le_sum
      intro h hh
      exact mul_le_mul_of_nonneg_left (hp h (hμ.2.1 h (Finsupp.mem_support_iff.mp hh))) (hμ.1 h)

/-- A nonzero universal price bound and a feasible original head satisfy both premises. -/
example : -1 ≤ matchingOutputPairing (fun _ : Fin 1 => (-1 : ℝ))
    (pairedMixtureSample (H := 1) (fun _ : Fin 1 => (id : Fin 2 → Fin 2)) (fun _ => 1)
      (fun _ => (0 : Fin 1)) (Finsupp.single (0 : MatchingHead 2 2 1) 1)) := by
  apply pairedMixture_price_lower (H := 1) (cap := 1)
  · intro h hh
    have hb := (pairedHeadOutput_bounds 1 h hh id 1 0).2
    simp only [pairedHeadPrice, matchingOutputPairing, pairedHeadSample,
      Fin.sum_univ_one, LinearMap.coe_mk, AddHom.coe_mk]
    linarith
  · exact matchingMixture_single_mem _ _ (matchingHeadBox_zero_mem _ _ _ _ (by norm_num))

/-- True contextual head pricing is continuous in free Q, both key roles and original values.
Source: supporting-output pricing following Appendix A.4, with actual sparsemax Eq. (1). -/
theorem pairedHeadPrice_continuous {V H D R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ) :
    Continuous (pairedHeadPrice (H := H) tokens rows channels g) := by
  have hs := pairedHeadSample_continuous (H := H) tokens rows channels
  change Continuous (fun h : MatchingHead V (2 * H) D => ∑ r, g r * pairedHeadSample tokens rows channels h r)
  apply continuous_finsetSum
  intro r hr
  exact continuous_const.mul ((continuous_apply r).comp hs)

/-- Global contextual pricing has an attained minimizer over the original bounded token tables.
Source: compact physical-head pricing after Appendix A.4, without expanding the free parameter set. -/
theorem pairedHeadPrice_minimum_exists {V H D R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ) :
    ∃ h ∈ matchingHeadBox V (2 * H) D cap, ∀ k ∈ matchingHeadBox V (2 * H) D cap,
      pairedHeadPrice tokens rows channels g h ≤ pairedHeadPrice tokens rows channels g k := by
  exact (matchingHeadBox_compact V (2 * H) D cap).exists_isMinOn
    ⟨0, matchingHeadBox_zero_mem V (2 * H) D cap hc⟩
    (pairedHeadPrice_continuous tokens rows channels g).continuousOn

/-- Nonconstant physical output pricing satisfies the attained-minimizer cap premise. -/
example : ∃ h ∈ matchingHeadBox 2 2 1 1, ∀ k ∈ matchingHeadBox 2 2 1 1,
    pairedHeadPrice (H := 1) (fun _ : Fin 1 => (id : Fin 2 → Fin 2)) (fun _ => 1)
      (fun _ => (0 : Fin 1)) (fun _ => (-2 : ℝ)) h ≤
    pairedHeadPrice (H := 1) (fun _ : Fin 1 => (id : Fin 2 → Fin 2)) (fun _ => 1)
      (fun _ => (0 : Fin 1)) (fun _ => (-2 : ℝ)) k :=
  pairedHeadPrice_minimum_exists (H := 1) 1 (by norm_num) _ _ _ _

end Transformer.GPTMini.Sparsemax
