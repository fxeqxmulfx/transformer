import Transformer.GPTMini.Sparsemax.AtomicMatchingCompression
import Mathlib.Analysis.Convex.Function

/-!
# Convex criteria and global certificates for freely learned matching

New first-order consequences of the genuine atomic model following
arXiv:2211.11052v1, Appendix A.4. The pricing function evaluates a whole
physical head with its freely selectable Q/K and independent values.
A certified lower price must hold for every head in the numerical box,
including heads absent from the current finite support.

A supporting linear functional and a global price bound certify the
objective gap. The certificate does not assert that gradient descent
on Q/K finds the required global head price. Convexity is proved only
when the supplied output criterion is convex; no FFN or nonlinear
trainable layer stack is included.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- A linear supporting functional on the actual observed scalar outputs.
Source: new first-order certificate for the atomic model after Appendix A.4. -/
def matchingOutputPairing {R : ℕ} (g : Fin R → ℝ) : (Fin R → ℝ) →ₗ[ℝ] ℝ where
  toFun y := ∑ r, g r * y r
  map_add' x y := by simp only [Pi.add_apply, mul_add, Finset.sum_add_distrib]
  map_smul' a y := by
    change (∑ r, g r * (a * y r)) = a * ∑ r, g r * y r
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro r hr
    ring

/-- Price a genuinely learned head; its index is not restricted to an existing interaction bank.
Source: new genuine-head pricing problem after Appendix A.4's atomic model. -/
def matchingHeadPrice {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ)
    (h : MatchingHead V H D) : ℝ :=
  matchingOutputPairing g (matchingHeadSample tokens rows channels h)

/-- Mixture pricing is exactly the weighted sum of physical head prices.
Source: the original sparsemax forward inside the new Appendix A.4 atomic mixture. -/
theorem matchingMixture_price {V H D R T : ℕ} (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ)
    (μ : MatchingMixture V H D) :
    matchingOutputPairing g (matchingMixtureSample tokens rows channels μ) =
      ∑ h ∈ μ.support, μ h * matchingHeadPrice tokens rows channels g h := by
  rw [matchingMixtureSample, Finsupp.linearCombination_apply]
  simp only [Finsupp.sum, map_sum, map_smul, smul_eq_mul, matchingHeadPrice]

/-- A global head-price bound extends to every feasible mixture, including new matching heads.
Source: the unit-mass atomic model after Appendix A.4; global pricing is an explicit hypothesis. -/
theorem matchingMixture_price_lower {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (g : Fin R → ℝ) (c : ℝ)
    (hp : ∀ h ∈ matchingHeadBox V H D cap, c ≤ matchingHeadPrice tokens rows channels g h)
    (μ : MatchingMixture V H D) (hμ : μ ∈ matchingMixtureDomain V H D cap) :
    c ≤ matchingOutputPairing g (matchingMixtureSample tokens rows channels μ) := by
  classical
  have hs := hμ.2.2
  rw [matchingMixtureMass_eq_sum] at hs
  rw [matchingMixture_price]
  calc
    c = (∑ h ∈ μ.support, μ h) * c := by rw [hs, one_mul]
    _ = ∑ h ∈ μ.support, μ h * c := Finset.sum_mul _ _ _
    _ ≤ ∑ h ∈ μ.support, μ h * matchingHeadPrice tokens rows channels g h := by
      apply Finset.sum_le_sum
      intro h hh
      exact mul_le_mul_of_nonneg_left (hp h (hμ.2.1 h (Finsupp.mem_support_iff.mp hh))) (hμ.1 h)

/-- A nonzero price bound holds for every possible head, not only the stored uniform head. -/
example : -1 ≤ matchingOutputPairing (fun _ : Fin 1 => -1)
    (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
      (Finsupp.single (matchingScalarHead 0) 1)) := by
  apply matchingMixture_price_lower (cap := 1)
  · intro h hh
    have hb := (matchingHeadOutput_bounds 1 h hh id 1 0).2
    simp only [matchingHeadPrice, matchingOutputPairing, matchingHeadSample,
      Fin.sum_univ_one, LinearMap.coe_mk, AddHom.coe_mk]
    linarith
  · exact matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num))

/-- Every convex output criterion gives a convex optimization problem on the full matching state.
Source: Appendix A.4's measure-linear convexity, proved for genuine sparsemax heads here. -/
theorem matchingMixture_criterion_convex {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (criterion : (Fin R → ℝ) → ℝ) (hc : ConvexOn ℝ Set.univ criterion) :
    ConvexOn ℝ (matchingMixtureDomain V H D cap)
      (criterion ∘ matchingMixtureSample tokens rows channels) :=
  (hc.comp_linearMap (matchingMixtureSample tokens rows channels)).subset
    (fun _ _ => Set.mem_univ _) (matchingMixtureDomain_convex V H D cap)

/-- A nonconstant linear criterion inhabits the genuine full-domain convexity premise. -/
example : ConvexOn ℝ (matchingMixtureDomain 2 1 1 1)
    ((fun y : Fin 1 → ℝ => y 0) ∘
      matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)) :=
  matchingMixture_criterion_convex _ _ _ _ _
    ((LinearMap.proj (R := ℝ) (φ := fun _ : Fin 1 => ℝ) 0).convexOn convex_univ)

/-- Supporting output gradients and certified global atom prices bound the true objective gap.
Source: new first-order consequence of Appendix A.4's atomic architecture. The global search
condition is explicit and ranges over all original Q/K/value heads, without fixed interactions. -/
theorem matchingMixture_gap_certificate {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (criterion : (Fin R → ℝ) → ℝ) (μ : MatchingMixture V H D)
    (hμ : μ ∈ matchingMixtureDomain V H D cap) (g : Fin R → ℝ) (c ε : ℝ)
    (hp : ∀ h ∈ matchingHeadBox V H D cap, c ≤ matchingHeadPrice tokens rows channels g h)
    (hg : ∀ y ∈ matchingPredictionSet H cap tokens rows channels,
      criterion (matchingMixtureSample tokens rows channels μ) +
        matchingOutputPairing g (y - matchingMixtureSample tokens rows channels μ) ≤ criterion y)
    (hgap : matchingOutputPairing g (matchingMixtureSample tokens rows channels μ) - c ≤ ε) :
    0 ≤ ε ∧ ∀ ν ∈ matchingMixtureDomain V H D cap,
      criterion (matchingMixtureSample tokens rows channels μ) ≤
        criterion (matchingMixtureSample tokens rows channels ν) + ε := by
  have hcur := matchingMixture_price_lower cap tokens rows channels g c hp μ hμ
  refine ⟨by linarith, ?_⟩
  intro ν hν
  have hnew := matchingMixture_price_lower cap tokens rows channels g c hp ν hν
  have ht := hg (matchingMixtureSample tokens rows channels ν) ⟨ν, hν, rfl⟩
  rw [map_sub] at ht
  linarith

/-- A genuine saturated matching head has a global squared-error certificate with nonzero gradient.
The target is two, so unit value bounds make the optimal residual one, not zero. -/
example : ∀ ν ∈ matchingMixtureDomain 2 1 1 1,
    (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
      (Finsupp.single (matchingScalarHead 1) 1) 0 - 2) ^ 2 ≤
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) ν 0 - 2) ^ 2 := by
  have hp : ∀ h ∈ matchingHeadBox 2 1 1 1,
      -2 ≤ matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) (fun _ => -2) h := by
    intro h hh
    have hb := (matchingHeadOutput_bounds 1 h hh id 1 0).2
    simp only [matchingHeadPrice, matchingOutputPairing, matchingHeadSample,
      Fin.sum_univ_one, LinearMap.coe_mk, AddHom.coe_mk]
    linarith
  have hg : ∀ y ∈ matchingPredictionSet 1 1 (fun _ : Fin 1 => (id : Fin 2 → Fin 2))
      (fun _ => 1) (fun _ => (0 : Fin 1)),
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (Finsupp.single (matchingScalarHead 1) 1) 0 - 2) ^ 2 +
        matchingOutputPairing (fun _ : Fin 1 => -2)
          (y - matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
            (Finsupp.single (matchingScalarHead 1) 1)) ≤ (y 0 - 2) ^ 2 := by
    intro y hy
    rw [matchingMixture_single_output]
    simp only [matchingHeadSample, matchingScalarHead_one_output, matchingOutputPairing,
      LinearMap.coe_mk, AddHom.coe_mk, Fin.sum_univ_one, Pi.sub_apply]
    nlinarith [sq_nonneg (y 0 - 1)]
  have hgap : matchingOutputPairing (fun _ : Fin 1 => -2)
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (Finsupp.single (matchingScalarHead 1) 1)) - (-2) ≤ 0 := by
    rw [matchingMixture_single_output]
    norm_num [matchingOutputPairing, matchingHeadSample, matchingScalarHead_one_output]
  have hcert := matchingMixture_gap_certificate 1 (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
    (fun y => (y 0 - 2) ^ 2) (Finsupp.single (matchingScalarHead 1) 1)
    (matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num)))
    (fun _ => -2) (-2) 0 hp hg hgap
  simpa only [add_zero] using hcert.2

end Transformer.GPTMini.Sparsemax
