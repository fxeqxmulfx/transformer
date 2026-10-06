import Transformer.GPTMini.Sparsemax.AtomicMatchingFamily
import Transformer.GPTMini.Sparsemax.AtomicMatchingOptimality

/-!
# Attained global training and pricing for freely learned matching

New attainment theorem for the genuine sparsemax atomic architecture
following arXiv:2211.11052v1, Appendix A.4. R scalar observations have
exactly the same prediction class as a compact chart with R+1 freely
selected heads. Continuity includes actual sparsemax support changes.

Every continuous output criterion has an attained global minimum with
at most R+1 physical heads. No optimal state or exact fit is supplied.
When the criterion is convex, the previously proved distributional
convexity also applies. Raw fixed-chart optimization is not proved convex.

Global physical-head pricing is continuous on the numerical head box,
so its optimum exists too. These are mathematical existence guarantees,
not an efficient method for finding a new globally optimal Q/K/value head.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The complete observed prediction class equals one compact finite physical chart.
Source: exact actual-head compression and padding after Appendix A.4, rather than a fixed atom dictionary. -/
theorem matchingPredictionSet_eq_family {V H D R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    matchingPredictionSet H cap tokens rows channels =
      matchingFamilySample tokens rows channels '' matchingFamilyDomain V H D (R + 1) cap := by
  ext y
  constructor
  · rintro ⟨μ, hμ, rfl⟩
    obtain ⟨ν, hν, he, hn⟩ := matchingMixture_compress cap tokens rows channels μ hμ
    obtain ⟨p, hp, hsample⟩ := matchingMixture_pad_family cap hc tokens rows channels ν hν hn
    exact ⟨p, hp, hsample.trans he⟩
  · rintro ⟨p, hp, rfl⟩
    exact ⟨matchingMixtureOfFamily p.1 p.2, matchingFamily_mem cap p hp,
      matchingFamily_reconstruct tokens rows channels p⟩

/-- A genuine token observation inhabits every finite-chart equality premise. -/
example : matchingPredictionSet 1 1 (fun _ : Fin 1 => (id : Fin 2 → Fin 2))
    (fun _ => 1) (fun _ => (0 : Fin 1)) =
      matchingFamilySample (fun _ : Fin 1 => (id : Fin 2 → Fin 2)) (fun _ => 1) (fun _ => (0 : Fin 1)) ''
        matchingFamilyDomain 2 1 1 2 1 :=
  matchingPredictionSet_eq_family 1 (by norm_num) _ _ _

/-- The full observed convex matching class is compact despite having a continuous family of atoms.
Source: actual R+1 physical-chart representation after Appendix A.4; no head bank is fixed. -/
theorem matchingPredictionSet_compact {V H D R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    IsCompact (matchingPredictionSet H cap tokens rows channels) := by
  rw [matchingPredictionSet_eq_family cap hc]
  exact (matchingFamilyDomain_compact V H D (R + 1) cap).image
    (matchingFamilySample_continuous tokens rows channels)

/-- A nontrivial token/channel observation satisfies the compactness cap premise. -/
example : IsCompact (matchingPredictionSet 1 1 (fun _ : Fin 1 => (id : Fin 2 → Fin 2))
    (fun _ => 1) (fun _ => (0 : Fin 1))) :=
  matchingPredictionSet_compact 1 (by norm_num) _ _ _

/-- The numerical cap gives a feasible true prediction without assuming a fitting answer table.
Source: zero physical tables inside the complete Appendix A.4 matching family. -/
theorem matchingPredictionSet_nonempty {V H D R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    (matchingPredictionSet H cap tokens rows channels).Nonempty := by
  refine ⟨matchingHeadSample tokens rows channels (0 : MatchingHead V H D),
    Finsupp.single (0 : MatchingHead V H D) 1, ?_, ?_⟩
  · exact matchingMixture_single_mem cap _ (matchingHeadBox_zero_mem V H D cap hc)
  · exact matchingMixture_single_output tokens rows channels _

/-- The cap-zero domain is still nonempty for an actual token observation. -/
example : (matchingPredictionSet 1 0 (fun _ : Fin 1 => (id : Fin 2 → Fin 2))
    (fun _ => 1) (fun _ => (0 : Fin 1))).Nonempty :=
  matchingPredictionSet_nonempty 0 (by norm_num) _ _ _

/-- Arbitrary continuous output training attains a global optimum with at most R+1 learned heads.
Source: the new bounded genuine-head version of Appendix A.4's sparse-optimum conclusion.
The paper uses positional simplex atoms and TV regularization; here heads are shared learned Q/K/values
with numerical coordinate bounds and probability mass. No optimum or exact-fit hypothesis is assumed. -/
theorem matchingMixture_minimum_exists {V H D R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (criterion : (Fin R → ℝ) → ℝ) (hcriterion : Continuous criterion) :
    ∃ μ ∈ matchingMixtureDomain V H D cap, μ.support.card ≤ R + 1 ∧
      ∀ ν ∈ matchingMixtureDomain V H D cap,
        criterion (matchingMixtureSample tokens rows channels μ) ≤
          criterion (matchingMixtureSample tokens rows channels ν) := by
  obtain ⟨y, hy, hm⟩ := (matchingPredictionSet_compact cap hc tokens rows channels).exists_isMinOn
    (matchingPredictionSet_nonempty cap hc tokens rows channels) hcriterion.continuousOn
  obtain ⟨μ, hμ, rfl⟩ := hy
  apply matchingMixture_optimum_compress cap tokens rows channels criterion μ hμ
  intro ν hν
  exact hm ⟨ν, hν, rfl⟩

/-- Ordinary squared error with an out-of-box target satisfies every attainment premise.
The existence result allows unavoidable positive training error. -/
example : ∃ μ ∈ matchingMixtureDomain 2 1 1 1, μ.support.card ≤ 2 ∧
    ∀ ν ∈ matchingMixtureDomain 2 1 1 1,
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) μ 0 - 2) ^ 2 ≤
        (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) ν 0 - 2) ^ 2 :=
  matchingMixture_minimum_exists 1 (by norm_num) _ _ _ (fun y => (y 0 - 2) ^ 2) (by fun_prop)

/-- A true physical head's linear observed price is continuous in all its independent tables.
Source: the supporting-output pricing problem following Appendix A.4. -/
theorem matchingHeadPrice_continuous {V H D R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ) :
    Continuous (matchingHeadPrice (H := H) tokens rows channels g) := by
  have hs := matchingHeadSample_continuous (H := H) tokens rows channels
  change Continuous (fun h : MatchingHead V H D => ∑ r, g r * matchingHeadSample tokens rows channels h r)
  apply continuous_finsetSum
  intro r hr
  exact continuous_const.mul ((continuous_apply r).comp hs)

/-- The complete Q/K/value pricing problem has an attained global physical-head minimizer.
Source: compact learned physical atoms after Appendix A.4; no search algorithm is assumed. -/
theorem matchingHeadPrice_minimum_exists {V H D R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) (g : Fin R → ℝ) :
    ∃ h ∈ matchingHeadBox V H D cap, ∀ k ∈ matchingHeadBox V H D cap,
      matchingHeadPrice tokens rows channels g h ≤ matchingHeadPrice tokens rows channels g k := by
  exact (matchingHeadBox_compact V H D cap).exists_isMinOn
    ⟨0, matchingHeadBox_zero_mem V H D cap hc⟩ (matchingHeadPrice_continuous tokens rows channels g).continuousOn

/-- Nonzero output pricing inhabits all global physical-head minimizer premises. -/
example : ∃ h ∈ matchingHeadBox 2 1 1 1, ∀ k ∈ matchingHeadBox 2 1 1 1,
    matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) (fun _ => (-2 : ℝ)) h ≤
      matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) (fun _ => (-2 : ℝ)) k :=
  matchingHeadPrice_minimum_exists 1 (by norm_num) _ _ _ _

/-- A supporting output functional admits an attained global gap certificate over all physical heads.
Source: the new complete-atom certificate following Appendix A.4. The minimizing head's existence
is proved; locating it numerically is still a separate optimization problem. -/
theorem matchingMixture_gap_exists {V H D R T : ℕ} (cap : ℝ) (hc : 0 ≤ cap)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (criterion : (Fin R → ℝ) → ℝ) (μ : MatchingMixture V H D)
    (hμ : μ ∈ matchingMixtureDomain V H D cap) (g : Fin R → ℝ)
    (hg : ∀ y ∈ matchingPredictionSet H cap tokens rows channels,
      criterion (matchingMixtureSample tokens rows channels μ) +
        matchingOutputPairing g (y - matchingMixtureSample tokens rows channels μ) ≤ criterion y) :
    ∃ h ∈ matchingHeadBox V H D cap,
      (∀ k ∈ matchingHeadBox V H D cap,
        matchingHeadPrice tokens rows channels g h ≤ matchingHeadPrice tokens rows channels g k) ∧
      0 ≤ matchingOutputPairing g (matchingMixtureSample tokens rows channels μ) -
        matchingHeadPrice tokens rows channels g h ∧
      ∀ ν ∈ matchingMixtureDomain V H D cap,
        criterion (matchingMixtureSample tokens rows channels μ) ≤
          criterion (matchingMixtureSample tokens rows channels ν) +
            (matchingOutputPairing g (matchingMixtureSample tokens rows channels μ) -
              matchingHeadPrice tokens rows channels g h) := by
  obtain ⟨h, hh, hp⟩ := matchingHeadPrice_minimum_exists cap hc tokens rows channels g
  exact ⟨h, hh, hp, matchingMixture_gap_certificate cap tokens rows channels criterion μ hμ g
    (matchingHeadPrice tokens rows channels g h)
    (matchingOutputPairing g (matchingMixtureSample tokens rows channels μ) -
      matchingHeadPrice tokens rows channels g h) hp hg le_rfl⟩

/-- A nonconstant ordinary linear output criterion satisfies every attained-gap premise. -/
example : ∃ h ∈ matchingHeadBox 2 1 1 1,
    (∀ k ∈ matchingHeadBox 2 1 1 1,
      matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) (fun _ => (-1 : ℝ)) h ≤
        matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) (fun _ => (-1 : ℝ)) k) ∧
    0 ≤ matchingOutputPairing (fun _ : Fin 1 => -1)
      (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
        (Finsupp.single (matchingScalarHead 0) 1)) -
      matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) (fun _ => (-1 : ℝ)) h ∧
    ∀ ν ∈ matchingMixtureDomain 2 1 1 1,
      matchingOutputPairing (fun _ : Fin 1 => -1)
        (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
          (Finsupp.single (matchingScalarHead 0) 1)) ≤
        matchingOutputPairing (fun _ : Fin 1 => -1)
          (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) ν) +
          (matchingOutputPairing (fun _ : Fin 1 => -1)
            (matchingMixtureSample (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0)
              (Finsupp.single (matchingScalarHead 0) 1)) -
            matchingHeadPrice (fun _ : Fin 1 => id) (fun _ => 1) (fun _ => 0) (fun _ => (-1 : ℝ)) h) := by
  apply matchingMixture_gap_exists (cap := 1) (by norm_num)
  · exact matchingMixture_single_mem _ _ (matchingScalarHead_mem _ (by norm_num))
  · intro y hy
    rw [map_sub]
    linarith

end Transformer.GPTMini.Sparsemax
