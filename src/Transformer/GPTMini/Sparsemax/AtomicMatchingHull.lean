import Transformer.GPTMini.Sparsemax.AtomicMatchingMixture

/-!
# Exact convexification by finite mixtures of unrestricted matching atoms

New genuine-sparsemax variant of arXiv:2211.11052v1, Appendix A.4.
The theorem below identifies the actual physical-mixture prediction set
with the convex hull of all bounded Q/K/value heads. This is an exact
statement for the new mixture architecture, not a claim that a fixed
number of raw heads or the paper's weight decay is convex.

The atom set is the image of every head in the numerical box. Neither
selected bit interactions nor teacher routes enter it. A finite family
is reconstructed as a stored finitely supported state; duplicate heads
are merged. Every reconstructed prediction still evaluates original
causal sparsemax attention and independent common values.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Assemble a learned finite head family without requiring its identities before optimization.
Source: the finite atomic measure following Appendix A.4. -/
def matchingMixtureOfFamily {ι : Type*} [Fintype ι] {V H D : ℕ}
    (heads : ι → MatchingHead V H D) (w : ι → ℝ) : MatchingMixture V H D :=
  ∑ i, Finsupp.single (heads i) (w i)

/-- Nonnegative learned family weights remain nonnegative after identical heads are merged.
Source: finite physical atom reconstruction after Appendix A.4. -/
theorem matchingMixtureOfFamily_nonneg {ι : Type*} [Fintype ι] {V H D : ℕ}
    (heads : ι → MatchingHead V H D) (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i) (h : MatchingHead V H D) :
    0 ≤ matchingMixtureOfFamily heads w h := by
  classical
  rw [matchingMixtureOfFamily, Finsupp.finsetSum_apply]
  apply Finset.sum_nonneg
  intro i hi
  simp only [Finsupp.single_apply]
  split_ifs
  · exact hw i
  · exact le_rfl

/-- Two actual matching heads with positive weights inhabit the reconstruction's nonnegativity premise. -/
example : 0 ≤ matchingMixtureOfFamily (fun i : Fin 2 => matchingScalarHead i.val)
    (fun _ => (1 / 2 : ℝ)) (matchingScalarHead 1) :=
  matchingMixtureOfFamily_nonneg _ _ (fun _ => by norm_num) _

/-- Reconstructed support contains only selected physical heads, even with signed coefficients.
Source: the exact finite atomic representation following Appendix A.4. -/
theorem matchingMixtureOfFamily_support {ι : Type*} [Fintype ι] {V H D : ℕ}
    (heads : ι → MatchingHead V H D) (w : ι → ℝ) :
    (matchingMixtureOfFamily heads w).support ⊆ Finset.univ.image heads := by
  classical
  intro h hh
  by_contra hn
  have hi : ∀ i, heads i ≠ h := by
    intro i he
    exact hn (Finset.mem_image.mpr ⟨i, Finset.mem_univ i, he⟩)
  have hz : matchingMixtureOfFamily heads w h = 0 := by
    simp [matchingMixtureOfFamily, Finsupp.finsetSum_apply, hi]
  exact (Finsupp.mem_support_iff.mp hh) hz

/-- Assembling a finite head family preserves its literal total coefficient mass.
Source: the unit-budget atomic state following Appendix A.4. -/
theorem matchingMixtureOfFamily_mass {ι : Type*} [Fintype ι] {V H D : ℕ}
    (heads : ι → MatchingHead V H D) (w : ι → ℝ) :
    matchingMixtureMass V H D (matchingMixtureOfFamily heads w) = ∑ i, w i := by
  simp [matchingMixtureOfFamily, map_sum, matchingMixtureMass]

/-- Finite assembly preserves the genuine learned heads' outputs on every observed coordinate.
Source: attention-only head summation inside the Appendix A.4 matching mixture. -/
theorem matchingMixtureOfFamily_sample {ι : Type*} [Fintype ι] {V H D R T : ℕ}
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D)
    (heads : ι → MatchingHead V H D) (w : ι → ℝ) :
    matchingMixtureSample tokens rows channels (matchingMixtureOfFamily heads w) =
      ∑ i, w i • matchingHeadSample tokens rows channels (heads i) := by
  simp [matchingMixtureOfFamily, map_sum, matchingMixtureSample]

/-- Every finite bounded family with probability weights reconstructs a feasible learned state.
Source: exact atomic reconstruction following Appendix A.4, with all Q/K/value identities free. -/
theorem matchingMixtureOfFamily_mem {ι : Type*} [Fintype ι] {V H D : ℕ} (cap : ℝ)
    (heads : ι → MatchingHead V H D) (w : ι → ℝ)
    (hh : ∀ i, heads i ∈ matchingHeadBox V H D cap) (hw : ∀ i, 0 ≤ w i) (hs : ∑ i, w i = 1) :
    matchingMixtureOfFamily heads w ∈ matchingMixtureDomain V H D cap := by
  classical
  refine ⟨matchingMixtureOfFamily_nonneg heads w hw, ?_, ?_⟩
  · intro h hn
    have hm := matchingMixtureOfFamily_support heads w (Finsupp.mem_support_iff.mpr hn)
    obtain ⟨i, hi, he⟩ := Finset.mem_image.mp hm
    rw [← he]
    exact hh i
  · rw [matchingMixtureOfFamily_mass, hs]

/-- Two genuine heads with different learned matching inhabit every finite-reconstruction premise. -/
example : matchingMixtureOfFamily (fun i : Fin 2 => matchingScalarHead i.val)
    (fun _ => (1 / 2 : ℝ)) ∈ matchingMixtureDomain 2 1 1 1 := by
  apply matchingMixtureOfFamily_mem
  · intro i
    fin_cases i <;> apply matchingScalarHead_mem <;> norm_num
  · intro i
    norm_num
  · norm_num [Fin.sum_univ_two]

/-- The complete nonlinear response family of every eligible physical matching head.
Source: the new genuine-sparsemax atomic variant of Appendix A.4. -/
def matchingSampleAtoms {V D R T : ℕ} (H : ℕ) (cap : ℝ) (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) : Set (Fin R → ℝ) :=
  matchingHeadSample tokens rows channels '' matchingHeadBox V H D cap

/-- Actual finite-mixture predictions, defined through the physical learned states.
Source: the candidate matching architecture following Appendix A.4, not a stipulated convex hull. -/
def matchingPredictionSet {V D R T : ℕ} (H : ℕ) (cap : ℝ) (tokens : Fin R → Fin T → Fin V)
    (rows : Fin R → Fin T) (channels : Fin R → Fin D) : Set (Fin R → ℝ) :=
  matchingMixtureSample tokens rows channels '' matchingMixtureDomain V H D cap

/-- The exact physical-mixture class is the convex hull of every eligible learned Q/K/value head.
Source: Appendix A.4's convex-mixture idea, extended to genuine shared causal sparsemax heads. -/
theorem matchingPredictionSet_eq_hull {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    matchingPredictionSet H cap tokens rows channels = convexHull ℝ (matchingSampleAtoms H cap tokens rows channels) := by
  classical
  ext y
  constructor
  · rintro ⟨μ, hμ, rfl⟩
    refine mem_convexHull_of_exists_fintype (fun h : μ.support => μ h)
      (fun h : μ.support => matchingHeadSample tokens rows channels h.val) (fun h => hμ.1 h) ?_ ?_ ?_
    · have hs := hμ.2.2
      rw [matchingMixtureMass_eq_sum] at hs
      calc
        (∑ h : μ.support, μ h) = ∑ h ∈ μ.support, μ h := Finset.sum_attach _ _
        _ = 1 := hs
    · intro h
      exact ⟨h, hμ.2.1 h (Finsupp.mem_support_iff.mp h.2), rfl⟩
    · rw [matchingMixtureSample, Finsupp.linearCombination_apply]
      change (∑ h : μ.support, μ h • matchingHeadSample tokens rows channels h.val) =
        ∑ h ∈ μ.support, μ h • matchingHeadSample tokens rows channels h
      exact Finset.sum_attach μ.support
        (fun h : MatchingHead V H D => μ h • matchingHeadSample tokens rows channels h)
  · intro hy
    obtain ⟨ι, hι, w, z, hw, hs, hz, hy⟩ := mem_convexHull_iff_exists_fintype.mp hy
    let heads (i : ι) : MatchingHead V H D := Classical.choose (hz i)
    have hh (i : ι) : heads i ∈ matchingHeadBox V H D cap := (Classical.choose_spec (hz i)).1
    have he (i : ι) : matchingHeadSample tokens rows channels (heads i) = z i := (Classical.choose_spec (hz i)).2
    refine ⟨matchingMixtureOfFamily heads w, matchingMixtureOfFamily_mem cap heads w hh hw hs, ?_⟩
    rw [matchingMixtureOfFamily_sample]
    simp_rw [he]
    exact hy

/-- Genuine prediction geometry is convex while the raw Q/K/value identities remain selectable.
Source: the proved exact physical-mixture reconstruction following Appendix A.4. -/
theorem matchingPredictionSet_convex {V H D R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (channels : Fin R → Fin D) :
    Convex ℝ (matchingPredictionSet H cap tokens rows channels) :=
  (matchingMixtureDomain_convex V H D cap).linear_image (matchingMixtureSample tokens rows channels)

end Transformer.GPTMini.Sparsemax
