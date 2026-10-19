module

public import Transformer.Clusters.Section12_ReluDynamics
public import Transformer.Clusters.Section4_SquareCounterexample

/-!
# The universal ReLU three-hyperplane extension is false

Section 12 of arXiv:2305.05465v6 proposes extending Theorem
`l:3hyperplanes11` to the feed-forward equation `eq: mlp.transfo`.
The top row of Figure `fig: last.pdf` uses ReLU and identity output
weights. Its caption describes particles eventually following
three hyperplanes. The universal extension formalized as
`mlpRelu_dist_tendsto_zero` is false, even for the identity good
triple in dimension one.

The preceding modules construct an actual global solution at
`Q = K = V = W = I₁`, with four initial coordinates
`(-8, -1, 0, 1)`. Three coordinates are frozen by ReLU. The
minimum coordinate moves monotonically but has integrable speed,
so its limit is `q ∈ [-8, -7]`. Thus the four distinct limiting
points are `(q, -1, 0, 1)`. A codimension-one affine hyperplane
in `ℝ¹` is a single point; covering these limits requires four.

This refutes the universal three-hyperplane extension, without
changing its dynamics, good-triple hypotheses, or the number
three. It does not refute the source's particular numerical
observations or assert that no broader clustering statement is
possible. The negative theorem below is exactly the negation of
the former universal assertion.
-/

@[expose] public section

open Real Filter Topology
namespace Transformer.Clusters

/-- A codimension-one subspace of the line is the zero subspace.
Source: arXiv:2305.05465v6, §12, the proposed extension of
`l:3hyperplanes11` in the dimension-one counterexample. -/
theorem codim_one_line_eq_bot (U : Submodule ℝ (EucSpace 1))
    (hU : Module.finrank ℝ U + 1 = 1) : U = ⊥ := by
  apply Submodule.finrank_eq_zero.mp
  omega

example : Module.finrank ℝ (⊥ : Submodule ℝ (EucSpace 1)) + 1 = 1 := by simp

/-- Any affine hyperplane approached by a convergent scalar token
must be its actual limiting point. Source: arXiv:2305.05465v6,
§12, the distance conclusion proposed in Figure `fig: last.pdf`. -/
theorem reluFourVectors_eq_of_infDist_tendsto_zero
    (Z : ℝ → Idx 4 → EucSpace 1) (q : ℝ) (i : Idx 4)
    (U : Submodule ℝ (EucSpace 1)) (p : EucSpace 1)
    (hU : Module.finrank ℝ U + 1 = 1)
    (hlim : Tendsto (fun t => Z t i) atTop (nhds (reluFourVectors q i)))
    (hp : Tendsto (fun t => Metric.infDist (Z t i) (affineShift U p)) atTop (nhds 0)) :
    reluFourVectors q i = p := by
  rw [codim_one_line_eq_bot U hU] at hp
  exact tendsto_nhds_unique hlim ((tendsto_infDist_affineShift_bot_iff _ p).mp hp)

example (i : Idx 4) :
    Module.finrank ℝ (⊥ : Submodule ℝ (EucSpace 1)) + 1 = 1 ∧
      Tendsto (fun _ : ℝ => reluFourVectors (-8) i) atTop
        (nhds (reluFourVectors (-8) i)) ∧
      Tendsto (fun _ : ℝ => Metric.infDist (reluFourVectors (-8) i)
        (affineShift (⊥ : Submodule ℝ (EucSpace 1)) (reluFourVectors (-8) i)))
          atTop (nhds 0) := by
  refine ⟨by simp, tendsto_const_nhds, ?_⟩
  exact (tendsto_infDist_affineShift_bot_iff _ _).mpr tendsto_const_nhds

/-- A family of affine limiting hyperplanes for the four distinct
ReLU limits has at least four members. Source: arXiv:2305.05465v6,
§12, the obstruction to its proposed three-hyperplane extension. -/
theorem four_le_card_of_relu_limits (Z : ℝ → Idx 4 → EucSpace 1) (q : ℝ)
    (hq : q ≤ -7)
    (hlim : ∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (reluFourVectors q i)))
    (U : Submodule ℝ (EucSpace 1)) (S : Finset (EucSpace 1))
    (hU : Module.finrank ℝ U + 1 = 1)
    (hS : ∀ i : Idx 4, ∃ p ∈ S,
      Tendsto (fun t => Metric.infDist (Z t i) (affineShift U p)) atTop (nhds 0)) :
    4 ≤ S.card := by
  classical
  have hsub : Finset.univ.image (reluFourVectors q) ⊆ S := by
    intro x hx
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hx
    obtain ⟨p, hpS, hp⟩ := hS i
    rw [reluFourVectors_eq_of_infDist_tendsto_zero Z q i U p hU (hlim i) hp]
    exact hpS
  simpa only [card_reluFourVectors_image hq] using Finset.card_le_card hsub

/-- The hypotheses of the cardinality obstruction have a concrete witness. -/
example : ∃ (Z : ℝ → Idx 4 → EucSpace 1) (q : ℝ)
    (U : Submodule ℝ (EucSpace 1)) (S : Finset (EucSpace 1)),
    q ≤ -7 ∧ (∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (reluFourVectors q i))) ∧
      Module.finrank ℝ U + 1 = 1 ∧ ∀ i : Idx 4, ∃ p ∈ S,
        Tendsto (fun t => Metric.infDist (Z t i) (affineShift U p)) atTop (nhds 0) := by
  classical
  refine ⟨fun _ => reluFourVectors (-8), -8, ⊥,
    Finset.univ.image (reluFourVectors (-8)), by norm_num,
    fun i => tendsto_const_nhds, by simp, fun i => ?_⟩
  refine ⟨reluFourVectors (-8) i, Finset.mem_image_of_mem _ (Finset.mem_univ i), ?_⟩
  exact (tendsto_infDist_affineShift_bot_iff _ _).mpr tendsto_const_nhds

/-- Four distinct scalar limits cannot approach only three affine hyperplanes.
Source: arXiv:2305.05465v6, §12, the distance conclusion of its
proposed extension of `l:3hyperplanes11`. -/
theorem not_three_hyperplanes_relu_limits (Z : ℝ → Idx 4 → EucSpace 1) (q : ℝ)
    (hq : q ≤ -7)
    (hlim : ∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (reluFourVectors q i))) :
    ¬ ∃ (U : Submodule ℝ (EucSpace 1)) (S : Finset (EucSpace 1)),
      Module.finrank ℝ U + 1 = 1 ∧ S.card ≤ 3 ∧
        ∀ i : Idx 4, ∃ p ∈ S,
          Tendsto (fun t => Metric.infDist (Z t i) (affineShift U p)) atTop (nhds 0) := by
  rintro ⟨U, S, hU, hcard, hS⟩
  have hfour := four_le_card_of_relu_limits Z q hq hlim U S hU hS
  omega

example : (-8 : ℝ) ≤ -7 ∧ ∀ i : Idx 4,
    Tendsto (fun _ : ℝ => reluFourVectors (-8) i) atTop (nhds (reluFourVectors (-8) i)) :=
  ⟨by norm_num, fun i => tendsto_const_nhds⟩

/-- An actual global solution refutes the three-hyperplane ReLU extension.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo`, Figure
`fig: last.pdf`, the universal claim formerly named
`mlpRelu_dist_tendsto_zero`. The initial tokens are explicitly distinct. -/
theorem mlpRelu_dist_tendsto_zero_counterexample :
    ∃ Z : ℝ → Idx 4 → EucSpace 1,
      IsGoodTriple (1 : ParamMatrix 1) 1 1 ∧
        MlpRescaledDynamics 1 relu (1 : ParamMatrix 1) 1 1 Z ∧
          Z 0 = reluFourVectors (-8) ∧
            ¬ ∃ (U : Submodule ℝ (EucSpace 1)) (S : Finset (EucSpace 1)),
              Module.finrank ℝ U + 1 = 1 ∧ S.card ≤ 3 ∧
                ∀ i : Idx 4, ∃ p ∈ S,
                  Tendsto (fun t => Metric.infDist (Z t i) (affineShift U p))
                    atTop (nhds 0) := by
  obtain ⟨Z, q, hZ, h0, hql, hqu, hlim⟩ := exists_mlp_relu_four_limits
  exact ⟨Z, isGoodTriple_one, hZ, h0, not_three_hyperplanes_relu_limits Z q hqu hlim⟩

/-- The exact universal three-hyperplane ReLU extension is false.
Source: arXiv:2305.05465v6, §12, `eq: mlp.transfo` and the
proposed extension of `l:3hyperplanes11` in Figure `fig: last.pdf`.
The counterexample uses the original hypotheses and conclusion. -/
theorem not_mlpRelu_dist_tendsto_zero :
    ¬ (∀ (d n : ℕ) (Q K V : ParamMatrix d), IsGoodTriple Q K V →
      ∀ Z : ℝ → Idx n → EucSpace d, MlpRescaledDynamics 1 relu Q K V Z →
        ∃ (U : Submodule ℝ (EucSpace d)) (S : Finset (EucSpace d)),
          Module.finrank ℝ U + 1 = d ∧ S.card ≤ 3 ∧
            ∀ i : Idx n, ∃ p ∈ S,
              Tendsto (fun t => Metric.infDist (Z t i) (affineShift U p)) atTop (nhds 0)) := by
  intro h
  obtain ⟨Z, hgood, hZ, h0, hnot⟩ := mlpRelu_dist_tendsto_zero_counterexample
  exact hnot (h 1 4 1 1 1 hgood Z hZ)

end Transformer.Clusters
