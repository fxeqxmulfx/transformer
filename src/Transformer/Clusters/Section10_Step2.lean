module

public import Transformer.Clusters.Section10_Remainder
public import Transformer.Clusters.Section10_SquareApproach
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# The emergence of clusters in self-attention dynamics — Step 2'

§10 of arXiv:2305.05465v6: Claim `cl:gamma'12` and Step 2', the lower bound on
the growth of `‖π_ℱ(Az_i(t))‖²` away from `𝒮_δ`, which replace Step 2
(`e:eq17`) of the proof of `t:Idcase11`.

**What the source says and what is carried here.**

* `𝒮_δ` "denotes the set of all points in `𝒦` at distance `≤ δ` to some point
  of `𝒮`"; that is `candidatesThickening`, with `𝒮` the `projLimitCandidates`
  of `Section10_ProjHull` seen in `ℝ^d`.

* Both statements are conditioned on `z_i(t) ∉ 𝒮_δ × 𝒢`, which through the
  splitting is `π_ℱ(z_i(t)) ∉ 𝒮_δ`.

* Both positive lower bounds are false with that definition of `𝒮_δ`.
  An actual identity-value square approaches its nonempty limiting hull
  from outside at every finite time. Each corner's competing scores and
  energy derivative are nonpositive. The hull is derived from the actual
  ODE and identified with the future-hull intersection.

* The first equality of `e:derivativenorm123`,
  `½ d/dt ‖π_ℱ(Az_i(t))‖² = ⟨π_ℱ(Aż_i(t)), π_ℱ(Az_i(t))⟩`, is proved; it is
  the chain rule on `e:Rres`. The false lower bounds are replaced below by
  counterexamples, without changing the source's restricted neighborhood.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`,
`cl:gamma'12`, Step 2', `e:derivativenorm123`.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The first equality of `e:derivativenorm123` -/

/-- **The first line of `e:derivativenorm123`.**

  `½ d/dt ‖π_ℱ(Az_i(t))‖² = ⟨π_ℱ(Aż_i(t)), π_ℱ(Az_i(t))⟩`,

with `ż_i(t)` the drift of `e:Rres`. This is the chain rule, and it is what
Step 2' bounds from below.

Source: arXiv:2305.05465v6, `e:derivativenorm123`. -/
theorem hasDerivAt_norm_sq_proj (Q K V A proj : ParamMatrix d)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => ‖proj (A (Z s i))‖ ^ 2)
      (2 * inner (𝕜 := ℝ) (proj (A (∑ j : Idx n,
          attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i))))
        (proj (A (Z t i)))) t := by
  have h1 : HasDerivAt (fun s => A (Z s i))
      (A (∑ j : Idx n,
        attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i))) t :=
    (A.hasFDerivAt (x := Z t i)).comp_hasDerivAt t (hZ t i)
  have hu : HasDerivAt (fun s => proj (A (Z s i)))
      (proj (A (∑ j : Idx n,
        attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i)))) t :=
    (proj.hasFDerivAt (x := A (Z t i))).comp_hasDerivAt t h1
  have h := hu.inner ℝ hu
  have hfun : (fun s => ‖proj (A (Z s i))‖ ^ 2)
      = fun s => inner (𝕜 := ℝ) (proj (A (Z s i))) (proj (A (Z s i))) := by
    funext s
    rw [real_inner_self_eq_norm_sq]
  rw [hfun]
  convert h using 1
  rw [real_inner_comm (proj (A (Z t i)))]
  ring

/-- The hypothesis of `hasDerivAt_norm_sq_proj` is satisfiable, by the
stationary configuration of `rescaledDynamics_one_const`. -/
example (z : EucSpace d) :
    RescaledDynamics (n := n) (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  rescaledDynamics_one_const _ _ z

/-! ### `𝒮_δ` -/

/-- **`𝒮_δ`**, the points of `𝒦` at distance `≤ δ` from `𝒮`.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`. -/
def candidatesThickening (A proj : ParamMatrix d) (F : Submodule ℝ (EucSpace d))
    (S : Finset F) (δ : ℝ) : Set (EucSpace d) :=
  {w ∈ polytopeIn F S |
    Metric.infDist w (((↑) : F → EucSpace d) '' projLimitCandidates A proj F S) ≤ δ}

/-- Every point of `𝒦` lies in `𝒮_δ` for `δ` large, so the family is not
empty of content; at `δ < 0` it is empty, which is why Step 2' quantifies over
`δ > 0`. -/
theorem candidatesThickening_subset (A proj : ParamMatrix d) (F : Submodule ℝ (EucSpace d))
    (S : Finset F) (δ : ℝ) : candidatesThickening A proj F S δ ⊆ polytopeIn F S :=
  fun _ hw => hw.1

/-! ### `cl:gamma'12` and Step 2' -/

/-- **Claim `cl:gamma'12` is false with the source's `Sδ ⊂ K`.**
The good triple, positive root and projection are all identities, with
`F = ℝ²`, `G = 0`. The actual square corner never belongs to the genuine
limit polytope, while every competing score is nonpositive.
Nonemptiness and exact future-hull equality are required as well as
Hausdorff convergence. Source: arXiv:2305.05465v6, §10, `cl:gamma'12`. -/
theorem not_exists_gamma_projScore :
    ¬ ∀ (Z : ℝ → Idx 4 → EucSpace 2) (S : Finset (⊤ : Submodule ℝ (EucSpace 2))),
      RescaledDynamics (1 : ParamMatrix 2) 1 1 Z → S.Nonempty →
      polytopeIn ⊤ S = {x | ∀ t : ℝ, 0 ≤ t → x ∈ (1 : ParamMatrix 2) '' tokenHull (Z t)} →
      Tendsto (fun t => Metric.hausdorffDist ((1 : ParamMatrix 2) '' tokenHull (Z t))
        (polytopeIn ⊤ S)) atTop (nhds 0) →
      ∃ γ δ₀ : ℝ, 0 < γ ∧ 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
        ∃ T : ℝ, 0 < T ∧ ∀ t : ℝ, T ≤ t → ∀ i : Idx 4,
          (1 : ParamMatrix 2) (Z t i) ∉ candidatesThickening 1 1 ⊤ S δ →
            ∃ j : Idx 4, γ * δ ≤ projScore 1 1 (Z t) i j := by
  intro h
  obtain ⟨Z, S, hZ, hS, hHull, hHaus, hout, hscore, _⟩ := exists_square_outside_limit_polytope
  obtain ⟨γ, δ₀, hγ, hδ, hbound⟩ := h Z S hZ hS hHull hHaus
  obtain ⟨T, hT, htime⟩ := hbound δ₀ hδ le_rfl
  have hnot : (1 : ParamMatrix 2) (Z T 3) ∉ candidatesThickening 1 1 ⊤ S δ₀ := by
    simp only [one_apply_eq_self]
    exact fun hm => hout T hT.le (candidatesThickening_subset 1 1 ⊤ S δ₀ hm)
  obtain ⟨j, hj⟩ := htime T le_rfl 3 hnot
  have hpos := mul_pos hγ hδ
  linarith [hscore T j]

/-- **Step 2' is false with the source's restricted neighborhood.**
The same actual identity-value square has nonpositive energy derivative
while its corner remains outside the genuine nonempty limit polytope.
Uniqueness of the derivative contradicts any positive bound `γ δ`.
Source: arXiv:2305.05465v6, §10, Step 2' and `e:derivativenorm123`. -/
theorem not_exists_gamma_deriv_norm_sq_proj :
    ¬ ∀ (Z : ℝ → Idx 4 → EucSpace 2) (S : Finset (⊤ : Submodule ℝ (EucSpace 2))),
      RescaledDynamics (1 : ParamMatrix 2) 1 1 Z → S.Nonempty →
      polytopeIn ⊤ S = {x | ∀ t : ℝ, 0 ≤ t → x ∈ (1 : ParamMatrix 2) '' tokenHull (Z t)} →
      Tendsto (fun t => Metric.hausdorffDist ((1 : ParamMatrix 2) '' tokenHull (Z t))
        (polytopeIn ⊤ S)) atTop (nhds 0) →
      ∃ γ δ₀ : ℝ, 0 < γ ∧ 0 < δ₀ ∧ ∀ δ : ℝ, 0 < δ → δ ≤ δ₀ →
        ∃ T : ℝ, 0 < T ∧ ∀ t : ℝ, T ≤ t → ∀ i : Idx 4,
          (1 : ParamMatrix 2) (Z t i) ∉ candidatesThickening 1 1 ⊤ S δ →
            ∃ D : ℝ, HasDerivAt (fun s => ‖(1 : ParamMatrix 2) ((1 : ParamMatrix 2)
              (Z s i))‖ ^ 2) D t ∧ γ * δ ≤ D := by
  intro h
  obtain ⟨Z, S, hZ, hS, hHull, hHaus, hout, _, hder⟩ := exists_square_outside_limit_polytope
  obtain ⟨γ, δ₀, hγ, hδ, hbound⟩ := h Z S hZ hS hHull hHaus
  obtain ⟨T, hT, htime⟩ := hbound δ₀ hδ le_rfl
  have hnot : (1 : ParamMatrix 2) (Z T 3) ∉ candidatesThickening 1 1 ⊤ S δ₀ := by
    simp only [one_apply_eq_self]
    exact fun hm => hout T hT.le (candidatesThickening_subset 1 1 ⊤ S δ₀ hm)
  obtain ⟨D, hD, hlow⟩ := htime T le_rfl 3 hnot
  obtain ⟨v, hv, hnonpos⟩ := hder T hT.le
  have hD' : HasDerivAt (fun s => ‖Z s 3‖ ^ 2) D T := by
    simpa only [one_apply_eq_self] using hD
  have heq := hD'.unique hv
  have hpos := mul_pos hγ hδ
  linarith

/-- Good-triple, root, projection, actual-ODE and hull inputs coexist for
stationary tokens, and for the square used by the counterexamples.
Source: arXiv:2305.05465v6, §10, the inputs of `cl:gamma'12` and Step 2'. -/
example (d : ℕ) (z : EucSpace d) (hz : z ∈ (⊤ : Submodule ℝ (EucSpace d))) :
    IsGoodTripleMulti (1 : ParamMatrix d) 1 1 ⊤ ⊥ 1 ∧
      IsAttentionRoot (1 : ParamMatrix d) 1 1 ∧
      IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
      RescaledDynamics (n := n + 1) (1 : ParamMatrix d) 1 1 (fun _ _ => z) ∧
      Tendsto (fun _ : ℝ => Metric.hausdorffDist
          ((1 : ParamMatrix d) '' tokenHull ((fun _ (_ : Idx (n + 1)) => z) 0))
          (polytopeIn ⊤ ({⟨z, hz⟩} : Finset (⊤ : Submodule ℝ (EucSpace d)))))
        atTop (nhds 0) := by
  refine ⟨isGoodTripleMulti_one d, isAttentionRoot_id d, isProjOnto_one d,
    rescaledDynamics_one_const _ _ z, ?_⟩
  have hset : (1 : ParamMatrix d) '' tokenHull ((fun _ (_ : Idx (n + 1)) => z) 0)
      = polytopeIn ⊤ ({⟨z, hz⟩} : Finset (⊤ : Submodule ℝ (EucSpace d))) := by
    have hrange : Set.range (fun _ : Idx (n + 1) => z) = {z} := by
      ext x
      simp [eq_comm]
    simp [tokenHull, polytopeIn, hrange, one_apply_eq_self, Set.image_id']
  rw [hset]
  simp

end Clusters
end Transformer
