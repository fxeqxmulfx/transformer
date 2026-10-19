module

public import Transformer.Clusters.Section5_Mix
public import Transformer.Clusters.Section8_Polytope

/-!
# The emergence of clusters in self-attention dynamics — the projected hull

§10 of arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`: the
first half of the proof of `t:multiplicity`, where the convex hull of §8 is
replaced by its projection onto `ℱ`.

**What the source says and what is carried here.**

* `π_ℱ` is "the projection onto `ℱ` parallel to `𝒢`". That is
  `IsProjOnto`: a linear map landing in `ℱ`, the identity on `ℱ`, zero on
  `𝒢`. It is carried as data rather than constructed from `IsCompl ℱ 𝒢`
  because the statements below quantify over it.

* "The set `π_ℱ(Cx({z_i(t)}))` is a convex subset of `ℱ` which is
  non-increasing with respect to `t` (the proof of this fact is identical to
  that of Proposition `p:noninc`)."  Both halves are proved here: convexity
  because a linear image of a convex set is convex, and monotonicity because
  images preserve inclusion — the monotonicity of the hull along the
  dynamics of §10 is taken as an explicit hypothesis.

* "It therefore converges toward some convex polytope `𝒦`" is proved in
  `Section10_Multiplicity`, together with the corrected candidate geometry.
  The source says to replace `A·` by `π_ℱ(A·)`; its original attention scores
  instead require `A π_ℱ`. The corrected candidates are
  `leadingLimitCandidates`, and the boundary conclusion retains `{0}` as
  in §8, `cl:propofS`. `projLimitCandidates` records the source's original
  expression for comparison with its faulty auxiliary estimates.

* `𝒦 ⊂ ℱ` and `∂𝒦` is its frontier **in `ℱ`**, so `𝒦` lives in `↥ℱ`, as in
  `polytopeTimesSubspace`; `polytopeIn` is the same polytope seen in `ℝ^d`,
  which is where the attention operator acts.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`,
`p:noninc`, `cl:propofS`.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### The projection onto `ℱ` parallel to `𝒢` -/

/-- **`π_ℱ`, the projection onto `ℱ` parallel to `𝒢`.**  It lands in `ℱ`,
fixes `ℱ` and kills `𝒢`.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`. -/
def IsProjOnto (proj : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) : Prop :=
  (∀ x : EucSpace d, proj x ∈ F) ∧ (∀ x ∈ F, proj x = x) ∧ (∀ x ∈ G, proj x = 0)

/-- At `ℱ = ℝ^d`, `𝒢 = 0` the projection is the identity: the case `V = I_d`
of §8, which `t:multiplicity` specializes to. -/
theorem isProjOnto_one (d : ℕ) : IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ := by
  refine ⟨fun x => Submodule.mem_top, fun x _ => one_apply_eq_self x, fun x hx => ?_⟩
  rw [one_apply_eq_self]
  simpa using hx

/-! ### The projected hull -/

/-- **`π_ℱ(Cx({z_i}))` is convex.**  A linear image of a convex set.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`. -/
theorem convex_image_tokenHull (proj : ParamMatrix d) (Z : Idx n → EucSpace d) :
    Convex ℝ (proj '' tokenHull Z) :=
  (convex_convexHull ℝ _).linear_image (proj : EucSpace d →ₗ[ℝ] EucSpace d)

/-- **`π_ℱ(Cx({z_i}))` lies in `ℱ`.** -/
theorem image_tokenHull_subset (proj : ParamMatrix d) {F G : Submodule ℝ (EucSpace d)}
    (hproj : IsProjOnto proj F G) (Z : Idx n → EucSpace d) :
    proj '' tokenHull Z ⊆ (F : Set (EucSpace d)) := by
  rintro _ ⟨x, -, rfl⟩
  exact hproj.1 x

/-- **`t ↦ π_ℱ(Cx({z_i(t)}))` is non-increasing.**  The conclusion of
`p:noninc` is taken as an explicit hypothesis; images preserve inclusion.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`, "the
proof of this fact is identical to that of Proposition `p:noninc`". -/
theorem image_tokenHull_antitone (proj : ParamMatrix d) (Z : ℝ → Idx n → EucSpace d)
    (hmono : ∀ s t : ℝ, s ≤ t → tokenHull (Z t) ⊆ tokenHull (Z s))
    (s t : ℝ) (hst : s ≤ t) :
    proj '' tokenHull (Z t) ⊆ proj '' tokenHull (Z s) :=
  Set.image_mono (hmono s t hst)

/-- The hypothesis of `image_tokenHull_antitone` is satisfiable: a
configuration that does not move has a constant hull. -/
example (z : EucSpace d) :
    ∀ s t : ℝ, s ≤ t → tokenHull ((fun _ (_ : Idx n) => z) t) ⊆
      tokenHull ((fun _ (_ : Idx n) => z) s) :=
  fun _ _ _ => subset_rfl

/-! ### The limiting polytope `𝒦` and its set `𝒮` -/

/-- The polytope `𝒦 ⊂ ℱ`, seen in `ℝ^d`. `𝒦` itself lives in `↥ℱ`, where
`frontier` is the relative one that `t:multiplicity` names.

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`. -/
def polytopeIn (F : Submodule ℝ (EucSpace d)) (S : Finset F) : Set (EucSpace d) :=
  ((↑) : F → EucSpace d) '' convexHull ℝ (S : Set F)

/-- Every generator of the relative polytope belongs to its ambient
image. Source: arXiv:2305.05465v6, §10, the limiting polytope `𝒦 ⊂ ℱ`. -/
theorem mem_polytopeIn_generator (F : Submodule ℝ (EucSpace d)) (S : Finset F)
    (p : F) (hp : p ∈ S) : (p : EucSpace d) ∈ polytopeIn F S :=
  ⟨p, subset_convexHull ℝ (S : Set F) hp, rfl⟩

/-- The generator-membership hypothesis has an actual nonempty instance.
Source: arXiv:2305.05465v6, §10, the limiting polytope. -/
example (F : Submodule ℝ (EucSpace d)) (p : F) : p ∈ ({p} : Finset F) :=
  Finset.mem_singleton_self p

/-- Enlarging the generators enlarges the same relative polytope in the
ambient space. Source: arXiv:2305.05465v6, §10, the projected hulls. -/
theorem polytopeIn_mono (F : Submodule ℝ (EucSpace d)) (S T : Finset F)
    (hST : S ⊆ T) : polytopeIn F S ⊆ polytopeIn F T :=
  Set.image_mono (convexHull_mono (Finset.coe_subset.mpr hST))

/-- Generator inclusion holds for a genuine singleton polytope.
Source: arXiv:2305.05465v6, §10, the projected hulls. -/
example (F : Submodule ℝ (EucSpace d)) (p : F) :
    ({p} : Finset F) ⊆ ({p} : Finset F) := Finset.Subset.refl _

/-- Nonemptiness of the generators is exactly nonemptiness of the
polytope. Source: arXiv:2305.05465v6, §10, the limiting projected hull;
this records nonemptiness independently of real Hausdorff distance. -/
theorem polytopeIn_nonempty_iff (F : Submodule ℝ (EucSpace d)) (S : Finset F) :
    (polytopeIn F S).Nonempty ↔ S.Nonempty := by
  rw [polytopeIn, Set.image_nonempty, convexHull_nonempty_iff, Finset.coe_nonempty]

/-- A singleton relative polytope has the expected ambient singleton.
Source: arXiv:2305.05465v6, §10, the coincident-token example. -/
theorem polytopeIn_singleton (F : Submodule ℝ (EucSpace d)) (p : F) :
    polytopeIn F {p} = {(p : EucSpace d)} := by
  rw [polytopeIn, Finset.coe_singleton, convexHull_singleton, Set.image_singleton]

/-- **The set `𝒮` of §10.**  The points `w ∈ 𝒦` with

  `‖π_ℱ(Aw)‖² = max_{j∈[m]} ⟨π_ℱ(Aw), π_ℱ(Av_j)⟩`,

which is `IsAttentionMax` of §8 at the operator `π_ℱ ∘ A` — the source's
"simply by replacing all occurrences of `A·` by `π_ℱ(A·)`".

Source: arXiv:2305.05465v6, `sec: clustering.hyperplanes.and.polytopes`. -/
def projLimitCandidates (A proj : ParamMatrix d) (F : Submodule ℝ (EucSpace d))
    (S : Finset F) : Set F :=
  {w ∈ convexHull ℝ (S : Set F) |
    IsAttentionMax (proj.comp A) (polytopeIn F S) (w : EucSpace d)}

/-- The hypotheses of `exists_polytope_tendsto_image_tokenHull` are
satisfiable: the triple `(I_d, I_d, I_d)` with `ℱ = ℝ^d`, `𝒢 = 0`, the
identity projection, and the stationary configuration of
`rescaledDynamics_one_const`. The complete theorem is in
`Section10_Multiplicity`. Source: arXiv:2305.05465v6, §10. -/
example (d : ℕ) (z : EucSpace d) :
    IsGoodTripleMulti (1 : ParamMatrix d) 1 1 ⊤ ⊥ 1 ∧
      IsAttentionRoot (1 : ParamMatrix d) 1 1 ∧
      IsProjOnto (1 : ParamMatrix d) ⊤ ⊥ ∧
      RescaledDynamics (n := n + 1) (1 : ParamMatrix d) 1 1 (fun _ _ => z) ∧ 0 < n + 1 :=
  ⟨isGoodTripleMulti_one d, isAttentionRoot_id d, isProjOnto_one d,
    rescaledDynamics_one_const _ _ z, Nat.zero_lt_succ n⟩

end Clusters
end Transformer
