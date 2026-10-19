module

public import Transformer.Clusters.Section4_Hyperplanes

/-!
# The emergence of clusters in self-attention dynamics — a mix of hyperplanes
  and polytopes

§5 of arXiv:2305.05465v6: the good triple with multiplicity `d:goodmulti`, and
`t:multiplicity`, clustering toward `(∂𝒦 ∪ {0}) × 𝒢`. The theorem itself
and its complete proof are in `Section10_Multiplicity`.

**What the source says and what is carried here.**

* `d:goodmulti` (ii) calls `V` *paranormal* when `ℝ^d` splits into two
  `V`-invariant subspaces `ℱ ⊕ 𝒢` with `V|_ℱ = λ Id`, `λ > 0`, and
  `ρ(V|_𝒢) < λ`. The spectral radius is taken through Gelfand's formula, as
  in `Section4_Hyperplanes` — see `IsSpectralRadiusLtOn`. `ℱ` and `𝒢` are
  parameters of the definition rather than existentials, because the
  conclusion of `t:multiplicity` names them.

* `t:multiplicity` sets `ℋ := (∂𝒦 ∪ {0}) × 𝒢` for a bounded convex polytope
  `𝒦 ⊂ ℱ`. The product is read through the splitting: `ℋ` is the set of
  `x ∈ ℝ^d` whose `ℱ`-part lies in `∂𝒦 ∪ {0}`, that is `x - p ∈ 𝒢` for some
  `p ∈ ∂𝒦 ∪ {0}`.

* `∂𝒦` is the frontier **in `ℱ`**, not in `ℝ^d`: `𝒦` is a subset of a proper
  subspace, so its frontier in `ℝ^d` would be all of `𝒦` and the conclusion
  would say strictly less. `𝒦` therefore lives in `↥ℱ`, where `frontier` is
  the relative one. "Bounded convex polytope" is the convex hull of a finite
  set, which is bounded and convex outright.

Source: arXiv:2305.05465v6, `d:goodmulti`, `t:multiplicity`.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **Definition (d:goodmulti) (ii), paranormality.**  `V` acts as `λ Id` on
`ℱ`, leaves `𝒢` invariant, and has spectral radius `< λ` there.

Source: arXiv:2305.05465v6, `d:goodmulti` (ii). -/
def IsParanormalOn (V : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ) : Prop :=
  (∀ w ∈ F, V w ∈ F) ∧ (∀ w ∈ G, V w ∈ G) ∧ IsCompl F G ∧ 0 < lam ∧
    (∀ w ∈ F, V w = lam • w) ∧ IsSpectralRadiusLtOn G V lam

/-- **Definition (d:goodmulti).**  A good triple with multiplicity: `QᵀK ≻ 0`
and `V` paranormal.

Source: arXiv:2305.05465v6, `d:goodmulti`. -/
def IsGoodTripleMulti (Q K V : ParamMatrix d) (F G : Submodule ℝ (EucSpace d))
    (lam : ℝ) : Prop :=
  IsPosDefQK Q K ∧ IsParanormalOn V F G lam

/-- `QᵀK ≻ 0` holds for `Q = K = I_d`. -/
theorem isPosDefQK_one (d : ℕ) : IsPosDefQK (1 : ParamMatrix d) 1 := by
  refine ⟨fun u v => ?_, fun u hu => ?_⟩
  · simpa using real_inner_comm v u
  · simpa using real_inner_self_pos.mpr hu

/-- **`(I_d, I_d, I_d)` is a good triple with multiplicity**, with `ℱ = ℝ^d`,
`𝒢 = 0` and `λ = 1`: the case `V = I_d` of §3, which `t:multiplicity`
specializes to `t:Idcase11int`. This witnesses `d:goodmulti`. -/
theorem isGoodTripleMulti_one (d : ℕ) :
    IsGoodTripleMulti (1 : ParamMatrix d) 1 1 ⊤ ⊥ 1 := by
  refine ⟨isPosDefQK_one d, fun w _ => Submodule.mem_top, fun w hw => by simpa using hw,
    isCompl_top_bot, one_pos, fun w _ => (one_smul ℝ w).symm,
    ⟨1, 0, one_pos, le_rfl, one_pos, ?_⟩⟩
  intro k w hw
  have hw0 : w = 0 := by simpa using hw
  rw [hw0]
  simp

/-- **`ℋ = (∂𝒦 ∪ {0}) × 𝒢`.**  The points of `ℝ^d` whose `ℱ`-component, taken
along `𝒢`, lies on the relative boundary of the polytope `𝒦` or at the origin.

Source: arXiv:2305.05465v6, `t:multiplicity`. -/
def polytopeTimesSubspace (F G : Submodule ℝ (EucSpace d)) (T : Finset F) :
    Set (EucSpace d) :=
  {x | ∃ p ∈ frontier (convexHull ℝ (T : Set F)) ∪ {0}, x - (p : EucSpace d) ∈ G}

/-- Powers of the value map on the leading space are exactly scalar
powers. Source: arXiv:2305.05465v6, §5, `d:goodmulti` (ii), the leading
restriction used by the exponential splitting in §10. -/
theorem IsParanormalOn.pow_apply_leading (V : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (k : ℕ) (w : EucSpace d) (hw : w ∈ F) :
    (V ^ k) w = lam ^ k • w := by
  induction k with
  | zero => simp only [pow_zero, one_apply_eq_self, one_smul]
  | succ k ih =>
    rw [pow_succ V k]
    change (V ^ k) (V w) = lam ^ (k + 1) • w
    rw [hV.2.2.2.2.1 w hw, map_smul, ih, smul_smul, pow_succ lam k]
    congr 1
    ring

/-- The full splitting and leading-vector membership coexist at every
identity-value vector. Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
example (d : ℕ) (w : EucSpace d) : IsParanormalOn (1 : ParamMatrix d) ⊤ ⊥ 1 ∧
    w ∈ (⊤ : Submodule ℝ (EucSpace d)) :=
  ⟨(isGoodTripleMulti_one d).2, Submodule.mem_top⟩

/-- Every positive scalar identity is paranormal, with the whole space
leading and a zero complement. Source: arXiv:2305.05465v6, §5,
`d:goodmulti` (ii); this is the identity-value example at arbitrary speed. -/
theorem isParanormalOn_smul_one (d : ℕ) (lam : ℝ) (hlam : 0 < lam) :
    IsParanormalOn (lam • (1 : ParamMatrix d)) ⊤ ⊥ lam := by
  refine ⟨fun _ _ => Submodule.mem_top, ?_, isCompl_top_bot, hlam, ?_,
    ⟨1, 0, one_pos, le_rfl, hlam, ?_⟩⟩
  · intro w hw
    have hw0 : w = 0 := by simpa only [Submodule.mem_bot] using hw
    rw [hw0, map_zero]
    exact Submodule.zero_mem _
  · intro w _
    simp only [smul_apply, one_apply_eq_self]
  · intro k w hw
    have hw0 : w = 0 := by simpa only [Submodule.mem_bot] using hw
    rw [hw0]
    simp only [map_zero, norm_zero, mul_zero, le_refl]

/-- The speed hypothesis has an actual positive example.
Source: arXiv:2305.05465v6, §5, `d:goodmulti` (ii). -/
example : (0 : ℝ) < 2 := by norm_num

/-- A positive attention form and positive scalar identity give a good
triple with multiplicity. Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
theorem isGoodTripleMulti_smul_one (Q K : ParamMatrix d) (hQK : IsPosDefQK Q K)
    (lam : ℝ) (hlam : 0 < lam) :
    IsGoodTripleMulti Q K (lam • (1 : ParamMatrix d)) ⊤ ⊥ lam :=
  ⟨hQK, isParanormalOn_smul_one d lam hlam⟩

/-- The full attention-positivity and speed inputs coexist.
Source: arXiv:2305.05465v6, §5, `d:goodmulti`. -/
example : IsPosDefQK (1 : ParamMatrix d) 1 ∧ (0 : ℝ) < 2 :=
  ⟨isPosDefQK_one d, by norm_num⟩

/-- Coincident tokens remain stationary in the rescaled variables at
any scalar identity speed. Source: arXiv:2305.05465v6, §3, `e:Rres`,
and §5, the positive-scalar identity example of `d:goodmulti`. -/
theorem rescaledDynamics_smul_one_const (Q K : ParamMatrix d) (lam : ℝ) (z : EucSpace d) :
    RescaledDynamics (n := n) Q K (lam • (1 : ParamMatrix d)) (fun _ _ => z) := by
  intro t i
  simpa only [sub_self, map_zero, smul_zero, Finset.sum_const_zero] using hasDerivAt_const t z

end Clusters
end Transformer
