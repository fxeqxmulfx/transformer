/-
# The emergence of clusters in self-attention dynamics — clustering toward
  hyperplanes

§4 of arXiv:2305.05465v6: the good triple `d:good`, and `l:3hyperplanes11`,
the convergence of every token to one of at most three parallel hyperplanes.

**What the source says and what is carried here.**

* `d:good` asks that the eigenvalue of `V` of largest modulus be real,
  positive and simple, `λ₁ > |λ₂| ≥ … ≥ |λ_d|`.  That is carried by the
  splitting it is equivalent to and that §9 works in: a line `ℝφ₁` on which
  `V` is `λ₁`, a complementary `V`-invariant `G`, and `ρ(V|_G) < λ₁`.  For a
  real operator `ρ` is the modulus of a possibly complex eigenvalue, so it is
  taken through Gelfand's formula — `ρ(V|_G) < λ` exactly when the powers of
  `V` are bounded on `G` by `C r^k` for some `r < λ` — which needs no
  complexification and is what `e:majorizeweights` uses the hypothesis for.

* The second half of `d:good`, `⟨Qφ₁, Kφ₁⟩ > 0` "for any `φ₁` lying on the
  line `ker(V - λ₁ Id)`", is `0 < ⟨Qφ, Kφ⟩` for one generator: the form is
  bilinear, so `⟨Q(cφ), K(cφ)⟩ = c²⟨Qφ, Kφ⟩` has the same sign for every
  `c ≠ 0`.  Note that `QᵀK ≻ 0` is *not* assumed here — positivity is required
  only along `φ₁`.

* "There exist at most three parallel hyperplanes … such that the distance of
  `z_i(t)` to one of them converges to `0`" is a subspace `W` of codimension
  `1`, at most three translates `p + W`, and `Metric.infDist` to one of them
  tending to `0`.  `affineShift` is that translate.  The source's own
  hyperplanes are the level sets of `φ₁*`, which are exactly the translates of
  `ker φ₁*`.

The convergence theorem is proved in `Section9_Hyperplanes`; this file
supplies its good-triple predicates and affine level-set geometry.

Source: arXiv:2305.05465v6, `d:good`, `l:3hyperplanes11`.
-/

import Transformer.Clusters.Section3_IdCase
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Topology.MetricSpace.HausdorffDistance

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-! ### Parallel affine subspaces -/

/-- The translate `p + W` of a linear subspace.  Two translates of the same
`W` are parallel, and `W` of codimension `1` makes them hyperplanes.

Source: arXiv:2305.05465v6, `l:3hyperplanes11`. -/
def affineShift (W : Submodule ℝ (EucSpace d)) (p : EucSpace d) : Set (EucSpace d) :=
  {x | x - p ∈ W}

/-- A point lies on the translate through it. -/
theorem mem_affineShift_self (W : Submodule ℝ (EucSpace d)) (p : EucSpace d) :
    p ∈ affineShift W p := by
  simp [affineShift]

/-! ### Good triples -/

/-- **`ρ(V|_G) < λ`, through Gelfand's formula.**  The powers of `V` are
dominated on `G` by a geometric sequence of ratio `r < λ`.  For an operator on
a finite-dimensional space this holds exactly when every eigenvalue of `V|_G`,
real or complex, has modulus `< λ`.

Source: arXiv:2305.05465v6, `d:good`, `d:goodmulti` (ii). -/
def IsSpectralRadiusLtOn (G : Submodule ℝ (EucSpace d)) (V : ParamMatrix d) (lam : ℝ) : Prop :=
  ∃ C r : ℝ, 0 < C ∧ 0 ≤ r ∧ r < lam ∧
    ∀ (k : ℕ) (w : EucSpace d), w ∈ G → ‖(V ^ k) w‖ ≤ C * r ^ k * ‖w‖

/-- **Definition (d:good), with its witnesses named.**  `λ₁ = lam` is real
and positive, `φ₁ = φ` spans its eigenline, `G` is the complementary
`V`-invariant subspace on which `V` has spectral radius `< λ₁`, and
`⟨Qφ₁, Kφ₁⟩ > 0`.  §9 needs the leading eigenpair by name, which is what
this form provides.

Source: arXiv:2305.05465v6, `d:good`. -/
def IsGoodTripleWith (Q K V : ParamMatrix d) (lam : ℝ) (φ : EucSpace d)
    (G : Submodule ℝ (EucSpace d)) : Prop :=
  0 < lam ∧ φ ≠ 0 ∧ V φ = lam • φ ∧ (∀ w ∈ G, V w ∈ G) ∧
    IsCompl (Submodule.span ℝ {φ}) G ∧ IsSpectralRadiusLtOn G V lam ∧
    0 < inner (𝕜 := ℝ) (Q φ) (K φ)

/-- **Definition (d:good).**  `(Q,K,V)` is a good triple: the leading
eigenvalue `λ₁` of `V` is real, positive and simple, and `⟨Q·, K·⟩` is
positive along its eigenvector.

Source: arXiv:2305.05465v6, `d:good`. -/
def IsGoodTriple (Q K V : ParamMatrix d) : Prop :=
  ∃ (lam : ℝ) (φ : EucSpace d) (G : Submodule ℝ (EucSpace d)), IsGoodTripleWith Q K V lam φ G

/-- Being a good triple is having such a witness. -/
theorem isGoodTriple_iff (Q K V : ParamMatrix d) :
    IsGoodTriple Q K V ↔
      ∃ (lam : ℝ) (φ : EucSpace d) (G : Submodule ℝ (EucSpace d)),
        IsGoodTripleWith Q K V lam φ G := Iff.rfl

/-- In dimension `1` a single non-zero vector spans everything. -/
theorem span_singleton_eq_top {φ : EucSpace 1} (hφ : φ ≠ 0) :
    Submodule.span ℝ {φ} = ⊤ :=
  Submodule.eq_top_of_finrank_eq
    ((finrank_span_singleton hφ).trans finrank_euclideanSpace_fin.symm)

/-- **`(I₁, I₁, I₁)` is a good triple**, the scalar case `d = 1`, `V = 1`: the
leading eigenvalue is `1`, its eigenspace is all of `ℝ`, the complement is
trivial, and `⟨φ, φ⟩ > 0`.  This witnesses `d:good`. -/
theorem isGoodTripleWith_one :
    IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1 (EuclideanSpace.single 0 (1 : ℝ)) ⊥ := by
  have hφ : (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1) ≠ 0 :=
    by simp
  refine ⟨one_pos, hφ, (one_smul ℝ _).symm,
    fun w hw => by simpa using hw, ?_, ⟨1, 0, one_pos, le_rfl, one_pos, ?_⟩,
    real_inner_self_pos.mpr hφ⟩
  · rw [span_singleton_eq_top hφ]
    exact isCompl_top_bot
  · intro k w hw
    have hw0 : w = 0 := by simpa using hw
    rw [hw0]
    simp

/-- **`(I₁, I₁, I₁)` is a good triple.** -/
theorem isGoodTriple_one : IsGoodTriple (1 : ParamMatrix 1) 1 1 :=
  ⟨1, EuclideanSpace.single 0 (1 : ℝ), ⊥, isGoodTripleWith_one⟩

/-! ### Geometry of the limiting coordinate level sets -/

/-- The level set `{x : f x = c}` contains `c • v` whenever `f v = 1`.
Source: arXiv:2305.05465v6, §9, the hyperplanes `H_0`, `H_a`, and `H_b`
in the proof of `l:3hyperplanes11`. -/
theorem apply_smul_eq (f : EucSpace d →L[ℝ] ℝ) {v : EucSpace d} (hv : f v = 1) (c : ℝ) :
    f (c • v) = c := by
  rw [map_smul, hv, smul_eq_mul, mul_one]

example : (EuclideanSpace.proj (𝕜 := ℝ) 0)
    (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1) = 1 := by simp

/-- `ker f` has codimension `1` when the normalized coordinate takes
the value `1`. Source: arXiv:2305.05465v6, §4, `l:3hyperplanes11`,
and §9, the parallel hyperplanes given by the level sets of `φ₁*`. -/
theorem finrank_ker_add_one (f : EucSpace d →L[ℝ] ℝ) {v : EucSpace d} (hv : f v = 1) :
    Module.finrank ℝ (LinearMap.ker (f : EucSpace d →ₗ[ℝ] ℝ)) + 1 = d := by
  have hsurj : LinearMap.range (f : EucSpace d →ₗ[ℝ] ℝ) = ⊤ :=
    LinearMap.range_eq_top.mpr fun c => ⟨c • v, apply_smul_eq f hv c⟩
  have hrk := LinearMap.finrank_range_add_finrank_ker (f : EucSpace d →ₗ[ℝ] ℝ)
  rw [hsurj, finrank_top, Module.finrank_self, finrank_euclideanSpace_fin] at hrk
  omega

example : (EuclideanSpace.proj (𝕜 := ℝ) 0)
    (EuclideanSpace.single 0 (1 : ℝ) : EucSpace 1) = 1 := by simp

end Clusters
end Transformer
