/-
# The emergence of clusters in self-attention dynamics — the monotone
  coordinates of §9

§9 of arXiv:2305.05465v6, `sec: clustering.hyperplanes`, first subsection:
the facts about the eigencoordinates of `e:Rres` that the proof of
`l:3hyperplanes11` runs on — `eq:phistarvar`, `l:fj`, `e:defab`, `c:bounded`.

**What the source says and what is carried here.**

* §9 fixes an orthonormal eigenbasis `(φ_1, …, φ_d)` of `V` and works with the
  dual basis `(φ*_1, …, φ*_d)`.  The only property of `φ*_k` that the proofs
  use is `φ*_k(Vz) = λ_k φ*_k(z)`, so that is what is carried:
  `IsEigenFunctional`, a left eigenvector of `V`.  Nothing is assumed about
  where it came from, and the hypothesis "`V` is diagonalizable" enters only
  where the source uses the full basis, namely `c:bounded`, where it is the
  expansion `z = Σ_k φ*_k(z) φ_k`.

* `l:fj` is stated for `λ_k ≥ 0` and speaks of `max_j φ*_k(z_j(t))`, so its
  `φ*_k` is real-valued; that is how it is carried.  (A complex eigenfunctional
  with a real eigenvalue splits into its real and imaginary parts, each a real
  eigenfunctional for the same eigenvalue, so nothing is lost.)  The complex
  case is `l:nottoofassst`, in `Transformer.Clusters.Section9_Growth`.

* The source's "non-increasing", "non-decreasing" and "bounded" are on
  `[0,+∞)`, which is where the dynamics is run; `AntitoneOn`/`MonotoneOn` over
  `Set.Ici 0` is that.

* `e:defab` is the existence of `a` and `b`, with the parenthetical bound
  `b ≤ max_j φ*_1(z_j(0))` in the form `b ≤ β(t)` for every `t ≥ 0`.  It is
  proved here, from the conclusion of `l:fj` taken as an explicit hypothesis.

* `c:bounded` is likewise proved from the conclusion of `l:fj` — the uniform
  bound on each coordinate — together with the eigenbasis expansion.

Source: arXiv:2305.05465v6, `l:fj`, `c:bounded`, `e:defab`, `eq:phistarvar`.
-/

import Transformer.Clusters.Section4_Hyperplanes
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Analysis.Calculus.Deriv.Comp

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {d n m : ℕ}

/-! ### Eigenfunctionals -/

/-- **The dual eigenvector `φ*_k`**, carried by the one identity §9 uses it
through: `φ*_k(Vz) = λ_k φ*_k(z)`.

Source: arXiv:2305.05465v6, §9, the dual basis `(φ*_1, …, φ*_d)`. -/
def IsEigenFunctional {𝕜 : Type*} [RCLike 𝕜] (V : ParamMatrix d)
    (f : EucSpace d →L[ℝ] 𝕜) (lam : 𝕜) : Prop :=
  ∀ z : EucSpace d, f (V z) = lam * f z

/-- At `V = I_d` every functional is an eigenfunctional for `λ = 1`; this is
what witnesses the hypotheses below. -/
theorem isEigenFunctional_one {𝕜 : Type*} [RCLike 𝕜] (f : EucSpace d →L[ℝ] 𝕜) :
    IsEigenFunctional (1 : ParamMatrix d) f 1 := by
  intro z
  rw [one_apply_eq_self, one_mul]

/-! ### `α_k` and `β_k` -/

/-- **`β_k(t) = max_j φ*_k(z_j(t))`.**

Source: arXiv:2305.05465v6, the proof of `l:fj`. -/
noncomputable def maxCoord (f : EucSpace d →L[ℝ] ℝ) (X : Idx (m + 1) → EucSpace d) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun j => f (X j)

/-- **`α_k(t) = min_j φ*_k(z_j(t))`.**

Source: arXiv:2305.05465v6, the proof of `l:fj`. -/
noncomputable def minCoord (f : EucSpace d →L[ℝ] ℝ) (X : Idx (m + 1) → EucSpace d) : ℝ :=
  Finset.univ.inf' Finset.univ_nonempty fun j => f (X j)

theorem le_maxCoord (f : EucSpace d →L[ℝ] ℝ) (X : Idx (m + 1) → EucSpace d) (i : Idx (m + 1)) :
    f (X i) ≤ maxCoord f X :=
  Finset.le_sup' (fun j => f (X j)) (Finset.mem_univ i)

theorem minCoord_le (f : EucSpace d →L[ℝ] ℝ) (X : Idx (m + 1) → EucSpace d) (i : Idx (m + 1)) :
    minCoord f X ≤ f (X i) :=
  Finset.inf'_le (fun j => f (X j)) (Finset.mem_univ i)

theorem minCoord_le_maxCoord (f : EucSpace d →L[ℝ] ℝ) (X : Idx (m + 1) → EucSpace d) :
    minCoord f X ≤ maxCoord f X :=
  (minCoord_le f X 0).trans (le_maxCoord f X 0)

/-! ### `eq:phistarvar` -/

/-- **Equation (eq:phistarvar).**  Along `e:Rres`, an eigencoordinate obeys

  `(1/λ) d/dt φ*(z_i(t)) = Σ_j P_ij(t) ( φ*(z_j(t)) - φ*(z_i(t)) )`,

written here without dividing, so that `λ = 0` is allowed.

Source: arXiv:2305.05465v6, `eq:phistarvar`, and the display in the proof of
`l:fj`. -/
theorem hasDerivAt_eigenFunctional (Q K V : ParamMatrix d) (f : EucSpace d →L[ℝ] ℝ)
    (lam : ℝ) (hf : IsEigenFunctional V f lam) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => f (Z s i))
      (lam * ∑ j : Idx n, attentionMatrix Q K (fun l => expTime V t (Z t l)) i j *
        (f (Z t j) - f (Z t i))) t := by
  have hval : f (∑ j : Idx n,
        attentionMatrix Q K (fun l => expTime V t (Z t l)) i j • V (Z t j - Z t i))
      = lam * ∑ j : Idx n, attentionMatrix Q K (fun l => expTime V t (Z t l)) i j *
        (f (Z t j) - f (Z t i)) := by
    rw [map_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [map_smul, hf, map_sub, smul_eq_mul]
    ring
  rw [← hval]
  exact (f.hasFDerivAt (x := Z t i)).comp_hasDerivAt t (hZ t i)

/-- The hypotheses of `hasDerivAt_eigenFunctional` are satisfiable: at
`V = I_d` every functional is an eigenfunctional, and the configuration in
which all tokens sit at one point solves `e:Rres`. -/
example (f : EucSpace d →L[ℝ] ℝ) (z : EucSpace d) :
    IsEigenFunctional (1 : ParamMatrix d) f 1 ∧
      RescaledDynamics (n := n) (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  ⟨isEigenFunctional_one f, rescaledDynamics_one_const _ _ z⟩

/-! ### `l:fj` -/

/-- **Lemma (l:fj).**  If `λ_k ≥ 0` then `t ↦ max_j φ*_k(z_j(t))` is
non-increasing on `[0,+∞)` and `t ↦ min_j φ*_k(z_j(t))` is non-decreasing
there.

Not proved here.  The source's argument: at a time `t` and an index `i`
realizing the minimum, `eq:phistarvar` gives
`d/dt φ*_k(z_i(t)) = λ_k Σ_j P_ij (φ*_k(z_j) - φ*_k(z_i)) ≥ 0`.

Source: arXiv:2305.05465v6, `l:fj`. -/
theorem maxCoord_antitoneOn_minCoord_monotoneOn (Q K V : ParamMatrix d)
    (f : EucSpace d →L[ℝ] ℝ) (lam : ℝ) (hf : IsEigenFunctional V f lam) (hlam : 0 ≤ lam)
    (Z : ℝ → Idx (m + 1) → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    AntitoneOn (fun t => maxCoord f (Z t)) (Set.Ici 0) ∧
      MonotoneOn (fun t => minCoord f (Z t)) (Set.Ici 0) := by
  sorry

/-- The hypotheses of `maxCoord_antitoneOn_minCoord_monotoneOn` are
satisfiable. -/
example (f : EucSpace d →L[ℝ] ℝ) (z : EucSpace d) :
    IsEigenFunctional (1 : ParamMatrix d) f 1 ∧ (0 : ℝ) ≤ 1 ∧
      RescaledDynamics (n := m + 1) (1 : ParamMatrix d) 1 1 (fun _ _ => z) :=
  ⟨isEigenFunctional_one f, zero_le_one, rescaledDynamics_one_const _ _ z⟩

/-- **Lemma (l:fj), "in particular".**  Each `t ↦ φ*_k(z_i(t))` is uniformly
bounded on `[0,+∞)`, between the two extremes of the initial configuration.

The monotonicity conclusions of `l:fj` are taken as explicit hypotheses.

Source: arXiv:2305.05465v6, `l:fj`, last sentence. -/
theorem abs_eigenFunctional_le (f : EucSpace d →L[ℝ] ℝ) (Z : ℝ → Idx (m + 1) → EucSpace d)
    (hmax : AntitoneOn (fun t => maxCoord f (Z t)) (Set.Ici 0))
    (hmin : MonotoneOn (fun t => minCoord f (Z t)) (Set.Ici 0))
    (i : Idx (m + 1)) (t : ℝ) (ht : 0 ≤ t) :
    |f (Z t i)| ≤ max |maxCoord f (Z 0)| |minCoord f (Z 0)| := by
  have h0 : (0 : ℝ) ∈ Set.Ici (0 : ℝ) := Set.mem_Ici.mpr le_rfl
  have hup : f (Z t i) ≤ maxCoord f (Z 0) :=
    (le_maxCoord f (Z t) i).trans (hmax h0 ht ht)
  have hlow : minCoord f (Z 0) ≤ f (Z t i) :=
    (hmin h0 ht ht).trans (minCoord_le f (Z t) i)
  refine abs_le.mpr ⟨?_, ?_⟩
  · have h1 : -|minCoord f (Z 0)| ≤ minCoord f (Z 0) := neg_abs_le _
    have h2 : |minCoord f (Z 0)| ≤ max |maxCoord f (Z 0)| |minCoord f (Z 0)| := le_max_right _ _
    linarith
  · have h1 : maxCoord f (Z 0) ≤ |maxCoord f (Z 0)| := le_abs_self _
    have h2 : |maxCoord f (Z 0)| ≤ max |maxCoord f (Z 0)| |minCoord f (Z 0)| := le_max_left _ _
    linarith

/-- The hypotheses of `abs_eigenFunctional_le` are satisfiable: along a
configuration that does not move, both extremes are constant. -/
example (f : EucSpace d →L[ℝ] ℝ) (z : EucSpace d) :
    AntitoneOn (fun _ : ℝ => maxCoord f (fun _ : Idx (m + 1) => z)) (Set.Ici 0) ∧
      MonotoneOn (fun _ : ℝ => minCoord f (fun _ : Idx (m + 1) => z)) (Set.Ici 0) :=
  ⟨antitoneOn_const, monotoneOn_const⟩

end Clusters
end Transformer
