/-
# Perceptrons and attention's mean-field landscape — polynomial transforms

Formalization of `lem: quadpol` (i), "in particular", of arXiv:2601.21366v2:
`f^μ` is a polynomial of degree `≤ k` exactly when `μ` has a density with
respect to `σ_d` that is a polynomial of degree `≤ k`.  The transform itself
is `Transform`.

**What the source says and what is carried here.**

* "`f^μ` is a polynomial of degree `k`" is carried as "`f^μ` is the
  restriction to the sphere of a polynomial of total degree `≤ k`".  Degree is
  not well defined for a *function* on the sphere: `Σ_i x_i² = 1` there, so
  every representing polynomial can be raised by two, and only the bound
  survives.  The same reading is used for the density, and the source's `iff`
  is then an equality of the two least such bounds.

* `σ_d` is `Metastability.IsUniformOn`, quantified over inside the
  conclusion, as in `thm: any.d`: the tree never constructs the uniform
  measure, and a hypothesis it cannot witness would be one nobody could
  discharge.

* The source's standing `d ≥ 2` is not assumed, as in `Transform`.  On
  `𝕊^0 = {±1}` every function is `a + b x`, so for `k ≥ 1` both sides hold for
  every `μ`, and for `k = 0` both say `μ({1}) = 1/2`; `𝒫(𝕊^{-1})` is empty.

* **What is not carried.**  The `L²(σ_d)` criterion of `lem: quadpol` (i) —
  `μ` has an `L²` density iff `Σ_{j,ℓ} |f̂^μ_{jℓ}/λ_j(β)|² < ∞`, with the
  density given by `eq:degreek` — is not stated.  Writing it down needs the
  spherical harmonics `Y_{jℓ}` and the Funk–Hecke eigenvalues
  `λ_j(β) = Γ(d/2)(2/β)^{d/2-1} I_{j+d/2-1}(β)`, which Mathlib does not have
  and which the source imports as standard; a definition supplying them would
  be the whole difficulty parked in a definition.  Its consequence — the
  polynomial characterization, which is what `rem:genericity-vacuity` uses —
  is `isPolyOfDegreeLE_attentionTransform_iff` below.

Source: arXiv:2601.21366v2, `lem: quadpol` (i).
-/

import Transformer.Perceptron.Transform
import Transformer.Metastability.InitialUniform
import Mathlib.Algebra.MvPolynomial.Degrees

open scoped ENNReal
open MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-- `f` is the restriction to `𝕊^{d-1}` of a polynomial of total degree `≤ k`.

This is the source's "polynomial of degree `k` on `𝕊^{d-1}`"; see the module
docstring on why only the bound is meaningful there. -/
def IsPolyOfDegreeLE (k : ℕ) (f : SSphere d → ℝ) : Prop :=
  ∃ P : MvPolynomial (Idx d) ℝ, P.totalDegree ≤ k ∧
    ∀ x : SSphere d, f x = MvPolynomial.eval (fun i => (x : EucSpace d) i) P

/-- A constant function is a polynomial of degree `0`. -/
theorem isPolyOfDegreeLE_const (c : ℝ) : IsPolyOfDegreeLE 0 (fun _ : SSphere d => c) :=
  ⟨MvPolynomial.C c, by simp [MvPolynomial.totalDegree_C], fun _ => by simp⟩

/-- **Lemma (lem: quadpol) (i), "in particular".**  `f^μ` is a polynomial of
degree `≤ k` if and only if `μ` has a density with respect to `σ_d` that is a
polynomial of degree `≤ k`.

Not proved here.

Source: arXiv:2601.21366v2, `lem: quadpol` (i). -/
theorem isPolyOfDegreeLE_attentionTransform_iff (β : ℝ) (hβ : 0 < β) (k : ℕ)
    (μ : Perspective.ProbSphere d) :
    ∀ ν : Measure (SSphere d), Metastability.IsUniformOn d ν →
      (IsPolyOfDegreeLE k (attentionTransform β μ) ↔
        ∃ g : SSphere d → ℝ, IsPolyOfDegreeLE k g ∧
          (μ : Measure (SSphere d))
            = ν.withDensity fun y => ENNReal.ofReal (g y)) := by
  sorry

/-- The hypothesis of `isPolyOfDegreeLE_attentionTransform_iff` is satisfiable:
`β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

end Perceptron
end Transformer
