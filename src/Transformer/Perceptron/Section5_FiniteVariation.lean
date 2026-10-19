module

public import Transformer.Perceptron.Kernel
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul

public import Mathlib.Analysis.Calculus.Deriv.Pow

/-!
# Differentiating the finite angular energy

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`, and §5.5,
`rem:strictSOPD-perceptron`. Finite support turns the actual energy
along a Monge geodesic into a finite sum with fixed probability
weights and fixed angular speeds. This file differentiates that
sum twice, keeping the pair-speed difference and both prefactors.

The angular potential is an explicit function `V`. Its first
derivative and its actual derivative's differentiability are inputs
to this calculus step, and will be supplied by `φ'=2σ` and `σ'`.
Continuity of the second derivative is not required.
The Hessian value is proved by differentiating the energy function;
it is not inserted into its definition.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Perceptron

variable {ι : Type*}

/-- **The finite-sum angular energy with fixed masses and speeds.**
Source: arXiv:2601.21366v2, §2.3, the coupled energy, and §5.2,
the energy differentiated in `eq:Hess-bilinear.std`. The preceding
geodesic identity identifies this expression with the actual energy. -/
noncomputable def finiteAngularEnergy (β : ℝ) (V : ℝ → ℝ)
    (F : Finset ι) (m θ s : ι → ℝ) (t : ℝ) : ℝ :=
  (2 * β)⁻¹ * (∑ i ∈ F, m i * ∑ j ∈ F, m j *
    kernelK β (θ i - θ j + t * (s i - s j))) +
      (1 / 2 : ℝ) * (∑ i ∈ F, m i * V (θ i + t * s i))

/-- **The first derivative after an affine angular shift.**
Source: arXiv:2601.21366v2, §5.2, the shifts used to compute
`eq:Hess-bilinear.std`. The slope can be zero or negative. -/
theorem hasDerivAt_affine_comp (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (u s t : ℝ) : HasDerivAt (fun r => f (u + r * s))
      (deriv f (u + t * s) * s) t := by
  have ha : HasDerivAt (fun r : ℝ => u + r * s) s t := by
    simpa only [id_eq, one_mul] using (((hasDerivAt_id t).mul_const s).const_add u)
  exact (hf (u + t * s)).hasDerivAt.comp t ha

/-- **The actual second derivative after an affine angular shift.**
Source: arXiv:2601.21366v2, §5.2, the squared speeds in
`eq:Hess-bilinear.std`. Differentiability of `deriv f` suffices. -/
theorem hasDerivAt_deriv_affine_comp (f : ℝ → ℝ)
    (hf : Differentiable ℝ f) (hf' : Differentiable ℝ (deriv f)) (u s t : ℝ) :
    HasDerivAt (deriv (fun r => f (u + r * s)))
      (deriv^[2] f (u + t * s) * s ^ 2) t := by
  change HasDerivAt _ (deriv (deriv f) (u + t * s) * s ^ 2) t
  have hfirst : deriv (fun r => f (u + r * s)) =
      fun r => deriv f (u + r * s) * s := by
    funext r
    exact (hasDerivAt_affine_comp f hf u s r).deriv
  rw [hfirst]
  have h := (hasDerivAt_affine_comp (deriv f) hf' u s t).mul_const s
  simpa only [pow_two, mul_assoc] using h

/-- The affine-composition hypotheses are jointly satisfied by the
quadratic function and its actual derivative at all real points. -/
example : Differentiable ℝ (fun r : ℝ => r ^ 2) ∧
    Differentiable ℝ (deriv (fun r : ℝ => r ^ 2)) := by
  constructor
  · exact fun r => (hasDerivAt_pow 2 r).differentiableAt
  · have h : deriv (fun r : ℝ => r ^ 2) = fun r => 2 * r := by
      funext r
      simp
    rw [h]
    exact fun r => ((hasDerivAt_id r).const_mul 2).differentiableAt

/-- **The actual first derivative of the finite angular energy.**
Source: arXiv:2601.21366v2, §5.2, the energy whose second
derivative is `eq:Hess-bilinear.std`. No mass normalization is
needed for this derivative identity. -/
theorem hasDerivAt_finiteAngularEnergy (β : ℝ) (V : ℝ → ℝ)
    (hV : Differentiable ℝ V) (F : Finset ι) (m θ s : ι → ℝ) (t : ℝ) :
    HasDerivAt (finiteAngularEnergy β V F m θ s)
      ((2 * β)⁻¹ * (∑ i ∈ F, m i * ∑ j ∈ F, m j *
        (deriv (kernelK β) (θ i - θ j + t * (s i - s j)) * (s i - s j))) +
        (1 / 2 : ℝ) * (∑ i ∈ F, m i * (deriv V (θ i + t * s i) * s i))) t := by
  have hK : Differentiable ℝ (kernelK β) :=
    fun u => (hasDerivAt_kernelK β u).differentiableAt
  have hI := (HasDerivAt.sum (u := F) fun i hi =>
    (HasDerivAt.sum (u := F) fun j hj =>
      (hasDerivAt_affine_comp (kernelK β) hK (θ i - θ j) (s i - s j) t).const_mul
        (m j)).const_mul (m i)).const_mul ((2 * β)⁻¹)
  have hP := (HasDerivAt.sum (u := F) fun i hi =>
    (hasDerivAt_affine_comp V hV (θ i) (s i) t).const_mul (m i)).const_mul (1 / 2 : ℝ)
  unfold finiteAngularEnergy
  simpa only [Finset.sum_apply, Pi.add_def] using hI.add hP

/-- **The actual first-derivative function is the differentiated sum.**
Source: arXiv:2601.21366v2, §5.2, the first step of
`eq:Hess-bilinear.std`. This equality is used before taking the
second derivative, so a formal Hessian value is not assumed. -/
theorem deriv_finiteAngularEnergy (β : ℝ) (V : ℝ → ℝ)
    (hV : Differentiable ℝ V) (F : Finset ι) (m θ s : ι → ℝ) :
    deriv (finiteAngularEnergy β V F m θ s) = fun t =>
      (2 * β)⁻¹ * (∑ i ∈ F, m i * ∑ j ∈ F, m j *
        (deriv (kernelK β) (θ i - θ j + t * (s i - s j)) * (s i - s j))) +
        (1 / 2 : ℝ) * (∑ i ∈ F, m i * (deriv V (θ i + t * s i) * s i)) := by
  funext t
  exact (hasDerivAt_finiteAngularEnergy β V hV F m θ s t).deriv

/-- **The second derivative at zero is the finite Hessian formula.**
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`.
The kernel term is its actual second derivative `kernelK2`, and
the potential term is the actual iterated derivative of `V`. -/
theorem hasDerivAt_deriv_finiteAngularEnergy (β : ℝ) (V : ℝ → ℝ)
    (hV : Differentiable ℝ V) (hV' : Differentiable ℝ (deriv V))
    (F : Finset ι) (m θ s : ι → ℝ) :
    HasDerivAt (deriv (finiteAngularEnergy β V F m θ s))
      ((2 * β)⁻¹ * (∑ i ∈ F, ∑ j ∈ F,
        m i * m j * kernelK2 β (θ i - θ j) * (s i - s j) ^ 2) +
          (1 / 2 : ℝ) * (∑ i ∈ F, m i * deriv^[2] V (θ i) * s i ^ 2)) 0 := by
  rw [deriv_finiteAngularEnergy β V hV]
  have hK (i j : ι) : HasDerivAt
      (fun t => deriv (kernelK β) (θ i - θ j + t * (s i - s j)) * (s i - s j))
      (kernelK2 β (θ i - θ j) * (s i - s j) * (s i - s j)) 0 := by
    have ha : HasDerivAt (fun t : ℝ => θ i - θ j + t * (s i - s j)) (s i - s j) 0 := by
      simpa only [id_eq, one_mul] using
        (((hasDerivAt_id (0 : ℝ)).mul_const (s i - s j)).const_add (θ i - θ j))
    have hd : HasDerivAt (deriv (kernelK β)) (kernelK2 β (θ i - θ j))
        (θ i - θ j + 0 * (s i - s j)) := by
      simpa only [zero_mul, add_zero] using hasDerivAt_deriv_kernelK β (θ i - θ j)
    exact (hd.comp 0 ha).mul_const (s i - s j)
  have hP (i : ι) : HasDerivAt (fun t => deriv V (θ i + t * s i) * s i)
      (deriv^[2] V (θ i) * s i * s i) 0 := by
    change HasDerivAt _ (deriv (deriv V) (θ i) * s i * s i) 0
    have h := (hasDerivAt_affine_comp (deriv V) hV' (θ i) (s i) 0).mul_const (s i)
    simpa only [zero_mul, add_zero] using h
  have hI := (HasDerivAt.sum (u := F) fun i hi =>
    (HasDerivAt.sum (u := F) fun j hj => (hK i j).const_mul (m j)).const_mul
      (m i)).const_mul ((2 * β)⁻¹)
  have hpot := (HasDerivAt.sum (u := F) fun i hi =>
    (hP i).const_mul (m i)).const_mul (1 / 2 : ℝ)
  simpa only [Finset.sum_apply, Finset.mul_sum, mul_assoc, pow_two, Pi.add_def] using hI.add hpot

/-- The finite-energy differentiability inputs are jointly realized
by the quadratic angular potential, including its actual second
derivative. Any finite masses, representatives and speeds are allowed. -/
example : Differentiable ℝ (fun θ : ℝ => θ ^ 2) ∧
    Differentiable ℝ (deriv (fun θ : ℝ => θ ^ 2)) := by
  have hd : deriv (fun θ : ℝ => θ ^ 2) = fun θ => 2 * θ := by
    funext θ
    simp
  refine ⟨fun θ => (hasDerivAt_pow 2 θ).differentiableAt, ?_⟩
  rw [hd]
  exact fun θ => ((hasDerivAt_id θ).const_mul 2).differentiableAt

end Transformer.Perceptron
