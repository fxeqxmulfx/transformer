module

public import Transformer.Perceptron.Section5_PolynomialSpace
public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Mathlib.Data.Finsupp.Multiset

/-!
# Powers of linear functions span the original spherical polynomial space

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
The proved original kernel moment formulas preserve powers of linear
functions. Here those actual powers span every original polynomial with
the same total-degree bound, including mixed coordinate monomials.

The span lies in the original finite-dimensional polynomial space, so
it is closed in the uniform norm. Differentiating actual power curves
adds one mixed linear factor. Induction supplies arbitrary products;
the original monomial support then supplies every polynomial.
Neither the kernel's inverse nor its degree preservation is assumed.
-/

@[expose] public section

open Real Set Filter
open scoped Topology

namespace Transformer.Perceptron

/-- The real span of actual powers of actual linear functions of degree
at most `k`. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
Its equality with the original polynomial space is proved below. -/
noncomputable def spherePowerSpan (d k : ℕ) : Submodule ℝ C(SSphere d, ℝ) :=
  Submodule.span ℝ {f | ∃ (b : EucSpace d) (n : ℕ), n ≤ k ∧ f = sphereLinearFunction b ^ n}

/-- Each original power is in its actual span at every larger degree
bound. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`. -/
theorem power_mem_spherePowerSpan {d n k : ℕ} (b : EucSpace d) (hn : n ≤ k) :
    sphereLinearFunction b ^ n ∈ spherePowerSpan d k :=
  Submodule.subset_span ⟨b, n, hn, rfl⟩

/-- The actual power span is contained in the actual original polynomial
space with the same degree bound. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i). Each generator has its proved original degree. -/
theorem spherePowerSpan_le_polynomialSpace (d k : ℕ) :
    spherePowerSpan d k ≤ spherePolynomialSpace d k := by
  apply Submodule.span_le.mpr
  rintro f ⟨b, n, hn, rfl⟩
  exact sphereLinearFunction_pow_mem b hn

/-- The actual power span is finite-dimensional, as a subspace of the
original bounded-degree polynomial space. Source: arXiv:2601.21366v2,
§5.1, the finite polynomial tests in `lem: quadpol` (i). -/
instance finiteDimensional_spherePowerSpan (d k : ℕ) :
    FiniteDimensional ℝ (spherePowerSpan d k) :=
  FiniteDimensional.of_injective (Submodule.inclusion (spherePowerSpan_le_polynomialSpace d k))
    (Submodule.inclusion_injective _)

/-- An actual derivative of a curve in the original power span remains
in that same span. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`,
mixed polynomial tests. Actual difference quotients give membership. -/
theorem deriv_mem_spherePowerSpan {d k : ℕ} (f : ℝ → C(SSphere d, ℝ))
    (f' : C(SSphere d, ℝ)) (hf : ∀ r, f r ∈ spherePowerSpan d k)
    (hd : HasDerivAt f f' 0) : f' ∈ spherePowerSpan d k := by
  exact (spherePowerSpan d k).closed_of_finiteDimensional.mem_of_tendsto
    hd.tendsto_slope_zero (Eventually.of_forall fun r =>
      (spherePowerSpan d k).smul_mem r⁻¹ ((spherePowerSpan d k).sub_mem (hf (0 + r)) (hf 0)))

/-- Every actual mixed product followed by a linear power is in the
original power span at its actual total degree. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, the finite polynomial tests of `lem: quadpol` (i).
The derivative at zero adds the original first factor without increasing
the original total degree; its nonzero integer coefficient is inverted. -/
theorem list_prod_mul_power_mem_spherePowerSpan {d k : ℕ}
    (l : List (EucSpace d)) (b : EucSpace d) (n : ℕ) (hn : l.length + n ≤ k) :
    (l.map sphereLinearFunction).prod * sphereLinearFunction b ^ n ∈ spherePowerSpan d k := by
  induction l generalizing b n with
  | nil => simpa using power_mem_spherePowerSpan b hn
  | cons v l ih =>
    let A := (l.map sphereLinearFunction).prod
    let B := sphereLinearFunction b
    let V := sphereLinearFunction v
    have hfun (r : ℝ) : sphereLinearFunction (b + r • v) = B + r • V := by
      apply ContinuousMap.ext
      intro x
      simp only [sphereLinearFunction, ContinuousMap.coe_mk, ContinuousMap.add_apply,
        ContinuousMap.smul_apply, inner_add_left, real_inner_smul_left, smul_eq_mul, B, V]
    have hcurve (r : ℝ) : A * (B + r • V) ^ (n + 1) ∈ spherePowerSpan d k := by
      rw [← hfun r]
      exact ih (b + r • v) (n + 1) (by simp only [List.length_cons] at hn; omega)
    have hd : HasDerivAt (fun r : ℝ => A * (B + r • V) ^ (n + 1))
        ((n + 1 : ℝ) • (A * B ^ n * V)) 0 := by
      have hbase : HasDerivAt (fun r : ℝ => B + r • V) V 0 := by
        simpa only [id_eq, one_smul] using ((hasDerivAt_id (0 : ℝ)).smul_const V).const_add B
      have hprod := (hbase.pow (n + 1)).const_mul A
      change HasDerivAt (fun r : ℝ => A * (B + r • V) ^ (n + 1)) _ 0 at hprod
      apply hprod.congr_deriv
      simp only [zero_smul, add_zero, Nat.add_sub_cancel]
      ext x
      simp only [ContinuousMap.smul_apply, ContinuousMap.mul_apply, ContinuousMap.pow_apply,
        ContinuousMap.natCast_apply, smul_eq_mul]
      push_cast
      ring
    have hm := deriv_mem_spherePowerSpan _ _ hcurve hd
    have hs := (spherePowerSpan d k).smul_mem ((n + 1 : ℝ)⁻¹) hm
    have hne : (n + 1 : ℝ) ≠ 0 := by positivity
    rw [smul_smul, inv_mul_cancel₀ hne, one_smul] at hs
    simpa only [List.map_cons, List.prod_cons, A, B, V, mul_assoc, mul_left_comm, mul_comm] using hs

/-- Every actual finite product of linear functions lies in the original
power span with its original number of factors. Source: arXiv:2601.21366v2,
§5.1, `lem: quadpol` (i), mixed polynomial tests. -/
theorem list_prod_mem_spherePowerSpan {d k : ℕ} (l : List (EucSpace d)) (hl : l.length ≤ k) :
    (l.map sphereLinearFunction).prod ∈ spherePowerSpan d k := by
  simpa only [pow_zero, mul_one] using list_prod_mul_power_mem_spherePowerSpan l 0 0
    (by simpa only [Nat.add_zero] using hl)

/-- An actual original coordinate monomial is the product of its original
coordinate factors, counted with their actual multiplicities.
Source: arXiv:2601.21366v2, §5.1, the polynomial tests of `lem: quadpol` (i). -/
theorem spherePolynomialEval_monomial_list {d : ℕ} (m : Idx d →₀ ℕ) :
    spherePolynomialEval d (MvPolynomial.monomial m 1) =
      ((m.toMultiset.toList.map (fun i => EuclideanSpace.single i (1 : ℝ))).map sphereLinearFunction).prod := by
  classical
  have hX (i : Idx d) : spherePolynomialEval d (MvPolynomial.X i) =
      sphereLinearFunction (EuclideanSpace.single i 1) := by
    apply ContinuousMap.ext
    intro x
    rw [spherePolynomialEval_apply, MvPolynomial.eval_X]
    simp only [sphereLinearFunction, ContinuousMap.coe_mk, EuclideanSpace.inner_single_left,
      RCLike.conj_to_real, one_mul]
  rw [List.map_map, Multiset.prod_map_toList, Finsupp.toMultiset_map, Finsupp.prod_toMultiset,
    Finsupp.prod_mapDomain_index (fun _ => pow_zero _) (fun _ _ _ => pow_add _ _ _),
    MvPolynomial.monic_monomial_eq, map_finsuppProd]
  simp only [map_pow, hX, Function.comp_def]

/-- Every actual original coordinate monomial belongs to the power span
at its actual total-degree bound. Source: arXiv:2601.21366v2, §5.1,
`lem: quadpol` (i). The original multiplicities give the list length. -/
theorem monomial_mem_spherePowerSpan {d k : ℕ} (m : Idx d →₀ ℕ)
    (hm : m.sum (fun _ e => e) ≤ k) :
    spherePolynomialEval d (MvPolynomial.monomial m 1) ∈ spherePowerSpan d k := by
  rw [spherePolynomialEval_monomial_list]
  apply list_prod_mem_spherePowerSpan
  rw [List.length_map, Multiset.length_toList, Finsupp.card_toMultiset]
  exact hm

/-- The powers of original linear functions span precisely all original
spherical polynomials with the same degree bound. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic` and `lem: quadpol` (i).
The original bounded-degree domain is spanned by its actual monomials;
their actual images give the reverse inclusion. -/
theorem spherePowerSpan_eq_polynomialSpace (d k : ℕ) :
    spherePowerSpan d k = spherePolynomialSpace d k := by
  apply le_antisymm (spherePowerSpan_le_polynomialSpace d k)
  rintro f ⟨P, rfl⟩
  have hspan : P.1 ∈ MvPolynomial.restrictSupport ℝ {m : Idx d →₀ ℕ | m.sum (fun _ e => e) ≤ k} := P.2
  rw [MvPolynomial.restrictSupport_eq_span] at hspan
  have hle : Submodule.span ℝ ((fun m : Idx d →₀ ℕ => MvPolynomial.monomial m (1 : ℝ)) ''
      {m | m.sum (fun _ e => e) ≤ k}) ≤
        (spherePowerSpan d k).comap (spherePolynomialEval d).toLinearMap := by
    apply Submodule.span_le.mpr
    rintro Q ⟨m, hm, rfl⟩
    exact monomial_mem_spherePowerSpan m hm
  exact hle hspan

/-- The actual mixed-product, monomial-degree and derivative hypotheses
hold jointly on the original circle, with a nonzero coordinate and a
constant curve. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i). -/
example : ([(basePoint 1 : EucSpace 2)].length + 1 ≤ 2) ∧
    (Finsupp.single (0 : Idx 2) 2).sum (fun _ e => e) ≤ 2 ∧
    (∀ r : ℝ, (fun _ : ℝ => sphereLinearFunction (basePoint 1 : EucSpace 2)) r ∈ spherePowerSpan 2 2) ∧
    HasDerivAt (fun _ : ℝ => sphereLinearFunction (basePoint 1 : EucSpace 2)) 0 0 := by
  refine ⟨by norm_num, by simp, fun _ => ?_, hasDerivAt_const 0 _⟩
  simpa only [pow_one] using power_mem_spherePowerSpan (basePoint 1 : EucSpace 2) (by norm_num : 1 ≤ 2)

end Transformer.Perceptron
