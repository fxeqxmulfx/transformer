/-
# Function representatives of monic relations between analytic germs

Only finitely many lower coefficient germs occur in a monic polynomial.
Their analytic representatives reconstruct the actual relation as a
function germ, without substituting formal series for convergent functions.
-/

import Transformer.RealAnalyticGerms.PreparedIntegralRelation

open Filter
open scoped BigOperators Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open AnalyticPreparation

/-- A monic polynomial relation between analytic germs has an actual
function representative with analytic lower parameter coefficients.
Auxiliary for finite projection in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem monic_analytic_germ_relation_representative {n : ℕ}
    (p : Polynomial (AnalyticGerm n)) (hp : p.Monic)
    (G : Base (n + 1) → ℝ) (hG : AnalyticAt ℝ G 0) :
    ∃ b : Fin p.natDegree → Base n → ℝ,
      (∀ i, AnalyticAt ℝ (b i) 0) ∧
      (Polynomial.aeval (AnalyticGerm.ofFunction G hG) p : FunctionGerm (n + 1)) =
        ((fun x : Base (n + 1) => G x ^ p.natDegree +
          ∑ i : Fin p.natDegree, b i (baseProjectionCLM n x) * G x ^ (i : ℕ)) :
          FunctionGerm (n + 1)) := by
  classical
  choose b hb hrep using fun i : Fin p.natDegree =>
    AnalyticGerm.exists_rep (p.coeff (i : ℕ))
  refine ⟨b, hb, ?_⟩
  have hsum : p = Polynomial.X ^ p.natDegree +
      ∑ i : Fin p.natDegree, Polynomial.C (p.coeff (i : ℕ)) * Polynomial.X ^ (i : ℕ) := by
    rw [Fin.sum_univ_eq_sum_range
      (fun i => Polynomial.C (p.coeff i) * Polynomial.X ^ i) p.natDegree]
    exact hp.as_sum
  have heval := congrArg (fun q : Polynomial (AnalyticGerm n) =>
    Polynomial.aeval (AnalyticGerm.ofFunction G hG) q) hsum
  rw [heval]
  simp only [map_add, map_pow, map_sum, map_mul, Polynomial.aeval_X, Polynomial.aeval_C]
  simp only [Subring.coe_add, Subring.coe_pow, Subring.coe_mul,
    AddSubmonoidClass.coe_finsetSum, algebraMap_analyticGermSucc_apply,
    lowerDimensionalInclusion, analyticGermPullbackHom_coe]
  simp_rw [← hrep, functionGermPullbackHom_coe]
  change (G : FunctionGerm (n + 1)) ^ p.natDegree +
      ∑ i : Fin p.natDegree,
        ((b i ∘ baseProjectionCLM n) : FunctionGerm (n + 1)) *
          (G : FunctionGerm (n + 1)) ^ (i : ℕ) = _
  let terms : Fin p.natDegree → Base (n + 1) → ℝ :=
    fun i => (b i ∘ baseProjectionCLM n) * G ^ (i : ℕ)
  change ((G ^ p.natDegree : Base (n + 1) → ℝ) : FunctionGerm (n + 1)) +
    ∑ i : Fin p.natDegree, (terms i : FunctionGerm (n + 1)) = _
  have hfinite : (((∑ i : Fin p.natDegree, terms i) : Base (n + 1) → ℝ) :
      FunctionGerm (n + 1)) =
      ∑ i : Fin p.natDegree, (terms i : FunctionGerm (n + 1)) :=
    map_sum (Filter.Germ.coeRingHom (𝓝 (0 : Base (n + 1)))) terms Finset.univ
  rw [← hfinite, ← Filter.Germ.coe_add]
  apply congrArg Filter.Germ.ofFun
  funext x
  simp only [terms, Pi.add_apply, Finset.sum_apply, Pi.mul_apply, Pi.pow_apply, Function.comp_apply]

end Transformer.RealAnalyticGerms
