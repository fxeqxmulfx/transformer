import Transformer.DoubleDescent.Section2_Risk
import Transformer.DoubleDescent.Section5_Curves
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Basic.ENNReal.BigOperators

/-!
# Translation likelihood and token loss

arXiv:1912.02292v1, Appendix B.3. The source's likelihood product uses
indices 1 through |y|, whereas its sampling description uses 0 through |y|.
We index actual target tokens by `Fin y.length`; an EOS token, if scored,
must be present in the target list. Thus no undefined endpoint is used.
The source explicitly notes that including every token of each sampled
sentence is not iid sampling from the token distribution of Definition 1.
-/

namespace Transformer.DoubleDescent

open scoped ENNReal

/-- Appendix B.3: next-token probabilities conditioned on the source and
the preceding target tokens. Probability normalization is a property of
such a model, not a hidden assumption of the product identity below. -/
abbrev TranslationModel (Source Target : Type*) := List Source → List Target → Target → ℝ

/-- Appendix B.3: the observed token probability at a valid zero-based index. -/
def tokenProbability {Source Target : Type*} (model : TranslationModel Source Target)
    (source : List Source) (target : List Target) (i : Fin target.length) : ℝ :=
  model source (target.take i) target[i]

/-- Appendix B.3: the autoregressive likelihood is the product of the
probabilities of all observed target tokens, each with its actual prefix. -/
noncomputable def translationLikelihood {Source Target : Type*}
    (model : TranslationModel Source Target) (source : List Source) (target : List Target) : ℝ :=
  ∏ i, tokenProbability model source target i

/-- Appendix B.3: nonnegative token NLL, with infinite loss at probability
zero. This corrects the undefined real-log expression at zero; Lean's
totalized `Real.log 0 = 0` must not turn an impossible token into zero loss. -/
noncomputable def tokenNLL (probability : ℝ) : ℝ≥0∞ :=
  if 0 < probability then ENNReal.ofReal (-Real.log probability) else ⊤

/-- Appendix B.3: an impossible token has infinite NLL. -/
theorem tokenNLL_zero : tokenNLL 0 = ⊤ := by simp [tokenNLL]

/-- Appendix B.3: a certain token has zero NLL. -/
theorem tokenNLL_one : tokenNLL 1 = 0 := by simp [tokenNLL]

/-- Appendix B.3: the sum of token NLLs is minus the log sentence
likelihood, for strictly positive observed token probabilities at most one. -/
theorem tokenNLL_sum_eq_sentence {Source Target : Type*}
    (model : TranslationModel Source Target) (source : List Source) (target : List Target)
    (hp : ∀ i, 0 < tokenProbability model source target i ∧
      tokenProbability model source target i ≤ 1) :
    ∑ i, tokenNLL (tokenProbability model source target i) =
      ENNReal.ofReal (-Real.log (translationLikelihood model source target)) := by
  have he (i : Fin target.length) : tokenNLL (tokenProbability model source target i) =
      ENNReal.ofReal (-Real.log (tokenProbability model source target i)) := by
    simp [tokenNLL, (hp i).1]
  have hn (i : Fin target.length) : 0 ≤ -Real.log (tokenProbability model source target i) :=
    neg_nonneg.mpr (Real.log_nonpos (hp i).1.le (hp i).2)
  simp_rw [he]
  rw [← ENNReal.ofReal_sum_of_nonneg (fun i _ => hn i)]
  unfold translationLikelihood
  rw [Real.log_prod (fun i _ => (hp i).1.ne'), Finset.sum_neg_distrib]

/-- Appendix B.3: a normalized two-token model with probability one half
for each label satisfies both strict positivity and the probability bound. -/
example : ∀ i : Fin ([false, true] : List Bool).length,
    0 < tokenProbability
      (fun (_ : List Bool) (_ : List Bool) (_ : Bool) => (1 / 2 : ℝ)) [] [false, true] i ∧
      tokenProbability
        (fun (_ : List Bool) (_ : List Bool) (_ : Bool) => (1 / 2 : ℝ)) [] [false, true] i ≤ 1 := by
  intro i
  norm_num [tokenProbability]

/-- Appendix B.3 and Section 7: passing from mean log loss to perplexity
preserves the finite double-descent inequalities because exponentiation is
strictly increasing. It does not identify the ambiguous units of a CSV column. -/
theorem doubleDescent_exp {I : Type*} [Preorder I] (error : I → ℝ) :
    HasDoubleDescent (fun i => Real.exp (error i)) ↔ HasDoubleDescent error := by
  unfold HasDoubleDescent
  simp only [Real.exp_lt_exp]

end Transformer.DoubleDescent
