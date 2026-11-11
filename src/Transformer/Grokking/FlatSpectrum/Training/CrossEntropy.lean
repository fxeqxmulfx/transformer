module

public import Transformer.Grokking.FlatSpectrum.Training.FactorSpace
public import Transformer.Grokking.FlatSpectrum.Training.Logits

/-!
# The cross-entropy of logits read off the product

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE1.
A sample `s` reads the product `X = U Vᴴ` through probes: the logit of the class `c` is
`z_{s,c}(X) = ⟨A s c, X⟩`, real-linear in `X`. The loss is the cross-entropy of the labels
`y s` (`ceLoss`),

    L(X) = Σ_s (log Σ_c exp z_{s,c}(X) − z_{s,y_s}(X)),

and with the softmax `p_s` of the logits of `s` its gradient field is (`ceGrad`)

    G(X) = Σ_s (Σ_c p_{s,c}(X) A s c − A s (y s)).

The probes `φ(c) e_ij` of a seen cell `(i, j)` read `Re (conj φ(c) X_ij)`: the kernel of the
note with an answer code `φ`. The probes `e_ic` read a product with one column per class.

`L` satisfies the first-order inequality with `G` (`ceLoss_first_order`), `G` is continuous
(`continuous_ceGrad`), and `L` is nonnegative (`ceLoss_nonneg`) and real-analytic in the
entries (`analyticAt_ceLoss`). So items WD1–WD5, WD9 and WD10 apply to it as they stand. At
`X = 0` the loss is `|S| log |C|` (`ceLoss_zero`). The witness of the examples is `signProbe`:
one sample of a `1 × 1` product read with the code `±1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix
open Landscape.Neural (crossEntropy sum_exp_pos crossEntropy_nonneg analyticAt_crossEntropy)
open Geometry (probability)

variable {m n S C : Type*} [Fintype m] [Fintype n] [Fintype S] [Fintype C]

/-- `⟨Σ_i A_i, B⟩ = Σ_i ⟨A_i, B⟩`. -/
lemma pairing_sum_left {ι : Type*} (s : Finset ι) (f : ι → Matrix m n ℂ) (B : Matrix m n ℂ) :
    pairing (∑ i ∈ s, f i) B = ∑ i ∈ s, pairing (f i) B := by
  simp only [pairing, conjTranspose_sum, Matrix.sum_mul, trace_sum, Complex.re_sum]

/-- The cross-entropy of the labels `y` when the logit of the class `c` at the sample `s` is
`⟨A s c, X⟩`: `Σ_s (log Σ_c exp ⟨A s c, X⟩ − ⟨A s (y s), X⟩)`, item XE1. -/
noncomputable def ceLoss (A : S → C → Matrix m n ℂ) (y : S → C) (X : Matrix m n ℂ) : ℝ :=
  ∑ s, crossEntropy (fun c => pairing (A s c) X) (y s)

/-- The gradient field `Σ_s (Σ_c p_{s,c} A s c − A s (y s))` of `ceLoss`, with `p_s` the
softmax of the logits of the sample `s`, item XE1. -/
noncomputable def ceGrad (A : S → C → Matrix m n ℂ) (y : S → C) (X : Matrix m n ℂ) :
    Matrix m n ℂ :=
  ∑ s, (∑ c, probability (fun c => pairing (A s c) X) c • A s c - A s (y s))

variable (A : S → C → Matrix m n ℂ) (y : S → C)

/-- `⟨G X, H⟩ = Σ_s (Σ_c p_{s,c} ⟨A s c, H⟩ − ⟨A s (y s), H⟩)`: the softmax mean of the logit
increments minus the increment of the label. -/
lemma pairing_ceGrad (X H : Matrix m n ℂ) :
    pairing (ceGrad A y X) H = ∑ s, (∑ c, probability (fun c => pairing (A s c) X) c *
      pairing (A s c) H - pairing (A s (y s)) H) := by
  rw [ceGrad, pairing_sum_left]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [pairing_sub_left, pairing_sum_left]
  simp only [pairing_smul_left]

/-- **The cross-entropy satisfies the first-order inequality** with the gradient `ceGrad`
(`review/flat_spectrum.md`, item XE1): it is convex in the product. -/
theorem ceLoss_first_order (X Y : Matrix m n ℂ) :
    ceLoss A y X + pairing (ceGrad A y X) (Y - X) ≤ ceLoss A y Y := by
  rw [pairing_ceGrad, ceLoss, ceLoss, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun s _ => ?_
  have h := crossEntropy_first_order (fun c => pairing (A s c) X) (fun c => pairing (A s c) Y)
    (y s)
  simpa only [pairing_sub_right] using h

/-- A softmax probability of the logits is continuous in the product. -/
lemma continuous_probability_pairing (B : C → Matrix m n ℂ) (c : C) :
    Continuous fun X : Matrix m n ℂ => probability (fun c => pairing (B c) X) c :=
  (Real.continuous_exp.comp (continuous_pairing continuous_const continuous_id)).div
    (continuous_finsetSum _ fun _ _ =>
      Real.continuous_exp.comp (continuous_pairing continuous_const continuous_id))
    fun _ => (sum_exp_pos _ c).ne'

/-- The gradient of the cross-entropy is continuous (`review/flat_spectrum.md`, item XE1). -/
theorem continuous_ceGrad : Continuous (ceGrad A y) :=
  continuous_finsetSum _ fun s _ => (continuous_finsetSum _ fun c _ =>
    (continuous_probability_pairing (A s) c).smul continuous_const).sub continuous_const

lemma ceLoss_nonneg (X : Matrix m n ℂ) : 0 ≤ ceLoss A y X :=
  Finset.sum_nonneg fun _ _ => crossEntropy_nonneg _ _

/-- At `X = 0` every logit vanishes and the loss is `|S| log |C|`. -/
lemma ceLoss_zero : ceLoss A y 0 = Fintype.card S * Real.log (Fintype.card C) := by
  simp only [ceLoss, pairing_zero_right, crossEntropy_zero, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul]

/-- The pairing with a fixed matrix is real-analytic in the entries: it is real-linear. -/
lemma analyticAt_pairing (B : Matrix m n ℂ) (y₀ : EuclideanSpace ℂ (m × n)) :
    AnalyticAt ℝ (fun x : EuclideanSpace ℂ (m × n) => pairing B (of fun i j => x (i, j)))
      y₀ := by
  have e : (fun x : EuclideanSpace ℂ (m × n) => pairing B (of fun i j => x (i, j))) =
      fun x => ∑ i, ∑ j, (star (B i j) * x (i, j)).re := by
    funext x
    rw [pairing, trace_conjTranspose_mul, Complex.re_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Complex.re_sum]
    rfl
  rw [e]
  exact Finset.analyticAt_fun_sum _ fun i _ => Finset.analyticAt_fun_sum _ fun j _ =>
    (Complex.reCLM.analyticAt _).comp (analyticAt_const.fun_mul
      (((EuclideanSpace.proj (i, j)).restrictScalars ℝ).analyticAt y₀))

/-- **The cross-entropy is real-analytic in the entries** of the product
(`review/flat_spectrum.md`, item XE1): its logits are real-linear and the class sum is
positive. -/
theorem analyticAt_ceLoss (y₀ : EuclideanSpace ℂ (m × n)) :
    AnalyticAt ℝ (fun x : EuclideanSpace ℂ (m × n) => ceLoss A y (of fun i j => x (i, j)))
      y₀ :=
  Finset.analyticAt_fun_sum _ fun s _ =>
    analyticAt_crossEntropy (fun c => analyticAt_pairing (A s c) y₀) (y s)

/-- The witness of the examples: one sample that reads a `1 × 1` product `X` with the code
`±1`, so that its two logits are `Re X` and `−Re X`. -/
def signProbe (_ : Fin 1) (c : Bool) : Matrix (Fin 1) (Fin 1) ℂ := if c then 1 else -1

lemma frobeniusSq_signProbe_le (s : Fin 1) (c : Bool) :
    frobeniusSq (signProbe s c) ≤ (1 : ℝ) ^ 2 := by
  cases c <;> simp [signProbe, frobeniusSq]

example := ceLoss_first_order signProbe (fun _ => true) 0 1

example := continuous_ceGrad signProbe fun _ => true

example := analyticAt_ceLoss signProbe (fun _ => true) 0

example : ceLoss signProbe (fun _ => true) 0 = 1 * Real.log 2 := by
  simpa using ceLoss_zero signProbe fun _ => true

end Transformer.Grokking.FlatSpectrum.Training
