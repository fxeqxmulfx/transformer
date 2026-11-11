module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation.Basic

/-!
# Input certificates transfer predictions to unseen dictionaries

Source: `review/flat_spectrum.md`, *Observable features determine unseen
predictions*, O2-O3. An explicit linear reconstruction of a test feature
vector from same-query training features is a checkable sufficient
identification certificate. Coefficients may be negative. No test label,
target computation, or learned parameter appears in this certificate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation

open scoped BigOperators

variable {Q K V C S : Type*} [Fintype K] [Fintype V] [Fintype S] [DecidableEq V]

/-- The input-only categorical feature vector used by the channel (O2). -/
def features (D : K → V) : K × V → ℝ :=
  fun kv => if D kv.1 = kv.2 then 1 else 0

/-- A finite certificate: coefficients are supported on the requested
query, and reconstruct every test binding indicator from training inputs (O2). -/
def Reconstructs (query : S → Q) (data : S → K → V) (q : Q) (D : K → V)
    (a : S → ℝ) : Prop :=
  (∀ s, query s ≠ q → a s = 0) ∧
  features D = ∑ s, a s • features (data s)

/-- The channel forward is the pairing of learned coefficients with
the actual fixed input features, before any classifier or loss (O2). -/
theorem logits_eq_dotProduct (W : Channel Q K V C) (q : Q) (D : K → V) (c : C) :
    logits W q D c = (fun kv : K × V => W q kv.1 kv.2 c) ⬝ᵥ features D := by
  simp [dotProduct, Fintype.sum_prod_type, features, mul_ite, logits, eq_comm]

/-- A test feature certificate reconstructs every logit of every model
in the family, not merely the training labels or a chosen optimum (O2). -/
theorem logits_reconstruct (query : S → Q) (data : S → K → V) (q : Q)
    (D : K → V) (a : S → ℝ) (hrec : Reconstructs query data q D a)
    (W : Channel Q K V C) (c : C) :
    logits W q D c = ∑ s, a s * logits W (query s) (data s) c := by
  classical
  rw [logits_eq_dotProduct, hrec.2, dotProduct_sum]
  apply Finset.sum_congr rfl
  intro s _
  rw [dotProduct_smul, ← logits_eq_dotProduct]
  by_cases hq : query s = q
  · simp [hq]
  · simp [hrec.1 s hq]

/-- Equality on observed logits determines an unseen prediction whenever
the observed input features reconstruct that test input (O2). -/
theorem logits_eq_of_reconstruct (query : S → Q) (data : S → K → V) (q : Q)
    (D : K → V) (a : S → ℝ) (hrec : Reconstructs query data q D a)
    {A W : Channel Q K V C}
    (hseen : ∀ s, logits W (query s) (data s) = logits A (query s) (data s)) :
    logits W q D = logits A q D := by
  funext c
  rw [logits_reconstruct query data q D a hrec W,
    logits_reconstruct query data q D a hrec A]
  simp only [hseen]

/-- Approximate observed logits transfer with the explicit amplification
factor `sum |a_s|`. A poorly conditioned reconstruction can amplify error (O3). -/
theorem logit_error_of_reconstruct (query : S → Q) (data : S → K → V) (q : Q)
    (D : K → V) (a : S → ℝ) (hrec : Reconstructs query data q D a)
    {A W : Channel Q K V C} (c : C) (ε : ℝ)
    (hseen : ∀ s, |logits W (query s) (data s) c - logits A (query s) (data s) c| ≤ ε) :
    |logits W q D c - logits A q D c| ≤ (∑ s, |a s|) * ε := by
  rw [logits_reconstruct query data q D a hrec W,
    logits_reconstruct query data q D a hrec A, ← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ s, |a s * logits W (query s) (data s) c -
        a s * logits A (query s) (data s) c| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ s, |a s| * ε := by
      apply Finset.sum_le_sum
      intro s _
      rw [← mul_sub, abs_mul]
      exact mul_le_mul_of_nonneg_left (hseen s) (abs_nonneg _)
    _ = _ := (Finset.sum_mul _ _ _).symm

variable [Fintype C] [DecidableEq C] [Nonempty V] [Nontrivial C]

/-- **Every global ordinary-CE optimizer makes the same unseen prediction**
when the labels are realizable and the test input has an observed-feature
certificate. No correctness premise about the optimizer or test set is used (O2). -/
theorem unseen_logits_of_min (query : S → Q) (data : S → K → V) (label : S → C)
    {A W : Channel Q K V C} (hA : Feasible A) (hfit : Fits query data label A)
    (hW : Feasible W) (hmin : IsMinOn (answerLoss query data label) {B | Feasible B} W)
    (q : Q) (D : K → V) (a : S → ℝ) (hrec : Reconstructs query data q D a) :
    logits W q D = logits A q D :=
  logits_eq_of_reconstruct query data q D a hrec
    (observed_logits_eq query data label hfit hW (hmin hA))

/-- The inferred unseen vector is an explicit combination of observed
labels. Its formula needs neither the reference channel's test logits nor
an unobserved answer supplied to the learner (O2). -/
theorem unseen_logits_eq_targets (query : S → Q) (data : S → K → V) (label : S → C)
    {A W : Channel Q K V C} (hA : Feasible A) (hfit : Fits query data label A)
    (hW : Feasible W) (hmin : IsMinOn (answerLoss query data label) {B | Feasible B} W)
    (q : Q) (D : K → V) (a : S → ℝ) (hrec : Reconstructs query data q D a) (c : C) :
    logits W q D c = ∑ s, a s * ConvexRecall.targetLogits (label s) c := by
  rw [unseen_logits_of_min query data label hA hfit hW hmin q D a hrec,
    logits_reconstruct query data q D a hrec A]
  apply Finset.sum_congr rfl
  intro s _
  rw [hfit s]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation
