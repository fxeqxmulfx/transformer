module

public import Transformer.Grokking.FiniteRecovery.VerifiedRestarts
public import Transformer.Grokking.FiniteRecovery.RecoveryStability
public import Mathlib.Probability.ProbabilityMassFunction.Constructions
public import Mathlib.MeasureTheory.Constructions.Pi

/-! A soft certificate for a finite identifiable class. Its arguments are
observed labels, structural rules on candidate tables, and the model scores.
The auxiliary probability of a violated rule is taken at fixed weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.ObservedConstraints

open MeasureTheory Set Finset
open scoped ENNReal

variable {X A J : Type*} [MeasurableSpace A] [Fintype J]

/-- Sum of marginal observed-label errors and probabilities of violating
the structural rules. A rule can inspect any subset of the predicted table. -/
noncomputable def risk (μ : Measure (X → A)) (S : Finset X) (y : S → A)
    (rule : J → Set (X → A)) : ℝ :=
  (∑ x : S, μ.real {g | g x ≠ y x}) + ∑ j, μ.real (rule j)ᶜ

/-- Every wrong table violates either the observations or a structural rule.
The union bound allows the rule violations to be arbitrarily dependent. -/
theorem wrong_table_le (μ : Measure (X → A)) [IsProbabilityMeasure μ] (S : Finset X)
    (rule : J → Set (X → A)) (id : InjOn (fun g : X → A => fun x : S => g x) (⋂ j, rule j))
    (f : X → A) (hf : ∀ j, f ∈ rule j) :
    μ.real {g | g ≠ f} ≤ risk μ S (fun x => f x) rule := by
  classical
  let O (x : S) := {g : X → A | g x ≠ f x}
  let R (j : J) := (rule j)ᶜ
  have cover : {g | g ≠ f} ⊆ (⋃ x, O x) ∪ ⋃ j, R j := by
    intro g ne
    by_contra h
    obtain ⟨o,r⟩ := not_or.mp h
    apply ne
    refine id ?_ (mem_iInter.mpr hf) ?_
    · exact mem_iInter.mpr fun j => by
        by_contra bad
        exact r (mem_iUnion.mpr ⟨j,bad⟩)
    · exact funext fun x => by
        by_contra bad
        exact o (mem_iUnion.mpr ⟨x,bad⟩)
  have ho : μ.real (⋃ x, O x) ≤ ∑ x, μ.real (O x) := by
    simpa using (measureReal_biUnion_finset_le (μ := μ) univ O)
  have hr : μ.real (⋃ j, R j) ≤ ∑ j, μ.real (R j) := by
    simpa using (measureReal_biUnion_finset_le (μ := μ) univ R)
  exact (measureReal_mono (μ := μ) cover).trans ((measureReal_union_le _ _).trans (add_le_add ho hr))

variable [Fintype X] [Fintype A] [Nonempty A] [MeasurableSingletonClass A]

omit [MeasurableSpace A] [MeasurableSingletonClass A] in
private theorem partition_pos (s : A → ℝ) : 0 < ∑ b, Real.exp (s b) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) univ_nonempty

/-- Softmax on a finite row of logits. Feeding in scaled logits permits
any positive inverse temperature without changing their ordering. -/
noncomputable def scorePMF (s : A → ℝ) : PMF A :=
  PMF.ofFintype (fun a => ENNReal.ofReal (Real.exp (s a) / ∑ b, Real.exp (s b))) (by
    rw [← ENNReal.ofReal_sum_of_nonneg (fun _ _ => div_nonneg (Real.exp_pos _).le (partition_pos s).le),
      ← Finset.sum_div,div_self (partition_pos s).ne',ENNReal.ofReal_one])

noncomputable def tableLaw (score : X → A → ℝ) : Measure (X → A) :=
  Measure.pi (fun x => (scorePMF (score x)).toMeasure)

instance (score : X → A → ℝ) : IsProbabilityMeasure (tableLaw score) := by
  unfold tableLaw
  infer_instance

theorem table_marginal (score : X → A → ℝ) (x : X) (a : A) :
    (tableLaw score).real {g | g x = a} = Real.exp (score x a) / ∑ b, Real.exp (score x b) := by
  let rows := fun j => (scorePMF (score j)).toMeasure
  have he := (measurePreserving_eval rows x).measure_preimage (measurableSet_singleton a).nullMeasurableSet
  change (Measure.pi rows ((Function.eval x) ⁻¹' {a})).toReal = _
  rw [he,PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton _)]
  change (ENNReal.ofReal _).toReal = _
  exact ENNReal.toReal_ofReal (div_nonneg (Real.exp_pos _).le (partition_pos _).le)

/-- Below one half, the correct token has larger softmax mass than any
rival at every input, hence strictly larger original logit. -/
theorem strict_of_risk (score : X → A → ℝ) (S : Finset X) (rule : J → Set (X → A))
    (id : InjOn (fun g : X → A => fun x : S => g x) (⋂ j, rule j))
    (f : X → A) (hf : ∀ j, f ∈ rule j) (h : risk (tableLaw score) S (fun x => f x) rule < 1/2) :
    StrictOn score f := by
  let μ := tableLaw score
  let : IsProbabilityMeasure μ := inferInstanceAs (IsProbabilityMeasure (tableLaw score))
  have bad := (wrong_table_le μ S rule id f hf).trans_lt h
  intro x a ha
  let C := {g : X → A | g x = f x}
  have hc : MeasurableSet C := measurableSet_eq_fun (measurable_pi_apply x) measurable_const
  have hmiss : μ.real Cᶜ < 1/2 := (measureReal_mono (μ := μ)
    (show Cᶜ ⊆ {g | g ≠ f} from fun g hg he => hg (he ▸ rfl))).trans_lt bad
  have hwrong : μ.real {g | g x = a} ≤ μ.real Cᶜ := measureReal_mono
    (fun g (hg : g x = a) (he : g x = f x) => ha (hg.symm.trans he))
  have sum := measureReal_compl (μ := μ) hc
  rw [probReal_univ] at sum
  have win : μ.real {g | g x = a} < μ.real C := by linarith
  change (tableLaw score).real {g | g x = a} < (tableLaw score).real {g | g x = f x} at win
  rw [table_marginal,table_marginal] at win
  exact Real.exp_lt_exp.mp ((div_lt_div_iff_of_pos_right (partition_pos (score x))).mp win)

omit [Fintype A] [Nonempty A] [MeasurableSingletonClass A] in
/-- For a feasible candidate, each violated rule requires at least one
sampled entry to differ. This bound uses just the one-site marginals and
can be evaluated even when the candidate loses at some sites. -/
theorem risk_le_of_feasible (μ : Measure (X → A)) [IsProbabilityMeasure μ] (S : Finset X)
    (rule : J → Set (X → A)) (g : X → A) (feasible : ∀ j, g ∈ rule j) :
    risk μ S (fun x => g x) rule ≤
      (∑ x : S, μ.real {k | k x ≠ g x}) + (Fintype.card J : ℝ)*∑ x, μ.real {k | k x ≠ g x} := by
  have cover : {k : X → A | k ≠ g} ⊆ ⋃ x, {k | k x ≠ g x} := by
    intro k hk
    by_contra h
    exact hk (funext (fun x => by by_contra bad; exact h (mem_iUnion.mpr ⟨x,bad⟩)))
  have bound : μ.real {k | k ≠ g} ≤ ∑ x, μ.real {k | k x ≠ g x} :=
    (measureReal_mono (μ := μ) cover).trans (by
      simpa using measureReal_biUnion_finset_le (μ := μ) univ (fun x => {k | k x ≠ g x}))
  have bad (j : J) : μ.real (rule j)ᶜ ≤ ∑ x, μ.real {k | k x ≠ g x} :=
    (measureReal_mono (μ := μ) (show (rule j)ᶜ ⊆ {k | k ≠ g} from fun k hk eq => hk (eq ▸ feasible j))).trans bound
  unfold risk
  apply add_le_add le_rfl
  simpa using sum_le_sum (fun j (_ : j ∈ (Finset.univ : Finset J)) => bad j)

/-- A feasible top table gives a computable upper bound using only its
own smallest gap, the observed labels, and membership in the given rules.
The gap is measured in the logits fed into the auxiliary softmax. -/
theorem risk_le_of_gap (score : X → A → ℝ) (S : Finset X) (rule : J → Set (X → A))
    (g : X → A) (feasible : ∀ j, g ∈ rule j) (m : ℝ)
    (gap : ∀ x b, b ≠ g x → m ≤ score x (g x) - score x b) :
    risk (tableLaw score) S (fun x => g x) rule ≤
      ((S.card : ℝ) + (Fintype.card J : ℝ)*Fintype.card X) * (Fintype.card A-1 : ℕ) * Real.exp (-m) := by
  classical
  let μ := tableLaw score
  let : IsProbabilityMeasure μ := inferInstanceAs (IsProbabilityMeasure (tableLaw score))
  let δ := ((Fintype.card A-1 : ℕ) : ℝ) * Real.exp (-m)
  have row (x : X) : μ.real {k | k x ≠ g x} ≤ δ := by
    let others := univ.erase (g x)
    have cover : {k : X → A | k x ≠ g x} ⊆ ⋃ b ∈ others, {k | k x = b} := by
      intro k hk
      exact mem_iUnion₂.mpr ⟨k x,Finset.mem_erase.mpr ⟨hk,mem_univ _⟩,rfl⟩
    have bound (b : A) (hb : b ∈ others) : μ.real {k | k x = b} ≤ Real.exp (-m) := by
      rw [show μ = tableLaw score from rfl,table_marginal]
      have den : Real.exp (score x (g x)) ≤ ∑ c, Real.exp (score x c) :=
        single_le_sum (fun c _ => (Real.exp_pos _).le) (mem_univ (g x))
      calc
        _ ≤ Real.exp (score x b) / Real.exp (score x (g x)) :=
          div_le_div_of_nonneg_left (Real.exp_pos _).le (Real.exp_pos _) den
        _ = Real.exp (score x b - score x (g x)) := (Real.exp_sub ..).symm
        _ ≤ Real.exp (-m) := Real.exp_le_exp.mpr (by linarith [gap x b (mem_erase.mp hb).1])
    calc
      _ ≤ ∑ b ∈ others, μ.real {k | k x = b} :=
        (measureReal_mono (μ := μ) cover (measure_ne_top μ _)).trans (measureReal_biUnion_finset_le others _)
      _ ≤ ∑ _b ∈ others, Real.exp (-m) := sum_le_sum bound
      _ = δ := by simp [δ,others,Finset.card_erase_of_mem]
  apply (risk_le_of_feasible μ S rule g feasible).trans
  calc
    _ ≤ (∑ _x : S, δ) + (Fintype.card J : ℝ) * ∑ _x : X, δ :=
      add_le_add (sum_le_sum (fun x _ => row x))
        (mul_le_mul_of_nonneg_left (sum_le_sum (fun x _ => row x)) (Nat.cast_nonneg _))
    _ = _ := by simp [δ]; ring

/-- The certificate is exact for strict argmax recovery if the verifier
may scale a fixed score table by a positive inverse temperature. -/
theorem strict_iff_scaled_risk (score : X → A → ℝ) (S : Finset X) (rule : J → Set (X → A))
    (id : InjOn (fun g : X → A => fun x : S => g x) (⋂ j, rule j))
    (f : X → A) (hf : ∀ j, f ∈ rule j) :
    StrictOn score f ↔ ∃ β > (0:ℝ), risk (tableLaw (fun x a => β*score x a)) S (fun x => f x) rule < 1/2 := by
  constructor
  · intro win
    let rivals := {p : X × A // p.2 ≠ f p.1}
    obtain ⟨m,hm,lower⟩ := exists_pos_le_all (fun p : rivals => score p.1.1 (f p.1.1)-score p.1.1 p.1.2)
      (fun p => sub_pos.mpr (win _ _ p.2))
    let C : ℝ := ((S.card : ℝ)+(Fintype.card J : ℝ)*Fintype.card X)*(Fintype.card A-1 : ℕ)
    have C₀ : 0 ≤ C := by dsimp [C]; positivity
    let β := (2*C+1)/m
    have pos : 0 < β := div_pos (by linarith) hm
    refine ⟨β,pos,(risk_le_of_gap _ S rule f hf (2*C+1) ?_).trans_lt ?_⟩
    · intro x b hb
      have h := mul_le_mul_of_nonneg_left (lower (⟨(x,b),hb⟩ : rivals)) pos.le
      rw [show β*m = 2*C+1 from div_mul_cancel₀ _ hm.ne'] at h
      simpa [mul_sub] using h
    · change C*Real.exp (-(2*C+1)) < 1/2
      rw [Real.exp_neg,← div_eq_mul_inv,div_lt_iff₀ (Real.exp_pos _)]
      linarith [Real.add_one_le_exp (2*C+1)]
  · rintro ⟨β,hβ,low⟩ x b hb
    have win := strict_of_risk _ S rule id f hf low x b hb
    exact lt_of_not_ge fun rev => win.not_ge (mul_le_mul_of_nonneg_left rev hβ.le)

end Transformer.Grokking.FiniteRecovery.ObservedConstraints
