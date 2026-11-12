module

public import Transformer.Grokking.Protocol.Target
public import Mathlib.Probability.Independence.InfinitePi

/-!
# The history of a sample through an epoch is independent of its later shuffles

Source: openai/grok at 3d64b1d8, `grok/data.py` (`ArithmeticIterator` draws a new permutation
of the training set for each epoch, whatever came before);
`experiments/grok_division/PROTOCOL.md`, *Randomness*. The paper, arXiv:2201.02177v1 (Power et
al.), names the batch size and the optimizer; it states no law of the shuffles.

A sample is a split, initial parameters and a shuffle for each epoch (`Sample`, `law`). Its
history through epoch `e` keeps the split, the initial parameters and the first `e` shuffles
(`history`, `firstShuffles`); what is left are the shuffles from `e` on (`laterShuffles`), and
the two give the sample back (`resume`, `resume_history_laterShuffles`).

Under the law of a sample the shuffles are independent and uniform, so:

* the first `e` shuffles and the later ones are independent, and the later ones have the law
  of the whole sequence (`indepFun_firstShuffles_laterShuffles`, `map_laterShuffles`);
* the pair of the history and the later shuffles has the product law
  (`law_map_history_laterShuffles`): given the history, the later shuffles are fresh.

`HistoryFiltration` makes the histories a filtration.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory ProbabilityTheory

variable {ι : Type*} {N k : ℕ}

/-- The first `e` shuffles of a sequence, followed by the identity. -/
def firstShuffles (e : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) : ℕ → Equiv.Perm (Fin k) :=
  fun n => if n < e then σ n else 1

/-- The shuffles from epoch `e` on. -/
def laterShuffles (e : ℕ) (σ : ℕ → Equiv.Perm (Fin k)) : ℕ → Equiv.Perm (Fin k) :=
  fun n => σ (e + n)

/-- **The history of a sample through epoch `e`**: its split, its initial parameters and its
first `e` shuffles. -/
def history (e : ℕ) (ω : Sample N k ι) : Sample N k ι := (ω.1, ω.2.1, firstShuffles e ω.2.2)

/-- The sample with the history of `ω` through epoch `e` and the later shuffles `τ`. -/
def resume (e : ℕ) (ω : Sample N k ι) (τ : ℕ → Equiv.Perm (Fin k)) : Sample N k ι :=
  (ω.1, ω.2.1, fun n => if n < e then ω.2.2 n else τ (n - e))

/-- A sample is its history through epoch `e` resumed by its later shuffles. -/
lemma resume_history_laterShuffles (e : ℕ) (ω : Sample N k ι) :
    resume e (history e ω) (laterShuffles e ω.2.2) = ω := by
  refine Prod.ext rfl (Prod.ext rfl (funext fun n => ?_))
  by_cases hn : n < e
  · simp [resume, history, firstShuffles, hn]
  · simp [resume, laterShuffles, hn, Nat.add_sub_cancel' (not_lt.1 hn)]

lemma measurable_firstShuffles (e : ℕ) : Measurable (firstShuffles (k := k) e) :=
  measurable_pi_iff.2 fun n => by
    unfold firstShuffles
    split_ifs
    · exact measurable_pi_apply n
    · exact measurable_const

lemma measurable_laterShuffles (e : ℕ) : Measurable (laterShuffles (k := k) e) :=
  measurable_pi_iff.2 fun n => measurable_pi_apply (e + n)

lemma measurable_history (e : ℕ) : Measurable (history (N := N) (k := k) (ι := ι) e) :=
  measurable_fst.prodMk ((measurable_fst.comp measurable_snd).prodMk
    ((measurable_firstShuffles e).comp (measurable_snd.comp measurable_snd)))

lemma measurable_resume (e : ℕ) :
    Measurable fun p : Sample N k ι × (ℕ → Equiv.Perm (Fin k)) => resume e p.1 p.2 :=
  (measurable_fst.comp measurable_fst).prodMk
    (((measurable_fst.comp measurable_snd).comp measurable_fst).prodMk
      (measurable_pi_iff.2 fun n => by
        by_cases hn : n < e
        · simp only [hn, ite_true]
          exact (measurable_pi_apply n).comp ((measurable_snd.comp measurable_snd).comp
            measurable_fst)
        · simp only [hn, ite_false]
          exact (measurable_pi_apply (n - e)).comp measurable_snd))

/-- **The first `e` shuffles and the later ones are independent.** -/
theorem indepFun_firstShuffles_laterShuffles (e : ℕ) :
    IndepFun (firstShuffles (k := k) e) (laterShuffles e)
      (Measure.infinitePi fun _ : ℕ => uniformPerm k) := by
  have hind := (iIndepFun_iff_iIndep _ _ _).1 (iIndepFun_infinitePi
    (P := fun _ : ℕ => uniformPerm k) (X := fun _ x => x) fun _ => measurable_id)
  have h := indep_iSup_of_disjoint (fun i => (measurable_pi_apply i).comap_le) hind
    (S := {i | i < e}) (T := {i | e ≤ i})
    (Set.disjoint_left.2 fun i (hi : i < e) (hi' : e ≤ i) => absurd hi (not_lt.2 hi'))
  rw [IndepFun_iff_Indep]
  refine indep_of_indep_of_le_left (indep_of_indep_of_le_right h ?_) ?_
  · rw [MeasurableSpace.pi, MeasurableSpace.comap_iSup]
    refine iSup_le fun n => ?_
    rw [MeasurableSpace.comap_comp]
    exact le_iSup₂ (f := fun i (_ : i ∈ {i | e ≤ i}) => MeasurableSpace.comap
      (fun σ : ℕ → Equiv.Perm (Fin k) => σ i) inferInstance) (e + n) (Nat.le_add_right e n)
  · rw [MeasurableSpace.pi, MeasurableSpace.comap_iSup]
    refine iSup_le fun n => ?_
    rw [MeasurableSpace.comap_comp]
    by_cases hn : n < e
    · have hfun : (fun σ : ℕ → Equiv.Perm (Fin k) => σ n) ∘ firstShuffles e = fun σ => σ n := by
        funext σ
        simp [firstShuffles, hn]
      rw [hfun]
      exact le_iSup₂ (f := fun i (_ : i ∈ {i | i < e}) => MeasurableSpace.comap
        (fun σ : ℕ → Equiv.Perm (Fin k) => σ i) inferInstance) n hn
    · have hfun : (fun σ : ℕ → Equiv.Perm (Fin k) => σ n) ∘ firstShuffles e = fun _ => 1 := by
        funext σ
        simp [firstShuffles, hn]
      rw [hfun, MeasurableSpace.comap_const]
      exact bot_le

/-- **The shuffles from an epoch on have the law of the whole sequence.** -/
theorem map_laterShuffles (e : ℕ) :
    (Measure.infinitePi fun _ : ℕ => uniformPerm k).map (laterShuffles e) =
      Measure.infinitePi fun _ : ℕ => uniformPerm k :=
  Measure.map_infinitePi_infinitePi_of_inj (P := fun _ : ℕ => uniformPerm k)
    (f := fun n => e + n) (add_right_injective e)

/-- **The history of a sample and its later shuffles have the product law**: given the
history through epoch `e`, the shuffles from `e` on are fresh. -/
theorem law_map_history_laterShuffles (ν : Measure (ι → ℝ)) [SFinite ν] (e : ℕ) :
    (law N k ν).map (fun ω => (history e ω, laterShuffles e ω.2.2)) =
      ((law N k ν).map (history e)).prod (Measure.infinitePi fun _ : ℕ => uniformPerm k) := by
  set c := Measure.infinitePi fun _ : ℕ => uniformPerm k with hc
  have hF := measurable_firstShuffles (k := k) e
  have hL := measurable_laterShuffles (k := k) e
  set c₁ := c.map (firstShuffles e) with hc₁
  have hS : c.map (fun σ => (firstShuffles e σ, laterShuffles e σ)) = c₁.prod c := by
    rw [(indepFun_iff_map_prod_eq_prod_map_map hF.aemeasurable hL.aemeasurable).1
      (indepFun_firstShuffles_laterShuffles e), map_laterShuffles]
  have hhist : (law N k ν).map (history e) = (uniformPerm N).prod (ν.prod c₁) := by
    have h1 := Measure.map_prod_map ν c measurable_id hF
    have h2 := Measure.map_prod_map (uniformPerm N) (ν.prod c) measurable_id
      (measurable_id.prodMap hF)
    rw [Measure.map_id] at h1 h2
    rw [hc₁, h1, h2]
    rfl
  have hpair : (law N k ν).map (fun ω => (ω.1, ω.2.1, firstShuffles e ω.2.2,
      laterShuffles e ω.2.2)) = (uniformPerm N).prod (ν.prod (c₁.prod c)) := by
    have h1 := Measure.map_prod_map ν c measurable_id (hF.prodMk hL)
    have h2 := Measure.map_prod_map (uniformPerm N) (ν.prod c) measurable_id
      (measurable_id.prodMap (hF.prodMk hL))
    rw [Measure.map_id] at h1 h2
    rw [← hS, h1, h2]
    rfl
  have hA1 : (ν.prod (c₁.prod c)).map MeasurableEquiv.prodAssoc.symm = (ν.prod c₁).prod c :=
    ((measurePreserving_prodAssoc ν c₁ c).symm _).map_eq
  have hA2 : ((uniformPerm N).prod ((ν.prod c₁).prod c)).map MeasurableEquiv.prodAssoc.symm =
      ((uniformPerm N).prod (ν.prod c₁)).prod c :=
    ((measurePreserving_prodAssoc (uniformPerm N) (ν.prod c₁) c).symm _).map_eq
  have hA3 := Measure.map_prod_map (uniformPerm N) (ν.prod (c₁.prod c)) measurable_id
    (MeasurableEquiv.measurable MeasurableEquiv.prodAssoc.symm)
  rw [Measure.map_id, hA1] at hA3
  rw [hhist, ← hA2, hA3, ← hpair, Measure.map_map (MeasurableEquiv.measurable _)
    (measurable_id.prodMap (MeasurableEquiv.measurable _)),
    Measure.map_map ((MeasurableEquiv.measurable _).comp
      (measurable_id.prodMap (MeasurableEquiv.measurable _))) (by fun_prop)]
  rfl

/-! The witnesses: sequences of shuffles of two training equations, and samples of four
equations of which two are trained on, under the law whose initial parameters are zero. -/

example (σ : ℕ → Equiv.Perm (Fin 2)) : firstShuffles 1 σ 0 = σ 0 ∧ firstShuffles 1 σ 1 = 1 ∧
    laterShuffles 1 σ 0 = σ 1 := ⟨rfl, rfl, rfl⟩

example (ω : Sample 4 2 Unit) := resume_history_laterShuffles 3 ω

example := indepFun_firstShuffles_laterShuffles (k := 2) 3

example := law_map_history_laterShuffles (N := 4) (k := 2) (Measure.dirac (0 : Unit → ℝ)) 3

end Transformer.Grokking.Protocol
