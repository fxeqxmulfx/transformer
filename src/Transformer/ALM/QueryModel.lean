/-
# SETH, proved

Everywhere else in this development the Strong Exponential Time Hypothesis is
a hypothesis, and `Transformer.ALM.Independence` shows why it has to be: the
`SATModel` interface leaves `cost` unrelated to `decides`, so it admits oracles
that answer instantly, and no argument quantifying over all models can settle
the conjecture.

Fix a model and the situation changes.  In the black-box model of
`Transformer.ALM.Probe` — algorithms that learn about the formula only by
evaluating it on assignments — the SETH bound is a **theorem**, proved here
with no hypotheses at all.  `four_pow_le_depth` gives the sharp form: any
correct probe algorithm makes `4^n = 2^{2n}` evaluations in the worst case,
which is every assignment.  `probeModel_SETH` is the consequence in the shape
`Transformer.ALM.SETH` consumes, and `OVHard_of_SETH` accepts it as it stands.

The argument is the adversary: answer "no" to every question.  If the
algorithm stops before asking about some assignment, it is shown the formula
whose unique satisfying assignment is that one — `uniqueCNF`, a conjunction of
`2n` unit clauses, so of linear density, exactly the regime SETH speaks about.
If it stops and says "satisfiable", it is shown a formula with an empty clause.

What this does **not** do is settle SETH.  SETH is about algorithms that read
the formula, and those can exploit its structure — that is the entire content
of the conjecture, and why it is open and implies `P ≠ NP`.  A black-box lower
bound says nothing about them; the restriction is what makes the proof
possible.  What it does settle is that the positive half of
`SETH_independent` is not an artifact of a degenerate model: `probeModel`
contains every adaptive query strategy, `scanAll` among them, and the bound
there is earned rather than assumed.

* Impagliazzo, Paturi, *On the complexity of k-SAT*, JCSS 62 (2001) — the
  conjecture whose black-box case is proved here.
* C. Bennett, E. Bernstein, G. Brassard, U. Vazirani, *Strengths and weaknesses
  of quantum computing*, SIAM J. Comput. 26 (1997), §3 — the adversary.
-/

import Transformer.ALM.Probe
import Transformer.ALM.Sparsification

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### The two formulas the adversary uses -/

/-- A formula with a single empty clause: nothing satisfies it. -/
def unsatCNF (n : ℕ) : CNF n 1 := fun _ => []

lemma unsatCNF_not_sat (n : ℕ) (α β : Fin n → Bool) : ¬ Sat (unsatCNF n) α β := by
  intro h
  simpa [unsatCNF] using h 0

lemma not_satisfiable_unsatCNF (n : ℕ) : ¬ Satisfiable (unsatCNF n) := by
  rintro ⟨α, β, h⟩
  exact unsatCNF_not_sat n α β h

/-- The conjunction of `2n` unit clauses pinning every variable: its unique
satisfying assignment is the pair `(α, β)`, and its density is linear. -/
def uniqueCNF (α β : Fin n → Bool) : CNF n (n + n) :=
  Fin.addCases (fun v => [((Sum.inl v : Fin n ⊕ Fin n), α v)])
    (fun v => [((Sum.inr v : Fin n ⊕ Fin n), β v)])

lemma sat_uniqueCNF_iff (α β α' β' : Fin n → Bool) :
    Sat (uniqueCNF α β) α' β' ↔ α' = α ∧ β' = β := by
  constructor
  · intro h
    refine ⟨funext fun v => ?_, funext fun v => ?_⟩
    · have hv := h (Fin.castAdd n v)
      rw [uniqueCNF, Fin.addCases_left] at hv
      simpa [lsat, rsat] using hv
    · have hv := h (Fin.natAdd n v)
      rw [uniqueCNF, Fin.addCases_right] at hv
      simpa [lsat, rsat] using hv
  · rintro ⟨rfl, rfl⟩ c
    induction c using Fin.addCases with
    | left v => rw [uniqueCNF, Fin.addCases_left]; simp [lsat]
    | right v => rw [uniqueCNF, Fin.addCases_right]; simp [rsat]

lemma satisfiable_uniqueCNF (α β : Fin n → Bool) : Satisfiable (uniqueCNF α β) :=
  ⟨α, β, (sat_uniqueCNF_iff α β α β).mpr ⟨rfl, rfl⟩⟩

lemma not_sat_uniqueCNF_of_ne (α β : Fin n → Bool)
    (q : (Fin n → Bool) × (Fin n → Bool)) (hq : q ≠ (α, β)) :
    ¬ Sat (uniqueCNF α β) q.1 q.2 := by
  intro h
  obtain ⟨h1, h2⟩ := (sat_uniqueCNF_iff α β q.1 q.2).mp h
  exact hq (Prod.ext h1 h2)

/-! ### The model -/

/-- **The black-box model of satisfiability.**  An algorithm is an adaptive
decision tree over assignment evaluations, and it costs the number of
evaluations it makes in the worst case. -/
def probeModel : SATModel where
  Alg := (n : ℕ) → Probe n
  decides := fun t => fun {n _m} φ => run φ (t n) = true
  cost := fun t n _ => (depth (t n) : ℝ)

lemma probeModel_decides {t : (n : ℕ) → Probe n} {n m : ℕ} (φ : CNF n m) :
    probeModel.decides t φ ↔ run φ (t n) = true := Iff.rfl

lemma card_pairs (n : ℕ) :
    Fintype.card ((Fin n → Bool) × (Fin n → Bool)) = 4 ^ n := by
  rw [Fintype.card_prod, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin,
    ← mul_pow]
  norm_num

/-! ### Exhaustive search is optimal, and is available -/

/-- **The lower bound, sharp.**  A correct probe algorithm evaluates the
formula on *every* one of the `4^n` assignments in the worst case.  Nothing is
assumed: this is the adversary argument, not a conjecture. -/
theorem four_pow_le_depth {t : (n : ℕ) → Probe n} (ht : probeModel.Solves t)
    (n : ℕ) : 4 ^ n ≤ depth (t n) := by
  by_contra hlt
  have hlt := not_le.mp hlt
  have hlen : (noPath (t n)).length < 4 ^ n :=
    lt_of_le_of_lt (noPath_length_le_depth (t n)) hlt
  cases hnv : noVal (t n) with
  | true =>
      have hrun : run (unsatCNF n) (t n) = true := by
        rw [run_eq_noVal _ (t n) fun q _ => unsatCNF_not_sat n q.1 q.2, hnv]
      exact not_satisfiable_unsatCNF n ((ht (unsatCNF n)).mp hrun)
  | false =>
      obtain ⟨p, hp⟩ : ∃ p : (Fin n → Bool) × (Fin n → Bool), p ∉ noPath (t n) := by
        by_contra hall
        have hall : ∀ q : (Fin n → Bool) × (Fin n → Bool), q ∈ noPath (t n) :=
          fun q => not_not.mp fun h => hall ⟨q, h⟩
        have hsub : (Finset.univ : Finset ((Fin n → Bool) × (Fin n → Bool)))
            ⊆ (noPath (t n)).toFinset := fun q _ => List.mem_toFinset.mpr (hall q)
        have h1 := Finset.card_le_card hsub
        rw [Finset.card_univ, card_pairs n] at h1
        exact absurd (h1.trans (List.toFinset_card_le _)) (not_le.mpr hlen)
      have hrun : run (uniqueCNF p.1 p.2) (t n) = false := by
        rw [run_eq_noVal _ (t n) fun q hq =>
          not_sat_uniqueCNF_of_ne p.1 p.2 q fun h => hp (by simpa [h] using hq), hnv]
      have hyes := (ht (uniqueCNF p.1 p.2)).mpr (satisfiable_uniqueCNF p.1 p.2)
      rw [probeModel_decides, hrun] at hyes
      exact absurd hyes (by simp)

/-- Exhaustive search, as a probe algorithm. -/
noncomputable def scanAll (n : ℕ) : Probe n := scan Finset.univ.toList

/-- It is correct, so `probeModel.Solves` is not empty and the bound above is
not a statement about nothing. -/
theorem probeModel_solves_scanAll : probeModel.Solves scanAll := by
  intro n m φ
  rw [probeModel_decides, scanAll, run_scan]
  constructor
  · rintro ⟨p, -, hsat⟩
    exact ⟨p.1, p.2, hsat⟩
  · rintro ⟨α, β, hsat⟩
    exact ⟨(α, β), Finset.mem_toList.mpr (Finset.mem_univ _), hsat⟩

/-! ### SETH as a theorem -/

/-- **SETH holds in the black-box model**, unconditionally.  The bound the
hypothesis asks for is `2^{2n(1-δ)}`; what the adversary gives is `2^{2n}`,
with no `δ` at all, so any density constant will do.

This is `SATModel.SETH` with every hypothesis discharged — the conjecture, for
the algorithms that cannot look at the formula. -/
theorem probeModel_SETH : probeModel.SETH := by
  intro δ hδ
  refine ⟨1, fun t ht N => ⟨N, le_refl N, ?_⟩⟩
  have hcount : (4 : ℝ) ^ N ≤ (depth (t N) : ℝ) := by
    exact_mod_cast four_pow_le_depth ht N
  have hN : (0 : ℝ) ≤ (N : ℝ) := Nat.cast_nonneg N
  have h1 : (2 : ℝ) ^ (2 * (N : ℝ) * (1 - δ)) ≤ (2 : ℝ) ^ (2 * (N : ℝ)) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by nlinarith)
  have hsq : (2 : ℝ) ^ (2 * (N : ℝ)) = ((2 : ℝ) ^ N) ^ 2 := by
    rw [show (2 : ℝ) * (N : ℝ) = (N : ℝ) * 2 by ring, Real.rpow_mul (by norm_num),
      Real.rpow_natCast, show ((2 : ℝ) : ℝ) = ((2 : ℕ) : ℝ) by norm_num,
      Real.rpow_natCast]
  have h4 : ((2 : ℝ) ^ N) ^ 2 = (4 : ℝ) ^ N := by rw [sq, ← mul_pow]; norm_num
  show (2 : ℝ) ^ (2 * (N : ℝ) * (1 - δ)) ≤ (depth (t N) : ℝ)
  rw [hsq, h4] at h1
  linarith

/-- And therefore in the form without a density restriction as well. -/
theorem probeModel_SETHGeneral : probeModel.SETHGeneral :=
  SATModel.SETHGeneral_of_SETH probeModel_SETH

end ALM
end Transformer
