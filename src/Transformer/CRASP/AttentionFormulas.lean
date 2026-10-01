/-
# Temporal predicates for the rounded attention vector

arXiv:2506.16055v3, Appendix B.2, division and its zero-denominator
fallback in `thm:rtfr_to_TLCl`. The tests distinguish every possible
attention value at one more counting level than the source states.
-/

import Transformer.CRASP.AttentionKernel
import Transformer.CRASP.RoundedCounts

namespace Transformer.CRASP.RTfr

universe u
variable {σ : Type u} {p s d k j : ℕ}

/-- The masked prefix has length `i+1`, including BOS (Appendix B.2). -/
def prefixLengthCount : LinearCount σ := ⟨1, [(1, Form.truth true)]⟩

/-- The length count uses only a depth-zero predicate (Appendix B.2). -/
theorem prefixLengthCount_good (j : ℕ) : (prefixLengthCount : LinearCount σ).Good j := by
  rintro x hx
  have he : x = (1, Form.truth true) := by simpa [prefixLengthCount] using hx
  subst x
  change (Form.truth true : Form σ) ∈ TLCl σ j
  exact Form.truth_mem true j

/-- The finite cell test for one attention coordinate (Appendix B.2). -/
noncomputable def attentionFormula (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    (ψ : (Fin d → Fx p s) → Form σ) (q : Fin d → Fx p s) (c : Fin d) (y : Fx p s) :
    Form σ :=
  let D := T.stateSum ℓ ψ (T.stateWeight ℓ q)
  let N := T.stateSum ℓ ψ (T.stateWeightedValue ℓ q c)
  let V := T.stateSum ℓ ψ (fun a => T.WV ℓ a c)
  Form.or
    (.and D.isZero (V.roundedQuotient prefixLengthCount y))
    (.and (D.scale (-1)).ltZero ((N.scale (2 ^ s)).roundedQuotient D y))

/-- The cell test costs exactly at most one new counting level (Appendix B.2). -/
theorem attentionFormula_mem (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    {ψ : (Fin d → Fx p s) → Form σ} (hψ : ∀ q, ψ q ∈ TLCl σ j)
    (q : Fin d → Fx p s) (c : Fin d) (y : Fx p s) :
    T.attentionFormula ℓ ψ q c y ∈ TLCl σ (j + 1) := by
  apply Form.or_mem
  · exact Form.and_mem (LinearCount.isZero_mem (T.stateSum_good ℓ j hψ _))
      (LinearCount.roundedQuotient_mem (T.stateSum_good ℓ j hψ _) (prefixLengthCount_good j) y)
  · exact Form.and_mem (LinearCount.ltZero_mem
        (LinearCount.good_scale (T.stateSum_good ℓ j hψ _) (-1)))
      (LinearCount.roundedQuotient_mem
        (LinearCount.good_scale (T.stateSum_good ℓ j hψ _) (2 ^ s))
        (T.stateSum_good ℓ j hψ _) y)

variable [DecidableEq σ]

/-- The length count evaluates to the number of attended positions (B.2). -/
@[simp] theorem val_prefixLengthCount (w : List σ) (i : ℕ) :
    (prefixLengthCount : LinearCount σ).val w i = (i : ℤ) + 1 := by
  simp [prefixLengthCount, LinearCount.val, Term.val, Form.sat_truth, add_comm]

/-- The temporal cell test defines the counted attention value (Appendix B.2). -/
theorem sat_attentionFormula (T : RTfr (Option σ) p s d k) (ℓ : ℕ)
    (ψ : (Fin d → Fx p s) → Form σ) (q : Fin d → Fx p s) (c : Fin d) (y : Fx p s)
    (w : List σ) (i : ℕ) :
    (T.attentionFormula ℓ ψ q c y).sat w i = true ↔
      T.countAttention ℓ ψ q w i c = y := by
  let D := T.stateSum ℓ ψ (T.stateWeight ℓ q)
  let N := T.stateSum ℓ ψ (T.stateWeightedValue ℓ q c)
  let V := T.stateSum ℓ ψ (fun a => T.WV ℓ a c)
  have hL : 0 < (prefixLengthCount : LinearCount σ).val w i := by simp
  have hfall := LinearCount.sat_roundedQuotient V prefixLengthCount y w i hL
  have hDnonneg : 0 ≤ D.val w i := T.stateSum_weight_nonneg ℓ ψ q w i
  have hs : (2 : ℝ) ^ s ≠ 0 := by positivity
  change (Form.or (.and D.isZero (V.roundedQuotient prefixLengthCount y))
    (.and (D.scale (-1)).ltZero ((N.scale (2 ^ s)).roundedQuotient D y))).sat w i = true ↔ _
  rw [Form.sat_or]
  simp only [Form.sat, Bool.or_eq_true, Bool.and_eq_true, LinearCount.sat_isZero,
    LinearCount.sat_ltZero, LinearCount.val_scale, neg_one_mul, decide_eq_true_eq]
  by_cases hz : D.val w i = 0
  · simp only [hz, neg_zero, lt_self_iff_false, false_and, or_false, true_and]
    rw [hfall, countAttention]
    simp only [show (T.stateSum ℓ ψ (T.stateWeight ℓ q)).val w i = 0 from hz, ite_true]
    rw [val_prefixLengthCount]
    push_cast
    rfl
  · have hpos : 0 < D.val w i := by omega
    have hnorm := LinearCount.sat_roundedQuotient (N.scale (2 ^ s)) D y w i hpos
    simp only [hz, false_and, false_or, neg_lt_zero, hpos, true_and]
    rw [hnorm, countAttention]
    simp only [show (T.stateSum ℓ ψ (T.stateWeight ℓ q)).val w i ≠ 0 from hz, ite_false]
    have he : (((N.scale (2 ^ s)).val w i : ℝ) / (D.val w i : ℝ)) / 2 ^ s =
        (N.val w i : ℝ) / (D.val w i : ℝ) := by
      rw [LinearCount.val_scale]
      push_cast
      field_simp
    rw [he]

/-- State-formula bounds used by the cell construction have witnesses (B.2). -/
example : ∀ _ : (Fin 1 → Fx 2 0), (Form.sym true : Form Bool) ∈ TLCl Bool 0 :=
  fun _ => ⟨rfl, rfl, le_rfl⟩

end Transformer.CRASP.RTfr
