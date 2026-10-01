/-
# Modulus zero gives an absolute-position predicate

arXiv:2506.16055v3, Appendix F, `thm:rtfr_eq_tlclmod`.
The former Lean `MOD` fragment also allowed modulus zero. In Lean's natural
remainder, `i % 0 = i`, so this is an exact-position test rather than a
periodic predicate. It can read the first letter at counting depth one.
-/

import Transformer.CRASP.PeriodicOutputPermutation
import Transformer.CRASP.PositionalPeriod

namespace Transformer.CRASP

/-- A first-letter test using the unintended modulus-zero atom (Appendix F). -/
def FormP.firstTrue : FormP Bool :=
  .neg (.lt (.countL (.and (.sym true) (.mod 0 1))) .one)

/-- Its counting depth is one and it uses no previous-position operators (F). -/
theorem FormP.firstTrue_mem : FormP.firstTrue ∈ TLClMod Bool 1 := ⟨rfl, le_rfl⟩

/-- The exact-position atom selects only the first letter (Appendix F). -/
theorem FormP.models_firstTrue_cons (a : Bool) (w : List Bool) :
    FormP.firstTrue.models (a :: w) ↔ a = true := by
  let ψ : FormP Bool := .and (.sym true) (.mod 0 1)
  have ht : (List.range' 2 w.length).filter (fun j => ψ.sat (a :: w) j) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro j hj
    have hjb := List.mem_range'.mp hj
    simp [ψ, FormP.sat, show j ≠ 1 by omega]
  simp only [FormP.models, FormP.firstTrue, List.length_cons, FormP.sat, TermP.val,
    List.range'_succ]
  change (!(decide (((1 :: List.range' 2 w.length).filter
    (fun j => ψ.sat (a :: w) j)).length < 1))) = true ↔ a = true
  rw [List.filter_cons, ht]
  cases a <;> simp [ψ, FormP.sat]

/-- The first-letter language is definable at depth one in the former fragment.
Source: arXiv:2506.16055v3, Appendix F, unintended `MOD_0^1` case. -/
theorem definableMod_firstTrue : DefinableMod {w : List Bool | w.head? = some true} 1 := by
  refine ⟨FormP.firstTrue, FormP.firstTrue_mem, ?_⟩
  ext w
  cases w with
  | nil => simp [FormP.lang, FormP.models, FormP.firstTrue, FormP.sat, TermP.val]
  | cons a w => simpa only [FormP.lang, Set.mem_ofPred_eq, List.head?_cons,
      Option.some.injEq] using FormP.models_firstTrue_cons a w

/-- No depth-one sinusoidal transformer at rational angles reads the first
letter on all words. Swapping positions one common period apart preserves
every source-state count and the final query but changes that first letter.

Source: arXiv:2506.16055v3, Appendix F, periodicity assumption in
`thm:rtfr_eq_tlclmod`. This refutes the former Lean signature admitting
`MOD_0^r`; the manuscript's periodic fragment needs positive moduli. -/
theorem not_sinusoidal_recognizes_firstTrue {p s d : ℕ} (θ : ℕ → ℝ)
    (hθ : (PosEnc.sinusoidal θ).RationalAngles) (T : PTfr (Option Bool) p s d 1)
    (hT : T.pe = .sinusoidal θ) :
    ¬ T.Recognizes {w : List Bool | w.head? = some true} := by
  obtain ⟨M, hM, hrot⟩ := exists_rotation_period d θ hθ
  have hemb : ∀ i, T.pe.emb p s d i = T.pe.emb p s d (i % M) := by
    intro i
    funext c
    simp only [hT, PosEnc.emb, sinusoidalVec, hrot i]
  have hlogit : ∀ i j (q a : Fin d → Fx p s),
      T.pe.logit i j q a = T.pe.logit (i % M) (j % M) q a := by
    intro i j q a
    simp only [hT, PosEnc.logit]
  have hout := PeriodicAttention.out_gap_swap T hM hemb hlogit true false false false
  intro hrec
  have hy := (hrec (true :: (List.replicate (M - 1) false ++ [false, false]))).mpr rfl
  have hn : ¬ T.Accepts (bos (false :: (List.replicate (M - 1) false ++ [true, false]))) := by
    rw [hrec]
    simp
  apply hn
  rw [PTfr.Accepts] at hy ⊢
  rw [← hout]
  exact hy

/-- The finite-alphabet, rational-angle and encoding hypotheses have witnesses.
Source: arXiv:2506.16055v3, Appendix F, periodic embeddings. -/
example : ∃ T : PTfr (Option Bool) 2 0 0 1,
    T.pe = .sinusoidal (fun _ => 0) ∧ (PosEnc.sinusoidal (fun _ => 0)).RationalAngles := by
  refine ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .sinusoidal (fun _ => 0) }, rfl, fun _ => 0, fun _ => ?_⟩
  simp

/-- Counterexample to the former Lean `thm:rtfr_eq_tlclmod` statement:
with modulus zero admitted, the depth-one implication to sinusoidal
transformers is false. The local paper calls the predicates periodic;
its corrected formal statement must require positive moduli.

Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_eq_tlclmod`. -/
theorem sinusoidal_equivalence_zero_modulus_counterexample :
    ∃ L : Set (List Bool), DefinableMod L 1 ∧
      ¬ ∃ (p s d : ℕ) (θ : ℕ → ℝ) (T : PTfr (Option Bool) p s d 1),
        T.pe = .sinusoidal θ ∧ (PosEnc.sinusoidal θ).RationalAngles ∧ T.Recognizes L := by
  refine ⟨_, definableMod_firstTrue, ?_⟩
  rintro ⟨p, s, d, θ, T, hT, hθ, hrec⟩
  exact not_sinusoidal_recognizes_firstTrue θ hθ T hT hrec

end Transformer.CRASP
