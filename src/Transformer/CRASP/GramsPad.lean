module

public import Transformer.CRASP.GramsCorrect

/-!
# Positional logits read the last tokens: every formula over the grams

In no paper; the construction is `Transformer.CRASP.Grams`. `thm:TLCl_to_rtfr` of
arXiv:2506.16055v3 compiles a formula `φ` of `TL[◁#]` into a rounded transformer that scores every
pair `0`, at the precision `capacity φ + 2` (`Transformer.CRASP.TemporalProgram.model`). The
construction needs `b < 2^(p-1)`, and a logit divided by `√d` or by `d` needs a `b` that grows
with the width `d`. The true comparison `1 < K + 1` conjoined to `φ` (`pad`) raises the capacity by
`K + 2` and the number of formula nodes by two at most, so the precision outgrows any `b` of the
width. Hence, for any logits `L D` and dot products `b D` that meet `ReadsPrev` at every width `D`,
some transformer of `n + k` layers with the logit `L D` recognizes the words whose grams satisfy a
formula of `TL[◁#]` of depth `k` (`exists_recognizes`); for ALiBi of slope `1` it is a `PTfr`
(`exists_alibi_recognizes`).
-/

@[expose] public section

namespace Transformer.CRASP.Grams

universe u v
variable {σ : Type u} {Γ : Type v} {n k : ℕ}

/-- The constant `K + 1`. -/
def constTerm : ℕ → Term Γ
  | 0 => .one
  | K + 1 => .add .one (constTerm K)

/-- `φ ∧ 1 < K + 1`. -/
def pad (K : ℕ) (φ : Form Γ) : Form Γ := .and φ (.lt .one (constTerm K))

theorem constTerm_props (K : ℕ) : (constTerm K : Term Γ).past = true ∧
    (constTerm K : Term Γ).pnpFree = true ∧ (constTerm K : Term Γ).depth = 0 ∧
    (constTerm K : Term Γ).subformulas = [] ∧ (constTerm K : Term Γ).constant = K + 1 ∧
    (constTerm K : Term Γ).countBodies = [] := by
  induction K with
  | zero => exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  | succ K ih =>
      obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih
      simp only [constTerm, Term.past, Term.pnpFree, Term.depth, Term.subformulas, Term.constant,
        Term.countBodies, h1, h2, h3, h4, h5, h6]
      exact ⟨rfl, rfl, rfl, rfl, by omega, rfl⟩

theorem val_constTerm [DecidableEq Γ] (w : List Γ) (i : ℕ) (K : ℕ) :
    (constTerm K : Term Γ).val w i = K + 1 := by
  induction K with
  | zero => rfl
  | succ K ih =>
      show 1 + (constTerm K : Term Γ).val w i = K + 1 + 1
      omega

/-- The padded formula holds where `φ` does. -/
theorem sat_pad [DecidableEq Γ] {K : ℕ} (hK : 1 ≤ K) (φ : Form Γ) (w : List Γ) (i : ℕ) :
    (pad K φ).sat w i = φ.sat w i := by
  have h : (Term.one : Term Γ).val w i < (constTerm K).val w i := by
    rw [val_constTerm]
    show 1 < K + 1
    omega
  simp only [pad, Form.sat, h, decide_true, Bool.and_true]

/-- The hypotheses of `sat_pad`, `lang_pad` and `pad_mem` hold for `K = 1` and a comparison of
constants. -/
example := sat_pad (K := 1) le_rfl (Form.lt .one .one : Form Bool) [true] 1

theorem lang_pad [DecidableEq Γ] {K : ℕ} (hK : 1 ≤ K) (φ : Form Γ) : (pad K φ).lang = φ.lang := by
  ext w
  simp only [Form.lang, Form.models, Set.mem_ofPred_eq, sat_pad hK]

example := lang_pad (K := 1) le_rfl (Form.lt .one .one : Form Bool)

theorem pad_mem {K : ℕ} {φ : Form Γ} (hφ : φ ∈ TLCl Γ k) : pad K φ ∈ TLCl Γ k := by
  obtain ⟨h1, h2, h3, -⟩ := constTerm_props (Γ := Γ) K
  exact Form.and_mem hφ ⟨by simp [Form.past, Term.past, h1], by simp [Form.pnpFree,
    Term.pnpFree, h2], by simp [Form.depth, Term.depth, h3]⟩

example := pad_mem (K := 1) (k := 0) (φ := (Form.lt .one (.add .one .one) : Form Bool))
  ⟨rfl, rfl, le_rfl⟩

/-- The padding raises the capacity by `K + 2`. -/
theorem capacity_pad (K : ℕ) (φ : Form Γ) : K + 2 ≤ (pad K φ).capacity := by
  obtain ⟨-, -, -, h4, h5, h6⟩ := constTerm_props (Γ := Γ) K
  have hmem : Form.lt .one (constTerm K) ∈ (pad K φ).subformulas := by
    simp [pad, Form.subformulas]
  have h := Form.weightSize_le_capacity hmem
  simp only [Form.weightSize, Term.weightSize, Term.constant, Term.countBodies, h5, h6,
    List.length_nil] at h
  omega

/-- The padding adds two formula nodes at most. -/
theorem dimension_pad_le (K : ℕ) (φ : Form Γ) :
    TemporalProgram.dimension (pad K φ) ≤ 2 * φ.subformulas.length + 5 := by
  obtain ⟨-, -, -, h4, -, -⟩ := constTerm_props (Γ := Γ) K
  have hlen : (pad K φ).subformulas.length = φ.subformulas.length + 2 := by
    simp [pad, Form.subformulas, Term.subformulas, h4]
  have hcard : Fintype.card (TemporalProgram.Node (pad K φ)) ≤ φ.subformulas.length + 2 := by
    rw [← hlen]
    refine (Fintype.card_le_of_surjective
      (fun i : Fin (pad K φ).subformulas.length =>
        (⟨(pad K φ).subformulas[i], List.getElem_mem _⟩ : TemporalProgram.Node (pad K φ)))
      fun ψ => ?_).trans (Fintype.card_fin _).le
    obtain ⟨i, hi, e⟩ := List.mem_iff_getElem.mp ψ.property
    exact ⟨⟨i, hi⟩, Subtype.ext e⟩
  simp only [TemporalProgram.dimension, TemporalProgram.Tag, Fintype.card_option,
    Fintype.card_sum]
  omega

variable [DecidableEq σ] [Fintype σ] [DecidableEq Γ]

/-- **Some transformer with the logit `L D` recognizes the words whose grams satisfy `φ`**, when
the logits `L D` and dot products `b D` meet `ReadsPrev` at every width `D`. -/
theorem exists_recognizes (L : ℕ → ℕ → ℕ → ℝ → ℝ) (b : ℕ → ℕ)
    (hL : ∀ D, 0 < D → ReadsPrev (L D) (b D)) (κ : Gram σ n → Γ) {φ : Form Γ}
    (hφ : φ ∈ TLCl Γ k) :
    ∃ (p D : ℕ) (T : LTfr (Option σ) p 0 D (n + k)), 0 < D ∧ T.logit = L D ∧
      T.Recognizes {w | grams n κ w ∈ φ.lang} := by
  let D₀ := 2 * φ.subformulas.length + 5 + Fintype.card (Gram σ n)
  let B := (Finset.range (D₀ + 1)).sup b
  let ψ := pad (B + 1) φ
  let D := width σ n (TemporalProgram.dimension ψ)
  have hD : D ≤ D₀ := by
    have h1 : TemporalProgram.dimension ψ ≤ 2 * φ.subformulas.length + 5 :=
      dimension_pad_le (B + 1) φ
    simp only [D, D₀, width]
    omega
  have hD0 : 0 < D := by
    have : 0 < Fintype.card (Gram σ n) := Fintype.card_pos
    simp only [D, width]
    omega
  have hbD : b D ≤ B := Finset.le_sup (Finset.mem_range.mpr (by omega))
  have hcap : B + 1 + 2 ≤ ψ.capacity := capacity_pad (B + 1) φ
  have hb : b D < 2 ^ (ψ.capacity + 2 - 1) := by
    calc b D < ψ.capacity + 1 := by omega
      _ < 2 ^ (ψ.capacity + 1) := Nat.lt_two_pow_self
      _ = 2 ^ (ψ.capacity + 2 - 1) := by rw [Nat.add_sub_assoc (by norm_num)]
  refine ⟨ψ.capacity + 2, D, model (L D) (b D) (TemporalProgram.model ψ k) κ, hD0, rfl, ?_⟩
  have hR := recognizes_model (L D) (b D) (TemporalProgram.model ψ k) κ (by omega) hb (hL D hD0)
    (fun ℓ x y => by simp [RTfr.score, TemporalProgram.model])
    (TemporalProgram.model_recognizes ψ k (pad_mem hφ))
  rwa [lang_pad (by omega)] at hR

/-- The hypotheses of `exists_recognizes` hold for ALiBi of slope `1` at every width. -/
example := exists_recognizes (n := 1) (k := 0) (fun _ => alibiLogit (1 : ℕ)) (fun _ => 1)
  (fun _ _ => readsPrev_alibi le_rfl) (fun c : Gram Bool 1 => decide (c 1 = some true))
  (φ := Form.lt .one (.add .one .one)) ⟨rfl, rfl, le_rfl⟩

/-- **Some transformer with ALiBi of slope `1` recognizes the words whose grams satisfy `φ`.** -/
theorem exists_alibi_recognizes (κ : Gram σ n → Γ) {φ : Form Γ} (hφ : φ ∈ TLCl Γ k) :
    ∃ (p D : ℕ) (T : PTfr (Option σ) p 0 D (n + k)), T.pe = .alibi 1 ∧
      T.Recognizes {w | grams n κ w ∈ φ.lang} := by
  obtain ⟨p, D, T, -, hT, hR⟩ := exists_recognizes (fun _ => alibiLogit (1 : ℕ)) (fun _ => 1)
    (fun _ _ => readsPrev_alibi le_rfl) κ hφ
  refine ⟨p, D, T.toAlibi 1, rfl, fun w => ?_⟩
  rw [LTfr.accepts_toAlibi T (by rw [hT]; simp)]
  exact hR w

/-- The hypothesis of `exists_alibi_recognizes` holds for a comparison of constants. -/
example := exists_alibi_recognizes (n := 1) (k := 0)
  (fun c : Gram Bool 1 => decide (c 1 = some true))
  (φ := Form.lt .one (.add .one .one)) ⟨rfl, rfl, le_rfl⟩

end Transformer.CRASP.Grams
