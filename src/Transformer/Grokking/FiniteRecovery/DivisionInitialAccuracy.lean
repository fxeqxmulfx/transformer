module

public import Transformer.Grokking.FiniteRecovery.FirstHitSymmetry
public import Transformer.Grokking.FiniteRecovery.DivisionMask

/-! Quantify how much strict modular division success can come from the
initial draw alone. The argument uses the vocabulary, the actual prompts
and the size of the hidden set. It applies before any gradient update. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set MeasureTheory
open scoped NNReal

variable {A : Type*} [Fintype A] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)

/-- At initialization, any answer absent from the prompt is exchangeable
with the other absent tokens. This counts strict winners, allowing ties. -/
theorem initial_absent_answer (r : ℕ) (x : Word A T) (i : Fin T) (a : A)
    (absent : ∀ k, x k ≠ a) (hT : T < Fintype.card A) (variance : ℝ≥0) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter A H d L => variance)
    let score (z : Draw) := (Model.fromWeights (sgdInitialWeights true z)).logits r x i
    μ.real {z | ∀ b, b ≠ a → score z b < score z a} ≤ 1/(Fintype.card A - T : ℕ) := by
  classical
  intro μ score
  let used := Finset.univ.image x
  let free := Finset.univ \ used
  have ha : a ∈ free := by simpa [free,used] using absent
  have perms (c : A) (hc : c ∈ free) : ∃ e : Equiv.Perm A, e a = c ∧ e ∘ x = x ∧
      ∀ _ : ℕ, ([] : List (LabeledPosition A)).map (relabelPosition e) = [] := by
    refine ⟨Equiv.swap a c,Equiv.swap_apply_left ..,funext fun k => ?_,fun _ => rfl⟩
    exact Equiv.swap_apply_of_ne_of_ne (absent k) (fun eq => (Finset.mem_sdiff.mp hc).2
      (Finset.mem_image.mpr ⟨k,Finset.mem_univ _,eq⟩))
  have h := uncorrectedAdam_first_hit_orbit (H := H) (d := d) (L := L) r 0 0 1 0
    (by norm_num) (fun _ => 0) (fun _ => []) x i a free ⟨a,ha⟩ perms {0} variance
  have hwin : μ.real {z | ∀ b, b ≠ a → score z b < score z a} ≤ 1/(free.card : ℝ) := by
    simpa [score,uncorrectedAdamRun,EqualMoment.uncorrectedRun] using h
  have hused : used.card ≤ T := by
    simpa only [Finset.card_univ,Fintype.card_fin] using (Finset.card_image_le (f := x) (s := Finset.univ))
  have hcount : Fintype.card A - T ≤ free.card := by
    rw [show free.card = Fintype.card A - used.card by simp [free,Finset.card_sdiff_of_subset]]
    omega
  exact hwin.trans (one_div_le_one_div_of_le (by exact_mod_cast Nat.sub_pos_of_lt hT) (by exact_mod_cast hcount))

set_option maxRecDepth 4096 in
/-- A quotient can occur among the prompt tokens only on the zero row,
the denominator-one column, or one square-numerator cell per denominator. -/
theorem exists_absent_division_answer (hidden : Finset DivisionInput) (hc : 288 < hidden.card) :
    ∃ x ∈ hidden, ∀ k, divisionPrompt x k ≠ divisionAnswer x := by
  classical
  let U : Finset DivisionUnit := Finset.univ
  let zeroRow := U.image (fun b => ((none,b) : DivisionInput))
  let ones := U.image (fun a => ((some a,1) : DivisionInput))
  let squares := U.image (fun b => ((some (b*b),b) : DivisionInput))
  have hu : U.card = 96 := (Finset.card_univ).trans ((Fintype.card_congr divisionUnitIndex).trans (Fintype.card_fin _))
  have hcard : ((zeroRow ∪ ones) ∪ squares).card ≤ 288 := by
    have h₀ : zeroRow.card ≤ U.card := Finset.card_image_le
    have h₁ : ones.card ≤ U.card := Finset.card_image_le
    have h₂ : squares.card ≤ U.card := Finset.card_image_le
    have h₃ := Finset.card_union_le zeroRow ones
    have h₄ := Finset.card_union_le (zeroRow ∪ ones) squares
    omega
  obtain ⟨⟨v,b⟩,hm,hout⟩ := Finset.exists_mem_notMem_of_card_lt_card (hcard.trans_lt hc)
  simp only [Finset.mem_union,not_or] at hout
  refine ⟨(v,b),hm,?_⟩
  cases v with
  | none => exact False.elim (hout.1.1 (Finset.mem_image.mpr ⟨b,Finset.mem_univ _,rfl⟩))
  | some a =>
    intro k h
    have lower : 22 ≤ (divisionAnswer (some a,b)).val := Nat.le_add_right _ _
    have value := congrArg Fin.val h
    fin_cases k
    · change 0 = _ at value
      omega
    · have eq : a / b = a := Units.val_injective (residueToken_injective h.symm)
      exact hout.1.2 (Finset.mem_image.mpr ⟨a,Finset.mem_univ _,by rw [div_eq_self.mp eq]⟩)
    · change 10 = _ at value
      omega
    · have eq : a / b = b := Units.val_injective (residueToken_injective h.symm)
      exact hout.2 (Finset.mem_image.mpr ⟨b,Finset.mem_univ _,by rw [div_eq_iff_eq_mul.mp eq]⟩)
    · change 1 = _ at value
      omega

local notation "DivisionDraw" => (ComplexParameter (Fin 239) 4 16 6 → ℂ)

def initiallySolvesDivision (hidden : Finset DivisionInput) : Set DivisionDraw :=
  {z | ∀ x ∈ hidden, ∀ b, b ≠ divisionAnswer x →
    (Model.fromWeights (sgdInitialWeights true z)).logits 1 (divisionPrompt x) (Fin.last 4) b <
    (Model.fromWeights (sgdInitialWeights true z)).logits 1 (divisionPrompt x) (Fin.last 4) (divisionAnswer x)}

/-- Every hidden half of this corpus satisfies the cardinality condition.
The bound holds with the full stock geometry and every constant variance. -/
theorem initial_division_success (hidden : Finset DivisionInput) (hc : 288 < hidden.card) (variance : ℝ≥0) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 239) 4 16 6 => variance)
    μ.real (initiallySolvesDivision hidden) ≤ 1/234 := by
  intro μ
  obtain ⟨x,hx,absent⟩ := exists_absent_division_answer hidden hc
  have h := initial_absent_answer (H := 4) (d := 16) (L := 6) 1 (divisionPrompt x) (Fin.last 4)
    (divisionAnswer x) absent (by simp) variance
  let s (z : DivisionDraw) := (Model.fromWeights (sgdInitialWeights true z)).logits 1 (divisionPrompt x) (Fin.last 4)
  have sub : initiallySolvesDivision hidden ⊆ {z | ∀ b, b ≠ divisionAnswer x → s z b < s z (divisionAnswer x)} :=
    fun _ hz => hz x hx
  exact (measureReal_mono (μ := μ) sub).trans (by simpa [s] using h)

/-- For any training procedure with success event E, at most 1/234 of
the mass can have been strictly correct throughout the hidden set at step zero. -/
theorem learned_division_success (hidden : Finset DivisionInput) (hc : 288 < hidden.card) (variance : ℝ≥0)
    (E : Set DivisionDraw) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter (Fin 239) 4 16 6 => variance)
    μ.real E - 1/234 ≤ μ.real (E \ initiallySolvesDivision hidden) := by
  classical
  intro μ
  have sub : E ⊆ (E \ initiallySolvesDivision hidden) ∪ initiallySolvesDivision hidden :=
    fun z hz => by by_cases h : z ∈ initiallySolvesDivision hidden; exact Or.inr h; exact Or.inl ⟨hz,h⟩
  have h := (measureReal_mono (μ := μ) sub).trans (measureReal_union_le _ _)
  linarith [initial_division_success hidden hc variance]

end Transformer.Grokking.FiniteRecovery.Argand
