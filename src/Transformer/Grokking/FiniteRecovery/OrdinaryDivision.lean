module

public import Transformer.Grokking.FiniteRecovery.ReadoutSimplex
public import Transformer.Grokking.FiniteRecovery.ProposalMass
public import Transformer.Grokking.FiniteRecovery.NativeFirstPerfect

/-! One ordinary bias-corrected AdamW trajectory on the original modular
division architecture. At every finite budget a positive-mass open set of
stock Gaussian initializations misses throughout, for every minibatch
order. The argument uses only the tied output rows and a bound on AdamW's
finite displacement. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.OrdinaryDivision

open Set MeasureTheory ProbabilityTheory Transformer.PhasorV2
open scoped NNReal ENNReal

abbrev Token := Fin 239
abbrev Draw := ComplexParameter Token 4 16 6 → ℂ
abbrev Coord := ReadoutCoord 64
abbrev Batches := ℕ → List (LabeledPosition Token)

private def vertex (i : Coord) : Token :=
  ⟨65+i.1.val+(if i.2 then 64 else 0),by have := i.1.isLt; split <;> omega⟩

private theorem vertex_injective : Function.Injective vertex := by
  rintro ⟨i,b⟩ ⟨j,c⟩ h
  have hv := congrArg Fin.val h
  cases b <;> cases c <;> simp [vertex] at hv ⊢ <;> first | exact Fin.ext hv | omega

private theorem vertex_ne (i : Coord) : vertex i ≠ 32 ∧ vertex i ≠ 238 := by
  rcases i with ⟨j,b⟩
  have hj := j.isLt
  cases b <;> constructor <;> intro h <;> have v := congrArg Fin.val h <;> simp [vertex] at v <;> omega

private def coefficient (a : Token) (j : Coord) : ℝ :=
  if a=238 then -1 else if a=vertex j then 1 else 0

private def template (M : ℝ) : Draw
  | .embedding a j => ⟨M*coefficient a (j,false),M*coefficient a (j,true)⟩
  | _ => 0

private def initialRow (z : Draw) (a : Token) (i : Coord) : ℝ :=
  readoutRow (sgdInitialWeights true z) a i

private theorem template_row (M : ℝ) (a : Token) (i : Coord) :
    initialRow (template M) a i = M*coefficient a i := by
  rcases i with ⟨i,b⟩
  cases b <;> rfl

private theorem decay_antitone (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t < 10) :
    Antitone (AdamW.decayFactor rate) := by
  apply antitone_nat_of_succ_le
  intro t
  exact mul_le_of_le_one_left (AdamW.decayFactor_bounds rate t (fun j _ => hr j)).1.le
    (by linarith [(hr t).1])

/-- For each cap, one open set of ordinary initializations makes token
32 (quotient 10) unwinnable until that cap, even if minibatches are
selected after seeing the draw. All 820,124 parameters can train. -/
theorem open_miss (cap : ℕ) (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t < 10)
    (variance : ComplexParameter Token 4 16 6 → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    ∃ V : Set Draw, IsOpen V ∧ 0 < independentGaussianDraws variance V ∧
      ∀ z ∈ V, ∀ (batch : Batches) t, t ≤ cap → ∀ h : CSpace 64,
      ¬ ∀ b : Token, b ≠ 32 → (Model.fromWeights (adamWRun 1 batch rate z t).weights).readout h b <
        (Model.fromWeights (adamWRun 1 batch rate z t).weights).readout h 32 := by
  let decay := AdamW.decayFactor rate
  have bounds t := AdamW.decayFactor_bounds rate t (fun i _ => hr i)
  let M := 63000 / decay cap
  have hM : 0 ≤ M := div_nonneg (by norm_num) (bounds cap).1.le
  let V : Set Draw := {z | ∀ a i, |initialRow z a i-M*coefficient a i| < 1}
  have vo : IsOpen V := by
    simp only [V,ofPred_forall]
    apply isOpen_iInter_of_finite
    intro a
    apply isOpen_iInter_of_finite
    intro i
    have hc : Continuous (fun z : Draw => initialRow z a i) :=
      (continuous_apply (.inl (.embedding a i.1,i.2) : Parameter Token 4 16 6)).comp (continuous_sgdInitialWeights true)
    exact isOpen_lt (hc.sub continuous_const).abs continuous_const
  let := independentGaussianDraws_openPos variance hv
  refine ⟨V,vo,vo.measure_pos _ ⟨template M,fun a i => by rw [template_row]; simp⟩,?_⟩
  intro z hz batch t ht h win
  let w := (adamWRun 1 batch rate z t).weights
  let K := decay t*M
  have hK : 63000 ≤ K := calc
    (63000 : ℝ) = decay cap*M := by
      have nz : decay cap ≠ 0 := (bounds cap).1.ne'
      dsimp only [M]; field_simp
    _ ≤ _ := mul_le_mul_of_nonneg_right (decay_antitone rate hr ht) hM
  have close (a : Token) (i : Coord) : |readoutRow w a i-K*coefficient a i| ≤ 121 := by
    have e := AdamW.decay_displacement rate (fun k => batchGradient 1 (batch k))
      (sgdInitialWeights true z) t (fun k _ => hr k) (.inl (.embedding a i.1,i.2))
    change |readoutRow w a i-decay t*initialRow z a i| ≤ 120*(1-decay t) at e
    have init := (hz a i).le
    have mult : |decay t*(initialRow z a i-M*coefficient a i)| ≤ 1 := by
      rw [abs_mul,abs_of_nonneg (bounds t).1.le]
      exact (mul_le_mul_of_nonneg_left init (bounds t).1.le).trans (by simpa using (bounds t).2)
    have eq : readoutRow w a i-K*coefficient a i =
        (readoutRow w a i-decay t*initialRow z a i) + decay t*(initialRow z a i-M*coefficient a i) := by
      dsimp only [K]; ring
    rw [eq]
    exact (abs_add_le ..).trans (by linarith [e,mult,(bounds t).1])
  refine (simplex_prevents_winner (readoutRow w) (32 : Token) 238 vertex (by decide) (fun i => (vertex_ne i).1)
    121 K (by norm_num) (by norm_num [Coord,ReadoutCoord,Fintype.card_prod]; linarith) ?_ ?_ ?_ (realFeature h)) ?_
  · intro i
    simpa [coefficient,Ne.symm (vertex_ne i).1] using close 32 i
  · intro i
    simpa [coefficient] using close 238 i
  · intro j i
    by_cases he : i=j
    · subst i; simpa [coefficient,(vertex_ne j).2] using close (vertex j) j
    · have ne : vertex j ≠ vertex i := fun v => he (vertex_injective v.symm)
      simpa [coefficient,(vertex_ne j).2,he,ne] using close (vertex j) i
  · intro b hb
    have he := win b hb
    change (Model.fromWeights w).readout h b < (Model.fromWeights w).readout h 32 at he
    simpa only [readout_real_dot,div_lt_div_iff_of_pos_right (by norm_num : (0:ℝ)<2)] using he

/-- The actual first-hit event on the native 256-update grid. The shuffle
may depend on the same draw. -/
def hit (B : ℕ) (rate : ℕ → ℝ) (batch : Draw → Batches) (hidden : Set DivisionInput) : Set Draw :=
  {z | ∃ t ∈ evaluationGrid 256 B, ∀ x ∈ hidden, ∀ b, b ≠ divisionAnswer x →
    (Model.fromWeights (adamWRun 1 (batch z) rate z t).weights).logits 1 (divisionPrompt x) (Fin.last 4) b <
      (Model.fromWeights (adamWRun 1 (batch z) rate z t).weights).logits 1 (divisionPrompt x) (Fin.last 4) (divisionAnswer x)}

/-- The lower side is supplied by an open set of strictly realizing
stock starts. The strict upper side holds through every update of any
finite budget, including the production WSD prefix and reshuffling.
It does not quantify typical improvement from an initially wrong table. -/
theorem finite_hit_probability (B : ℕ) (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t < 10)
    (batch : Draw → Batches) (hidden : Set DivisionInput) (x : DivisionInput) (hx : x ∈ hidden)
    (answer : divisionAnswer x = 32) (variance : ComplexParameter Token 4 16 6 → ℝ≥0)
    (hv : ∀ i, 0 < variance i) :
    let μ := independentGaussianDraws variance
    0 < μ (hit B rate batch hidden) ∧ μ (hit B rate batch hidden) < 1 := by
  intro μ
  let := independentGaussianDraws_openPos variance hv
  constructor
  · let U : Set Draw := {z | StrictOn (fun x => (Model.fromWeights (sgdInitialWeights true z)).logits
      1 (divisionPrompt x) (Fin.last 4)) divisionAnswer}
    have ho : IsOpen U := (strict_prompt_weights_open 1 divisionPrompt divisionAnswer).preimage
      (continuous_sgdInitialWeights true)
    obtain ⟨w,hw⟩ := Division.division_native_realization
    exact (ho.measure_pos μ ⟨w,hw⟩).trans_le (measure_mono fun z hz =>
      ⟨0,by simp [evaluationGrid],fun x _ => hz x⟩)
  · obtain ⟨V,ho,pos,bad⟩ := open_miss B rate hr variance hv
    have sub : hit B rate batch hidden ⊆ Vᶜ := by
      rintro z ⟨t,ht,win⟩ hV
      have time : t ≤ B := by
        rcases Finset.mem_insert.mp ht with rfl | h
        · exact le_rfl
        · exact Nat.le_of_lt_succ (Finset.mem_range.mp (Finset.mem_filter.mp h).1)
      apply bad z hV (batch z) t time ((Model.fromWeights (adamWRun 1 (batch z) rate z t).weights).hidden
        1 (divisionPrompt x) (Fin.last 4))
      simpa only [← answer,Model.logits] using win x hx
    apply (measure_mono sub).trans_lt
    rw [prob_compl_eq_one_sub ho.measurableSet]
    exact ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero pos.ne'

theorem tuned_wsd_finite_hit (B : ℕ) (batch : Draw → Batches) (hidden : Set DivisionInput)
    (x : DivisionInput) (hx : x ∈ hidden) (answer : divisionAnswer x = 32) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter Token 4 16 6 => (1/50 : ℝ≥0)^2)
    let p := μ (hit B (fun t => 3*wsdRate 20000 t) batch hidden)
    0 < p ∧ p < 1 := by
  apply finite_hit_probability B _ _ batch hidden x hx answer _ (by intro; norm_num)
  intro t
  obtain ⟨lo,hi⟩ := wsdRate_bounds 20000 t
  exact ⟨by linarith,by linarith⟩

end Transformer.Grokking.FiniteRecovery.Argand.OrdinaryDivision
