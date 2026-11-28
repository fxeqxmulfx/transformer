module

public import Transformer.Grokking.FiniteRecovery.FirstHitSymmetry
public import Transformer.Grokking.FiniteRecovery.NormalizedRecovery

/-! A five-cell staircase identifies a rank-one phase table with nine
entries. Two hidden inputs need different, unobserved answer tokens.
Ordinary AdamW and its normalized form have 53 × 52 equally likely
ordered pairs of these winners at each evaluation. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

/-- The native rule also evaluates the cap when it falls between two
regular checks. This upper bound includes initialization and that endpoint. -/
theorem evaluationGrid_card_bound {interval : ℕ} (hi : 0 < interval) (cap : ℕ) :
    (evaluationGrid interval cap).card ≤ cap / interval + 2 := by
  let F := (Finset.range (cap+1)).filter fun t => interval ∣ t
  let multiples := (Finset.range (cap/interval+1)).image fun k => interval*k
  have hs : F ⊆ multiples := by
    intro t ht
    obtain ⟨hc,k,rfl⟩ := Finset.mem_filter.mp ht
    apply Finset.mem_image.mpr
    refine ⟨k,Finset.mem_range.mpr (Nat.lt_succ_of_le ((Nat.le_div_iff_mul_le hi).2 ?_)),rfl⟩
    simpa [Nat.mul_comm] using Nat.le_of_lt_succ (Finset.mem_range.mp hc)
  have hc := (Finset.card_le_card hs).trans Finset.card_image_le
  simp only [Finset.card_range] at hc
  exact (Finset.card_insert_le cap F).trans (Nat.add_le_add_right hc 1)

namespace Argand.TwoHiddenAnswers

open MeasureTheory Set Matrix FlatSpectrum
open scoped NNReal

abbrev Token := Fin 64
abbrev Cell := Fin 3 × Fin 3
local notation "Draw" => (ComplexParameter Token 4 16 6 → ℂ)

def label (x : Cell) : Token := ⟨(65+x.1.val-x.2.val)%64, Nat.mod_lt _ (by decide)⟩
def token (x : Cell) : Token := ⟨8+3*x.1.val+x.2.val,by omega⟩
def prompt (x : Cell) : Word Token 1 := fun _ => token x
def observedCell : Fin 5 → Cell := ![(0,0),(1,1),(2,2),(0,1),(1,2)]
def observation (j : Fin 5) : LabeledPosition Token := ⟨0,prompt (observedCell j),0,label (observedCell j)⟩
def probe : Fin 2 → Cell := ![(1,0),(2,0)]
def pairLabels (p : Token × Token) : Fin 2 → Token := ![p.1,p.2]
def used : Finset Token := {0,1,8,9,10,11,12,13,14,15,16}
def unused : Finset Token := Finset.univ \ used

theorem phase_recovered : NuclearRecovers (range observedCell)
    (fun i j => (ZMod.stdAddChar ((label (i,j)).val : ZMod 64) : ℂ)) := by
  have hs : range observedCell = staircase 2 := by
    ext ⟨i,j⟩
    simp only [staircase,mem_range,mem_union]
    fin_cases i <;> fin_cases j <;> decide
  let u (i : Fin 3) := (ZMod.stdAddChar (i.val : ZMod 64) : ℂ)
  let v (j : Fin 3) := (ZMod.stdAddChar (1-(j.val : ZMod 64)) : ℂ)
  have he : (fun i j => (ZMod.stdAddChar ((label (i,j)).val : ZMod 64) : ℂ)) = vecMulVec u v := by
    funext i j
    have ex : ((label (i,j)).val : ZMod 64) = (i.val : ZMod 64) + (1-(j.val : ZMod 64)) := by
      fin_cases i <;> fin_cases j <;> decide
    simp [ex,vecMulVec_apply,u,v,AddChar.map_add_eq_mul]
  rw [hs,he]
  exact forall_isMinOn_vecMulVec_staircase (fun i j => by simp [vecMulVec_apply,u,v])

theorem representable : ∃ w : Weights Token 4 16 6, StrictOn
    (fun x => (Model.fromWeights w).logits 1 (prompt x) 0) label := by
  classical
  have hinj : Function.Injective token := by
    intro ⟨i,j⟩ ⟨k,l⟩ h
    have hval := congrArg Fin.val h
    change 8+3*i.val+j.val=8+3*k.val+l.val at hval
    exact Prod.ext (Fin.ext (show i.val = k.val by omega)) (Fin.ext (show j.val = l.val by omega))
  let next := Function.extend token label (fun _ => 0)
  obtain ⟨w,hw⟩ := sixLayer_transition_realization (T := 0) next
  refine ⟨w,fun x a ha => ?_⟩
  have he : transitionTask (T := 0) next (prompt x) = label x := hinj.extend_apply label (fun _ => 0) x
  simpa only [he,Fin.last_zero] using hw (prompt x) a (fun h => ha (h.trans he))

private theorem move_pair (p : Token × Token) (hp : p ∈ unused.offDiag) :
    ∃ e : Equiv.Perm Token, (∀ k, e (label (probe k)) = pairLabels p k) ∧ ∀ x ∈ used, e x = x := by
  obtain ⟨h0,h1,hne⟩ := Finset.mem_offDiag.mp hp
  let a := Equiv.swap (2 : Token) p.1
  let b := Equiv.swap (a 3) p.2
  have fixed {x : Token} (hx : x ∈ used) : a x = x := Equiv.swap_apply_of_ne_of_ne
    (fun h => (by decide : (2 : Token) ∉ used) (h ▸ hx)) (fun h => (Finset.mem_sdiff.mp h0).2 (h ▸ hx))
  have hfirst : b p.1 = p.1 := Equiv.swap_apply_of_ne_of_ne
    (fun h => (by decide : (2 : Token) ≠ 3) (a.injective ((Equiv.swap_apply_left ..).trans h))) hne
  refine ⟨a.trans b,?_,fun x hx => ?_⟩
  · intro k
    fin_cases k
    · exact (congrArg b (Equiv.swap_apply_left ..)).trans hfirst
    · exact Equiv.swap_apply_left ..
  · change b (a x) = x
    rw [fixed hx]
    exact Equiv.swap_apply_of_ne_of_ne (fun h => (by decide : (3 : Token) ∉ used)
      ((a.injective ((fixed hx).trans h)) ▸ hx)) (fun h => (Finset.mem_sdiff.mp h1).2 (h ▸ hx))

def batch (stream : ℕ → List (Fin 5)) (t : ℕ) := (stream t).map observation

noncomputable def score (stream : ℕ → List (Fin 5)) (z : Draw) (t : ℕ) (x : Cell) : Token → ℝ :=
  (Model.fromWeights (normalizedAdamRun 1 (batch stream) (fun k => 3*wsdRate 20000 k) z t).weights).logits 1 (prompt x) 0

/-- Strict success on just the two designated hidden inputs has at most
the number of checks divided by 2,756 as its mass, for any batch stream. -/
theorem two_hit_bound (stream : ℕ → List (Fin 5)) (grid : Finset ℕ) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter Token 4 16 6 => (1/50 : ℝ≥0)^2)
    μ.real {z | ∃ t ∈ grid, StrictOn (fun k => score stream z t (probe k)) (label ∘ probe)} ≤ (grid.card : ℝ)/2756 := by
  intro μ
  let W t p := {z : Draw | StrictOn (fun k => score stream z t (probe k)) (pairLabels p)}
  have labels : label ∘ probe = pairLabels (2,3) := by funext k; fin_cases k <;> decide
  rw [labels]
  have card : unused.offDiag.card = 2756 := by decide +kernel
  rw [← show (unused.offDiag.card : ℝ) = 2756 by exact_mod_cast card]
  apply first_hit_mass_le_checks_div_card μ grid unused.offDiag ⟨(2,3),by decide⟩ (2,3) W
  · intro t p
    simpa only [W,score,normalizedAdam_weights,Fin.last_zero,Function.comp_apply] using (adamW_strict_open (H := 4) (d := 16) (L := 6)
      1 t (fun k => 3*wsdRate 20000 k) (batch stream) (prompt ∘ probe) (pairLabels p)).measurableSet
  · intro t p _ q _ hne
    apply Set.disjoint_left.mpr
    intro z h g
    have eq (k : Fin 2) : pairLabels p k = pairLabels q k := by
      by_contra hn
      exact lt_asymm (h k _ (Ne.symm hn)) (g k _ hn)
    exact hne (Prod.ext (eq 0) (eq 1))
  · intro t p hp
    obtain ⟨e,answers,fix⟩ := move_pair p hp
    have token_used (x : Cell) : token x ∈ used := by obtain ⟨i,j⟩ := x; fin_cases i <;> fin_cases j <;> decide
    have prompt_eq (x : Cell) : e ∘ prompt x = prompt x := funext fun _ => fix _ (token_used x)
    have obs (j : Fin 5) : relabelPosition e (observation j) = observation j := by
      have hy : label (observedCell j) ∈ used := by fin_cases j <;> decide
      simp only [observation,relabelPosition,prompt_eq,fix _ hy]
    have batches : (fun k => (batch stream k).map (relabelPosition e)) = batch stream := by
      funext k
      simp only [batch,List.map_map,show relabelPosition e ∘ observation = observation from funext obs]
    have perm (z : Draw) (k : Fin 2) (a : Token) : score stream (relabelDraw e z) t (probe k) (e a) =
        score stream z t (probe k) a := by
      simp only [score,normalizedAdam_weights]
      simpa only [batches,prompt_eq] using adamW_logits_relabel e 1 t (batch stream) (fun k => 3*wsdRate 20000 k)
        z (prompt (probe k)) 0 a
    have pre : (relabelDraw e) ⁻¹' W t p = W t (2,3) := by
      ext z
      change (∀ k b, b ≠ pairLabels p k → _ < _) ↔ ∀ k b, b ≠ pairLabels (2,3) k → _ < _
      have target (k : Fin 2) : e (pairLabels (2,3) k) = pairLabels p k := by rw [← labels]; exact answers k
      constructor
      · intro h k b hb
        have v := h k (e b) (fun he => hb (e.injective (he.trans (target k).symm)))
        simpa only [← target,perm] using v
      · intro h k v hv
        obtain ⟨b,rfl⟩ := e.surjective v
        simp only [← target,perm]
        exact h k b (fun he => hv ((congrArg e he).trans (target k)))
    exact (gaussian_relabel_preimage e (fun _ => (1/50 : ℝ≥0)^2) (W t p)).trans (congrArg μ pre)

/-- Even 200,000 updates and stopping at the first perfect evaluation
have success probability below one half on this identifiable table. -/
theorem hidden_hit_200000 (stream : ℕ → List (Fin 5)) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter Token 4 16 6 => (1/50 : ℝ≥0)^2)
    μ.real {z | ∃ t ∈ evaluationGrid 256 200000, ∀ x ∉ range observedCell, ∀ a, a ≠ label x →
        score stream z t x a < score stream z t x (label x)} ≤ 783/2756 := by
  intro μ
  have hsub : {z | ∃ t ∈ evaluationGrid 256 200000, ∀ x ∉ range observedCell, ∀ a, a ≠ label x →
      score stream z t x a < score stream z t x (label x)} ⊆ {z | ∃ t ∈ evaluationGrid 256 200000,
      StrictOn (fun k => score stream z t (probe k)) (label ∘ probe)} := by
    rintro z ⟨t,ht,hw⟩
    refine ⟨t,ht,fun k => hw (probe k) ?_⟩
    have hne : ∀ i : Fin 2, ∀ j : Fin 5, observedCell j ≠ probe i := by decide
    exact fun ⟨j,h⟩ => hne k j h
  apply (measureReal_mono (μ := μ) hsub).trans ((two_hit_bound stream _).trans ?_)
  have h : (evaluationGrid 256 200000).card ≤ 783 := evaluationGrid_card_bound (by decide) _
  exact div_le_div_of_nonneg_right (by exact_mod_cast h) (by norm_num)

example : (783 : ℝ)/2756 < 1/2 := by norm_num

end Argand.TwoHiddenAnswers
end Transformer.Grokking.FiniteRecovery
