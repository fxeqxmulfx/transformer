module

public import Transformer.CRASP.PositionalStateCounts

/-!
# Counting layers with arbitrary local normalization and multiple heads

Extension of arXiv:2506.16055v3, Appendix B.2 and Appendix F, to the source
architecture in `review/flat_spectrum.md`, *Depth, width and precision*, R3.
All heads and value coordinates are counted in parallel. The local map may
include RMSNorm, XSA, SwiGLU, residual additions and projection. It is a function
on finite states, not an additional counting operation. A zero denominator
produces zero, rather than the paper's uniform-attention fallback.
-/

@[expose] public section

namespace Transformer.CRASP.Normalized

universe u v w
variable {Q : Type u} {C : Type v} {σ : Type w} {p s k : ℕ}

/-- Finite tables for rounded attention, followed by an arbitrary local map. -/
structure Layer (Q : Type u) (C : Type v) (p s : ℕ) where
  denominator : Q → C → Q → Fx p s
  numerator : Q → C → Q → Fx p s
  finish : Q → (C → Fx p s) → Q

/-- Nonnegative rounded softmax coefficients; no condition on value signs. -/
def Layer.Nonnegative (L : Layer Q C p s) : Prop :=
  ∀ q c a, 0 ≤ (L.denominator q c a).m

/-- Each coordinate sums over the entire prefix before rounding the quotient. -/
noncomputable def Layer.aggregate (L : Layer Q C p s) (bos : Q)
    (q : ℕ → Q) (i : ℕ) (c : C) : Fx p s :=
  CountCells.value bos (L.denominator (q i) c) (L.numerator (q i) c)
    (fun _ => 0) q i

/-- A parallel attention layer and its local postprocessing. -/
noncomputable def Layer.step (L : Layer Q C p s) (bos : Q)
    (q : ℕ → Q) (i : ℕ) : Q := L.finish (q i) (L.aggregate bos q i)

variable [Fintype Q] [Fintype C]

open scoped Classical in
/-- Enumerate the query and all rounded head coordinates, then test the local map. -/
noncomputable def Layer.formula (L : Layer Q C p s) (bos : Q)
    (ψ : Q → FormP σ) (r : Q) : FormP σ :=
  FormP.any (Finset.univ.toList.map fun z : Q × (C → Fx p s) =>
    .and (ψ z.1) (.and (FormP.truth (decide (L.finish z.1 z.2 = r)))
      (FormP.all (Finset.univ.toList.map fun c : C =>
        CountCells.stateFormula bos (L.denominator z.1 c) (L.numerator z.1 c)
          (fun _ => 0) ψ (z.2 c)))))

/-- Parallel heads and arbitrary finite local maps cost at most one counting
level. Extension of arXiv:2506.16055v3, Appendix B.2, `eq:phi_h`. -/
theorem Layer.formula_mem (L : Layer Q C p s) (bos : Q)
    {ψ : Q → FormP σ} (hψ : ∀ q, ψ q ∈ TLClMod σ k) (r : Q) :
    L.formula bos ψ r ∈ TLClMod σ (k + 1) := by
  classical
  apply FormP.any_mem
  rintro φ hφ
  obtain ⟨z, _, rfl⟩ := List.mem_map.mp hφ
  apply FormP.and_mem (FormP.mem_mono (hψ z.1) (Nat.le_succ k))
  apply FormP.and_mem (FormP.truth_mem _ _)
  apply FormP.all_mem
  rintro φ hφ
  obtain ⟨c, _, rfl⟩ := List.mem_map.mp hφ
  exact CountCells.stateFormula_mem _ _ _ _ _ hψ _

variable [DecidableEq σ] [DecidableEq Q]

/-- The counting formula computes the actual finite-state layer, including
every head coordinate and its local nonlinear map (Appendix B.2/F extension). -/
theorem Layer.sat_formula (L : Layer Q C p s) (bos : Q) (hL : L.Nonnegative)
    (ψ : Q → FormP σ) (q : ℕ → Q) (w : List σ) {i : ℕ} (hi : 0 < i)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a, (ψ a).sat w j = decide (q j = a))
    (r : Q) : (L.formula bos ψ r).sat w i = true ↔ L.step bos q i = r := by
  classical
  have hcur := hψ i (Finset.mem_Icc.mpr ⟨hi, le_rfl⟩)
  have hcell (a : Q) (c : C) (y : Fx p s) :=
    CountCells.sat_stateFormula bos (L.denominator a c) (L.numerator a c)
      (fun _ => 0) (hL a c) ψ q w hi hψ y
  rw [Layer.formula, FormP.sat_any]
  constructor
  · rintro ⟨φ, hφ, hsat⟩
    obtain ⟨z, _, rfl⟩ := List.mem_map.mp hφ
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, decide_eq_true_eq] at hsat
    have hq : q i = z.1 := of_decide_eq_true (by rw [← hcur]; exact hsat.1)
    have ha : L.aggregate bos q i = z.2 := by
      funext c
      have hc := (FormP.sat_all _ w i).mp hsat.2.2 _
        (List.mem_map_of_mem (Finset.mem_toList.mpr (Finset.mem_univ c)))
      have he := (hcell z.1 c (z.2 c)).mp hc
      simpa only [Layer.aggregate, CountCells.value, hq] using he
    rw [Layer.step, hq, ha]
    exact hsat.2.1
  · intro hr
    have hcoords : (FormP.all (Finset.univ.toList.map fun c : C =>
        CountCells.stateFormula bos (L.denominator (q i) c) (L.numerator (q i) c)
          (fun _ => 0) ψ (L.aggregate bos q i c))).sat w i = true := by
      apply (FormP.sat_all _ w i).mpr
      rintro φ hφ
      obtain ⟨c, _, rfl⟩ := List.mem_map.mp hφ
      exact (hcell (q i) c _).mpr rfl
    refine ⟨_, List.mem_map_of_mem (a := (q i, L.aggregate bos q i))
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, decide_eq_true_eq]
    exact ⟨by rw [hcur]; simp, hr, hcoords⟩

private def witness : Layer Bool Unit 2 0 :=
  ⟨fun _ _ _ => 0, fun _ _ _ => 0, fun q _ => q⟩

/-- The nonnegativity, prefix agreement and depth assumptions are inhabited. -/
example : witness.Nonnegative ∧ 0 < 1 ∧
    (∀ a : Bool, (FormP.sym a) ∈ TLClMod Bool 0) ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Bool,
      (FormP.sym a).sat [true] j = decide (true = a)) := by
  refine ⟨by simp [Layer.Nonnegative, witness], by decide, ?_, ?_⟩
  · intro a; exact ⟨rfl, le_rfl⟩
  · intro j hj a
    have : j = 1 := by have := Finset.mem_Icc.mp hj; omega
    subst j
    simp [FormP.sat]

end Transformer.CRASP.Normalized

noncomputable section

namespace Transformer.CRASP.Normalized

universe u v
variable {Q : Type u} {σ : Type v} [Fintype Q]

open scoped Classical in
/-- Enumerate precisely the hidden states accepted by a local readout test. -/
def outputFormula (accept : Q → Bool) (ψ : Q → FormP σ) : FormP σ :=
  FormP.any (Finset.univ.toList.map fun r : Q =>
    .and (ψ r) (FormP.truth (accept r)))

/-- Final local decoding adds no counting level (B.2/F extension). -/
theorem outputFormula_mem (accept : Q → Bool) (ψ : Q → FormP σ) {k : ℕ}
    (hψ : ∀ r, ψ r ∈ TLClMod σ k) : outputFormula accept ψ ∈ TLClMod σ k := by
  apply FormP.any_mem
  rintro φ hφ
  obtain ⟨r, _, rfl⟩ := List.mem_map.mp hφ
  exact FormP.and_mem (hψ r) (FormP.truth_mem _ _)

variable [DecidableEq Q] [DecidableEq σ]

/-- The compiled output equals the actual local decision, including ties. -/
theorem sat_outputFormula (accept : Q → Bool) (ψ : Q → FormP σ)
    (w : List σ) (i : ℕ) (q : Q)
    (hψ : ∀ r, (ψ r).sat w i = decide (q = r)) :
    (outputFormula accept ψ).sat w i = accept q := by
  classical
  rw [Bool.eq_iff_iff, outputFormula, FormP.sat_any]
  constructor
  · rintro ⟨φ, hφ, hs⟩
    obtain ⟨r, _, rfl⟩ := List.mem_map.mp hφ
    simp only [FormP.sat, Bool.and_eq_true, FormP.sat_truth, hψ,
      decide_eq_true_eq] at hs
    exact hs.1 ▸ hs.2
  · intro hq
    refine ⟨_, List.mem_map_of_mem (a := q)
      (Finset.mem_toList.mpr (Finset.mem_univ _)), ?_⟩
    simp only [FormP.sat, FormP.sat_truth, hψ, decide_true, Bool.true_and, hq]


end Transformer.CRASP.Normalized
