/-
# IC-EoT: convexity of recursive predictions

arXiv:2603.22095v2, §4.1, Corollary 2, Eqs. (33)–(39).
The induction retains affine control histories separately. Thus it needs
monotonicity only in predicted variables, as the paper states, rather than
the stronger monotonicity in every predictor input.
-/

import Transformer.ICEoT.Section4_RolloutModel

noncomputable section

namespace Transformer.ICEoT

variable {E : Type*} [AddCommGroup E] [Module ℝ E]
variable {P C O : Type*}

/-- The mixed composition rule used by §4.1, Corollary 2: predicted columns
are convex, control columns affine, and only the former need monotonicity. -/
theorem selective_composition (f : ((P → ℝ) × (C → ℝ)) → O → ℝ)
    (p : E → P → ℝ) (c : E → C → ℝ)
    (hf : ConvexOn ℝ Set.univ f) (hm : PredictedMonotone f)
    (hp : ComponentwiseConvex p) (hc : AffineCombinations c) :
    ComponentwiseConvex (fun x => f (p x, c x)) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  calc
    f (p (a • x + b • y), c (a • x + b • y)) ≤
        f (a • p x + b • p y, c (a • x + b • y)) :=
      hm _ _ _ (hp.2 hx hy ha hb hab)
    _ ≤ a • f (p x, c x) + b • f (p y, c y) := by
      rw [hc x y a b hab]
      exact hf.2 (Set.mem_univ (p x, c x)) (Set.mem_univ (p y, c y)) ha hb hab

example : PredictorConditions witnessPredictor ∧
    ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
    AffineCombinations (fun x : Sequence 1 1 => fun r : Fin 1 × (Bool × Fin 1) => x (r.1, r.2.2)) :=
  ⟨witnessPredictor_conditions, convexOn_id convex_univ, fun _ _ _ _ _ => rfl⟩

/-- The shift operation is a linear coordinate selection/concatenation;
§4.1, Eq. (35). -/
theorem shiftHistory_combination {n : ℕ} {J : Type*}
    (H K : (Fin (n + 1) × J) → ℝ) (r s : J → ℝ) (a b : ℝ) :
    shiftHistory (a • H + b • K) (a • r + b • s) =
      a • shiftHistory H r + b • shiftHistory K s := by
  funext ir
  by_cases hi : ir.1.val < n <;> simp [shiftHistory, hi]

/-- Convex predicted histories remain convex after Eq. (35), §4.1. -/
theorem shiftHistory_convex {n : ℕ} {J : Type*}
    (H : E → (Fin (n + 1) × J) → ℝ) (r : E → J → ℝ)
    (hH : ComponentwiseConvex H) (hr : ComponentwiseConvex r) :
    ComponentwiseConvex (fun x => shiftHistory (H x) (r x)) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab ir
  by_cases hi : ir.1.val < n
  · simpa [shiftHistory, hi] using hH.2 hx hy ha hb hab (⟨ir.1.val + 1, by omega⟩, ir.2)
  · simpa [shiftHistory, hi] using hr.2 hx hy ha hb hab ir.2

example : ComponentwiseConvex (id : Sequence 1 1 → Sequence 1 1) ∧
    ComponentwiseConvex (fun x : Sequence 1 1 => fun r : Fin 1 => x (0, r)) :=
  ⟨convexOn_id convex_univ, ⟨convex_univ, fun _ _ _ _ _ _ _ _ _ _ => le_rfl⟩⟩

/-- Affine control histories remain affine after Eq. (35), §4.1. -/
theorem shiftHistory_affine {n : ℕ} {J : Type*}
    (H : E → (Fin (n + 1) × J) → ℝ) (r : E → J → ℝ)
    (hH : AffineCombinations H) (hr : AffineCombinations r) :
    AffineCombinations (fun x => shiftHistory (H x) (r x)) := by
  intro x y a b hab
  dsimp only
  rw [hH x y a b hab, hr x y a b hab, shiftHistory_combination]

example : AffineCombinations (id : Sequence 1 1 → Sequence 1 1) ∧
    AffineCombinations (fun x : Sequence 1 1 => fun r : Fin 1 => x (0, r)) :=
  ⟨fun _ _ _ _ _ => rfl, fun _ _ _ _ _ => rfl⟩

/-- Selective stage-control expansion is affine in the complete finite
control sequence; §4.1, Eqs. (32), (33). -/
theorem extendedStage_affine {np u : ℕ} (k : ℕ) :
    AffineCombinations (fun U : Controls np u => extendControl (stageControl U k)) := by
  intro U V a b hab
  funext r
  rcases r with ⟨s, r⟩
  by_cases hk : k < np <;> cases s <;>
    simp [extendControl, stageControl, hk, add_comm]

/-- Induction on the actual history shift and neural substitution proves
every component of every predicted stage convex in the full control
sequence. Source: §4.1, Corollary 2, Eqs. (35), (36), (39). -/
theorem mpcRun_properties {np n y u : ℕ} (f : History n y u → Fin y → ℝ)
    (H0 : History n y u) (y0 : Fin y → ℝ) (hf : PredictorConditions f) (k : ℕ) :
    ComponentwiseConvex (fun U : Controls np u => (mpcRun f H0 y0 U k).predictedHistory) ∧
      AffineCombinations (fun U : Controls np u => (mpcRun f H0 y0 U k).controlHistory) ∧
      ComponentwiseConvex (fun U : Controls np u => (mpcRun f H0 y0 U k).prediction) := by
  induction k with
  | zero =>
    refine ⟨convexOn_const _ convex_univ, ?_, convexOn_const _ convex_univ⟩
    intro x y a b hab
    change H0.2 = a • H0.2 + b • H0.2
    rw [← add_smul, hab, one_smul]
  | succ k ih =>
    have hp := shiftHistory_convex _ _ ih.1 ih.2.2
    have hc := shiftHistory_affine _ _ ih.2.1 (extendedStage_affine k)
    exact ⟨hp, hc, selective_composition f _ _ hf.1 hf.2 hp hc⟩

example : PredictorConditions witnessPredictor := witnessPredictor_conditions

/-- Each scalar predicted temperature or electricity coordinate is convex;
§4.1, Corollary 2, Eqs. (37)–(39). -/
theorem mpc_prediction_convex {np n y u : ℕ} (f : History n y u → Fin y → ℝ)
    (H0 : History n y u) (y0 : Fin y → ℝ) (hf : PredictorConditions f)
    (k : ℕ) (r : Fin y) :
    ConvexOn ℝ Set.univ (fun U : Controls np u => (mpcRun f H0 y0 U k).prediction r) :=
  (componentwiseConvex_iff _).mp (mpcRun_properties f H0 y0 hf k).2.2 r

example : PredictorConditions witnessPredictor := witnessPredictor_conditions

end Transformer.ICEoT
