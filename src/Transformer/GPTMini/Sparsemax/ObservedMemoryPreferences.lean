import Transformer.GPTMini.Sparsemax.IncidentMemorySelection

/-!
# An additional convex geometry criterion computed from observations

Derived data criterion before arXiv:1602.02068v2, Eq. (1). Adjacent observed
prototypes provide preferred edge weight `scale/(1+distance)`. Independent
fixed query/key feature maps provide their preferred squared-norm additions.
Only adjacent distances and per-prototype feature energies are needed; no
dense reference Gram or target attention labels are supplied.

The metric, feature maps, prototype order and scale are explicit choices.
They are held fixed during the convex optimization. Their data-derived
reference need not satisfy the structural domain; unique constrained
projection still exists. Positive scale and nonnegative distances prefer
positive edges, with stronger preferences for closer observed neighbors.
This selects otherwise equivalent geometry, not semantic output-only routes.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Preferred compact geometry from adjacent distances and separate observed feature energies.
Source: the derived additional data criterion before arXiv:1602.02068v2, Eq. (1).
Fixed data feature maps are inputs, not claimed to be learned jointly. -/
def observedMemoryReference {Key : Type*} {N Q K : ℕ} (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queryFeatures : Key → Fin Q → ℝ)
    (keyFeatures : Key → Fin K → ℝ) (scale : ℝ) : LocalMemoryParameters N :=
  ((fun e => scale / (1 + distance (prototypes e.castSucc) (prototypes e.succ))),
    Sum.elim (fun i => ∑ d, (queryFeatures (prototypes i) d) ^ 2)
      (fun j => ∑ d, (keyFeatures (prototypes j) d) ^ 2))

/-- Nonnegative scale and observed edge distance imply a nonnegative preferred edge.
Source: the observation-derived criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryReference_edge_nonneg {Key : Type*} {N Q K : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (queryFeatures : Key → Fin Q → ℝ) (keyFeatures : Key → Fin K → ℝ) (scale : ℝ)
    (hs : 0 ≤ scale) (e : Fin N)
    (hd : 0 ≤ distance (prototypes e.castSucc) (prototypes e.succ)) :
    0 ≤ (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).1 e := by
  exact div_nonneg hs (by linarith)

/-- Actual unequal observations and nonconstant features inhabit edge nonnegativity. -/
example : 0 ≤ (observedMemoryReference (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (j.val : ℝ)) (fun x => fun _ : Fin 1 => x / 4)
    (fun x => fun _ : Fin 1 => x / 8) (1 / 8)).1 0 :=
  observedMemoryReference_edge_nonneg _ _ _ _ _ (by norm_num) 0 (by norm_num)

/-- Positive scale gives a positive edge preference at every nonnegative finite distance.
Source: the data-derived compact geometry criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryReference_edge_pos {Key : Type*} {N Q K : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (queryFeatures : Key → Fin Q → ℝ) (keyFeatures : Key → Fin K → ℝ) (scale : ℝ)
    (hs : 0 < scale) (e : Fin N)
    (hd : 0 ≤ distance (prototypes e.castSucc) (prototypes e.succ)) :
    0 < (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).1 e := by
  exact div_pos hs (by linarith)

/-- A positive data preference has concrete admissible scale and distance premises. -/
example : 0 < (observedMemoryReference (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (j.val : ℝ)) (fun x => fun _ : Fin 1 => x / 4)
    (fun x => fun _ : Fin 1 => x / 8) (1 / 8)).1 0 :=
  observedMemoryReference_edge_pos _ _ _ _ _ (by norm_num) 0 (by norm_num)

/-- Nonnegative observation distances bound preferred edges by the declared scale.
Source: the explicit reciprocal-distance data criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryReference_edge_le_scale {Key : Type*} {N Q K : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (queryFeatures : Key → Fin Q → ℝ) (keyFeatures : Key → Fin K → ℝ) (scale : ℝ)
    (hs : 0 ≤ scale) (e : Fin N)
    (hd : 0 ≤ distance (prototypes e.castSucc) (prototypes e.succ)) :
    (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).1 e ≤ scale := by
  exact div_le_self hs (by linarith)

/-- A genuine data edge preference lies below its positive scale. -/
example : (observedMemoryReference (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (j.val : ℝ)) (fun x => fun _ : Fin 1 => x / 4)
    (fun x => fun _ : Fin 1 => x / 8) (1 / 8)).1 0 ≤ 1 / 8 :=
  observedMemoryReference_edge_le_scale _ _ _ _ _ (by norm_num) 0 (by norm_num)

/-- Closer observed adjacent pairs receive no smaller preferred edge weights.
Source: the explicit data geometry preference for arXiv:1602.02068v2, Eq. (1).
This compares preferences; coupling constraints may affect their projected values. -/
theorem observedMemoryReference_edge_antitone {Key : Type*} {N Q K : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (queryFeatures : Key → Fin Q → ℝ) (keyFeatures : Key → Fin K → ℝ) (scale : ℝ)
    (hs : 0 ≤ scale) (e f : Fin N)
    (hd : 0 ≤ distance (prototypes e.castSucc) (prototypes e.succ))
    (hef : distance (prototypes e.castSucc) (prototypes e.succ) ≤
      distance (prototypes f.castSucc) (prototypes f.succ)) :
    (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).1 f ≤
      (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).1 e := by
  exact div_le_div_of_nonneg_left hs (by linarith) (by linarith)

/-- Unequal neighbor distances and nonconstant feature maps inhabit the preference-order premises. -/
example : (observedMemoryReference (fun x y : ℝ => |x - y|)
    (fun j : Fin 3 => if j = 2 then (3 : ℝ) else j.val)
    (fun x => fun _ : Fin 1 => x / 4) (fun x => fun _ : Fin 1 => x / 8) (1 / 8)).1 1 ≤
    (observedMemoryReference (fun x y : ℝ => |x - y|)
      (fun j : Fin 3 => if j = 2 then (3 : ℝ) else j.val)
      (fun x => fun _ : Fin 1 => x / 4) (fun x => fun _ : Fin 1 => x / 8) (1 / 8)).1 0 :=
  observedMemoryReference_edge_antitone _ _ _ _ _ (by norm_num) 0 1 (by norm_num) (by norm_num)

/-- Strictly closer adjacent observations give strictly stronger positive-scale preferences.
Source: the reciprocal-distance observation criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryReference_edge_strictAntitone {Key : Type*} {N Q K : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (queryFeatures : Key → Fin Q → ℝ) (keyFeatures : Key → Fin K → ℝ) (scale : ℝ)
    (hs : 0 < scale) (e f : Fin N)
    (hd : 0 ≤ distance (prototypes e.castSucc) (prototypes e.succ))
    (hef : distance (prototypes e.castSucc) (prototypes e.succ) <
      distance (prototypes f.castSucc) (prototypes f.succ)) :
    (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).1 f <
      (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).1 e := by
  exact div_lt_div_of_pos_left hs (by linarith) (by linarith)

/-- Distinct observation distances inhabit the strict preference-order premises. -/
example : (observedMemoryReference (fun x y : ℝ => |x - y|)
    (fun j : Fin 3 => if j = 2 then (3 : ℝ) else j.val)
    (fun x => fun _ : Fin 1 => x / 4) (fun x => fun _ : Fin 1 => x / 8) (1 / 8)).1 1 <
    (observedMemoryReference (fun x y : ℝ => |x - y|)
      (fun j : Fin 3 => if j = 2 then (3 : ℝ) else j.val)
      (fun x => fun _ : Fin 1 => x / 4) (fun x => fun _ : Fin 1 => x / 8) (1 / 8)).1 0 :=
  observedMemoryReference_edge_strictAntitone _ _ _ _ _ (by norm_num) 0 1 (by norm_num) (by norm_num)

/-- Both preferred norm-addition families are nonnegative energies of observed data features.
Source: the independent data-derived Q/K preferences for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryReference_norm_nonneg {Key : Type*} {N Q K : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (queryFeatures : Key → Fin Q → ℝ) (keyFeatures : Key → Fin K → ℝ) (scale : ℝ)
    (x : Sum (Fin (N + 1)) (Fin (N + 1))) :
    0 ≤ (observedMemoryReference distance prototypes queryFeatures keyFeatures scale).2 x := by
  rcases x with i | j
  · change 0 ≤ ∑ d, (queryFeatures (prototypes i) d) ^ 2
    exact Finset.sum_nonneg (fun d _ => sq_nonneg (queryFeatures (prototypes i) d))
  · change 0 ≤ ∑ d, (keyFeatures (prototypes j) d) ^ 2
    exact Finset.sum_nonneg (fun d _ => sq_nonneg (keyFeatures (prototypes j) d))

/-- Observations determine a unique constrained minimizer without supplying target attention routes.
Source: the additional data criterion and proved compact domain for arXiv:1602.02068v2, Eq. (1).
The reference is not assumed feasible and no full model task objective is selected. -/
theorem observedMemoryReference_existsUnique {Key : Type*} {N Q K : ℕ}
    (distance : Key → Key → ℝ) (prototypes : Fin (N + 1) → Key)
    (queryFeatures : Key → Fin Q → ℝ) (keyFeatures : Key → Fin K → ℝ) (scale cap floor : ℝ)
    (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    ∃! p, p ∈ incidentMemoryParameterDomain N cap floor ∧
      IsMinOn (localMemoryQuadratic
        (observedMemoryReference distance prototypes queryFeatures keyFeatures scale))
        (incidentMemoryParameterDomain N cap floor) p :=
  incidentMemoryQuadratic_existsUnique cap floor _ hc hf

/-- Real data preferences outside the norm cap still have a unique constrained minimum. -/
example : ∃! p, p ∈ incidentMemoryParameterDomain 1 4 (3 / 4) ∧
    IsMinOn (localMemoryQuadratic (observedMemoryReference (fun x y : ℝ => |x - y|)
      (fun j : Fin 2 => (4 * j.val : ℝ)) (fun x => fun _ : Fin 1 => x)
      (fun x => fun _ : Fin 1 => x / 2) 1)) (incidentMemoryParameterDomain 1 4 (3 / 4)) p :=
  observedMemoryReference_existsUnique _ _ _ _ _ _ _ (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
