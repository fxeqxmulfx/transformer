module

public import Transformer.Phasor.TripleTransport.OrbitFit

/-!
# A cyclic fit and a dense successor at one learned gate

Source: the Phasor experiment section “Interpolation in a learned cyclic
order”. A fitted prototype fills a missing static proposal. A completed
dense-prefix match still supplies its actual successor. Both candidates
enter the same existing affine read gate.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.OrbitInduction
open DenseInduction OrbitInterpolation Polynomial

/-- All positions first offer the existing static candidate. A learned orbit
chart can fill an absent entry. A completed dense prefix donor can override
either candidate at the same structural gate. -/
def offer {A : Type*} (existing learned : ℕ → Option A) (t : ℕ) : Option A :=
  (existing t).or (learned t)

theorem short_offer {d : ℕ} (xs residual : ℕ → Fin d → ℝ)
    (existing learned : ℕ → Option (Fin d → ℝ)) (g : ℕ → ℝ) (t : ℕ) (c : Fin d)
    (early : t ≤ 1) :
    feature xs residual (offer existing learned) g t c =
      residual t c + g t * SharedStructuralGate.value id (offer existing learned t) c := by
  have empty : t-1=0 := by omega
  simp [feature, donor, empty, findDonor]

/-- If the already available static operations abstain and a learned
polynomial chart fits enough anchors, the second dense output uses its
unique predicted prototype. The same block still accepts arbitrary streams. -/
theorem chart_at_second {X F : Type*} [Field F] [DecidableEq X] {d : ℕ}
    (chart : X ≃ F) (anchors : Finset X) (seen truth : X → X) (poly : F[X])
    (small : poly.degree < anchors.card)
    (law : ∀ x, chart (truth x) = poly.eval (chart x))
    (agrees : ∀ x ∈ anchors, seen x = truth x) (q : X) (unit : X → Fin d → ℝ)
    (xs residual : ℕ → Fin d → ℝ) (existing learned : ℕ → Option (Fin d → ℝ))
    (g : ℕ → ℝ) (c : Fin d) (empty : existing 1 = none)
    (present : learned 1 = some (unit (read chart anchors seen q))) :
    feature xs residual (offer existing learned) g 1 c =
      residual 1 c + g 1 * unit (truth q) c := by
  rw [short_offer _ _ _ _ _ _ _ (by decide)]
  simp [offer, empty, present, recover chart anchors seen truth poly small law agrees q,
    SharedStructuralGate.value]

/-- Every latest completed dense match makes this coordinate independent
of both static tables. There is one forward rule and one learned multiplier. -/
theorem latest_override {d : ℕ} (xs residual : ℕ → Fin d → ℝ)
    (a b : ℕ → Option (Fin d → ℝ)) (g : ℕ → ℝ) (i t : ℕ) (c : Fin d)
    (past : i+1<t) (hit : accepts (xs t) (xs i) = true)
    (last : ∀ j, i < j → j < t-1 → accepts (xs t) (xs j) = false) :
    feature xs residual (offer a b) g t c = residual t c + g t * xs (i+1) c :=
  feature_of_latest xs residual (offer a b) g i t c past hit last

example : offer (fun _ => some false) (fun _ => some true) 1 = some false := by decide

example : feature (fun _ => axis (0 : Fin 2)) (fun _ _ => 0)
    (offer (fun _ => none) (fun _ => some (axis 1))) (fun _ => 1) 1 1 = 1 := by
  rw [short_offer _ _ _ _ _ _ _ (by decide)]
  norm_num [offer, SharedStructuralGate.value, axis]

end Transformer.Phasor.TripleTransport.OrbitInduction

namespace Transformer.Phasor.TripleTransport.OrbitInduction
open DenseInduction OrbitInterpolation Polynomial

/-- In a fixed learned chart, a query omitted from a redundant fit may train the
same gain used for novel queries. This holds at any prefix with no completed
dense match and no earlier static proposal. -/
theorem erased_fallback {X F : Type*} [Field F] [DecidableEq X] {d : ℕ}
    (chart : X ≃ F) (anchors : Finset X) (q : X) (stored other : X → X)
    (p : F[X]) (member : q ∈ anchors) (deg : p.degree < ↑(anchors.card-1))
    (fit : ∀ i ∈ anchors, chart (stored i) = p.eval (chart i))
    (others : ∀ i ∈ anchors.erase q, other i = stored i)
    (prototype : X → Fin d → ℝ) (xs residual : ℕ → Fin d → ℝ)
    (existing : ℕ → Option (Fin d → ℝ)) (g : ℕ → ℝ) (t : ℕ) (c : Fin d)
    (unmatched : donor accepts xs t = none) (absent : existing t = none) :
    feature xs residual (offer existing
      (fun _ => some (prototype (read chart (anchors.erase q) other q)))) g t c =
        residual t c + g t * prototype (stored q) c := by
  simp [feature, unmatched, offer, absent, SharedStructuralGate.value,
    read_after_erasing chart anchors q stored other p member deg fit others]

example (t : ℕ) : feature (fun _ _ => (0 : ℝ)) (fun _ _ => 0)
    (offer (fun _ => none) (fun _ => some (fun _ : Fin 1 =>
      (read (Equiv.refl ℝ) (({0,1,2} : Finset ℝ).erase 1)
        (fun x => if x=1 then 99 else x+2) 1)))) (fun _ => 1) t 0 = 3 := by
  have unmatched : donor accepts (fun (_ : ℕ) (_ : Fin 1) => (0 : ℝ)) t = none := by
    unfold donor
    generalize t-1 = b
    induction b with
    | zero => rfl
    | succ b ih => simpa [findDonor, accepts, eligible, product] using ih
  have val := erased_fallback (Equiv.refl ℝ) ({0,1,2} : Finset ℝ) 1
    (fun x => x+2) (fun x => if x=1 then 99 else x+2) (Polynomial.X+Polynomial.C (2 : ℝ))
    (by norm_num) (by apply lt_of_le_of_lt (Polynomial.degree_add_le ..); norm_num [Polynomial.degree_C])
    (by simp) (by intro i hi; simp [(Finset.mem_erase.mp hi).1])
    (fun v (_ : Fin 1) => v) (fun _ _ => 0) (fun _ _ => 0) (fun _ => none)
    (fun _ => 1) t 0 unmatched rfl
  norm_num at val ⊢
  exact val

end Transformer.Phasor.TripleTransport.OrbitInduction
namespace Transformer.Phasor.TripleTransport.OrbitInduction
open DenseInduction OrbitInterpolation Polynomial

/-- For fixed learned maps the complete mixture is causal when its residual,
gain and static proposals are causal. The statement allows arbitrary dense
sequences and arbitrary changes beyond the query. -/
theorem feature_congr {d : ℕ} (x y rx ry : ℕ → Fin d → ℝ)
    (a b a' b' : ℕ → Option (Fin d → ℝ)) (g g' : ℕ → ℝ) (t : ℕ) (c : Fin d)
    (same : ∀ i, i ≤ t → x i = y i) (residual : rx t = ry t)
    (gate : g t = g' t) (base : a t = a' t) (learned : b t = b' t) :
    feature x rx (offer a b) g t c = feature y ry (offer a' b') g' t c := by
  have selected := DenseInduction.prefix_congr accepts x y x y t same (fun i hi => same i hi.le)
  simp only [feature, offer, selected, residual, gate, base, learned]

example (x : ℕ → Fin 2 → ℝ) (t : ℕ) :
    feature x x (offer (fun _ => none) (fun i => some (x i))) (fun _ => 1) t 0 =
      feature (fun i => if i≤t then x i else axis 1) x
        (offer (fun _ => none) (fun i => some (x i))) (fun _ => 1) t 0 := by
  apply feature_congr
  · intro i hi
    simp [hi]
  all_goals rfl

example : ∃ learned : ℕ → Option (Fin 1 → ℝ),
    feature (fun _ _ => 0) (fun _ _ => 0) (offer (fun _ => none) learned)
      (fun _ => 1) 1 0 = 4 := by
  let s : Finset ℝ := {0,1,3}
  let chart := Equiv.refl ℝ
  let seen := fun x : ℝ => x^2
  let unit := fun (x : ℝ) (_ : Fin 1) => x
  let learned := fun _ : ℕ => some (unit (read chart s seen 2))
  refine ⟨learned, ?_⟩
  have h := chart_at_second (d := 1) chart s seen seen (Polynomial.X^2)
    (by norm_num [s, Polynomial.degree_X_pow]) (by simp [chart,seen]) (by simp)
    2 unit (fun _ _ => 0) (fun _ _ => 0) (fun _ => none) learned (fun _ => 1) (0 : Fin 1)
    (by rfl) (by rfl)
  simpa [unit,seen,show (2 : ℝ)^2 = 4 by norm_num] using h

example : feature (fun _ => axis (0 : Fin 2)) (fun _ _ => 0)
    (offer (fun _ => some (axis 1)) (fun _ => none)) (fun _ => 1) 2 0 = 1 := by
  rw [latest_override _ _ _ _ _ 0 2 _ (by omega) (by simp [accepts_axis]) (by omega)]
  norm_num [axis]

end Transformer.Phasor.TripleTransport.OrbitInduction
