module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Counts
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Learning

/-!
# Input-only paired features and the forward selection boundary

Source: `review/flat_spectrum.md`, CFt4. An ordered pair of visible categorical
slots is another fixed categorical feature. A single forward decoder of this
feature can express rules that no individual-slot decoder can fit. This keeps
the joint channel domain and ordinary CE convex; the discrete candidate search
is not a jointly convex optimization. Presence features ignore token position,
but preferring them is a prior, not a consequence of observed-label consistency.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C : Type*}

/-- Pair two supplied input slots without computing an output rule (CFt4). -/
def pairedData (D : K → V) : K × K → V × V := fun k => (D k.1, D k.2)

/-- A possible learned pair decoder, only a parameter witness (CFt4). -/
def pairedChannel [DecidableEq K] [DecidableEq C]
    (key : Q → K × K) (rule : Q → V × V → C) : Channel Q (K × K) (V × V) C :=
  vertex key (fun q _ v => rule q v)

/-- Any fixed two-slot rule has a feasible one-key representation after
pairing. The rule is not an input to the feature map or learner (CFt4). -/
theorem feasible_pairedChannel [Fintype K] [Fintype C] [DecidableEq K] [DecidableEq C]
    (key : Q → K × K) (rule : Q → V × V → C) : Feasible (pairedChannel key rule) := by
  exact feasible_vertex key (fun q _ v => rule q v)

/-- The actual logits of the paired witness are its one-hot decoded answer
(CFt4); no product of separately trained parameters occurs. -/
theorem logits_pairedChannel [Fintype K] [DecidableEq K] [DecidableEq C]
    (key : Q → K × K) (rule : Q → V × V → C) (q : Q) (D : K → V) (c : C) :
    logits (pairedChannel key rule) q (pairedData D) c =
      if c = rule q (D (key q).1, D (key q).2) then 1 else 0 := by
  simp [logits, pairedChannel, vertex, pairedData]
  split_ifs <;> simp_all

/-- Pairing changes only fixed inputs. Ordinary full-vocabulary CE stays
jointly convex in the complete channel tensor, CFt4. -/
theorem paired_answerLoss_convex [Fintype K] [Fintype C] {S : Type*} [Fintype S]
    (query : S → Q) (data : S → K → V) (labels : S → C) :
    ConvexOn ℝ Set.univ (answerLoss query (fun s => pairedData (data s)) labels) := by
  exact answerLoss_convex query (fun s => pairedData (data s)) labels

/-- Preserve every raw sample label while pairing its observed inputs (CFt4). -/
def pairedSamples (samples : List (Forward.Sample K V C)) :
    List (Forward.Sample (K × K) (V × V) C) :=
  samples.map (fun s => (pairedData s.1, s.2))

/-- A consistent pair in the fixed catalogue guarantees a successful actual
forward scan, without a CE derivative or a supplied decoder (CFt4). -/
theorem paired_learn_complete [DecidableEq V] [DecidableEq C]
    (keys : List (K × K)) (samples : List (Forward.Sample K V C)) (default : C)
    (pair : K × K) (rule : V × V → C) (hk : pair ∈ keys)
    (hfit : ∀ s ∈ samples, rule (s.1 pair.1, s.1 pair.2) = s.2) :
    ∃ k g, Forward.learn keys (pairedSamples samples) default = some (k, g) := by
  apply Forward.learn_complete keys (pairedSamples samples) default pair rule hk
  intro t ht
  obtain ⟨s, hs, rfl⟩ := List.mem_map.mp ht
  exact hfit s hs

/-- A returned pair predicts a covered value correctly if the observed labels
follow that pair's rule. Coverage concerns the actual selected pair; it does
not follow just from training fit or from the existence of another good pair. -/
theorem paired_prediction_of_coverage [DecidableEq V] [DecidableEq C]
    (keys : List (K × K)) (samples : List (Forward.Sample K V C)) (default : C)
    {k : K × K} {g : V × V → C}
    (h : Forward.learn keys (pairedSamples samples) default = some (k, g))
    (rule : V × V → C) (hfit : ∀ s ∈ samples, rule (pairedData s.1 k) = s.2)
    (D : K → V) (hcover : ∃ s ∈ samples, pairedData s.1 k = pairedData D k) :
    g (pairedData D k) = rule (pairedData D k) := by
  obtain ⟨s, hs, hv⟩ := hcover
  have hg := (Forward.learn_sound keys (pairedSamples samples) default h).2
    (pairedData s.1, s.2) (List.mem_map_of_mem hs)
  rw [← hv]
  exact hg.trans (hfit s hs).symm

/-- Four ordinary categorical examples; every two-slot output occurs (CFt4). -/
def pairWitness : List (Forward.Sample Bool Bool (Bool × Bool)) :=
  [(pairData false false, (false, false)), (pairData false true, (false, true)),
   (pairData true false, (true, false)), (pairData true true, (true, true))]

/-- Neither individual key can explain the witness, although the paired
key yields a successful forward scan (CFt4). -/
theorem pairWitness_separates :
    (∀ k : Bool, Forward.trainKey k pairWitness = none) ∧
    (∃ k g, Forward.learn [(false, true)] (pairedSamples pairWitness) (false, false) =
      some (k, g)) := by
  constructor
  · intro k
    cases k <;> simp [Forward.trainKey, Forward.scan, Forward.step, pairWitness, pairData]
  · apply paired_learn_complete _ _ _ (false, true) id (by simp)
    intro s hs
    simp only [pairWitness, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl <;> rfl

/-- Visible token presence has no positions, labels or target rule (CFt4). -/
def presenceData [DecidableEq V] (xs : List V) : V → Bool := fun v => decide (v ∈ xs)

/-- Permuting the visible prefix preserves every presence coordinate (CFt4).
It need not preserve the target of an order-sensitive task. -/
theorem presenceData_perm [DecidableEq V] {xs ys : List V} (h : xs.Perm ys) :
    presenceData xs = presenceData ys := by
  funext v
  simp only [presenceData, h.mem_iff]

/-- Any fixed channel restricted to presence inputs inherits their invariance,
CFt4. The query is held fixed; this theorem does not identify the correct rule. -/
theorem presence_logits_perm [Fintype V] [DecidableEq V]
    (W : Channel Q V Bool C) (q : Q) {xs ys : List V} (h : xs.Perm ys) :
    logits W q (presenceData xs) = logits W q (presenceData ys) := by
  rw [presenceData_perm h]

/-- Two consistent candidate orders fit the same observations and disagree
at an unseen dictionary. Thus arbitrary order is not an identification theorem
(CFt4); preferring invariance is an explicit structural prior. -/
theorem candidate_order_obstruction :
    let samples : List (Forward.Sample Bool Bool Bool) :=
      [(fun _ => false, false), (fun _ => true, true)]
    ∃ f g, Forward.learn [false, true] samples false = some (false, f) ∧
      Forward.learn [true, false] samples false = some (true, g) ∧ f false ≠ g true := by
  dsimp only
  let samples : List (Forward.Sample Bool Bool Bool) :=
    [(fun _ => false, false), (fun _ => true, true)]
  have hf (k : Bool) : ∀ s ∈ samples, (id : Bool → Bool) (s.1 k) = s.2 := by
    intro s hs
    simp only [samples, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl <;> rfl
  obtain ⟨a, ha⟩ := Forward.trainKey_complete false samples id (hf false)
  obtain ⟨b, hb⟩ := Forward.trainKey_complete true samples id (hf true)
  refine ⟨Forward.decode a false, Forward.decode b false, ?_, ?_, ?_⟩
  · change Forward.learn [false, true] samples false = _
    simp only [Forward.learn, ha]
  · change Forward.learn [true, false] samples false = _
    simp only [Forward.learn, hb]
  · have h0 := Forward.trainKey_sound false samples false ha
      (fun _ => false, false) (by simp [samples])
    have h1 := Forward.trainKey_sound true samples false hb
      (fun _ => true, true) (by simp [samples])
    simpa only [h0, h1] using (Bool.false_ne_true)

example := feasible_pairedChannel (fun _ : Unit => (false, true))
  (fun _ (v : Bool × Bool) => v)

example := paired_learn_complete [(false, true)] pairWitness (false, false)
  (false, true) id (by simp)
  (by intro s hs
      simp only [pairWitness, List.mem_cons, List.not_mem_nil, or_false] at hs
      rcases hs with rfl | rfl | rfl | rfl <;> rfl)

example : ∃ k g,
    Forward.learn [(false, true)] (pairedSamples pairWitness) (false, false) = some (k, g) ∧
    g (pairedData (pairData false true) k) = (false, true) := by
  obtain ⟨k, g, h⟩ := pairWitness_separates.2
  have hk : k = (false, true) := by
    simpa using (Forward.learn_sound _ _ _ h).1
  subst k
  refine ⟨_, _, h, ?_⟩
  apply paired_prediction_of_coverage _ _ _ h id
  · intro s hs
    simp only [pairWitness, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl <;> rfl
  · exact ⟨(pairData false true, (false, true)), by simp [pairWitness], rfl⟩

example : presenceData [false, true] = presenceData [true, false] :=
  presenceData_perm (List.Perm.swap _ _ _)

example : logits (vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))
    () (presenceData [false, true]) =
    logits (vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))
      () (presenceData [true, false]) :=
  presence_logits_perm _ _ (List.Perm.swap _ _ _)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
