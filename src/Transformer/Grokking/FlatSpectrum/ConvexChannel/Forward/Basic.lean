module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward.Defs

/-!
# Exact information retained by the forward scan

Source: `review/flat_spectrum.md`, *Forward-only channel learning*, CF1-CF2.
The invariant describes precisely all decoders consistent with the observed
answers. Consequently a realizable candidate survives, every returned table
fits the observations, and an observed contradiction eliminates the key.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward

variable {K V C : Type*} [DecidableEq V] [DecidableEq C]

/-- A single update retains exactly the old decoders predicting this answer (CF2). -/
theorem allows_step (state : Option (Record V C)) (v : V) (c : C) (f : V → C) :
    Allows (step state v c) f ↔ Allows state f ∧ f v = c := by
  cases state with
  | none => simp [step, Allows]
  | some record =>
    cases hv : record v with
    | none =>
      simp only [step, hv, Allows]
      constructor
      · intro h
        refine ⟨?_, ?_⟩
        · intro w a hw
          by_cases he : w = v
          · subst w
            simp [hv] at hw
          · exact h w a (by simpa [Function.update_of_ne he] using hw)
        · exact h v c (Function.update_self _ _ _)
      · rintro ⟨h, hc⟩ w a hw
        by_cases he : w = v
        · subst w
          simp only [Function.update_self, Option.some.injEq] at hw
          exact hc.trans hw
        · exact h w a (by simpa [Function.update_of_ne he] using hw)
    | some previous =>
      by_cases he : previous = c
      · subst c
        simp only [step, hv, Allows]
        exact ⟨fun h => ⟨h, h v previous hv⟩, And.left⟩
      · simp only [step, hv, ite_eq_right he, Allows]
        constructor
        · exact False.elim
        · rintro ⟨h, hc⟩
          exact he ((h v previous hv).symm.trans hc)

/-- The left-to-right scan retains exactly the decoders fitting every sample (CF2). -/
theorem allows_scan (k : K) (samples : List (Sample K V C))
    (state : Option (Record V C)) (f : V → C) :
    Allows (scan k samples state) f ↔
      Allows state f ∧ ∀ s ∈ samples, f (s.1 k) = s.2 := by
  induction samples generalizing state with
  | nil => simp [scan]
  | cons s rest ih =>
    rw [scan, ih, allows_step]
    simp only [List.mem_cons, forall_eq_or_imp]
    exact and_assoc

/-- Starting from an empty table adds no prior restriction on the decoder (CF2). -/
theorem allows_trainKey (k : K) (samples : List (Sample K V C)) (f : V → C) :
    Allows (trainKey k samples) f ↔ ∀ s ∈ samples, f (s.1 k) = s.2 := by
  unfold trainKey
  rw [allows_scan]
  simp [Allows, Extends]

omit [DecidableEq V] [DecidableEq C] in
/-- Filling unseen values preserves every recorded answer (CF2). -/
theorem extends_decode (record : Record V C) (default : C) :
    Extends record (decode record default) := by
  intro v c hc
  simp [decode, hc]

/-- A surviving table gives the right answer on each observed dictionary (CF2). -/
theorem trainKey_sound (k : K) (samples : List (Sample K V C))
    (default : C) {record : Record V C} (h : trainKey k samples = some record) :
    ∀ s ∈ samples, decode record default (s.1 k) = s.2 := by
  apply (allows_trainKey k samples _).mp
  rw [h]
  exact extends_decode record default

/-- Any key whose values determine the observed answers survives the scan (CF2). -/
theorem trainKey_complete (k : K) (samples : List (Sample K V C)) (f : V → C)
    (h : ∀ s ∈ samples, f (s.1 k) = s.2) :
    ∃ record, trainKey k samples = some record := by
  have ha := (allows_trainKey k samples f).mpr h
  cases hs : trainKey k samples with
  | none => simp [hs, Allows] at ha
  | some record => exact ⟨record, rfl⟩

/-- Two conflicting labels at the same candidate value eliminate that key (CF2). -/
theorem trainKey_conflict (k : K) (samples : List (Sample K V C)) (default : C)
    {s t : Sample K V C} (hs : s ∈ samples) (ht : t ∈ samples)
    (hv : s.1 k = t.1 k) (hc : s.2 ≠ t.2) : trainKey k samples = none := by
  cases h : trainKey k samples with
  | none => rfl
  | some record =>
    have hfit := trainKey_sound k samples default h
    exact False.elim (hc ((hfit s hs).symm.trans ((congrArg _ hv).trans (hfit t ht))))

example := trainKey_sound false [(id, false)] false
  (record := Function.update (fun _ : Bool => none) false (some false)) (by rfl)

example := trainKey_complete false [(id, false)] (id : Bool → Bool) (by simp)

example := trainKey_conflict false [(id, false), (id, true)] false
  (s := (id, false)) (t := (id, true)) (by simp) (by simp) rfl (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Forward
