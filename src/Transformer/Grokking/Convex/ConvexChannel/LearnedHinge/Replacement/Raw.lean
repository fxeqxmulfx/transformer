module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement.Basic
public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Projection.Raw

/-!
# Local replacement preserves every original raw-input constraint

Source: `review/convex/moments.md`, CM83. The old model satisfies the entire
raw-prefix domain. The replacement vector is checked against every original
oriented row of its chosen slot and component. These two facts establish
feasibility of the whole new model without rechecking unchanged vectors.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement

open PatternReduction Projection

variable {O J : Type*} [Fintype O] [DecidableEq O] [DecidableEq J] {T : ℕ}

/-- Exact component replacement retains the full complex activation cone,
CM83. All prefixes, classes, signs and both components remain quantified. -/
theorem replace_mem (p : Parameters J O (ObjectWeights O T))
    (mask : J → Pattern (Prefix O T)) (slot : J) (object : O) (sign part : Bool)
    (value : Coordinate O T → ℝ)
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hv : Feasible (fun x => signedRow (mask slot (x, part)) x) value) :
    replace p slot object sign part value ∈ complexDomain rawInput Set.univ mask := by
  intro j c s x hx
  rcases x with ⟨tokens, k⟩
  change 0 ≤ orient (mask j (tokens, k))
    (components rawInput (tokens, k) (replace p slot object sign part value j c s))
  rw [raw_components, replace_project]
  split_ifs with h
  · rcases h with ⟨rfl, rfl, rfl, rfl⟩
    simpa only [signedRow_dot] using hv tokens
  · have hh := hp j c s (tokens, k) hx
    change 0 ≤ orient (mask j (tokens, k)) (components rawInput (tokens, k) (p j c s)) at hh
    rw [raw_components] at hh
    exact hh

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement
