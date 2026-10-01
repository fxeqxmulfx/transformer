/-
# Lower bounds for periodic position encodings, including depth zero

arXiv:2506.16055v3, §4.5, `thm:rtfr_pes_depth_hierarchy`, and Appendix F.
The proved reverse simulations give the positive-depth lower bounds.
At depth zero, equal-length words with the same final letter have identical
outputs under every position encoding.
-/

import Transformer.CRASP.PeriodicEncodingsToLogic
import Transformer.CRASP.PeriodicOutputPermutation
import Transformer.CRASP.PositionalDepth
import Transformer.CRASP.AlibiToLogic

namespace Transformer.CRASP

/-- No depth-zero positional transformer recognizes `E_1`, for any encoding.
Source: arXiv:2506.16055v3, §4.5, depth-zero case of
`thm:rtfr_pes_depth_hierarchy`, not argued by the manuscript. -/
theorem not_recognizes_altPlusNeutral_one_zero {p s d : ℕ}
    (T : PTfr (Option (Option Bool)) p s d 0) :
    ¬ T.Recognizes (altPlusNeutral 1) := by
  have hy : [some false, some false] ∈ altPlusNeutral 1 := by
    change [false, false] ∈ altPlus false 1
    rw [altPlus_one]
    exact ⟨2, by omega, rfl⟩
  have hn : [some true, some false] ∉ altPlusNeutral 1 := by
    change [true, false] ∉ altPlus false 1
    rw [altPlus_one]
    rintro ⟨m, hm, heq⟩
    have h0 := congrArg (·[0]?) heq
    cases m <;> simp at h0
  have hout : T.out (bos [some false, some false]) =
      T.out (bos [some true, some false]) := by
    rw [PTfr.out_bos, PTfr.out_bos]
    change T.Wout (T.actAt ([some false] ++ [some false]) 0 (1 + 1)) =
      T.Wout (T.actAt ([some true] ++ [some false]) 0 (1 + 1))
    have hl := PeriodicAttention.initial_last T [some false] (some false)
    have hr := PeriodicAttention.initial_last T [some true] (some false)
    simp only [List.length_singleton] at hl hr
    rw [hl, hr]
  intro hrec
  have ha := (hrec _).mpr hy
  apply hn
  apply (hrec _).mp
  rwa [PTfr.Accepts, ← hout]

/-- Sinusoidal encodings at rational angles preserve the negative depth bound.
Source: arXiv:2506.16055v3, §4.5 and Appendix F, sinusoidal hierarchy. -/
theorem not_recognizes_altPlusNeutral_sinusoidal (k : ℕ) {p s d : ℕ}
    (θ : ℕ → ℝ) (hθ : (PosEnc.sinusoidal θ).RationalAngles)
    (T : PTfr (Option (Option Bool)) p s d k) (hT : T.pe = .sinusoidal θ) :
    ¬ T.Recognizes (altPlusNeutral (k + 1)) := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · exact not_recognizes_altPlusNeutral_one_zero T
  · intro hrec
    obtain ⟨φ, hφ, hlang⟩ := exists_mem_TLClMod_of_sinusoidal θ hθ T hT
    apply (definablePos_altPlusNeutral k hk).2
    exact ⟨φ, TLClMod_subset_TLClPos _ _ hφ, hlang.trans (Set.ext hrec)⟩

/-- RoPE at rational angles preserves the negative depth bound.
Source: arXiv:2506.16055v3, §4.5 and Appendix F, RoPE hierarchy. -/
theorem not_recognizes_altPlusNeutral_rope (k : ℕ) {p s d : ℕ}
    (θ : ℕ → ℝ) (hθ : (PosEnc.rope θ).RationalAngles)
    (T : PTfr (Option (Option Bool)) p s d k) (hT : T.pe = .rope θ) :
    ¬ T.Recognizes (altPlusNeutral (k + 1)) := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · exact not_recognizes_altPlusNeutral_one_zero T
  · intro hrec
    obtain ⟨φ, hφ, hlang⟩ := exists_mem_TLClMod_of_rope θ hθ T hT
    apply (definablePos_altPlusNeutral k hk).2
    exact ⟨φ, TLClMod_subset_TLClPos _ _ hφ, hlang.trans (Set.ext hrec)⟩

/-- ALiBi of any slope preserves the negative depth bound.
Source: arXiv:2506.16055v3, §4.5 and Appendix F, ALiBi hierarchy. -/
theorem not_recognizes_altPlusNeutral_alibi (k : ℕ) {p s d : ℕ} (a : ℝ)
    (T : PTfr (Option (Option Bool)) p s d k) (hT : T.pe = .alibi a) :
    ¬ T.Recognizes (altPlusNeutral (k + 1)) := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · exact not_recognizes_altPlusNeutral_one_zero T
  · intro hrec
    obtain ⟨φ, hφ, hlang⟩ := exists_mem_TLClY_of_alibi a T hT
    apply (definablePos_altPlusNeutral k hk).2
    exact ⟨φ, TLClY_subset_TLClPos _ _ hφ, hlang.trans (Set.ext hrec)⟩

/-- The ALiBi encoding hypothesis has a zero-model witness (Appendix F). -/
example : ∃ T : PTfr (Option (Option Bool)) 2 0 0 0, T.pe = .alibi 1 :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0,
     pe := .alibi 1 }, rfl⟩

/-- Both encoding and rationality hypotheses have zero-angle witnesses (F). -/
example : ∀ pe ∈ [PosEnc.sinusoidal (fun _ => 0), PosEnc.rope (fun _ => 0)],
    pe.RationalAngles ∧ ∃ T : PTfr (Option (Option Bool)) 2 0 0 0, T.pe = pe := by
  intro pe hpe
  refine ⟨?_, ⟨{
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := pe }, rfl⟩⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hpe
  rcases hpe with rfl | rfl <;> exact ⟨fun _ => 0, fun _ => by simp⟩

end Transformer.CRASP
