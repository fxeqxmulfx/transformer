/-
# Reverse simulations for sinusoidal embeddings and RoPE

arXiv:2506.16055v3, Appendix F, `thm:rtfr_eq_tlclmod` (reverse direction)
and `thm:rtfr_to_TLClmod`. Rational angles have a common period across the
finitely many coordinate blocks, and finite decorated states give formulas
with exactly the original counting-depth bound.
-/

import Transformer.CRASP.PeriodicToLogic
import Transformer.CRASP.PositionalPeriod

namespace Transformer.CRASP

universe u
variable {σ : Type u} [Fintype σ] [DecidableEq σ] {p s d k : ℕ}

/-- **Proposition `thm:rtfr_to_TLClmod`.** A RoPE transformer at rational
angles over the paper's finite alphabet has a same-depth `MOD` formula.

The former signature omitted the finite alphabet fixed in Section 2.3.
This assumption is needed for the embedding disjunction in Appendix B.2.

Source: arXiv:2506.16055v3, Appendix F, `sec:rope_pes`,
`thm:rtfr_to_TLClmod`. -/
theorem exists_mem_TLClMod_of_rope (θ : ℕ → ℝ)
    (hθ : (PosEnc.rope θ).RationalAngles) (T : PTfr (Option σ) p s d k)
    (hT : T.pe = .rope θ) :
    ∃ φ ∈ TLClMod σ k, φ.lang = {w : List σ | T.Accepts (bos w)} := by
  obtain ⟨M, hM, hrot⟩ := exists_rotation_period d θ hθ
  apply exists_mem_TLClMod_of_periodic T hM
  · intro i
    simp [hT, PosEnc.emb]
  · intro i j q a
    simp only [hT, PosEnc.logit, hrot i, hrot j]

/-- The reverse half of `thm:rtfr_eq_tlclmod`, including depth zero.
Source: arXiv:2506.16055v3, Appendix F, `sec:sinusoidal_pes`. -/
theorem exists_mem_TLClMod_of_sinusoidal (θ : ℕ → ℝ)
    (hθ : (PosEnc.sinusoidal θ).RationalAngles) (T : PTfr (Option σ) p s d k)
    (hT : T.pe = .sinusoidal θ) :
    ∃ φ ∈ TLClMod σ k, φ.lang = {w : List σ | T.Accepts (bos w)} := by
  obtain ⟨M, hM, hrot⟩ := exists_rotation_period d θ hθ
  apply exists_mem_TLClMod_of_periodic T hM
  · intro i
    funext c
    simp only [hT, PosEnc.emb, sinusoidalVec, hrot i]
  · intro i j q a
    simp only [hT, PosEnc.logit]

/-- Both periodic encodings meet the rational-angle and encoding hypotheses.
Source: arXiv:2506.16055v3, Appendix F, periodic encodings. -/
example : ∀ pe ∈ [PosEnc.rope (fun _ => 0), PosEnc.sinusoidal (fun _ => 0)],
    pe.RationalAngles ∧ ∃ T : PTfr (Option Bool) 2 0 0 0, T.pe = pe := by
  intro pe hpe
  have ht : ∃ T : PTfr (Option Bool) 2 0 0 0, T.pe = pe :=
    ⟨{
      E := fun _ _ => 0
      WQ := fun _ _ => 0
      WK := fun _ _ => 0
      WV := fun _ _ => 0
      ff := fun _ _ => 0
      Wout := fun _ => 0
      pe := pe }, rfl⟩
  refine ⟨?_, ht⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hpe
  rcases hpe with rfl | rfl <;> exact ⟨fun _ => 0, fun _ => by simp⟩

end Transformer.CRASP
