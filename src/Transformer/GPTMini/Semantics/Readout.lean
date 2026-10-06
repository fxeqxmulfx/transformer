import Transformer.GPTMini.TokenInterface.Correctness

/-!
# Semantic codes survive the actual final normalization and tied readout

Source: GPTMini.forward at f11b6e2, including its final RMSNorm and tied
embedding matrix. If the answer embedding has maximal norm and is
separated from every competitor by distance delta, a state within
scale * delta / 2 of its positive scaled code decodes to that answer.

These hypotheses concern Euclidean codes and approximation error before
final normalization. No correct actual logit or task-solving predicate
is assumed. Routing supplies a source of the approximation error; the
semantic algorithm must supply which code the head should carry.

A second result makes representation distinctions explicit: disjoint code
neighborhoods cannot contain the same hidden state. This applies before
unembedding, so it detects loss of order or binding in the residual stream.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface

/-- A separated maximal-norm code has a robust positive inner-product advantage.
Source: the new geometric certificate for the tied GPTMini readout at f11b6e2. -/
theorem code_inner_pos {d : ℕ} (x answer competitor : EucSpace d) (scale delta : ℝ)
    (hscale : 0 < scale) (hdelta : 0 < delta)
    (hnorm : ‖competitor‖ ≤ ‖answer‖)
    (hseparated : delta ≤ ‖answer - competitor‖)
    (hclose : ‖x - scale • answer‖ < scale * delta / 2) :
    0 < inner (𝕜 := ℝ) x (answer - competitor) := by
  have hdist : 0 < ‖answer - competitor‖ := hdelta.trans_le hseparated
  have hprototype : ‖answer - competitor‖ ^ 2 / 2 ≤
      inner (𝕜 := ℝ) answer (answer - competitor) := by
    rw [inner_sub_right, real_inner_self_eq_norm_sq]
    nlinarith [norm_sub_sq_real answer competitor, norm_nonneg answer,
      norm_nonneg competitor]
  have herror := (abs_le.mp (abs_real_inner_le_norm
    (x - scale • answer) (answer - competitor))).1
  rw [inner_sub_left, real_inner_smul_left] at herror
  have hbudget : ‖x - scale • answer‖ < scale * ‖answer - competitor‖ / 2 :=
    hclose.trans_le (by gcongr)
  have herrorbudget := mul_lt_mul_of_pos_right hbudget hdist
  have hscaled := mul_le_mul_of_nonneg_left hprototype hscale.le
  nlinarith

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    ‖(0 : EucSpace 64)‖ ≤ ‖controlUnit‖ ∧ 1 ≤ ‖controlUnit - 0‖ ∧
    ‖controlUnit - (1 : ℝ) • controlUnit‖ < (1 : ℝ) * 1 / 2 := by
  norm_num [controlUnit_norm]

/-- A geometrically certified actual final hidden state emits its semantic answer token.
Source: GPTMini.forward and the checked List Int adapter at f11b6e2/cbafbe9.
The theorem quantifies over given parameters; it does not assert existence of task-solving weights. -/
theorem tokenFunction_of_code (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (heps : 0 < eps) (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
    (answer : Fin cfg.vocab_size) (scale delta : ℝ)
    (hscale : 0 < scale) (hdelta : 0 < delta)
    (hlen : tail.length + 1 ≤ cfg.max_seq_len)
    (hgeometry : ∀ v, v ≠ answer → ‖params.embedding v‖ ≤ ‖params.embedding answer‖ ∧
      delta ≤ ‖params.embedding answer - params.embedding v‖)
    (hclose : ‖hidden cfg params eps (fun i => (i.val : ℝ)) (head :: tail).get
        cfg.n_layers ⟨tail.length, by simp⟩ - scale • params.embedding answer‖ <
      scale * delta / 2) :
    tokenFunction cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.decodeTokens (head :: tail) ++ [(answer.val : ℤ)] := by
  apply tokenFunction_of_margin cfg params eps head tail answer hlen
  intro v hv
  let state := hidden cfg params eps (fun i => (i.val : ℝ)) (head :: tail).get
    cfg.n_layers ⟨tail.length, by simp⟩
  have hinner := code_inner_pos state (params.embedding answer) (params.embedding v)
    scale delta hscale hdelta (hgeometry v hv).1 (hgeometry v hv).2 hclose
  rw [inner_sub_right] at hinner
  have hd : (0 : ℝ) < (cfg.d_model : ℝ) := Nat.cast_pos.mpr cfg.d_model_pos
  have hnorm : 0 < Real.sqrt (cfg.d_model : ℝ) /
      Real.sqrt (‖state‖ ^ 2 + (cfg.d_model : ℝ) * eps) := by positivity
  change inner (𝕜 := ℝ) (rmsNormEps eps state) (params.embedding v) <
    inner (𝕜 := ℝ) (rmsNormEps eps state) (params.embedding answer)
  rw [rmsNormEps, real_inner_smul_left, real_inner_smul_left]
  exact mul_lt_mul_of_pos_left (by linarith) hnorm

/-- The concrete depth control has a separated maximal-norm REJECT code.
Source: controlEmbedding's actual 36-token tied embedding table, not an output-logit premise. -/
theorem control_code_geometry : ∀ v : Fin controlConfig.vocab_size, v ≠ ⟨15, by decide⟩ →
    ‖controlParams.embedding v‖ ≤ ‖controlParams.embedding ⟨15, by decide⟩‖ ∧
      (1 : ℝ) ≤ ‖controlParams.embedding ⟨15, by decide⟩ - controlParams.embedding v‖ := by
  intro v hv
  have hv15 : v.val ≠ 15 := fun h => hv (Fin.ext h)
  change ‖controlEmbedding v‖ ≤ ‖controlEmbedding ⟨15, by decide⟩‖ ∧
    (1 : ℝ) ≤ ‖controlEmbedding ⟨15, by decide⟩ - controlEmbedding v‖
  simp only [controlEmbedding, ite_eq_right hv15, ↓reduceIte]
  split_ifs
  · have he : (2 : ℝ) • controlUnit - controlUnit = controlUnit := by module
    rw [he]
    simp [norm_smul, controlUnit_norm]
  · simp [norm_smul, controlUnit_norm]

example : (⟨9, by decide⟩ : Fin controlConfig.vocab_size) ≠ ⟨15, by decide⟩ := by decide

/-- Before final normalization, the actual BOS,a control state equals half the REJECT embedding.
Source: hidden_control on the original small Basis depth architecture. -/
theorem control_hidden_code :
    ‖hidden controlConfig controlParams (1 / 100000) (fun i => (i.val : ℝ))
        ([⟨1, by decide⟩, ⟨9, by decide⟩] : List (Fin controlConfig.vocab_size)).get
        controlConfig.n_layers ⟨1, by decide⟩ -
      (1 / 2 : ℝ) • controlParams.embedding ⟨15, by decide⟩‖ < (1 / 2 : ℝ) * 1 / 2 := by
  rw [hidden_control]
  change ‖controlEmbedding ⟨9, by decide⟩ - (1 / 2 : ℝ) • controlEmbedding ⟨15, by decide⟩‖ < _
  have he : (1 / 2 : ℝ) • ((2 : ℝ) • controlUnit) = controlUnit := by
    rw [smul_smul]
    norm_num
  simp only [controlEmbedding, ↓reduceIte, he]
  norm_num

/-- A genuinely supervised depth prefix satisfies every internal code certificate premise.
Source: the same concrete control, now verified before RMSNorm rather than from its final logits. -/
example : tokenFunction controlConfig controlParams (1 / 100000) [1, 9] = [1, 9, 15] := by
  have h := tokenFunction_of_code controlConfig controlParams (1 / 100000) (by norm_num)
    ⟨1, by decide⟩ [⟨9, by decide⟩] ⟨15, by decide⟩ (1 / 2) 1
    (by norm_num) (by norm_num) (by decide) control_code_geometry control_hidden_code
  simpa [Transformer.Basis.decodeTokens] using h

/-- Distinct separated semantic codes force distinct actual hidden states under a common error budget.
Source: the new Euclidean decoding certificate, applicable to order, binding and overwrite pairs. -/
theorem semantic_states_distinct {d : ℕ} (x y left right : EucSpace d) (radius : ℝ)
    (hx : ‖x - left‖ ≤ radius) (hy : ‖y - right‖ ≤ radius)
    (hseparated : 2 * radius < ‖left - right‖) : x ≠ y := by
  intro heq
  subst y
  have htriangle := norm_sub_le_norm_sub_add_norm_sub left x right
  rw [norm_sub_rev left x] at htriangle
  linarith

example : ‖controlUnit - controlUnit‖ ≤ (0 : ℝ) ∧
    ‖(0 : EucSpace 64) - 0‖ ≤ (0 : ℝ) ∧ 2 * (0 : ℝ) < ‖controlUnit - 0‖ := by
  simp [controlUnit_norm]

/-- Separated code neighborhoods have no common hidden state within the stated error budget.
Source: semantic_states_distinct; a semantic distinction cannot collapse before logits. -/
theorem no_common_semantic_state {d : ℕ} (left right : EucSpace d) (radius : ℝ)
    (hseparated : 2 * radius < ‖left - right‖) :
    ¬∃ x, ‖x - left‖ ≤ radius ∧ ‖x - right‖ ≤ radius := by
  rintro ⟨x, hleft, hright⟩
  exact semantic_states_distinct x x left right radius hleft hright hseparated rfl

example : 2 * (0 : ℝ) < ‖controlUnit - 0‖ := by
  simp [controlUnit_norm]

end Transformer.GPTMini.Semantics
