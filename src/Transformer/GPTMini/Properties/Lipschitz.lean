/-
# Property: the representation is Lipschitz in the input

Two `gpt-mini` streams driven by the same parameters and started from two
configurations `x_0, y_0 : Fin T → ℝ^{d_model}` stay within

  `L(params, eps, α_max) · max_i ‖x_0(i) - y_0(i)‖`

of each other at every depth, `L` being the product `endToEndLipschitz` of the
per-block constants of `Properties.LipschitzConstants`.  That is
`stream_lipschitz`, proved here by induction on the depth: each per-block
constant is at least `1`, so the partial products increase and the same `L`
bounds every depth, and past `n_layers` no block is applied any more.

The recursion is taken as a hypothesis (`IsStream`) rather than read off
`GPTMini.hidden`, so that the initial configuration is arbitrary and not only
one a token lookup can produce; `hidden_isStream` is the witness that the
hypothesis is satisfiable.

The bound stops at the representation: the logits are one `rmsNormEps` and one
unembedding further on, and this file says nothing about that last step.

`α_max` is a uniform upper bound for every head's `log α` in every block —
the only thing the per-block constants need to know about the temperatures.

The bound matters for adversarial robustness (perturbation amplification),
mean-field continuity (Wasserstein-style stability) and numerical analysis
(error propagation under `bf16` rounding).
-/

import Transformer.GPTMini.Properties.LipschitzConstants

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (eps : ℝ)

/-- **The layer recursion, as a property of an arbitrary sequence.**

`GPTMini.hidden` satisfies it (`hidden_isStream`), but so does the stream
started from any configuration whatever, which is what makes
`stream_lipschitz` a statement about perturbations of the representation and
not only about those a token lookup can produce.  Source: `reference/model.py`
(`GPTMini.forward`). -/
def IsStream (params : ModelParams cfg) {T : ℕ} (positions : Fin T → ℝ)
    (x : ℕ → Fin T → EucSpace cfg.d_model) : Prop :=
  ∀ L : ℕ,
    x (L + 1)
      = if h : L < cfg.n_layers then
          blockForward cfg (params.blocks ⟨L, h⟩) eps positions (x L)
        else x L

/-- The stream of `GPTMini.hidden` is one. -/
theorem hidden_isStream
    (params : ModelParams cfg) {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) :
    IsStream cfg eps params positions (hidden cfg params eps positions tokens) := by
  intro L
  rw [hidden]

/-- **Lipschitz dependence of the representation on the input.**

Two streams driven by the same parameters and differing at depth `0` stay
within

  `endToEndLipschitz · max_i ‖x_0(i) - y_0(i)‖`

of each other at every depth: the per-block constants multiply, and past
`n_layers` nothing is applied any more.  The bound holds at every `L`, not
only at `L = n_layers`, because each per-block constant is at least `1`
(`one_le_perBlockLipschitz`).

The logits are one RMSNorm and one unembedding further on, and this file says
nothing about that last step: `rmsNormEps` has no Lipschitz bound here, only
the bound on its norm of `Properties.StreamGrowth`.

`α_max` has to dominate the `log α` of every head of every block: the head
constant grows with the temperature, so a single hot head sets the pace.

Source: `reference/model.py` (`GPTMini.forward`), by induction from
`blockForward_lipschitz`. -/
theorem stream_lipschitz
    (params : ModelParams cfg) (alpha_max : ℝ) (heps : 0 < eps)
    (halpha : ∀ (l : Fin cfg.n_layers) (h : Fin cfg.n_heads),
      (params.blocks l).attn.log_alpha h ≤ alpha_max)
    {T : ℕ} [Nonempty (Fin T)] (positions : Fin T → ℝ)
    (x y : ℕ → Fin T → EucSpace cfg.d_model)
    (hx : IsStream cfg eps params positions x) (hy : IsStream cfg eps params positions y)
    (L : ℕ) (i : Fin T) :
    ‖x L i - y L i‖
      ≤ endToEndLipschitz cfg eps params alpha_max
        * (Finset.univ : Finset (Fin T)).sup' Finset.univ_nonempty
            (fun i' => ‖x 0 i' - y 0 i'‖) := by
  classical
  set δ : ℝ := (Finset.univ : Finset (Fin T)).sup' Finset.univ_nonempty
    (fun i' => ‖x 0 i' - y 0 i'‖) with hδdef
  have hδ0 : 0 ≤ δ :=
    le_trans (norm_nonneg _)
      (Finset.le_sup' (fun i' => ‖x 0 i' - y 0 i'‖) (Finset.mem_univ i))
  set P : ℕ → ℝ := fun l =>
    if h : l < cfg.n_layers then perBlockLipschitz cfg eps (params.blocks ⟨l, h⟩) alpha_max
    else 1 with hPdef
  have hP1 : ∀ l : ℕ, 1 ≤ P l := by
    intro l
    by_cases h : l < cfg.n_layers
    · simpa [hPdef, h] using
        one_le_perBlockLipschitz cfg eps (params.blocks ⟨l, h⟩) alpha_max heps
    · simp [hPdef, h]
  have hprod1 : ∀ n : ℕ, 1 ≤ ∏ l ∈ Finset.range n, P l := by
    intro n
    induction n with
    | zero => simp
    | succ n ih => rw [Finset.prod_range_succ]; nlinarith [hP1 n]
  have hmono : ∀ a b : ℕ, a ≤ b →
      (∏ l ∈ Finset.range a, P l) ≤ ∏ l ∈ Finset.range b, P l := by
    intro a b hab
    induction b, hab using Nat.le_induction with
    | base => exact le_refl _
    | succ b hab ih =>
        rw [Finset.prod_range_succ]
        nlinarith [hprod1 b, hP1 b, ih, hprod1 a]
  have hstop : ∀ b : ℕ, cfg.n_layers ≤ b →
      (∏ l ∈ Finset.range b, P l) = ∏ l ∈ Finset.range cfg.n_layers, P l := by
    intro b hb
    induction b, hb using Nat.le_induction with
    | base => rfl
    | succ b hb ih =>
        rw [Finset.prod_range_succ, ih, hPdef]
        simp [Nat.not_lt.mpr hb]
  have hend : (∏ l ∈ Finset.range cfg.n_layers, P l)
      = endToEndLipschitz cfg eps params alpha_max := by
    rw [endToEndLipschitz, ← Fin.prod_univ_eq_prod_range]
    refine Finset.prod_congr rfl fun l _ => ?_
    simp [hPdef, l.isLt]
  have key : ∀ L : ℕ, ∀ i : Fin T,
      ‖x L i - y L i‖ ≤ (∏ l ∈ Finset.range L, P l) * δ := by
    intro L
    induction L with
    | zero =>
        intro i
        rw [Finset.prod_range_zero, one_mul]
        exact Finset.le_sup' (fun i' => ‖x 0 i' - y 0 i'‖) (Finset.mem_univ i)
    | succ L ih =>
        intro i
        have hsup : (Finset.univ : Finset (Fin T)).sup' Finset.univ_nonempty
            (fun i' => ‖x L i' - y L i'‖) ≤ (∏ l ∈ Finset.range L, P l) * δ :=
          Finset.sup'_le _ _ fun i' _ => ih i'
        by_cases h : L < cfg.n_layers
        · have hstep := blockForward_lipschitz cfg eps (params.blocks ⟨L, h⟩) alpha_max heps
            (halpha ⟨L, h⟩) positions (x L) (y L) i
          have hPL : P L = perBlockLipschitz cfg eps (params.blocks ⟨L, h⟩) alpha_max := by
            simp [hPdef, h]
          rw [show x (L + 1) i
              = blockForward cfg (params.blocks ⟨L, h⟩) eps positions (x L) i from by
                rw [hx L]; simp [h],
            show y (L + 1) i
              = blockForward cfg (params.blocks ⟨L, h⟩) eps positions (y L) i from by
                rw [hy L]; simp [h],
            Finset.prod_range_succ, hPL]
          nlinarith [hstep, hsup, hprod1 L,
            one_le_perBlockLipschitz cfg eps (params.blocks ⟨L, h⟩) alpha_max heps]
        · rw [show x (L + 1) i = x L i from by rw [hx L]; simp [h],
            show y (L + 1) i = y L i from by rw [hy L]; simp [h],
            Finset.prod_range_succ]
          have hPL : P L = 1 := by simp [hPdef, h]
          rw [hPL]
          simpa using ih i
  refine le_trans (key L i) ?_
  have hle : (∏ l ∈ Finset.range L, P l) ≤ endToEndLipschitz cfg eps params alpha_max := by
    by_cases hL : L ≤ cfg.n_layers
    · rw [← hend]; exact hmono L cfg.n_layers hL
    · rw [← hend, hstop L (Nat.le_of_lt (Nat.lt_of_not_le hL))]
  exact mul_le_mul_of_nonneg_right hle hδ0

/-- The hypotheses are satisfiable: the streams of two token sequences at the
default config are `IsStream`, and `Fin 1` is nonempty. -/
example (params : ModelParams Config.default) (alpha_max : ℝ)
    (halpha : ∀ (l : Fin Config.default.n_layers) (h : Fin Config.default.n_heads),
      (params.blocks l).attn.log_alpha h ≤ alpha_max)
    (tokens tokens' : Fin 1 → Fin Config.default.vocab_size) :
    ‖hidden Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) tokens 3 0
        - hidden Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) tokens' 3 0‖
      ≤ endToEndLipschitz Config.default 1e-6 params alpha_max
        * (Finset.univ : Finset (Fin 1)).sup' Finset.univ_nonempty
            (fun i' => ‖hidden Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) tokens 0 i'
              - hidden Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) tokens' 0 i'‖) :=
  stream_lipschitz Config.default 1e-6 params alpha_max (by norm_num) halpha _ _ _
    (hidden_isStream Config.default 1e-6 params _ tokens)
    (hidden_isStream Config.default 1e-6 params _ tokens') 3 0

end Properties
end GPTMini
end Transformer
