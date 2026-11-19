module

public import Transformer.Sandwich.Section3_Transformer
public import Transformer.Sandwich.Section3_Biases
public import Transformer.Sandwich.Section5_Locality
public import Transformer.Sandwich.Section5_Causal
public import Transformer.Sandwich.Section5_WindowLayer
public import Transformer.Sandwich.Section5_Averaging
public import Transformer.Sandwich.Section5_ReceptiveField
public import Transformer.Sandwich.Section5_Extrapolation
public import Transformer.Sandwich.Section5_Gradient
public import Transformer.Sandwich.Section5_Sinusoidal
public import Transformer.Sandwich.Section6_Sandwich
public import Transformer.Sandwich.Section6_KERPLE
public import Transformer.Sandwich.SectionE_Implementation

/-!
# Sandwich: length extrapolation through receptive fields

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, "Dissecting Transformer Length Extrapolation via
the Lens of Receptive Field Analysis" (ACL 2023). The TeX source is in `papers/arXiv-2212.10356/`.

A model is a causal map from input sequences `ℕ → ℝ^d` to next-token distributions; a positional
scheme is a bias of the distance added to the scaled logit, `−∞` masking the key.

* `Section3_Transformer`: §3.1, Eq. (1) and the model of Eq. (3): heads, `W_o`, LayerNorm, layers
  and the readout; the weights are the Softmax over the keys `n ≤ m`.
* `Section3_Biases`: §3.2–3.3, ALiBi's bias of Eq. (2) and its slopes, and windowed attention.
* `Section5_Locality`, `Section5_Causal`: the window bookkeeping of sequence maps; windows compose
  to `w + w' − 1`; the model is causal for every bias.
* `Section5_WindowLayer`: a model with windowed attention of size `w` and `R` layers reads the
  `R(w − 1) + 1` most recent tokens and does not see absolute positions.
* `Section5_Averaging`: in an averaging model every token of that window contributes.
* `Section5_ReceptiveField`: the TRF of §1 and §5 is `R(w − 1) + 1`, not `wR`: the maximum over
  all models, attained by the averaging model; `L_tr = 512` covers the TRF of 12 layers exactly when
  `w ≤ 43`, and `w = 100` gives `1189`, not `1200`.
* `Section5_Extrapolation`: §1 and §5, a model whose TRF fits in `L_tr` predicts at every position
  as at a position `< L_tr`.
* `Section5_Gradient`: §5.1, the normalized gradients and their cumulative sums; the printed
  `ERF = min {m | c_m > 0.99}` is always the first position, and the ERF that counts the most
  recent tokens is at most the TRF.
* `Section5_Sinusoidal`: §5.3, Eqs. (4)–(5); with an APE or invertible Rotary matrices every token
  `n ≤ m` contributes, so the receptive field is not bounded by `L_tr`.
* `Section6_Sandwich`: §6.1, `p_m^⊤ p_n = Σ_i cos((m − n) θ_i)` with the maximum `d̄/2` at
  `m = n` only, and Eq. (6): a relative bias, at most `0`, `0` only at the distance `0`, and at
  least `−d̄/h`.
* `Section6_KERPLE`: §6.2–6.3, the smoothed Sandwich is KERPLE's logarithmic bias kernel, and its
  attention is KERPLE's with `r₁ = 0.825 √k` since Sandwich adds its bias after the scaling; the
  exact Sandwich is no logarithmic bias; the plateau; T5's distinct and shared buckets.
* `SectionE_Implementation`: Appendix E, the reference code is `p_m^⊤ p_n / h` with Sinusoidal's
  frequencies `1/10000^{2i/d̄}`, `i = 0, …, d̄/2 − 1`. Omitting `−d̄/2` does not change the
  attention, but the frequency `1` in place of `1/10000` adds `(d̄/2 + cos t − cos(t/10000))/h`,
  which does.
-/
