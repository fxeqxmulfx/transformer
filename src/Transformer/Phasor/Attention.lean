module

public import Transformer.Phasor.Router
public import Transformer.LeRoPEBlock.Bound
public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies

/-!
# The attention of the main model: LeRoPE heads with the squared-imaginary phase score

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`,
*Architecture*, and `model.json`. At the position `i` of a layer the attention sublayer reads, from
the layer's input state `x_j` of every position `j ≤ i`, the vector `z_j = Router.read x_j`, and
each of `H` heads of real width `d` computes

    q_ih = LeRoPE_i(R_d(W_Q^h z_i)),   k_jh = LeRoPE_j(R_d(W_K^h z_j)),   v_jh = W_V^h z_j,
    p_ijh = softmax_{j ≤ i}((ln N_i / d)(Re Z_ij + β_h (Im Z_ij)² / d)),   N_i = i + 1,

with `Z_ij = ∑_c conj(q_ic) k_jc` over the `d/2` complex coordinates of the half-split pairing
(`Transformer.MHC.PeriRMS.pairLayout`), `β_h = a_h / √(1 + a_h²)` (`phaseCoefficient`), and the
frequencies `10000^{−2m/d} e^{α_m}` of the rotation (`leropeFrequencies`), the `α_m` learned and
shared by Q, K, heads and layers. The heads' averages `∑_j p_ijh v_jh` are concatenated and
written by `W_O` (`Attention.output`).

`Attention.block` is `Transformer.LeRoPEBlock.block` for these scores and values, with the four
streams of a position as the token: every theorem of `Transformer.LeRoPEBlock` applies to it, and
`Attention.block_congr` is causality. The projections are real-linear maps; the model's are the
complex-linear ones (stored `[[A, −B], [B, A]]`), a subclass, and no bound below needs it.

* `headVector_dotProduct_le`: after the per-head parameter-free RMS the pair layout of a query or
  key has squared length at most `d`;
* `Attention.sublayer`, `Attention.step_apply`: the attention residual of the README,
  `R_D(W_O heads)` written back with the router's gates and phases;
* `Attention.logit_abs_le`: **every logit of every head is at most `(5/4) ln (i + 1)`** at every
  distance and every state, for every projection, frequency scale and coefficient `a_h`.

The coefficients `a_h` start at `0` (`Attention.zero`), the ordinary SSMax score.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.LeRoPEBlock

variable {D d H : ℕ}

/-- LeRoPE's frequencies `θ_m = 10000^{−2m/d} e^{α_m}` for the `d/2` channels of a head
(arXiv:2607.10134v1, §3.1, Eq. (5)), `α_m` the learned logarithmic scales. -/
noncomputable def leropeFrequencies (d : ℕ) (alpha : Fin (d / 2) → ℝ) : Fin (d / 2) → ℝ :=
  fun m => RoPE.LeRoPE.learnedFrequency (RoPE.frequency 10000 d m) (alpha m)

/-- A head's query or key before the rotation: the parameter-free RMS of the projection `W z`,
laid out in the pairs of the half-split pairing. -/
noncomputable def headVector (eps : ℝ) (W : EucSpace D →ₗ[ℝ] EucSpace d) (z : EucSpace D) :
    Fin 2 × Fin (d / 2) → ℝ :=
  pairLayout d (GPTMini.rmsNormEps eps (W z))

/-- **After the head RMS a query or key has squared length at most `d`.** -/
theorem headVector_dotProduct_le (eps : ℝ) (heps : 0 < eps) (hd : 0 < d)
    (W : EucSpace D →ₗ[ℝ] EucSpace d) (z : EucSpace D) :
    headVector eps W z ⬝ᵥ headVector eps W z ≤ (d : ℝ) := by
  have hn := GPTMini.rmsNormEps_norm_le eps heps hd (W z)
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  calc headVector eps W z ⬝ᵥ headVector eps W z
      ≤ ‖GPTMini.rmsNormEps eps (W z)‖ ^ 2 := pairLayout_dotProduct_le d _
    _ ≤ Real.sqrt d ^ 2 := pow_le_pow_left₀ (norm_nonneg _) hn 2
    _ = d := Real.sq_sqrt hd'.le

/-- The parameters of the attention of one layer: the `H` heads' query, key and value projections
from the width `D` of the streams to the head width `d`, the output projection `W_O` of the `H`
concatenated heads, and the coefficients `a_h` of the squared imaginary part. -/
structure Attention (D d H : ℕ) where
  query : Fin H → EucSpace D →ₗ[ℝ] EucSpace d
  key : Fin H → EucSpace D →ₗ[ℝ] EucSpace d
  value : Fin H → EucSpace D →ₗ[ℝ] EucSpace d
  output : (Fin H → EucSpace d) →ₗ[ℝ] EucSpace D
  coefficient : Fin H → ℝ

/-- Zero projections and coefficients `a_h = 0`, the model's initialization of the score
coordinates: uniform attention over the visible keys, the ordinary SSMax logit. -/
noncomputable def Attention.zero (D d H : ℕ) : Attention D d H where
  query := fun _ => 0
  key := fun _ => 0
  value := fun _ => 0
  output := 0
  coefficient := fun _ => 0

/-- **The logit of head `h`** at the query position `i`, for the query state `x`, the key state `y`
at the distance `δ`: SSMax's gain `ln (i + 1) / d` times `Re Z + β_h (Im Z)² / d` of the head
vectors of what the router `R` reads in each state, the key turned back by `δ` (`leropeLogit`). -/
noncomputable def Attention.logit (A : Attention D d H) (R : Router D) (eps : ℝ)
    (alpha : Fin (d / 2) → ℝ) (h : Fin H) : ℕ → State D → State D → ℕ → ℝ :=
  leropeLogit (leropeFrequencies d alpha) (phaseCoefficient (A.coefficient h)) d
    (fun x => headVector eps (A.query h) (R.read eps x))
    (fun x => headVector eps (A.key h) (R.read eps x))

/-- **The attention block at the position `i`**: `Φ` of the query's state and of the `H` heads'
softmax averages over the states at the positions `j ≤ i`, the head `h` scoring the key `j` by
`Attention.logit` and reading the value `W_V^h z_j`. -/
noncomputable def Attention.block {Y : Type*} (A : Attention D d H) (R : Router D) (eps : ℝ)
    (alpha : Fin (d / 2) → ℝ) (Φ : State D → (Fin H → EucSpace d) → Y) (x : ℕ → State D)
    (i : ℕ) : Y :=
  LeRoPEBlock.block Φ (A.logit R eps alpha) (fun h y => A.value h (R.read eps y)) x i

/-- **The heads' softmax averages** at the position `i`: `∑_{j ≤ i} p_ijh W_V^h z_j` for each head
`h`, the input of `Φ` in `Attention.block`. -/
noncomputable def Attention.heads (A : Attention D d H) (R : Router D) (eps : ℝ)
    (alpha : Fin (d / 2) → ℝ) (x : ℕ → State D) (i : ℕ) : Fin H → EucSpace d :=
  A.block R eps alpha (fun _ heads => heads) x i

/-- The attention block at `i` is `Φ` of the query's state and of the heads' averages. -/
theorem Attention.block_eq {Y : Type*} (A : Attention D d H) (R : Router D) (eps : ℝ)
    (alpha : Fin (d / 2) → ℝ) (Φ : State D → (Fin H → EucSpace d) → Y) (x : ℕ → State D) (i : ℕ) :
    A.block R eps alpha Φ x i = Φ (x i) (A.heads R eps alpha x i) :=
  rfl

/-- **The attention sublayer for given heads' averages**: the router `R` of the sublayer, and as raw
branch `W_O` of the concatenated averages `heads`. The raw branch ignores the state and the read:
they enter through `heads`, which `Attention.block` has computed from the reads of the positions
`j ≤ i` (`Router.read`), the query side from the state at `i`. -/
noncomputable def Attention.sublayer (A : Attention D d H) (R : Router D) (eps : ℝ)
    (heads : Fin H → EucSpace d) : Sublayer D where
  routing := R.routing eps
  raw := fun _ _ => A.output heads

/-- The block of the attention sublayer is the phase-conjugated skip plus the twice-sigmoid write
gate times the depth scale times `R_D(W_O heads)`: the README's attention residual. -/
theorem Attention.step_apply (A : Attention D d H) (R : Router D) (eps scale : ℝ)
    (heads : Fin H → EucSpace d) (x : State D) (i : Fin 4) :
    step eps scale (A.sublayer R eps heads) x i =
      phaseMix (headPhaseAction D) (finiteSinkhorn (R.routing eps x).skip 20)
          (R.routing eps x).phase x i +
        writeGate ((R.routing eps x).write i) •
          (scale • GPTMini.rmsNormEps eps (A.output heads)) :=
  rfl

/-- **Every logit of every head is at most `(5/4) ln (i + 1)`**, at every distance, for every pair
of states, projection, frequency scale and coefficient `a_h`, for `ε > 0`. -/
theorem Attention.logit_abs_le (A : Attention D d H) (R : Router D) (eps : ℝ) (heps : 0 < eps)
    (hd : 0 < d) (alpha : Fin (d / 2) → ℝ) (h : Fin H) (i : ℕ) (x y : State D) (δ : ℕ) :
    |A.logit R eps alpha h i x y δ| ≤ 5 / 4 * Real.log ((i : ℝ) + 1) :=
  leropeLogit_abs_le _ _ _ (Nat.cast_pos.2 hd) (phaseCoefficient_abs_lt_one _).le _ _ i x y δ
    (headVector_dotProduct_le eps heps hd _ _) (headVector_dotProduct_le eps heps hd _ _)

/-- **Causality**: the attention block at the position `i` reads only the states at `j ≤ i`. -/
theorem Attention.block_congr {Y : Type*} (A : Attention D d H) (R : Router D) (eps : ℝ)
    (alpha : Fin (d / 2) → ℝ) (Φ : State D → (Fin H → EucSpace d) → Y) {x x' : ℕ → State D}
    (i : ℕ) (hx : ∀ j ≤ i, x j = x' j) :
    A.block R eps alpha Φ x i = A.block R eps alpha Φ x' i :=
  block_congr_of_le _ _ _ i hx

/-- The model's sizes, `D = 128`, `d = 32`, `H = 4`, `ε = 2⁻²³`: the logit of any head, at any
position, state and distance, is at most `(5/4) ln (i + 1)`. -/
example (R : Router 128) (alpha : Fin (32 / 2) → ℝ) (h : Fin 4) (i δ : ℕ) (x y : State 128) :
    |(Attention.zero 128 32 4).logit R (1 / 8388608) alpha h i x y δ| ≤
      5 / 4 * Real.log ((i : ℝ) + 1) :=
  Attention.logit_abs_le _ R _ (by norm_num) (by norm_num) alpha h i x y δ

/-- The block at the position `3` is the same on two state sequences that agree up to `3`. -/
example (R : Router 128) (alpha : Fin (32 / 2) → ℝ) (x : ℕ → State 128) :
    (Attention.zero 128 32 4).block R (1 / 8388608) alpha (fun _ heads => heads 0) x 3 =
      (Attention.zero 128 32 4).block R (1 / 8388608) alpha (fun _ heads => heads 0)
        (fun j => if j ≤ 3 then x j else 0) 3 :=
  Attention.block_congr _ R _ alpha _ 3 (fun j hj => by simp [hj])

end Transformer.Phasor
