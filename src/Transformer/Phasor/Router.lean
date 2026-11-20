module

public import Transformer.MHC.PeriRMS.Basic

/-!
# The dynamic router of a sublayer

Every attention and feed-forward sublayer of DiffGPT's main model has its own router
(`research/main_model/README.txt`, *Architecture*, and `mhc.json`: `router_input`,
`router_projection`): the parameter-free RMS of the flattened four-stream state
(`flatRMS`, `R_{4D}(vec x)`), a bias-free real projection of it to 24 logits, one learned scale
for each of the three groups (4 read, 4 write and 16 skip logits) and 24 biases, as in the
`logits` kernel of `src/native_transformer/src/mhc.rs`. The paper's `tanh` of
arXiv:2512.24880v2, Eqs. (5)–(7) is not part of the main model. Three real coordinates `b_s`
give the stream phases `r_s = (1 + i b_s)/√(1 + b_s²) = e^{i arctan b_s}` for `s = 1, 2, 3`; the
phase of stream `0` is fixed at one.

`Router.routing` turns the state into the `Routing` of `Transformer.MHC.PeriRMS`: read and write
logits for the gates, skip logits for the finite Sinkhorn map and the angles `arctan b_s`.

* `routing_phase_zero`, `routing_phase_succ`: the phase of stream `0` is `0`, of stream `s + 1` it
  is `arctan b_{s+1}`; `routing_rot_succ` shows `e^{i arctan b} = (1 + i b)/√(1 + b²)`
  (`MHC.unitPhase`). All phases are zero when the `b_s` are, as at initialization.
* `flatRMS_energy_le`: the router input has mean square at most `1`.

The three scales start at `0.01`, the biases at `0` for read and write and `2 I` for skip
(`Router.initial`), the projections random; exact real arithmetic, no FP16 operands.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS

variable {D : ℕ}

/-- The parameter-free RMS of the flattened four-stream state, `R_{4D}(vec x)`: the vector
`x √(4D) / √(‖x‖² + 4D ε)` with `‖x‖² = ∑_s ‖x_s‖²`. -/
noncomputable def flatRMS (eps : ℝ) (x : State D) : State D :=
  (Real.sqrt (4 * (D : ℝ)) / Real.sqrt (∑ s, ‖x s‖ ^ 2 + 4 * (D : ℝ) * eps)) • x

/-- The router of one sublayer: the projections of the normalized state to the read, write and
skip logits, the three learned scales (read, write, skip), the 24 biases and the three real phase
coordinates `b_1, b_2, b_3`. -/
structure Router (D : ℕ) where
  readProj : State D →ₗ[ℝ] (Fin 4 → ℝ)
  writeProj : State D →ₗ[ℝ] (Fin 4 → ℝ)
  skipProj : State D →ₗ[ℝ] Matrix (Fin 4) (Fin 4) ℝ
  scale : Fin 3 → ℝ
  readBias : Fin 4 → ℝ
  writeBias : Fin 4 → ℝ
  skipBias : Matrix (Fin 4) (Fin 4) ℝ
  phase : Fin 3 → ℝ

/-- The routing of a state: `logit = scale_group · (projection of the normalized state) + bias`,
and the angles `0, arctan b_1, arctan b_2, arctan b_3` of the stream phases. -/
noncomputable def Router.routing (R : Router D) (eps : ℝ) (x : State D) : Routing where
  read i := R.scale 0 * R.readProj (flatRMS eps x) i + R.readBias i
  write i := R.scale 1 * R.writeProj (flatRMS eps x) i + R.writeBias i
  skip i j := R.scale 2 * R.skipProj (flatRMS eps x) i j + R.skipBias i j
  phase := Fin.cases 0 fun s => Real.arctan (R.phase s)

/-- The phase of stream `0` is fixed at one. -/
theorem Router.routing_phase_zero (R : Router D) (eps : ℝ) (x : State D) :
    (R.routing eps x).phase 0 = 0 :=
  rfl

/-- The angle of stream `s + 1` is `arctan b_{s+1}`. -/
theorem Router.routing_phase_succ (R : Router D) (eps : ℝ) (x : State D) (s : Fin 3) :
    (R.routing eps x).phase s.succ = Real.arctan (R.phase s) :=
  rfl

/-- The phase `e^{i φ_s}` of a learned stream is the implemented `(1 + i b)/√(1 + b²)`. -/
theorem Router.routing_rot_succ (R : Router D) (eps : ℝ) (x : State D) (s : Fin 3) :
    complexPhase.rot ((R.routing eps x).phase s.succ) 1 = unitPhase (R.phase s) := by
  rw [routing_phase_succ, unitPhase_eq_exp_arctan]
  exact mul_one _

/-- Zero phase coordinates give zero angles on every stream: the paper's skip. -/
theorem Router.routing_phase_eq_zero (R : Router D) (eps : ℝ) (x : State D) (h : R.phase = 0) :
    (R.routing eps x).phase = fun _ => 0 := by
  funext s
  refine Fin.cases ?_ (fun t => ?_) s
  · rfl
  · rw [routing_phase_succ, h]
    simp

/-- The router input has mean square at most one: `∑_s ‖x̃_s‖² ≤ 4D`, for every `ε > 0`. -/
theorem flatRMS_energy_le (eps : ℝ) (heps : 0 < eps) (hD : 0 < D) (x : State D) :
    ∑ s, ‖flatRMS eps x s‖ ^ 2 ≤ 4 * (D : ℝ) := by
  have hD' : (0 : ℝ) < D := by exact_mod_cast hD
  have hS0 : 0 ≤ ∑ s, ‖x s‖ ^ 2 := Finset.sum_nonneg fun s _ => sq_nonneg _
  have hden : 0 < ∑ s, ‖x s‖ ^ 2 + 4 * (D : ℝ) * eps := by positivity
  have hc : (Real.sqrt (4 * (D : ℝ)) / Real.sqrt (∑ s, ‖x s‖ ^ 2 + 4 * (D : ℝ) * eps)) ^ 2 =
      4 * (D : ℝ) / (∑ s, ‖x s‖ ^ 2 + 4 * (D : ℝ) * eps) := by
    rw [div_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt hden.le]
  have hsum : ∑ s, ‖flatRMS eps x s‖ ^ 2 =
      (4 * (D : ℝ) / (∑ s, ‖x s‖ ^ 2 + 4 * (D : ℝ) * eps)) * ∑ s, ‖x s‖ ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun s _ => ?_
    simp only [flatRMS, Pi.smul_apply, norm_smul, Real.norm_eq_abs, mul_pow, sq_abs, hc]
  rw [hsum, div_mul_eq_mul_div, div_le_iff₀ hden]
  nlinarith [mul_pos hD' heps]

/-- The model's initialization with given projections: scales `0.01`, read and write biases `0`,
skip bias `2 I`, phase coordinates `0` (`mhc.json`, `initialization`). -/
noncomputable def Router.initial (readProj writeProj : State D →ₗ[ℝ] (Fin 4 → ℝ))
    (skipProj : State D →ₗ[ℝ] Matrix (Fin 4) (Fin 4) ℝ) : Router D where
  readProj := readProj
  writeProj := writeProj
  skipProj := skipProj
  scale := fun _ => 1 / 100
  readBias := 0
  writeBias := 0
  skipBias := 2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)
  phase := 0

/-- At initialization every stream phase is zero, whatever the projections and the state. -/
example (readProj writeProj : State 128 →ₗ[ℝ] (Fin 4 → ℝ))
    (skipProj : State 128 →ₗ[ℝ] Matrix (Fin 4) (Fin 4) ℝ) (x : State 128) :
    ((Router.initial readProj writeProj skipProj).routing (1 / 8388608) x).phase = fun _ => 0 :=
  Router.routing_phase_eq_zero _ _ _ rfl

/-- The router input of the model, `D = 128`, `ε = 2⁻²³`: mean square at most one. -/
example (x : State 128) : ∑ s, ‖flatRMS (1 / 8388608) x s‖ ^ 2 ≤ 4 * ((128 : ℕ) : ℝ) :=
  flatRMS_energy_le _ (by norm_num) (by norm_num) x

end Transformer.Phasor
