module

public import Transformer.Grokking.FiniteRecovery.ArgandHiddenLearnability
public import Transformer.Precision.AdamWKernel
public import Mathlib.Probability.UniformOn
public import Mathlib.MeasureTheory.Integral.Bochner.SumMeasure

/-! # Hidden success for any trajectory, the u32 seed and the binary32 kernel

`three_quarters_of_trajectory` is `three_quarters_of_marginal_jacobian` for any weight
sequence on any probability space, so it applies to the binary32 weights the CUDA runner
stores, read as real numbers. `seed_kernel_three_quarters` instantiates it on a/b mod 97
under the uniform u32 seed, which fixes the initialization and the minibatch shuffle;
with ordered gradient reductions the run is a function of the seed. From step `B` on the
weights follow the binary32 AdamW kernel at rates in `[0, 2^{-22}]`, the WSD floor
`floorRate32` among them, where the decay factor rounds to one (`kernel_decay_eq_one`):
the Jacobian bound is required along `fl(p - step) - p`, without decay. The moments and
clipped gradients the kernel reads are arbitrary. The logits are real at the stored
weights; the runner's FP32 forward pass is not modeled. At a zero gradient real AdamW
sends every weight to zero (`AdamW.run_zero_gradient_tendsto`), the kernel keeps it
(`adamKernel_rest`).
-/

@[expose] public section

open Set MeasureTheory Filter ProbabilityTheory Transformer.Precision
open scoped Topology

namespace Transformer.Grokking.FiniteRecovery.Argand

/-- A nearest rounding to binary32, with the tie rule `Classical.choose` picks. -/
noncomputable def nearest32 : ℝ → ℝ := (exists_isNearest_grid (E := 8) (by norm_num) 23).choose

theorem nearest32_spec : IsNearest (grid 8 23) nearest32 :=
  (exists_isNearest_grid (E := 8) (by norm_num) 23).choose_spec

theorem floorRate32_small : 0 ≤ floorRate32 ∧ floorRate32 ≤ 1/2^22 :=
  ⟨by norm_num [floorRate32, binary32Value], by norm_num [floorRate32, binary32Value]⟩

/-- **The kernel keeps what real AdamW decays.** At a zero gradient with zero moments and
a rate in `[0, 2^{-22}]`, the binary32 kernel returns every binary32 weight and both
moments unchanged. -/
theorem adamKernel_rest {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q) {p lr bc1 bc2 : ℝ}
    (hp : p ∈ grid 8 23) (h0 : 0 ≤ lr) (h : lr ≤ 1/2^22) :
    adamKernel Q p 0 0 0 lr tenth32 bc1 bc2 = (p, 0, 0) := by
  have z : Q 0 = 0 := hQ.eq_self (zero_mem_grid (by norm_num) 23)
  simp [adamKernel, kernelStep, z, kernel_decay_eq_one hQ h0 h, hQ.eq_self hp]

example := adamKernel_rest nearest32_spec one_mem_grid (bc1 := 1) (bc2 := 1) floorRate32_small.1
  floorRate32_small.2

namespace ObservedDrift

open ObservedConstraints

variable {X A J : Type*} [Fintype X] [Fintype A] [Nonempty A] {H d L T : ℕ}
local notation "Seed" => Fin (2^32)
local notation "W" => Weights (Fin 239) 4 16 6

/-- **Three quarters along any trajectory, conditionally.** The proof of
`three_quarters_of_marginal_jacobian` with the real AdamW run replaced by any weight
sequence `w` of a draw of any law: the Jacobian bound on `U` is required along the actual
displacement `w z (t+1) - w z t`, whatever arithmetic produced it. -/
theorem three_quarters_of_trajectory [MeasurableSpace A] [MeasurableSingletonClass A] [Fintype J]
    {D : Type*} [MeasurableSpace D] (μ : Measure D) [IsProbabilityMeasure μ] (r : ℕ) (S : Finset X)
    (y : S → A) (rule : J → Set (X → A)) (g : X → A) (feasible : ∀ j, g ∈ rule j)
    (fit : ∀ x : S, g x = y x) (id : InjOn (fun v : X → A => fun x : S => v x) (⋂ j, rule j))
    (prompt : X → Word A (T+1)) (pos : X → Fin (T+1)) (w : D → ℕ → Weights A H d L)
    (B : ℕ) {ρ floor : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : floor < 1/2)
    (e : D → ℝ) (hm : Measurable e) (hi : Integrable e μ) (hp : ∀ᵐ z ∂μ, 0 ≤ e z)
    (mean : ∫ z, e z ∂μ < (1-ρ)*(1/2-floor)/4) :
    let U := marginalBound (H := H) (d := d) (L := L) r (Fintype.card J) S y g prompt pos
    (∀ z t, B ≤ t → ∀ u ∈ Ioo (0:ℝ) 1, fderiv ℝ U (w z t + u • (w z (t+1) - w z t))
      (w z (t+1) - w z t) ≤ -(1-ρ)*(U (w z t)-floor)+e z) →
    ∃ G : Set D, MeasurableSet G ∧ 3/4 < μ.real G ∧ ∀ z ∈ G, ∀ᶠ t in atTop,
      ∀ f : X → A, (∀ j, f ∈ rule j) → (∀ x : S, f x = y x) →
        StrictOn (fun x => (Model.fromWeights (w z t)).logits r (prompt x) (pos x)) f := by
  intro U jac
  let score (v : Weights A H d L) x := (Model.fromWeights v).logits r (prompt x) (pos x)
  let Δ z t := w z (t+1) - w z t
  have hU : Differentiable ℝ U := differentiable_marginalBound r (Fintype.card J) S y g prompt pos
  have major (v : Weights A H d L) : risk (tableLaw (score v)) S (fun x => g x) rule ≤ U v := by
    rw [funext fit]
    exact risk_le_marginalBound r S y g rule feasible fit prompt pos v
  obtain ⟨G,mG,pG,ok⟩ := permanent_of_observed_drift μ score S rule g feasible id U major w Δ
    (fun z t => (add_sub_cancel (w z t) (w z (t+1))).symm)
    B hρ hlt hr e hm hi hp (q := 3/4) (by linarith only [mean]) (by
      intro z t ht
      have along (u : ℝ) : HasDerivAt (fun v : ℝ => w z t + v • Δ z t) (Δ z t) u := by
        simpa using ((hasDerivAt_id u).smul_const (Δ z t)).const_add (w z t)
      have slope (u : ℝ) : HasDerivAt (segmentRisk U (w z t) (Δ z t))
          (fderiv ℝ U (w z t + u • Δ z t) (Δ z t)) u := (hU _).hasFDerivAt.comp_hasDerivAt u (along u)
      have hd : Differentiable ℝ (segmentRisk U (w z t) (Δ z t)) := fun u => (slope u).differentiableAt
      exact ⟨hd.continuous.continuousOn,hd.differentiableOn,fun u hu => (slope u).deriv ▸ jac z t ht u hu⟩)
  refine ⟨G,mG,pG,fun z hz => ?_⟩
  filter_upwards [ok z hz] with t win
  intro f hf hy
  have he : f = g := id (mem_iInter.mpr hf) (mem_iInter.mpr feasible) ((funext hy).trans (funext fit).symm)
  exact he ▸ win

/-- On a mask with a connected unit graph and one zero-row equation, the observations
determine the members of the division class. -/
theorem division_injOn (S : Finset DivisionInput)
    (connected : (FlatSpectrum.patternGraph (unitMask (S : Set DivisionInput))).Preconnected)
    (hz : ∃ j, (none, j) ∈ S) :
    InjOn (fun v : DivisionInput → Fin 239 => fun x : S => v x) (⋂ _ : Unit, factorClass unitToken) := by
  rw [iInter_const]
  intro f hf g hg eq
  exact factorClass_identified unitToken _ connected hz hf hg (funext fun x => congrFun eq ⟨x.val, x.property⟩)

/-- The mean under the uniform u32 seed is the average over all `2^32` seeds. -/
theorem integral_uniform_seed (e : Seed → ℝ) :
    ∫ z, e z ∂(uniformOn univ : Measure Seed) = (∑ z, e z) / 2^32 := by
  rw [integral_fintype Integrable.of_finite]
  have h (z : Seed) : (uniformOn univ : Measure Seed).real {z} = 1/2^32 := by
    simp [measureReal_def, uniformOn_univ]; norm_num
  simp only [h, smul_eq_mul, ← Finset.mul_sum]
  ring

/-- **Three quarters under the CUDA runner's law and update, conditionally.** The seed
`z` is uniform on u32; `p z t` holds the binary32 weights after `t` updates, which from
step `B` on follow the binary32 AdamW kernel `adamKernel` coordinatewise, with moments
`first`, `second`, the clipped gradient `grad` the runner passes, rates `rate t` in
`[0, 2^{-22}]` (the WSD floor `floorRate32` among them) and bias corrections `bc1`,
`bc2`. On a mask with a connected unit graph and one zero-row equation, each member `F`
of the class is learned on the hidden inputs, for the real logits at the stored weights,
with probability above three quarters if the Jacobian bound on `U` holds along the
kernel's displacement `fl(p - step) - p` from `B` on. That bound and `mean` are
hypotheses; neither is proved for this runner. -/
theorem seed_kernel_three_quarters (S : Finset DivisionInput)
    (connected : (FlatSpectrum.patternGraph (unitMask (S : Set DivisionInput))).Preconnected)
    (hz : ∃ j, (none, j) ∈ S) {F : DivisionInput → Fin 239} (hF : F ∈ factorClass unitToken)
    {Q : ℝ → ℝ} (hQ : IsNearest (grid 8 23) Q) (p first second grad : Seed → ℕ → W)
    (rate bc1 bc2 : ℕ → ℝ) (B : ℕ) (small : ∀ t, B ≤ t → 0 ≤ rate t ∧ rate t ≤ 1/2^22)
    (stored : ∀ z t i, p z t i ∈ grid 8 23) (kernel : ∀ z t, B ≤ t → ∀ i, p z (t+1) i =
      (adamKernel Q (p z t i) (first z t i) (second z t i) (grad z t i) (rate t) tenth32 (bc1 t) (bc2 t)).1)
    {ρ floor : ℝ} (hρ : 0 ≤ ρ) (hlt : ρ < 1) (hr : floor < 1/2) (e : Seed → ℝ) (hp : ∀ z, 0 ≤ e z)
    (mean : (∑ z, e z) / 2^32 < (1-ρ)*(1/2-floor)/4) :
    let U := marginalBound (H := 4) (d := 16) (L := 6) 1 1 S (fun x => F x) F divisionPrompt (fun _ => Fin.last 4)
    (∀ z t, B ≤ t → ∀ u ∈ Ioo (0:ℝ) 1,
      let Δ : W := fun i =>
        Q (p z t i - kernelStep Q (first z t i) (second z t i) (grad z t i) (rate t) (bc1 t) (bc2 t)) - p z t i
      fderiv ℝ U (p z t + u • Δ) Δ ≤ -(1-ρ)*(U (p z t)-floor)+e z) →
    3/4 < (uniformOn univ : Measure Seed).real {z | HiddenSuccess (S : Set DivisionInput) (fun t x =>
      (Model.fromWeights (p z t)).logits 1 (divisionPrompt x) (Fin.last 4)) F} := by
  intro U jac
  have step (z : Seed) (t : ℕ) (ht : B ≤ t) : p z (t+1) - p z t = fun i =>
      Q (p z t i - kernelStep Q (first z t i) (second z t i) (grad z t i) (rate t) (bc1 t) (bc2 t)) - p z t i := by
    funext i
    rw [Pi.sub_apply, kernel z t ht i,
      adamKernel_weight_of_small_rate hQ (stored z t i) (small t ht).1 (small t ht).2]
  obtain ⟨G, -, pG, ok⟩ := three_quarters_of_trajectory (uniformOn univ) 1 S (fun x => F x)
    (fun _ : Unit => factorClass unitToken) F (fun _ => hF) (fun _ => rfl) (division_injOn S connected hz)
    divisionPrompt (fun _ => Fin.last 4) p B hρ hlt hr e .of_discrete Integrable.of_finite (ae_of_all _ hp)
    (by rwa [integral_uniform_seed]) (fun z t ht u hu => by rw [step z t ht]; exact jac z t ht u hu)
  refine pG.trans_le (measureReal_mono fun z hz => ?_)
  obtain ⟨N, hN⟩ := eventually_atTop.mp (ok z hz)
  exact ⟨N, fun t ht x _ a ha => hN t ht F (fun _ => hF) (fun _ => rfl) x a ha⟩

private def sparseRows (i : Fin 97) : ℕ := if i = 1 then 2^96 - 1 else 1

private noncomputable def sparseSites : Finset DivisionInput := (bitDivisionMask sparseRows).toFinite.toFinset

private theorem sparse_graph : (FlatSpectrum.patternGraph (unitMask (sparseSites : Set DivisionInput))).Preconnected ∧
    ∃ j, (none, j) ∈ sparseSites := by
  have h := bitMask_identifying_graph sparseRows (fun _ => 0) (fun _ => 0) 0 (by decide) (by decide) (by decide)
  rw [sparseSites, Set.Finite.coe_toFinset]
  exact ⟨h.1, h.2.imp fun _ h => (Set.Finite.mem_toFinset _).mpr h⟩

example := division_injOn sparseSites sparse_graph.1 sparse_graph.2

example (p : Seed → ℕ → W) := three_quarters_of_trajectory (uniformOn univ) 1 sparseSites
  (fun x => divisionAnswer x) (fun _ : Unit => factorClass unitToken) divisionAnswer
  (fun _ => divisionAnswer_mem) (fun _ => rfl) (division_injOn sparseSites sparse_graph.1 sparse_graph.2)
  divisionPrompt (fun _ => Fin.last 4) p 0 (ρ := 0) (floor := 0) le_rfl one_pos (by norm_num) (fun _ => 0)
  measurable_const (integrable_const 0) (ae_of_all _ fun _ => le_rfl) (by norm_num)

/-- The kernel's weights from zero under `nearest32`, for given moments and gradients. -/
noncomputable def kernelRun (first second grad : Seed → ℕ → W) (rate bc1 bc2 : ℕ → ℝ) (z : Seed) : ℕ → W
  | 0 => 0
  | t+1 => fun i => (adamKernel nearest32 (kernelRun first second grad rate bc1 bc2 z t i) (first z t i)
      (second z t i) (grad z t i) (rate t) tenth32 (bc1 t) (bc2 t)).1

theorem kernelRun_stored (first second grad : Seed → ℕ → W) (rate bc1 bc2 : ℕ → ℝ) (z : Seed) :
    ∀ t i, kernelRun first second grad rate bc1 bc2 z t i ∈ grid 8 23
  | 0, _ => zero_mem_grid (by norm_num) 23
  | _+1, _ => (nearest32_spec _).1

example (first second grad : Seed → ℕ → W) (bc1 bc2 : ℕ → ℝ) :=
  seed_kernel_three_quarters sparseSites sparse_graph.1 sparse_graph.2 divisionAnswer_mem nearest32_spec
    (kernelRun first second grad (fun _ => floorRate32) bc1 bc2) first second grad (fun _ => floorRate32) bc1 bc2
    0 (fun _ _ => floorRate32_small) (kernelRun_stored first second grad _ bc1 bc2) (fun _ _ _ _ => rfl)
    (ρ := 0) (floor := 0) le_rfl one_pos (by norm_num) (fun _ => 0) (fun _ => le_rfl)
    (by simp only [Finset.sum_const_zero, zero_div]; norm_num)

end ObservedDrift

end Transformer.Grokking.FiniteRecovery.Argand
