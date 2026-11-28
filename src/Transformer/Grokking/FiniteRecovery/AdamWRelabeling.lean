module

public import Transformer.Grokking.FiniteRecovery.ArgandRelabeling
public import Transformer.Grokking.FiniteRecovery.AdamWTrials
public import Mathlib.Dynamics.Ergodic.MeasurePreserving

/-! Alphabet invariance of native Argand training in real arithmetic.
The full clipped AdamW trajectory, its WSD endpoint and the probability
of strict generalization transfer to any bijective encoding of the task.
The minibatch order may depend on the draw if it is transported with it. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.AdamW

open Transformer.AMSGradW.Kernite

variable {I J : Type*} [Fintype I] [Fintype J]

def State.reindex (e : I ≃ J) (s : State I) : State J :=
  ⟨s.weights ∘ e.symm, s.first ∘ e.symm, s.second ∘ e.symm⟩

theorem clip_reindex (e : I ≃ J) (g : I → ℝ) : clip (g ∘ e.symm) = clip g ∘ e.symm := by
  have hn : ‖(WithLp.toLp 2 (g ∘ e.symm) : Space J)‖ = ‖(WithLp.toLp 2 g : Space I)‖ :=
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ e).norm_map (WithLp.toLp 2 g)
  funext j
  simp only [clip,Function.comp_apply,hn]

/-- Global clipping, squared moments and decoupled decay respect the same
coordinate permutation at every step, for any transported gradient stream. -/
theorem run_reindex (e : I ≃ J) (rate : ℕ → ℝ)
    (g : ℕ → (I → ℝ) → I → ℝ) (h : ℕ → (J → ℝ) → J → ℝ)
    (hg : ∀ t w, h t (w ∘ e.symm) = g t w ∘ e.symm) (w : I → ℝ) (n : ℕ) :
    run rate h (w ∘ e.symm) n = (run rate g w n).reindex e := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [run,ih,State.reindex,hg,clip_reindex,Function.comp_apply]
    rfl

example (e : Fin 3 ≃ Fin 3) (rate : ℕ → ℝ) (w : Fin 3 → ℝ) (n : ℕ) :=
  run_reindex e rate (fun _ x => x) (fun _ x => x) (fun _ _ => rfl) w n

end Transformer.Grokking.FiniteRecovery.AdamW

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set
open scoped NNReal

variable {A B X : Type*} [Fintype A] [Fintype B] {H d L T : ℕ}

theorem adamWRun_relabel (e : A ≃ B) (r : ℕ) (batch : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (z : ComplexParameter A H d L → ℂ) (n : ℕ) :
    adamWRun r (fun t => (batch t).map (relabelPosition e)) rate (relabelDraw e z) n =
      (adamWRun r batch rate z n).reindex (parameterTokenEquiv e) := by
  unfold adamWRun
  rw [initialWeights_relabel]
  exact AdamW.run_reindex (parameterTokenEquiv e) rate _ _
    (fun t => batchGradient_relabel e r (batch t)) _ n

theorem adamW_logits_relabel (e : A ≃ B) (r K : ℕ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (z : ComplexParameter A H d L → ℂ) (x : Word A T) (i : Fin T) (a : A) :
    (Model.fromWeights (adamWRun r (fun t => (batch t).map (relabelPosition e)) rate
      (relabelDraw e z) K).weights).logits r (e ∘ x) i (e a) =
    (Model.fromWeights (adamWRun r batch rate z K).weights).logits r x i a := by
  rw [adamWRun_relabel]
  change (Model.fromWeights (relabelWeights e _)).logits _ _ _ _ = _
  rw [fromWeights_relabel,Model.logits_relabel]

/-- Choosing X to be the unobserved inputs checks every hidden prompt.
The order of supervised minibatches is allowed to depend on the raw draw. -/
noncomputable def terminalStrictDraws (r K : ℕ) (rate : ℕ → ℝ)
    (batch : (ComplexParameter A H d L → ℂ) → ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) : Set (ComplexParameter A H d L → ℂ) :=
  {z | StrictOn (fun x => (Model.fromWeights (adamWRun r (batch z) rate z K).weights).logits r
    (prompt x) (Fin.last T)) f}

omit [Fintype A] [Fintype B] in
theorem relabelDraw_inverse (e : A ≃ B) (z : ComplexParameter A H d L → ℂ) :
    relabelDraw e.symm (relabelDraw e z) = z := by
  funext p
  change z ((tensorTokenEquiv e).symm ((tensorTokenEquiv e) p)) = z p
  rw [Equiv.symm_apply_apply]

theorem mem_terminalStrictDraws_relabel (e : A ≃ B) (r K : ℕ) (rate : ℕ → ℝ)
    (batch : (ComplexParameter A H d L → ℂ) → ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) (z : ComplexParameter A H d L → ℂ) :
    relabelDraw e z ∈ terminalStrictDraws r K rate
      (fun y t => (batch (relabelDraw e.symm y) t).map (relabelPosition e))
      (fun x => e ∘ prompt x) (e ∘ f) ↔ z ∈ terminalStrictDraws r K rate batch prompt f := by
  simp only [terminalStrictDraws,Set.mem_ofPred_eq,relabelDraw_inverse,StrictOn,Function.comp_apply]
  constructor
  · intro h x a ha
    simpa only [adamW_logits_relabel] using h x (e a) (fun he => ha (e.injective he))
  · intro h x b hb
    obtain ⟨a,rfl⟩ := e.surjective b
    simpa only [adamW_logits_relabel] using h x a (fun ha => hb (congrArg e ha))

/-- For any seed generator, coupling the drawn tensors and the data order
preserves which seeds succeed. The seed may determine both independently
of whether the raw tensor alone determines the order. -/
theorem terminal_success_seeds_relabel {D : Type*} (e : A ≃ B) (r K : ℕ) (rate : ℕ → ℝ)
    (draw : D → ComplexParameter A H d L → ℂ) (batch : D → ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) :
    {s | relabelDraw e (draw s) ∈ terminalStrictDraws r K rate
      (fun _ t => (batch s t).map (relabelPosition e)) (fun x => e ∘ prompt x) (e ∘ f)} =
    {s | draw s ∈ terminalStrictDraws r K rate (fun _ => batch s) prompt f} :=
  Set.ext fun s => mem_terminalStrictDraws_relabel e r K rate (fun _ => batch s) prompt f (draw s)

/-- Transporting each native coordinate's variance preserves the exact
one-run probability at the stated horizon, including an unscaled WSD run.
No restriction on the cumulative learning rate is needed for this equality. -/
theorem adamW_relabel_probability (e : A ≃ B) (r K : ℕ) (rate : ℕ → ℝ)
    (batch : (ComplexParameter A H d L → ℂ) → ℕ → List (LabeledPosition A))
    (prompt : X → Word A (T+1)) (f : X → A) (variance : ComplexParameter A H d L → ℝ≥0) :
    independentGaussianDraws (variance ∘ (tensorTokenEquiv e).symm)
      (terminalStrictDraws r K rate (fun y t => (batch (relabelDraw e.symm y) t).map (relabelPosition e))
        (fun x => e ∘ prompt x) (e ∘ f)) =
    independentGaussianDraws variance (terminalStrictDraws r K rate batch prompt f) := by
  let E := tensorTokenEquiv (H := H) (d := d) (L := L) e
  let M := MeasurableEquiv.piCongrLeft (fun _ : ComplexParameter B H d L => ℂ) E
  have hm : MeasurePreserving M (independentGaussianDraws variance)
      (independentGaussianDraws (variance ∘ E.symm)) := by
    simpa only [independentGaussianDraws,Function.comp_apply,E.symm_apply_apply,M] using
      (measurePreserving_piCongrLeft (fun i => complexGaussianDraw (variance (E.symm i))) E)
  have fe (z : ComplexParameter A H d L → ℂ) : M z = relabelDraw e z := by
    funext j
    obtain ⟨i,rfl⟩ := E.surjective j
    change (MeasurableEquiv.piCongrLeft (fun _ => ℂ) E z) (E i) = z (E.symm (E i))
    rw [MeasurableEquiv.piCongrLeft_apply_apply,E.symm_apply_apply]
  rw [← hm.measure_preimage_equiv]
  apply congrArg (independentGaussianDraws variance)
  ext z
  rw [Set.mem_preimage,fe]
  exact mem_terminalStrictDraws_relabel e r K rate batch prompt f z

/-- Transport the admissible functions with the answer tokens. -/
def relabelFamily (e : A ≃ B) (family : Set (X → A)) : Set (X → B) :=
  (fun g => e ∘ g) '' family

omit [Fintype A] [Fintype B] in
theorem relabelFamily_identified (e : A ≃ B) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) :
    InjOn (fun g : X → B => fun x : S => g x) (relabelFamily e family) := by
  rintro _ ⟨f,hf,rfl⟩ _ ⟨g,hg,rfl⟩ h
  exact congrArg (fun u => e ∘ u) (hid hf hg (funext fun x => e.injective (congrFun h x)))

/-- One strict real training witness in the original encoding suffices for
every relabeling. Each restart uses the same fixed minibatch stream and
unmodified schedule. The geometric miss probability uses exactly the
single-run mass in the original encoding. -/
theorem adamW_relabel_certified_restarts [Finite X] (e : A ≃ B) (r K : ℕ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (S : Set X)
    (family : Set (X → A)) (hid : InjOn (fun g : X → A => fun x : S => g x) family)
    (f : X → A) (hf : f ∈ family) (v : ℝ≥0) (hv : 0 < v)
    (witness : ComplexParameter A H d L → ℂ)
    (hw : witness ∈ terminalStrictDraws r K rate (fun _ => batch) prompt f) :
    let scores (z : ComplexParameter B H d L → ℂ) x := (Model.fromWeights
      (adamWRun r (fun t => (batch t).map (relabelPosition e)) rate z K).weights).logits r
      (e ∘ prompt x) (Fin.last T)
    let ok z := classCertificate S (relabelFamily e family) (fun x => e (f x)) (scores z)
    let μ := independentGaussianDraws (fun _ : ComplexParameter B H d L => v)
    let p := independentGaussianDraws (fun _ : ComplexParameter A H d L => v)
      (terminalStrictDraws r K rate (fun _ => batch) prompt f)
    0 < p ∧ (∀ m, Measure.infinitePi (fun _ : ℕ => μ) {z | retained ok z m = none} = (1-p)^m) ∧
      ∀ᵐ z ∂Measure.infinitePi (fun _ : ℕ => μ), ∃ j, StrictOn (scores (z j)) (e ∘ f) ∧
        ∀ k, retained ok z (j+1+k) = some (z j) := by
  intro scores ok μ p
  let := independentGaussianDraws_openPos (fun _ : ComplexParameter A H d L => v) (fun _ => hv)
  have hp : 0 < p := (adamW_strict_open r K rate batch prompt f).measure_pos _ ⟨witness,hw⟩
  have he : μ {z | StrictOn (scores z) (e ∘ f)} = p :=
    adamW_relabel_probability e r K rate (fun _ => batch) prompt f (fun _ => v)
  refine ⟨hp,?_⟩
  rw [← he]
  exact certified_restarts μ S (relabelFamily e family) (relabelFamily_identified e S family hid)
    (e ∘ f) ⟨f,hf,rfl⟩ scores (adamW_strict_open r K rate _ _ _).measurableSet (he.symm ▸ hp)

end Transformer.Grokking.FiniteRecovery.Argand
