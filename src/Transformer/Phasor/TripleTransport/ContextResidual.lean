module

public import Transformer.Phasor.TripleTransport.ReadBasis

/-!
# One contextual residual and two bank reads

Source: “One context residual and two learned gates” in the triple-transport
experiment. The class fixes the prefix control when transforming or translating
the exposed values. The native readout supplies its current hidden state as
both the control and the contextual value.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.ContextResidual
open ReadBasis
open scoped BigOperators

abbrev Reader (C : Type*) (d : ℕ) := C → Vec d → (Fin 2 → Vec d) → Vec d

/-- At fixed control, a common linear transformation acts on the context,
the direct value, the proposed value and the result. -/
def Natural {C : Type*} {d : ℕ} (f : Reader C d) : Prop :=
  ∀ c (a : Vec d →ₗ[ℝ] Vec d) h v, f c (a h) (fun i => a (v i)) = a (f c h v)

/-- An independent displacement of the contextual value passes to the output. -/
def Translates {C : Type*} {d : ℕ} (f : Reader C d) : Prop :=
  ∀ c h δ v, f c (h + δ) v = f c h v + δ

/-- With two exposed bank reads, width at least two, and arbitrary scalar
maps of the fixed control, these two laws characterize one context residual
plus two learned gates. -/
theorem complete {C : Type*} {d : ℕ} (axis : Fin 2 → Fin d) (inj : Function.Injective axis)
    (f : Reader C d) : Natural f ∧ Translates f ↔
      ∃ g : C → Fin 2 → ℝ, ∀ c h v, f c h v = h + ReadBasis.mix (g c) v := by
  constructor
  · rintro ⟨natural, translate⟩
    have origin : Equivariant (fun c v => f c 0 v) := by
      intro c a v
      simpa only [map_zero] using natural c a 0 v
    obtain ⟨g, atZero⟩ := (ReadBasis.complete axis inj _).1 origin
    refine ⟨g, ?_⟩
    intro c h v
    simpa only [zero_add, atZero, add_comm h] using translate c 0 h v
  · rintro ⟨g, read⟩
    constructor
    · intro c a h v
      simp [read, ReadBasis.mix, map_add, map_smul]
    · intro c h δ v
      simp only [read, add_right_comm h δ]

/-- The contextual value itself is necessary at positive width. Holding the
control and both bank reads fixed, a unit displacement changes the result. -/
theorem context_needed {C : Type*} {d : ℕ} (f : Reader C d) (law : Translates f)
    (c : C) (j : Fin d) (v : Fin 2 → Vec d) : ¬ ∀ h, f c h v = f c 0 v := by
  intro fixed
  have shift := congrFun (law c 0 (fun _ => 1) v) j
  rw [zero_add, fixed] at shift
  simp only [Pi.add_apply] at shift
  linarith

/-- Each bank read is necessary in this smaller class, even with arbitrary
processing of the other read and the context. Use the allowable readout
that adds the missing bank value to the context and fix the context to zero. -/
theorem necessary {d : ℕ} (k : Fin 2) (j : Fin d) :
    Natural (fun (_ : Unit) (h : Vec d) (v : Fin 2 → Vec d) => h + v k) ∧
    Translates (fun (_ : Unit) (h : Vec d) (v : Fin 2 → Vec d) => h + v k) ∧
    ¬ Omits k (fun v : Fin 2 → Vec d => (0 : Vec d) + v k) := by
  refine ⟨fun _ a h v => (map_add a h (v k)).symm, ?_, ?_⟩
  · intro c h δ v
    exact add_right_comm h δ (v k)
  · simpa only [zero_add] using (ReadBasis.necessary k j).2

example : Natural (fun (_ : Unit) (h : Vec 2) (v : Fin 2 → Vec 2) => h + v 1) ∧
    Translates (fun (_ : Unit) (h : Vec 2) (v : Fin 2 → Vec 2) => h + v 1) :=
  ⟨(necessary 1 (0 : Fin 2)).1,(necessary 1 (0 : Fin 2)).2.1⟩

example : ∃ g : Unit → Fin 2 → ℝ, ∀ c (h : Vec 2) (v : Fin 2 → Vec 2),
    (fun (_ : Unit) x y => x + y 1) c h v = h + ReadBasis.mix (g c) v := by
  apply (complete id Function.injective_id _).1
  exact ⟨(necessary 1 (0 : Fin 2)).1,(necessary 1 (0 : Fin 2)).2.1⟩

example : ¬ Translates (fun (_ : Unit) (h : Vec 2) (v : Fin 2 → Vec 2) => h + h + v 0) := by
  intro law
  have h := congrFun (law () 0 (fun _ => 1) (fun _ => 0)) 0
  norm_num at h

private theorem on_frame {d : ℕ} (axis : Fin 2 → Fin d) (inj : Function.Injective axis)
    (g : Fin 2 → ℝ) (i : Fin 2) : ReadBasis.mix g (frame axis) (axis i) = g i := by
  simp [ReadBasis.mix, frame, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, inj.eq_iff, mul_ite]

/-- For the native affine gate budget, only the two origin-frame coordinates
need to be affine in the control. The same weights then implement the
complete read at every context and every pair of values. -/
theorem affine {d k : ℕ} (axis : Fin 2 → Fin d) (inj : Function.Injective axis)
    (f : Reader (Vec k) d) (natural : Natural f) (translate : Translates f) :
    (∃ (b : Fin 2 → ℝ) (w : Fin 2 → Fin k → ℝ), ∀ c i,
       f c 0 (frame axis) (axis i) = b i + ∑ j, w i j * c j) ↔
    (∃ (b : Fin 2 → ℝ) (w : Fin 2 → Fin k → ℝ), ∀ c h v,
       f c h v = h + ReadBasis.mix (fun i => b i + ∑ j, w i j * c j) v) := by
  constructor
  · rintro ⟨b, w, law⟩
    have origin : Equivariant (fun c v => f c 0 v) := by
      intro c a v
      simpa only [map_zero] using natural c a 0 v
    refine ⟨b, w, ?_⟩
    intro c h v
    have read := ReadBasis.affine_complete axis inj (fun c v => f c 0 v) origin b w law c v
    simpa only [zero_add, read, add_comm h] using translate c 0 h v
  · rintro ⟨b, w, read⟩
    exact ⟨b, w, fun c i => by rw [read, zero_add, on_frame axis inj]⟩

example : ∃ (b : Fin 2 → ℝ) (w : Fin 2 → Fin 2 → ℝ),
    ∀ (c : Vec 2) (h : Vec 2) (v : Fin 2 → Vec 2),
    h + v 1 = h + ReadBasis.mix (fun i => b i + ∑ j, w i j * c j) v := by
  apply (affine id Function.injective_id (fun _ h v => h + v 1)
    (fun _ a h v => (map_add a h (v 1)).symm)
    (fun _ h δ v => add_right_comm _ _ _)).1
  exact ⟨fun i => if i = 1 then 1 else 0, fun _ _ => 0, fun c i => by simp [frame]⟩

example (v : Fin 2 → Vec 2) : ¬ ∀ h : Vec 2, h + v 1 = (0 : Vec 2) + v 1 :=
  context_needed (fun (_ : Unit) h (x : Fin 2 → Vec 2) => h + x 1)
    (necessary 1 (0 : Fin 2)).2.1 () 0 v

end Transformer.Phasor.TripleTransport.ContextResidual
