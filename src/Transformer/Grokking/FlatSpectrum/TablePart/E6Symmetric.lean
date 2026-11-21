module

public import Transformer.Grokking.FlatSpectrum.CertificateGram
public import Transformer.Grokking.FlatSpectrum.CertificateTests
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Lines

/-!
# PT6: the certificate of a symmetric table that keeps its form under the flip

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. An invertible kernel
has one certificate. So when the table is symmetric and keeps its form under `x ↦ −1 − x`, up to
phases, the certificate has the same two symmetries, and it vanishes on every image of an unseen
cell.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Matrix

attribute [local instance] fact_prime_97

lemma line_val (u : Fin 1 → ZMod 97) : line 97 (u 0).val = u := by
  funext t
  rw [Subsingleton.elim t 0]
  simp [line]

example (u : Fin 1 → ZMod 97) : line 97 (u 0).val = u := line_val u

/-- The flip `x ↦ −1 − x` of the lines. -/
def flipLine : Equiv.Perm (Fin 1 → ZMod 97) := Equiv.subLeft fun _ => -1

lemma flipLine_line {a : ℕ} (ha : a < 97) : flipLine (line 97 a) = line 97 (96 - a) := by
  funext t
  have h : ((96 : ℕ) : ZMod 97) = -1 := by decide +kernel
  simp only [flipLine, Equiv.subLeft_apply, Pi.sub_apply, line]
  rw [Nat.cast_sub (by omega), h]

example : flipLine (line 97 1) = line 97 95 := flipLine_line (by norm_num)

/-- **The certificate of an invertible symmetric kernel that keeps its form under the flip** is
symmetric and keeps the same form. -/
theorem certificate_symmetries {f : ZMod 97 → ZMod 97 → ZMod 97}
    {Y : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ} (hdet : (tableKernel f).det ≠ 0)
    (hs : ∀ a b, f b a = f a b) {α β : ZMod 97 → ℂ} (hα : ∀ x, ‖α x‖ = 1) (hβ : ∀ x, ‖β x‖ = 1)
    (hF : ∀ a b, (ZMod.stdAddChar (f (-1 - a) (-1 - b)) : ℂ) =
      α a * ZMod.stdAddChar (f a b) * β b)
    (hY : IsContraction Y) (htr : (Yᴴ * tableKernel f).trace = nuclearNorm (tableKernel f)) :
    (∀ u v, Y v u = Y u v) ∧ ∀ a b, a < 97 → b < 97 → Y (line 97 a) (line 97 b) =
      star (α (a : ZMod 97)) * Y (line 97 (96 - a)) (line 97 (96 - b)) *
        star (β (b : ZMod 97)) := by
  set K := tableKernel f with hKdef
  have hKt : Kᵀ = K := by
    ext u v
    simp [hKdef, tableKernel, hs (u 0) (v 0)]
  have htrT : ((Yᵀ)ᴴ * K).trace = nuclearNorm K := by
    have h := trace_transpose_certificate K Y
    rw [hKt] at h
    rw [h, htr]
  have hYT : Y = Yᵀ := eq_of_certificates hdet hY htr hY.transpose htrT
  refine ⟨fun u v => ?_, fun a b ha hb => ?_⟩
  · have h : Y v u = Yᵀ v u := by rw [← hYT]
    rwa [transpose_apply] at h
  have hsym' (i j : Fin 1 → ZMod 97) :
      K (flipLine i) (flipLine j) = α (i 0) * K i j * β (j 0) := by
    simpa [hKdef, tableKernel, flipLine] using hF (i 0) (j 0)
  have hZ := hY.phased flipLine flipLine (a := fun i => star (α (i 0)))
    (b := fun j => star (β (j 0))) (fun i => by simp [hα]) (fun j => by simp [hβ])
  have hZtr := trace_phased (Y := Y) hsym'
  rw [htr] at hZtr
  have hYZ := eq_of_certificates hdet hY htr hZ hZtr
  have h : Y (line 97 a) (line 97 b) = star (α ((line 97 a) 0)) *
      Y (flipLine (line 97 a)) (flipLine (line 97 b)) * star (β ((line 97 b) 0)) := by
    conv_lhs => rw [hYZ]
    rfl
  rw [flipLine_line ha, flipLine_line hb] at h
  exact h

end Transformer.Grokking.FlatSpectrum.TablePart.E6
