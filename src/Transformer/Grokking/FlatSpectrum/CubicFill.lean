module

public import Transformer.Grokking.FlatSpectrum.Decoding
public import Transformer.Grokking.FlatSpectrum.RankOneFill

/-!
# One missing cell of `a³ + a b` and of `a + b`

Source: `review/flat_spectrum.md`, Corollary. The `a³ + a b` kernel is flat (`Kernels.lean`),
so the minimum-nuclear-norm completion of one missing cell `(a, b)` fills it with
`−ψ(a³ + a b)` (`FlatFill.lean`), which decodes to `a³ + a b + (p ± 1)/2` for odd `p` and
never to the truth (`Decoding.lean`). The `a + b` kernel has rank one; its fill is the truth
(`RankOneFill.lean`), which decodes to `a + b` alone.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

variable {p : ℕ} [NeZero p]

/-- **Corollary** of `review/flat_spectrum.md` for `a³ + a b`: with one cell `(a, b)` missing,
the minimum-nuclear-norm completion of `ψ(a³ + a b)` fills it with the opposite phase
`−ψ(a³ + a b)`, and no other fill attains the minimum. -/
theorem isMinOn_cubic_fill_iff (hp : 2 ≤ p) {a b : ZMod p} {X : Matrix (ZMod p) (ZMod p) ℂ}
    (hX : ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), X kl.1 kl.2 = cubicKernel p kl.1 kl.2) :
    IsMinOn nuclearNorm
        {Y | ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), Y kl.1 kl.2 = cubicKernel p kl.1 kl.2}
        X ↔
      X a b = -cubicKernel p a b :=
  isMinOn_flat_fill_iff (by rw [conjTranspose_mul_cubicKernel, ZMod.card])
    (by simp [cubicKernel, taskKernel]) (by rw [ZMod.card]; exact hp) hX

example := isMinOn_cubic_fill_iff (p := 5) (by norm_num) (a := 1) (b := 2) (X := cubicKernel 5)
  (fun _ _ => rfl)

/-- **Corollary** of `review/flat_spectrum.md`: for odd `p ≥ 3`, the minimum-nuclear-norm fill
of one missing cell `(a, b)` of `a³ + a b` decodes exactly to `a³ + a b + (p − 1)/2` and
`a³ + a b + (p + 1)/2`. -/
theorem isMinOn_decode_cubic_fill_iff (hodd : Odd p) (hp : 2 ≤ p) {a b : ZMod p}
    {X : Matrix (ZMod p) (ZMod p) ℂ}
    (hX : ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), X kl.1 kl.2 = cubicKernel p kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), Y kl.1 kl.2 = cubicKernel p kl.1 kl.2} X)
    (c : ZMod p) :
    IsMinOn (fun c => ‖X a b - ZMod.stdAddChar c‖) Set.univ c ↔
      c = a ^ 3 + a * b + ((p - 1) / 2 : ℕ) ∨ c = a ^ 3 + a * b + ((p + 1) / 2 : ℕ) := by
  rw [(isMinOn_cubic_fill_iff hp hX).mp hmin]
  simpa [cubicKernel, taskKernel] using isMinOn_norm_neg_stdAddChar_sub_iff hodd _ c

example := isMinOn_decode_cubic_fill_iff (p := 5) ⟨2, rfl⟩ (by norm_num) (a := 1) (b := 2)
  (add_smul_single_apply_of_mem (-2 * cubicKernel 5 1 2))
  ((isMinOn_cubic_fill_iff (by norm_num) (add_smul_single_apply_of_mem _)).mpr
    (by simp; ring)) 0

/-- **Corollary** of `review/flat_spectrum.md` ("always wrong"): for `p ≥ 2`, the
minimum-nuclear-norm fill of one missing cell `(a, b)` of `a³ + a b` never decodes to the
truth `a³ + a b`. -/
theorem not_isMinOn_decode_cubic_fill (hp : 2 ≤ p) {a b : ZMod p}
    {X : Matrix (ZMod p) (ZMod p) ℂ}
    (hX : ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), X kl.1 kl.2 = cubicKernel p kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), Y kl.1 kl.2 = cubicKernel p kl.1 kl.2} X) :
    ¬IsMinOn (fun c => ‖X a b - ZMod.stdAddChar c‖) Set.univ (a ^ 3 + a * b) := by
  rw [(isMinOn_cubic_fill_iff hp hX).mp hmin]
  simpa [cubicKernel, taskKernel] using not_isMinOn_norm_neg_stdAddChar_sub hp (a ^ 3 + a * b)

example := not_isMinOn_decode_cubic_fill (p := 4) (by norm_num) (a := 1) (b := 2)
  (add_smul_single_apply_of_mem (-2 * cubicKernel 4 1 2))
  ((isMinOn_cubic_fill_iff (by norm_num) (add_smul_single_apply_of_mem _)).mpr
    (by simp; ring))

/-- The fill `ψ(f)` decodes to `f` alone. -/
lemma isMinOn_norm_stdAddChar_sub_iff (f c : ZMod p) :
    IsMinOn (fun c => ‖ZMod.stdAddChar f - ZMod.stdAddChar c‖) Set.univ c ↔ c = f := by
  rw [isMinOn_univ_iff]
  constructor
  · intro h
    have h1 := h f
    rw [sub_self, norm_zero] at h1
    exact (ZMod.injective_stdAddChar (sub_eq_zero.mp (norm_le_zero_iff.mp h1))).symm
  · rintro rfl c'
    simp

example := isMinOn_norm_stdAddChar_sub_iff (p := 5) 1 1

/-- **Corollary** of `review/flat_spectrum.md` for `a + b` ("which is right"): the
minimum-nuclear-norm fill of one missing cell `(a, b)` decodes to the truth `a + b` alone. -/
theorem isMinOn_decode_add_fill_iff (hp : 2 ≤ p) {a b : ZMod p}
    {X : Matrix (ZMod p) (ZMod p) ℂ}
    (hX : ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), X kl.1 kl.2 = addKernel p kl.1 kl.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ kl ∈ ({(a, b)}ᶜ : Set (ZMod p × ZMod p)), Y kl.1 kl.2 = addKernel p kl.1 kl.2} X)
    (c : ZMod p) :
    IsMinOn (fun c => ‖X a b - ZMod.stdAddChar c‖) Set.univ c ↔ c = a + b := by
  rw [(isMinOn_add_fill_iff hp hX).mp hmin]
  simpa [addKernel, taskKernel] using isMinOn_norm_stdAddChar_sub_iff (a + b) c

example := isMinOn_decode_add_fill_iff (p := 5) (by norm_num) (a := 1) (b := 2)
  (X := addKernel 5) (fun _ _ => rfl)
  ((isMinOn_add_fill_iff (by norm_num) (fun _ _ => rfl)).mpr rfl) 0

end Transformer.Grokking.FlatSpectrum
