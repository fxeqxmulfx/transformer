module

public import Transformer.Grokking.FiniteRecovery.DivisionAttentionWeights

/-! The two-head causal routing step works for any linear value features
and any query functional with the specified five positional amplitudes.
Only the numerator and denominator positions carry value features. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2 Finset
open scoped ComplexConjugate

noncomputable def selector (q : Stream →ₗ[ℂ] ℂ) : Stream →ₗ[ℂ] Heads 4 16 :=
  LinearMap.pi fun h => LinearMap.pi fun m => if h = 1 ∧ m = 0 then q else 0

noncomputable def headSlot (h : Fin 4) (m : Fin 16) : Heads 4 16 →ₗ[ℂ] ℂ :=
  (LinearMap.proj (R := ℂ) (φ := fun _ : Fin 16 => ℂ) m).comp
    (LinearMap.proj (R := ℂ) (φ := fun _ : Fin 4 => Fin 16 → ℂ) h)

noncomputable def operandSlot (numerator : Bool) (m : Fin 16) : Heads 4 16 →ₗ[ℂ] ℂ :=
  let u := mix 0 1
  let p := mix 1 1
  let q := mix 1 3
  (p-q)⁻¹ • (if numerator then headSlot 1 m - (q/u) • headSlot 0 m
    else (p/u) • headSlot 0 m - headSlot 1 m)

noncomputable def routeMap (q : Stream →ₗ[ℂ] ℂ) (value : Fin 16 → Stream →ₗ[ℂ] ℂ) :
    Attention 64 4 16 where
  WQ := selector q
  WK := selector q
  WV := LinearMap.pi fun _ => LinearMap.pi value
  WO := Parity.assemble fun i => if i = 5 then operandSlot true 0 else
    if i = 6 then operandSlot false 1 else if i = 7 then operandSlot true 2 else 0

noncomputable def weightedValues (v : Fin 5 → Fin 16 → ℂ) (h : Fin 4) (m : Fin 16) : ℂ :=
  ∑ j, mix h j * v j m

theorem native_weightedValues (q : Stream →ₗ[ℂ] ℂ) (value : Fin 16 → Stream →ₗ[ℂ] ℂ)
    (z : Fin 5 → Stream) (hq : ∀ j, q (z j) = (posGain j : ℂ)) :
    attentionWith (causalGeometry 0) (routeMap q value) Parity.offset 1 z (4 : Fin 5) =
      (routeMap q value).WO (weightedValues fun j m => value m (z j)) := by
  have native (j : Fin 5) (h : Fin 4) : selector q (z j) h = rawHead h j := by
    funext m
    change (if h=1 ∧ m=0 then q else (0 : Stream →ₗ[ℂ] ℂ)) _ = _
    by_cases he : h=1 ∧ m=0 <;> simp [rawHead,he,hq]
  let geo := causalGeometry (T := 5) (0 : Fin 16 → ℝ)
  let head (j : Fin 5) (h : Fin 4) := geo.rotate j (normalizeHead Parity.offset (selector q (z j) h))
  let score (h : Fin 4) (j : Fin 5) :=
    (1 : ℂ) * ((Real.log ((geo.time (4 : Fin 5) : ℝ)+1) / Real.sqrt 16 : ℝ) : ℂ) *
      ∑ m, conj (head 4 h m) * head j h m
  have hs (h : Fin 4) : score h = posScore h := by
    funext j
    have ht : (geo.time (4:Fin 5) : ℝ)+1 = 5 := by change ((4:ℕ):ℝ)+1 = 5; norm_num
    dsimp only [score,head]
    rw [native 4 h,native j h,ht,one_mul]
    rfl
  apply congrArg (routeMap q value).WO
  funext h m
  change (∑ j, (⟨maskedSoftmax (geo.visible 4) (fun k => (score h k).re) j,
    maskedSoftmax (geo.visible 4) (fun k => (score h k).im) j⟩ : ℂ) * value m (z j)) = _
  rw [hs h]
  rfl

private theorem solve_mixture (v : Fin 5 → Fin 16 → ℂ)
    (hz : ∀ m, v 0 m = 0 ∧ v 2 m = 0 ∧ v 4 m = 0) (m : Fin 16) :
    operandSlot true m (weightedValues v) = v 1 m ∧ operandSlot false m (weightedValues v) = v 3 m := by
  let p := mix 1 1
  let q := mix 1 3
  let u := mix 0 1
  have hu : u ≠ 0 := by
    intro h
    have he := congrArg Complex.re h
    norm_num [u,mix_uniform] at he
  have hd : p-q ≠ 0 := sub_ne_zero.mpr mix_unequal
  have h3 : mix 0 3 = u := (mix_uniform 3).trans (mix_uniform 1).symm
  have two (h : Fin 4) : weightedValues v h m = mix h 1 * v 1 m + mix h 3 * v 3 m := by
    unfold weightedValues
    rw [Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_one]
    change mix h 0 * v 0 m + (mix h 1 * v 1 m + (mix h 2 * v 2 m + (mix h 3 * v 3 m + mix h 4 * v 4 m))) = _
    rw [(hz m).1,(hz m).2.1,(hz m).2.2]
    ring
  change (p-q)⁻¹ * (weightedValues v 1 m - (q/u)*weightedValues v 0 m) = _ ∧
    (p-q)⁻¹ * ((p/u)*weightedValues v 0 m - weightedValues v 1 m) = _
  rw [two,two,h3]
  change (p-q)⁻¹ * (p*_+q*_-(q/u)*(u*_+u*_)) = _ ∧
    (p-q)⁻¹ * ((p/u)*(u*_+u*_)-(p*_+q*_)) = _
  constructor <;> field_simp [hu,hd] <;> ring

/-- A native head pair writes three arbitrary linear operand features into
scratch channels 5, 6 and 7, without changing any classifier channel. -/
theorem route_features (q : Stream →ₗ[ℂ] ℂ) (value : Fin 16 → Stream →ₗ[ℂ] ℂ) (z : Fin 5 → Stream)
    (hq : ∀ j, q (z j) = (posGain j : ℂ))
    (hz : ∀ m, value m (z 0) = 0 ∧ value m (z 2) = 0 ∧ value m (z 4) = 0) (i : Fin 64) :
    attentionWith (causalGeometry 0) (routeMap q value) Parity.offset 1 z (4 : Fin 5) i =
      if i = 5 then value 0 (z 1) else if i = 6 then value 1 (z 3)
      else if i = 7 then value 2 (z 1) else 0 := by
  rw [native_weightedValues _ _ _ hq]
  let v := fun j m => value m (z j)
  change (if i=5 then operandSlot true 0 else if i=6 then operandSlot false 1 else if i=7 then operandSlot true 2 else
    (0 : Heads 4 16 →ₗ[ℂ] ℂ)) (weightedValues v) = _
  split
  · exact (solve_mixture v hz 0).1
  · split
    · exact (solve_mixture v hz 1).2
    · split
      · exact (solve_mixture v hz 2).1
      · rfl

end Transformer.Grokking.FiniteRecovery.Argand.Division
