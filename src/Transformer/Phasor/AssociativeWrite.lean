module

public import Transformer.Phasor.ReflectionPhaseRead
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Learning associations between successive embeddings

Source: the associative write experiment in `experiments/phasor_value_transport/README.md`.
Every adjacent pair contributes to a matrix of associations. A projected
previous embedding is the key and the current embedding supplies the value.
The write corrects the old prediction at that key. The current query reads
the updated matrix. Every map is shared over the sequence.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer
open scoped InnerProductSpace

variable {D d H F K V m : ℕ}

/-- A smooth, bounded key even when the incoming projection is zero. -/
noncomputable def boundedAssocKey (x : EucSpace K) : EucSpace K :=
  (Real.sqrt (1 + ‖x‖^2))⁻¹ • x

/-- A normalized key has squared norm strictly below one. The amount left
over in a repeated write is exactly the reciprocal denominator. -/
theorem boundedAssocKey_complement (x : EucSpace K) :
    1 - ‖boundedAssocKey x‖^2 = (1+‖x‖^2)⁻¹ := by
  have hp : 0 < 1+‖x‖^2 := by positivity
  rw [boundedAssocKey, norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity),
    mul_pow, inv_pow, Real.sq_sqrt hp.le]
  field_simp
  ring

/-- One row of the association matrix is a linear functional of the key. -/
noncomputable def assocWrite (s k : EucSpace K) (v : ℝ) : EucSpace K :=
  s + (v - ⟪s,k⟫_ℝ) • k

/-- A query changes by the key-query inner product times the error of the
old value at the written key. This applies to arbitrary dense queries. -/
theorem assocWrite_read (s k q : EucSpace K) (v : ℝ) :
    ⟪assocWrite s k v, q⟫_ℝ = ⟪s,q⟫_ℝ + (v-⟪s,k⟫_ℝ)*⟪k,q⟫_ℝ := by
  rw [assocWrite, inner_add_left, real_inner_smul_left]

/-- After writing a normalized key, the residual on that key is the old
residual divided by one plus the squared norm of the raw key. -/
theorem assocWrite_key_error (s raw : EucSpace K) (v : ℝ) :
    let k := boundedAssocKey raw
    v - ⟪assocWrite s k v,k⟫_ℝ = (v-⟪s,k⟫_ℝ)/(1+‖raw‖^2) := by
  dsimp only
  rw [assocWrite_read, real_inner_self_eq_norm_sq]
  rw [div_eq_mul_inv, ← boundedAssocKey_complement]
  ring

/-- Simultaneously perturb the previous state, key and value. Both occurrences
of the key contribute to the directional derivative. -/
theorem assocWrite_hasDerivAt (s k ds dk : EucSpace K) (v dv : ℝ) :
    HasDerivAt (fun t : ℝ => assocWrite (s+t • ds) (k+t • dk) (v+t*dv))
      (ds + (dv - ⟪ds,k⟫_ℝ - ⟪s,dk⟫_ℝ) • k + (v-⟪s,k⟫_ℝ) • dk) 0 := by
  have line (u du : EucSpace K) : HasDerivAt (fun t : ℝ => u+t • du) du 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const du).const_add u
  have val : HasDerivAt (fun t : ℝ => v+t*dv) dv 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const dv).const_add v
  have read := HasDerivAt.inner ℝ (line s ds) (line k dk)
  have write := (line s ds).add ((val.sub read).smul (line k dk))
  convert write using 1 <;> first | rfl | (simp; module)

/-- The transpose of the linearized write transfers a cotangent to the old
state, scalar value and key. The key occurs in both the read and the write. -/
theorem assocWrite_adjoint (s k g ds dk : EucSpace K) (v dv : ℝ) :
    ⟪g, ds + (dv - ⟪ds,k⟫_ℝ - ⟪s,dk⟫_ℝ) • k + (v-⟪s,k⟫_ℝ) • dk⟫_ℝ =
      ⟪g - ⟪g,k⟫_ℝ • k, ds⟫_ℝ + ⟪g,k⟫_ℝ*dv +
      ⟪(v-⟪s,k⟫_ℝ) • g - ⟪g,k⟫_ℝ • s, dk⟫_ℝ := by
  simp only [inner_add_right, inner_sub_left, real_inner_smul_left, real_inner_smul_right]
  rw [real_inner_comm k ds]
  ring

/-- The first input has no preceding key. Subsequent writes use every
adjacent pair, including pairs where the two embeddings agree. -/
noncomputable def assocPrefix (k : ℕ → EucSpace K) (v : ℕ → Fin V → ℝ) :
    ℕ → Fin V → EucSpace K
  | 0 => 0
  | 1 => 0
  | n+2 => fun c => assocWrite (assocPrefix k v (n+1) c) (k n) (v (n+1) c)

/-- Agreement of the projected inputs in a prefix suffices for all its
associative state rows, including empty and single-vector sequences. -/
theorem assocPrefix_congr (k h : ℕ → EucSpace K) (v w : ℕ → Fin V → ℝ) (n : ℕ)
    (same : ∀ j < n, k j = h j ∧ v j = w j) : assocPrefix k v n = assocPrefix h w n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    cases n with
    | zero => rfl
    | succ n =>
      have prev := ih (fun j hj => same j (by omega))
      funext c
      simp only [assocPrefix, prev, (same n (by omega)).1, (same (n+1) (by omega)).2]

example (k : ℕ → EucSpace 8) (v : ℕ → Fin 16 → ℝ) (n : ℕ) :
    assocPrefix k v n = assocPrefix (fun j => if j<n then k j else 0)
      (fun j => if j<n then v j else 0) n :=
  assocPrefix_congr _ _ _ _ _ (fun j hj => by simp [hj])

/-- Four trainable linear projections shared by positions and classes. -/
structure AssocProjection (D K V : ℕ) where
  key : EucSpace D →ₗ[ℝ] EucSpace K
  query : EucSpace D →ₗ[ℝ] EucSpace K
  value : EucSpace D →ₗ[ℝ] (Fin V → ℝ)
  output : (Fin V → ℝ) →ₗ[ℝ] EucSpace D

/-- A single additional residual on the mixed rotation and reflection
backbone. Both paths take the same arbitrary sequence of embeddings. -/
noncomputable def EmbeddingBackbone.assocReflectionFeature (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (gate : Fin m → ℝ) (B : AssocProjection D K V)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  let vectors := fun j => GPTMini.rmsNormEps eps (x j)
  let states := assocPrefix (fun j => boundedAssocKey (B.key (vectors j)))
    (fun j => B.value (vectors j)) (i+1)
  A.reflectionPhaseFeature P gate θ eps scale x l i +
    B.output (fun c => ⟪states c, boundedAssocKey (B.query (vectors i))⟫_ℝ)

/-- A zero output projection preserves the entire mixed reflection model
at initialization for all values of the other three projections. -/
theorem EmbeddingBackbone.assocReflectionFeature_zero (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (gate : Fin m → ℝ) (B : AssocProjection D K V)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    A.assocReflectionFeature P gate {B with output := 0} θ eps scale x l i =
      A.reflectionPhaseFeature P gate θ eps scale x l i := by
  simp [EmbeddingBackbone.assocReflectionFeature]

/-- All writes, the query and the complete backbone output depend only on
the prefix through the current embedding. -/
theorem EmbeddingBackbone.assocReflectionFeature_congr_prefix (A : EmbeddingBackbone D d H F)
    (P : PhasePrefixRead D m) (gate : Fin m → ℝ) (B : AssocProjection D K V)
    (θ : StreamReadWeights D) (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ)
    (h : ∀ j ≤ i, x j = y j) :
    A.assocReflectionFeature P gate B θ eps scale x l i =
      A.assocReflectionFeature P gate B θ eps scale y l i := by
  let key := fun z => boundedAssocKey (B.key (GPTMini.rmsNormEps eps z))
  let val := fun z => B.value (GPTMini.rmsNormEps eps z)
  have states : assocPrefix (key ∘ x) (val ∘ x) (i+1) = assocPrefix (key ∘ y) (val ∘ y) (i+1) :=
    assocPrefix_congr _ _ _ _ _ (fun j hj => by simp only [Function.comp_apply, h j (by omega), and_self])
  unfold EmbeddingBackbone.assocReflectionFeature
  rw [A.reflectionPhaseFeature_congr_prefix P gate θ eps scale x y l i h]
  let query := fun z => boundedAssocKey (B.query (GPTMini.rmsNormEps eps z))
  change _ + B.output (fun c => ⟪assocPrefix (key ∘ x) (val ∘ x) (i+1) c, query (x i)⟫_ℝ) =
    _ + B.output (fun c => ⟪assocPrefix (key ∘ y) (val ∘ y) (i+1) c, query (y i)⟫_ℝ)
  rw [states, h i (le_refl i)]

example (A : EmbeddingBackbone 128 32 4 171) (P : PhasePrefixRead 128 32)
    (g : Fin 32 → ℝ) (B : AssocProjection 128 8 4) (θ : StreamReadWeights 128)
    (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.assocReflectionFeature P g B θ 1 1 x l 5 =
      A.assocReflectionFeature P g B θ 1 1 (fun j => if j ≤ 5 then x j else fill j) l 5 :=
  A.assocReflectionFeature_congr_prefix _ _ _ _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor
