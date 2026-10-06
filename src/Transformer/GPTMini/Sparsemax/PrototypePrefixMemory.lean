import Transformer.GPTMini.Sparsemax.PrototypeKernelTraining
import Transformer.GPTMini.Sparsemax.CausalPrefixKeys
import Transformer.GPTMini.Sparsemax.BoundedGramWidth

/-!
# Compact kernel memory from actual causal prefixes

Derived architecture for sparsemax arXiv:1602.02068v2, Eq. (1), and
`attn @ v` at `73f8a0b`. Register finitely many observed prefixes and
their fixed position/token features. Complete masked signatures supply
identity comparisons directly; they are never enumerated into a dictionary
of every possible text. Kernel profiles address the registered memory.

Future changes in either queries or registered prototypes leave the codes
unchanged. On distinct registered prefixes the actual learned-memory block
realizes arbitrary vector targets with one common original value table.
The guarantee applies to every feasible learned Gram and the earlier convex
joint chart, with one slot per registered prefix and no teacher routes.

The exact guarantee concerns these registered observations. Unseen queries
receive their normalized kernel prediction, without a generalization theorem.
Features and prototype selection are fixed from data during optimization;
the learned memory Gram is separate. Duplicate observations can share a slot.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Direct prefix comparisons and feature kernels define a compact memory input.
Source: the derived causal prototype architecture for arXiv:1602.02068v2, Eq. (1).
The memory index depends on registered prototypes, not all possible signatures. -/
def prefixPrototypeCodes {R T V N F : ℕ} (key : Fin T → Fin V → Fin (F + 1))
    (queries : Fin R → Fin T → Fin V) (queryRows : Fin R → Fin T)
    (prototypes : Fin (N + 1) → Fin T → Fin V) (prototypeRows : Fin (N + 1) → Fin T) :
    Matrix (Fin R) (Fin (N + 1)) ℝ :=
  prototypeKernelCodes (fun r => causalPrefixSignature (queries r) (queryRows r))
    (fun j => causalPrefixSignature (prototypes j) (prototypeRows j))
    (contextEncodedPrefixCodes key queries queryRows)
    (contextEncodedPrefixCodes key prototypes prototypeRows)

/-- Every observed or unseen query produces a valid probability input to learned memory.
Source: positive normalization of the derived arXiv:1602.02068v2, Eq. (1) kernel. -/
theorem prefixPrototypeCodes_mem {R T V N F : ℕ} (key : Fin T → Fin V → Fin (F + 1))
    (queries : Fin R → Fin T → Fin V) (queryRows : Fin R → Fin T)
    (prototypes : Fin (N + 1) → Fin T → Fin V) (prototypeRows : Fin (N + 1) → Fin T) :
    prefixPrototypeCodes key queries queryRows prototypes prototypeRows ∈ contextCodeDomain R N :=
  prototypeKernelCodes_mem _ _ _ _

/-- Query codes depend only on the observed query prefix, including for unseen observations.
Source: masked signatures and causal feature probabilities before arXiv:1602.02068v2, Eq. (1). -/
theorem prefixPrototypeCodes_causal {R T V N F : ℕ} (key : Fin T → Fin V → Fin (F + 1))
    (queries other : Fin R → Fin T → Fin V) (queryRows : Fin R → Fin T)
    (prototypes : Fin (N + 1) → Fin T → Fin V) (prototypeRows : Fin (N + 1) → Fin T)
    (h : ∀ r j, j ≤ queryRows r → queries r j = other r j) :
    prefixPrototypeCodes key queries queryRows prototypes prototypeRows =
      prefixPrototypeCodes key other queryRows prototypes prototypeRows := by
  have he : (fun r => causalPrefixSignature (queries r) (queryRows r)) =
      (fun r => causalPrefixSignature (other r) (queryRows r)) := by
    funext r
    exact (causalPrefixSignature_eq_iff _ _ _ _).mpr ⟨rfl, h r⟩
  unfold prefixPrototypeCodes
  rw [he, contextEncodedPrefixCodes_eq_of_visible key queries other queryRows h]

/-- The compact block's actual joint forward is causal for all parameter assignments.
Source: the derived prefix-kernel input before sparsemax arXiv:1602.02068v2, Eq. (1). -/
theorem jointPrefixPrototypeForward_causal {R T V N F D : ℕ}
    (key : Fin T → Fin V → Fin (F + 1)) (queries other : Fin R → Fin T → Fin V)
    (queryRows : Fin R → Fin T) (prototypes : Fin (N + 1) → Fin T → Fin V)
    (prototypeRows : Fin (N + 1) → Fin T)
    (p : EmbeddingGram (N + 1) × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (h : ∀ r j, j ≤ queryRows r → queries r j = other r j) :
    jointContextMemoryForward (prefixPrototypeCodes key queries queryRows prototypes prototypeRows) p =
      jointContextMemoryForward (prefixPrototypeCodes key other queryRows prototypes prototypeRows) p := by
  rw [prefixPrototypeCodes_causal key queries other queryRows prototypes prototypeRows h]

/-- A future query change, feasible learned Gram and nonconstant values inhabit causality. -/
example : jointContextMemoryForward (prefixPrototypeCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
    (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0)
    (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 1))
    (memoryIdentityGram 1, Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 3 : ℝ))) =
    jointContextMemoryForward (prefixPrototypeCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
      (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2)) (fun _ => 0)
      (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 1))
      (memoryIdentityGram 1, Matrix.of (fun r : Fin 2 => fun _ : Fin 1 => (r.val + 3 : ℝ))) := by
  apply jointPrefixPrototypeForward_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- Actual future query changes inhabit the compact-code causality premise. -/
example : prefixPrototypeCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
    (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0)
    (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 1) =
    prefixPrototypeCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
      (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2)) (fun _ => 0)
      (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 1) := by
  apply prefixPrototypeCodes_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- Registering a prototype uses only its own observed prefix, not its hidden future tokens.
Source: the derived signature and feature encoder before arXiv:1602.02068v2, Eq. (1). -/
theorem prefixPrototypeCodes_prototypes_causal {R T V N F : ℕ}
    (key : Fin T → Fin V → Fin (F + 1)) (queries : Fin R → Fin T → Fin V)
    (queryRows : Fin R → Fin T) (prototypes other : Fin (N + 1) → Fin T → Fin V)
    (prototypeRows : Fin (N + 1) → Fin T)
    (h : ∀ r j, j ≤ prototypeRows r → prototypes r j = other r j) :
    prefixPrototypeCodes key queries queryRows prototypes prototypeRows =
      prefixPrototypeCodes key queries queryRows other prototypeRows := by
  have he : (fun r => causalPrefixSignature (prototypes r) (prototypeRows r)) =
      (fun r => causalPrefixSignature (other r) (prototypeRows r)) := by
    funext r
    exact (causalPrefixSignature_eq_iff _ _ _ _).mpr ⟨rfl, h r⟩
  unfold prefixPrototypeCodes
  rw [he, contextEncodedPrefixCodes_eq_of_visible key prototypes other prototypeRows h]

/-- Different hidden prototype continuations inhabit the registration-causality premise. -/
example : prefixPrototypeCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
    (fun _ : Fin 1 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 1)
    (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 0) =
    prefixPrototypeCodes (fun _ : Fin 2 => fun k : Fin 2 => k)
      (fun _ : Fin 1 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 1)
      (fun r : Fin 2 => fun j : Fin 2 => if j = 0 then r else (1 : Fin 2)) (fun _ => 0) := by
  apply prefixPrototypeCodes_prototypes_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- One compact original value table fits all targets on distinct registered causal prefixes.
Source: arXiv:1602.02068v2, Eq. (1), the proved data inverse and the common value decoder. -/
theorem contextMemory_prefixPrototype_targets {T V N F D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (key : Fin T → Fin V → Fin (F + 1))
    (prototypes : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (Y : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hs : Function.Injective (fun r => causalPrefixSignature (prototypes r) (rows r)))
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor) :
    ∃ values : Matrix (Fin (N + 1)) (Fin D) ℝ,
      contextMemoryValueOutput G (prefixPrototypeCodes key prototypes rows prototypes rows) values =
        Y := by
  refine ⟨recoverMemoryValues G (prototypeTrainingRightInverse
    (fun r => causalPrefixSignature (prototypes r) (rows r))
    (contextEncodedPrefixCodes key prototypes rows) * Y), ?_⟩
  exact contextMemory_prototype_target cap floor G _ _ Y hs hf hG

/-- The earlier three repeated-token prefixes are genuinely distinct causal observations.
Source: the concrete arXiv:1602.02068v2, Eq. (1) data witness, now used as prototypes. -/
theorem memoryExamplePrefix_injective :
    Function.Injective (fun r => causalPrefixSignature (memoryExampleTokens r) (2 : Fin 3)) := by
  intro r s h
  have he := (causalPrefixSignature_eq_iff _ _ _ _).mp h
  have h0 := he.2 0 (by norm_num)
  have h1 := he.2 1 (by norm_num)
  fin_cases r <;> fin_cases s <;>
    first | rfl | exact absurd h0 (by decide) | exact absurd h1 (by decide)

/-- Three memory slots fit the previously excluded triple on the actual causal prefixes.
Source: the compact arXiv:1602.02068v2, Eq. (1) architecture, with all premises inhabited.
Earlier position/token memory used six slots and full signatures used twenty-seven. -/
example : ∃ values : Matrix (Fin 3) (Fin 1) ℝ,
    contextMemoryValueOutput (memoryIdentityGram 2)
      (prefixPrototypeCodes positionExampleKey memoryExampleTokens (fun _ => 2)
        memoryExampleTokens (fun _ => 2)) values =
      Matrix.of (fun r : Fin 3 => fun _ : Fin 1 => if r = 2 then 1 else 0) :=
  contextMemory_prefixPrototype_targets 1 (3 / 4) _ _ _ _ _ memoryExamplePrefix_injective
    (by norm_num) (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

/-- Exact prototype interpolation needs only P memory slots and Q/K width 2P.
Source: the derived compact arXiv:1602.02068v2, Eq. (1) architecture.
Both embedding recovery and the common value table are proved for every feasible Gram. -/
theorem contextMemory_prefixPrototype_compact {T V N F D : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (key : Fin T → Fin V → Fin (F + 1))
    (prototypes : Fin (N + 1) → Fin T → Fin V) (rows : Fin (N + 1) → Fin T)
    (Y : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hs : Function.Injective (fun r => causalPrefixSignature (prototypes r) (rows r)))
    (hf : 1 / 2 < floor) (hG : G ∈ memoryGramDomain N cap floor) :
    ∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
      ∃ values : Matrix (Fin (N + 1)) (Fin D) ℝ, G = featureGram features ∧
        contextMemoryValueOutput G (prefixPrototypeCodes key prototypes rows prototypes rows) values =
          Y := by
  obtain ⟨features, he⟩ := memoryGram_fixedWidth cap floor G hG
  obtain ⟨values, hv⟩ := contextMemory_prefixPrototype_targets cap floor G key prototypes rows Y hs hf hG
  exact ⟨features, values, he, hv⟩

/-- Three actual prefix targets use three shared memory slots and at most six Q/K coordinates. -/
example : ∃ features : Fin 6 → Sum (Fin 3) (Fin 3) → ℝ,
    ∃ values : Matrix (Fin 3) (Fin 1) ℝ, memoryIdentityGram 2 = featureGram features ∧
      contextMemoryValueOutput (memoryIdentityGram 2)
        (prefixPrototypeCodes positionExampleKey memoryExampleTokens (fun _ => 2)
          memoryExampleTokens (fun _ => 2)) values =
        Matrix.of (fun r : Fin 3 => fun _ : Fin 1 => if r = 2 then 1 else 0) :=
  contextMemory_prefixPrototype_compact 1 (3 / 4) _ _ _ _ _ memoryExamplePrefix_injective
    (by norm_num) (memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num))

end Transformer.GPTMini.Sparsemax
