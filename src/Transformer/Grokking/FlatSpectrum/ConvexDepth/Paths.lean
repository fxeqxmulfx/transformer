module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Basic
public import Transformer.Residual.Basic

/-!
# Joint weights on genuinely composed nonlinear block paths

Source: `review/convex_depth.md`, CD5. The trainable object is a joint tensor over
complete block choices. Every path evaluates its own nonlinear hidden states; no
block is evaluated at a prematurely averaged state. This is a convex lifting of
finite block selection, not convex training of independent continuous block factors.
Plain residuals retain exactly the recurrence in arXiv:2512.24880v2, Eq. (1).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Landscape.Neural (crossEntropy)

/-- A full sequence of choices, with the latest block first in the representation. -/
def Path (J : Type) : ℕ → Type
  | 0 => Unit
  | d + 1 => J × Path J d

instance pathFintype (J : Type) [Fintype J] (d : ℕ) : Fintype (Path J d) := by
  induction d with
  | zero => exact inferInstanceAs (Fintype Unit)
  | succ d ih =>
    letI := ih
    exact inferInstanceAs (Fintype (J × Path J d))

variable {J : Type} {E : Type*}

/-- Execute all chosen blocks on their own preceding hidden state (CD5).
Block functions may be nonlinear and input-dependent. -/
def pathRun (B : ℕ → J → E → E) : (d : ℕ) → Path J d → E → E
  | 0, _, x => x
  | d + 1, p, x => B d p.1 (pathRun B d p.2 x)

/-- Embed a fixed layer-by-layer choice into the joint path coordinates (CD5). -/
def choicePath (choices : ℕ → J) : (d : ℕ) → Path J d
  | 0 => ()
  | d + 1 => (choices d, choicePath choices d)

variable [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Plain residual candidates, with unrestricted nonlinear branches (CD5). -/
def residualBlocks (F : ℕ → J → E → E) (k : ℕ) (j : J) (h : E) : E :=
  h + F k j h

/-- Flat-spectrum connection candidates, now permitting branches that read and
interact with earlier path-specific hidden features (CD5). -/
def connectionBlocks (U : ℕ → E →L[ℝ] E) (F : ℕ → J → E → E)
    (k : ℕ) (j : J) (h : E) : E := U k h + F k j h

/-- Identity skip transport recovers ordinary residual connections exactly (CD5). -/
theorem connection_identity (F : ℕ → J → E → E) :
    connectionBlocks (fun _ => ContinuousLinearMap.id ℝ E) F = residualBlocks F := by
  funext k j h
  simp [connectionBlocks, residualBlocks]

omit [NormedSpace ℝ E] in
/-- Every deterministic nonlinear residual stack is embedded without changing any
hidden state or branch evaluation (CD5); there is no frozen-feature substitution. -/
theorem pathRun_residual (F : ℕ → J → E → E) (choices : ℕ → J) (x : E) (d : ℕ) :
    pathRun (residualBlocks F) d (choicePath choices d) x =
      Transformer.Residual.residualRun (fun k => F k (choices k)) x d := by
  induction d with
  | zero => rfl
  | succ d ih =>
    simp only [pathRun, choicePath, residualBlocks, Transformer.Residual.residualRun, ih]

variable [Fintype J]

/-- The entire learned joint tensor is nonnegative and normalized (CD5).
No product-factorization constraint is imposed on different depths. -/
def pathDomain (d : ℕ) : Set (Path J d → ℝ) :=
  {w | (∀ p, 0 ≤ w p) ∧ ∑ p, w p = 1}

/-- The joint path domain is convex (CD5). -/
theorem pathDomain_convex (d : ℕ) : Convex ℝ (pathDomain (J := J) d) := by
  intro w hw v hv a b ha hb hab
  constructor
  · intro p
    exact add_nonneg (mul_nonneg ha (hw.1 p)) (mul_nonneg hb (hv.1 p))
  · change (∑ p, (a * w p + b * v p)) = 1
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hw.2, hv.2]
    simpa using hab

/-- Average only after complete nonlinear path evaluation (CD5). -/
noncomputable def pathPredict (B : ℕ → J → E → E) (d : ℕ)
    (w : Path J d → ℝ) (x : E) : E := ∑ p, w p • pathRun B d p x

/-- Nonlinear interactions within each path do not spoil linearity in the joint
path tensor (CD5). This property would fail for independent layer factors. -/
theorem pathPredict_linear (B : ℕ → J → E → E) (d : ℕ)
    (w v : Path J d → ℝ) (x : E) (a b : ℝ) :
    pathPredict B d (a • w + b • v) x =
      a • pathPredict B d w x + b • pathPredict B d v x := by
  simp only [pathPredict, Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_smul,
    mul_smul, Finset.sum_add_distrib, Finset.smul_sum]

/-- A deterministic stack is a vertex of the learned tensor domain (CD5). -/
noncomputable def pathVertex {d : ℕ} (p : Path J d) : Path J d → ℝ := by
  classical
  exact fun q => if q = p then 1 else 0

/-- Every complete block choice is feasible in the lifting (CD5). -/
theorem pathVertex_mem {d : ℕ} (p : Path J d) : pathVertex p ∈ pathDomain d := by
  classical
  constructor
  · intro q
    simp only [pathVertex]
    split_ifs <;> norm_num
  · simp [pathVertex]

/-- The lifted prediction reproduces a deterministic deep network exactly (CD5). -/
theorem pathPredict_vertex (B : ℕ → J → E → E) (d : ℕ) (p : Path J d) (x : E) :
    pathPredict B d (pathVertex p) x = pathRun B d p x := by
  classical
  simp [pathPredict, pathVertex]

/-- The generic dense lifting has exponentially many coordinates in depth (CD5).
This is a proved cost of the construction, not an efficient-training claim. -/
theorem path_card (d : ℕ) : Fintype.card (Path J d) = Fintype.card J ^ d := by
  induction d with
  | zero => rfl
  | succ d ih =>
    change Fintype.card (J × Path J d) = _
    rw [Fintype.card_prod, ih, pow_succ, Nat.mul_comm]

end Transformer.Grokking.FlatSpectrum.ConvexDepth
