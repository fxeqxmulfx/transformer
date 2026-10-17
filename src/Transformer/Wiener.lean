/-
# Stochastic calculus for the Wiener process

Brownian motion, the cylindrical Wiener process on `L²(ρ)` and the filtrations
built from it (`Wiener/Defs.lean`); their moments, the martingale property, the
covariance `𝔼[W(t)[h] W(s)[g]] = (s ∧ t) ⟪h, g⟫` and the bridge to Mathlib's
`IsBrownianReal` (`Wiener/Basic.lean`).  The usual augmentation keeps a Brownian
motion Brownian (`Wiener/Augment.lean`), so a cylindrical Wiener process is one for
`ℱ^W` and for the filtration of an independent initial datum and the noise
(`Wiener/Filtration.lean`).  Doob's `L²` inequalities for martingales
indexed by `ℝ≥0` with right-continuous paths: on a finite grid
(`Wiener/DoobFinite.lean`), along nested dyadic grids (`Wiener/DoobGrid.lean`),
and over `[0, T]` (`Wiener/Doob.lean`).

For: arXiv:2604.01978v1, App. A.2 (the stochastic integral against a
cylindrical Wiener process) and §2.4.
-/

import Transformer.Wiener.Defs
import Transformer.Wiener.Basic
import Transformer.Wiener.Augment
import Transformer.Wiener.Filtration
import Transformer.Wiener.DoobFinite
import Transformer.Wiener.DoobGrid
import Transformer.Wiener.Doob
