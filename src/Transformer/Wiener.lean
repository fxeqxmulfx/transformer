/-
# Stochastic calculus for the Wiener process

Doob's `L²` inequalities for martingales indexed by `ℝ≥0` with right-continuous
paths: on a finite grid (`Wiener/DoobFinite.lean`), along nested dyadic grids
(`Wiener/DoobGrid.lean`), and over `[0, T]` (`Wiener/Doob.lean`).

For: arXiv:2604.01978v1, App. A.2 (the stochastic integral against a
cylindrical Wiener process) and §2.4.
-/

import Transformer.Wiener.DoobFinite
import Transformer.Wiener.DoobGrid
import Transformer.Wiener.Doob
