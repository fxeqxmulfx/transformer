module

public import Transformer.CRASP.Rotary.Model

/-!
# Uniform counting theory for exact nonperiodic RoPE

Extension of arXiv:2506.16055v3, Appendix B.2/F, specified in
`review/flat_spectrum.md`, *Uniform nonperiodic RoPE*, U1--U4.
Position-dependent weighted counts replace the finite periodic position code.
One fixed depth-D formula simulates the rounded normalized source stack on
every finite original-token word, with the exact source geometric phases.
Full attention quotients and block updates agree with the earlier rounded
model on every finite context. No strict depth hierarchy or IEEE error bound
is claimed by this simulation theorem.
-/
