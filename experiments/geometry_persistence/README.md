# Geometry persistence during continued training

Can a bound using the recorded AdamW updates certify the useful geometry
through the low-rate hold, and can old prediction certificates survive finite
parameter changes? The [frozen protocol](PROTOCOL.md) tests all three existing
eight-seed cohorts, every hold update, and 192 fixed endpoint windows.

The study is in progress. It separates observed-path guarantees from a
predictive theorem about unknown future gradients, and separates readout
energy concentration from full nonlinear prediction certificates.
