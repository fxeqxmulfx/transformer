# Unscored initial protocol

This is the original four-case P97 protocol frozen at commit 63c2d22a.
No GPU learning ran under it. The actual-size source audit found a last-bit
FP32 backward difference in the inherited attention port. The replacement
protocol uses the exact source subtraction graph, adds P193 before any GPU
outcomes are observed, and records per-parameter diagnostics every 1,000 updates.
The initial protocol is retained unchanged; its SHA-256 identifies the original
proposal, not a learning result. The synthetic cohort remains at 63c2d22a.
