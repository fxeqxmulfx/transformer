//! The processor's treatment of subnormal numbers (`PROTOCOL.md`, Amendment 2).
//!
//! IEEE 754's gradual underflow is the default and the source's arithmetic. On x86-64 an
//! operation with a subnormal operand or result leaves the fast path and costs some hundred
//! times an ordinary one. Two bits of the register `MXCSR` replace such results
//! (flush-to-zero) and such operands (denormals-are-zero) by zero instead: the mode of
//! PyTorch's `torch.set_flush_denormal(True)`. The register belongs to a thread, so every
//! thread of a step sets it for itself.
//!
//! Rust's reference keeps the register's control bits at their defaults, so compiled Rust code
//! that runs in this mode is outside the language's guarantees: the compiler may evaluate an
//! expression of constants by the default rules. No constant of the trainer is subnormal, and
//! the overlap of Amendment 2 measures what the mode changes in a run.

use std::arch::asm;

/// The bits flush-to-zero (15) and denormals-are-zero (6) of `MXCSR`.
const FLUSH: u32 = 0x8040;

/// The thread's `MXCSR`.
fn mxcsr() -> u32 {
    let mut value = 0u32;
    // SAFETY: `stmxcsr` stores the register's 32 bits at the address given, a live `u32`.
    unsafe { asm!("stmxcsr dword ptr [{}]", in(reg) &mut value, options(nostack, preserves_flags)) };
    value
}

/// Whether this thread flushes subnormal results and operands to zero.
pub fn flushes_subnormals() -> bool {
    mxcsr() & FLUSH == FLUSH
}

/// Sets whether this thread flushes subnormal results and operands to zero. The rounding mode
/// and the exception masks stay as they are.
pub fn flush_subnormals(on: bool) {
    let value = if on { mxcsr() | FLUSH } else { mxcsr() & !FLUSH };
    // SAFETY: `ldmxcsr` loads the register from a live `u32`, the register's own value but for
    // the two bits: no reserved bit is set and no exception is unmasked.
    unsafe { asm!("ldmxcsr dword ptr [{}]", in(reg) &value, options(nostack, readonly, preserves_flags)) };
}

/// Gradual underflow again in the thread that drops it.
pub struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        flush_subnormals(false);
    }
}
