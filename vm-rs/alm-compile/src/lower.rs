//! Lowering the hard WebAssembly operations, ported from
//! `compilation/lower.py`.
//!
//! The machine executes a fixed dispatch table: control flow, locals, globals,
//! byte and word memory, the comparisons, `add` and `sub`.  Everything else —
//! `mul`, the divisions and remainders, the bitwise operations, the shifts and
//! rotates, the counting instructions, the sign extensions — is rewritten here
//! into that subset before a program is turned into tokens.
//!
//! Almost every expansion is a subtraction loop, and the ones that are not use
//! byte 0 of linear memory as a shift register: storing a word and reloading it
//! at an offset is a byte shift, and `load8_s` / `load16_s` is a sign
//! extension.  The port is instruction for instruction, because the token
//! prefix is a transcription of this stream and a different-but-equivalent
//! expansion is a different program.
//!
//! The constant cases are reached only when the operand is a literal or a
//! local that is only ever assigned one literal (`find_const_locals`); a
//! genuinely dynamic operand falls to the loop forms at the end of
//! `lower_hard_ops`.

use std::collections::HashMap;

use crate::decoder::{op, FuncBody, Imm, Instr, Val};

/// Byte 0 of linear memory, the scratch word every byte-wise expansion uses.
pub const SCRATCH_ADDR: Val = 0;

/// The four temporaries `lower_hard_ops` appends to every lowered function.
const NUM_TEMPS: u32 = 4;

// ── Building instructions ───────────────────────────────────────────

fn nul(opcode: u8) -> Instr {
    Instr { opcode, imm: Imm::None }
}

/// `i32.const`.  The argument is an `i64` so the negative literals of the
/// original read the same here; the immediate itself is a 32-bit word.
fn konst(v: i64) -> Instr {
    Instr { opcode: op::I32_CONST, imm: Imm::I32(v as u32) }
}

/// An instruction carrying one index: the locals, the globals, `br`, `call`.
fn idx(opcode: u8, index: u32) -> Instr {
    Instr { opcode, imm: Imm::Index(index) }
}

/// A `block`, `loop` or `if`.  Every one the lowering emits is void (`0x40`).
fn blk(opcode: u8) -> Instr {
    Instr { opcode, imm: Imm::Block(0x40) }
}

/// A load or a store at the scratch word.  Every one the lowering emits is
/// unaligned (`align = 0`); the offset is the byte being addressed.
fn mem(opcode: u8, offset: u32) -> Instr {
    Instr { opcode, imm: Imm::Mem { align: 0, offset } }
}

// ── Arithmetic ──────────────────────────────────────────────────────

/// `x * C` by an addition chain over the bits of `C`, most significant first.
/// `x` is in `local_a` and the result is left on the stack.
fn expand_mul(c: Val, local_a: u32) -> Vec<Instr> {
    if c == 0 {
        return vec![konst(0)];
    }
    if c == 1 {
        return vec![idx(op::LOCAL_GET, local_a)];
    }
    if c == 0xFFFF_FFFF {
        return vec![konst(0), idx(op::LOCAL_GET, local_a), nul(op::I32_SUB)];
    }

    // Least significant bit first, so `bits.last()` is the leading one.
    let mut bits: Vec<bool> = Vec::new();
    let mut v = c;
    while v != 0 {
        bits.push(v & 1 == 1);
        v >>= 1;
    }

    let tmp = local_a + 1;
    let mut instrs = vec![idx(op::LOCAL_GET, local_a), idx(op::LOCAL_SET, tmp)];
    for i in (0..bits.len() - 1).rev() {
        instrs.extend([
            idx(op::LOCAL_GET, tmp),
            idx(op::LOCAL_GET, tmp),
            nul(op::I32_ADD),
            idx(op::LOCAL_SET, tmp),
        ]);
        if bits[i] {
            instrs.extend([
                idx(op::LOCAL_GET, tmp),
                idx(op::LOCAL_GET, local_a),
                nul(op::I32_ADD),
                idx(op::LOCAL_SET, tmp),
            ]);
        }
    }
    instrs.push(idx(op::LOCAL_GET, tmp));
    instrs
}

/// Unsigned `x / C` by repeated subtraction.  `x` is on the stack.
fn expand_div_u(c: Val, local_a: u32) -> Vec<Instr> {
    let local_n = local_a;
    let local_q = local_a + 1;
    vec![
        idx(op::LOCAL_SET, local_n),
        konst(0),
        idx(op::LOCAL_SET, local_q),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_n),
        konst(c as i64),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_n),
        konst(c as i64),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_n),
        idx(op::LOCAL_GET, local_q),
        konst(1),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_q),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_q),
    ]
}

/// Unsigned `x % C`: the same loop, keeping the dividend instead.
fn expand_rem_u(c: Val, local_a: u32) -> Vec<Instr> {
    let local_n = local_a;
    vec![
        idx(op::LOCAL_SET, local_n),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_n),
        konst(c as i64),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_n),
        konst(c as i64),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_n),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_n),
    ]
}

/// Signed `x / C`, truncating toward zero: take both absolute values, divide
/// unsigned, negate if the signs differed.
fn expand_div_s(c: Val, local_a: u32) -> Vec<Instr> {
    let c_signed = if c >= 1 << 31 { c as i64 - (1i64 << 32) } else { c as i64 };
    let abs_c = (c_signed.unsigned_abs() & 0xFFFF_FFFF) as Val;
    let c_negative = c_signed < 0;

    let local_x = local_a;
    let local_q = local_a + 1;
    let local_neg = local_a + 2;

    let mut instrs = vec![
        idx(op::LOCAL_SET, local_x),
        idx(op::LOCAL_GET, local_x),
        konst(0),
        nul(op::I32_LT_S),
        idx(op::LOCAL_SET, local_neg),
        blk(op::BLOCK),
        idx(op::LOCAL_GET, local_neg),
        nul(op::I32_EQZ),
        idx(op::BR_IF, 0),
        konst(0),
        idx(op::LOCAL_GET, local_x),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_x),
        nul(op::END),
    ];

    if c_negative {
        instrs.extend([
            idx(op::LOCAL_GET, local_neg),
            nul(op::I32_EQZ),
            idx(op::LOCAL_SET, local_neg),
        ]);
    }

    instrs.extend([
        konst(0),
        idx(op::LOCAL_SET, local_q),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_x),
        konst(abs_c as i64),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_x),
        konst(abs_c as i64),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_x),
        idx(op::LOCAL_GET, local_q),
        konst(1),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_q),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        blk(op::BLOCK),
        idx(op::LOCAL_GET, local_neg),
        nul(op::I32_EQZ),
        idx(op::BR_IF, 0),
        konst(0),
        idx(op::LOCAL_GET, local_q),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_q),
        nul(op::END),
        idx(op::LOCAL_GET, local_q),
    ]);

    instrs
}

// ── Counting ────────────────────────────────────────────────────────

/// `i32.clz`: double `x` until it is negative, counting the doublings.
fn expand_clz(local_a: u32) -> Vec<Instr> {
    let local_x = local_a;
    let local_count = local_a + 1;
    vec![
        idx(op::LOCAL_SET, local_x),
        konst(0),
        idx(op::LOCAL_SET, local_count),
        blk(op::BLOCK),
        blk(op::BLOCK),
        idx(op::LOCAL_GET, local_x),
        idx(op::BR_IF, 0),
        konst(32),
        idx(op::LOCAL_SET, local_count),
        idx(op::BR, 1),
        nul(op::END),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_x),
        konst(0),
        nul(op::I32_LT_S),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_x),
        idx(op::LOCAL_GET, local_x),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_x),
        idx(op::LOCAL_GET, local_count),
        konst(1),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_count),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_count),
    ]
}

/// `i32.ctz`: take the low bit through the scratch byte and a mod-2 loop,
/// halve, repeat.
fn expand_ctz(local_a: u32) -> Vec<Instr> {
    let local_x = local_a;
    let local_count = local_a + 1;
    let local_byte = local_a + 2;
    let local_q = local_a + 3;
    vec![
        idx(op::LOCAL_SET, local_x),
        konst(0),
        idx(op::LOCAL_SET, local_count),
        blk(op::BLOCK),
        blk(op::BLOCK),
        idx(op::LOCAL_GET, local_x),
        idx(op::BR_IF, 0),
        konst(32),
        idx(op::LOCAL_SET, local_count),
        idx(op::BR, 1),
        nul(op::END),
        blk(op::LOOP),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_x),
        mem(op::I32_STORE8, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD8_U, 0),
        idx(op::LOCAL_SET, local_byte),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_byte),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_byte),
        idx(op::BR_IF, 1),
        konst(0),
        idx(op::LOCAL_SET, local_q),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_x),
        konst(2),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_x),
        konst(2),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_x),
        idx(op::LOCAL_GET, local_q),
        konst(1),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_q),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_q),
        idx(op::LOCAL_SET, local_x),
        idx(op::LOCAL_GET, local_count),
        konst(1),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_count),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_count),
    ]
}

/// `i32.popcnt`: the same low bit, accumulated instead of stopped at.
fn expand_popcnt(local_a: u32) -> Vec<Instr> {
    let local_x = local_a;
    let local_count = local_a + 1;
    let local_byte = local_a + 2;
    let local_q = local_a + 3;
    vec![
        idx(op::LOCAL_SET, local_x),
        konst(0),
        idx(op::LOCAL_SET, local_count),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_x),
        nul(op::I32_EQZ),
        idx(op::BR_IF, 1),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_x),
        mem(op::I32_STORE8, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD8_U, 0),
        idx(op::LOCAL_SET, local_byte),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_byte),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_count),
        idx(op::LOCAL_GET, local_byte),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_count),
        konst(0),
        idx(op::LOCAL_SET, local_q),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_x),
        konst(2),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_x),
        konst(2),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_x),
        idx(op::LOCAL_GET, local_q),
        konst(1),
        nul(op::I32_ADD),
        idx(op::LOCAL_SET, local_q),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_q),
        idx(op::LOCAL_SET, local_x),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_count),
    ]
}

// ── Shifts and rotates ──────────────────────────────────────────────

/// `x << C` with `x` already in `local_a`.
///
/// For `C >= 8` the byte-aligned part is a store at `SCRATCH` and a load at
/// `SCRATCH + q`, which is what a little-endian word gives for free; the
/// remaining `r` bits are doublings.
fn expand_shl(c: u32, local_a: u32) -> Vec<Instr> {
    if c == 0 {
        return vec![idx(op::LOCAL_GET, local_a)];
    }
    if c >= 32 {
        return vec![konst(0)];
    }
    let (q, r) = (c / 8, c % 8);

    if q > 0 {
        let mut instrs = vec![
            konst(SCRATCH_ADDR as i64),
            konst(0),
            mem(op::I32_STORE, 0),
            konst(SCRATCH_ADDR as i64),
            idx(op::LOCAL_GET, local_a),
            mem(op::I32_STORE, q),
            konst(SCRATCH_ADDR as i64),
            mem(op::I32_LOAD, 0),
        ];
        for _ in 0..r {
            instrs.extend([
                idx(op::LOCAL_TEE, local_a),
                idx(op::LOCAL_GET, local_a),
                nul(op::I32_ADD),
            ]);
        }
        return instrs;
    }

    let mut instrs = vec![idx(op::LOCAL_GET, local_a)];
    for _ in 0..c {
        instrs.extend([idx(op::LOCAL_TEE, local_a), idx(op::LOCAL_GET, local_a), nul(op::I32_ADD)]);
    }
    instrs
}

/// `x << C` with `x` on the stack: `local.tee` saves the set-and-get of the
/// form above for the small shifts.
fn expand_shl_from_stack(c: u32, local_a: u32) -> Vec<Instr> {
    if c == 0 {
        return Vec::new();
    }
    if c >= 32 {
        return vec![nul(op::DROP), konst(0)];
    }
    let q = c / 8;

    if q > 0 {
        let mut instrs = vec![idx(op::LOCAL_SET, local_a)];
        instrs.extend(expand_shl(c, local_a));
        return instrs;
    }

    let mut instrs = Vec::new();
    for _ in 0..c {
        instrs.extend([idx(op::LOCAL_TEE, local_a), idx(op::LOCAL_GET, local_a), nul(op::I32_ADD)]);
    }
    instrs
}

/// Unsigned `x >> C` with `x` on the stack: the byte-aligned part is a load at
/// an offset with the word above zeroed, the rest is a division loop.
fn expand_shr_u(c: u32, local_a: u32) -> Vec<Instr> {
    if c >= 32 {
        return vec![nul(op::DROP), konst(0)];
    }
    let (q, r) = (c / 8, c % 8);

    if q > 0 {
        let mut instrs = vec![
            idx(op::LOCAL_SET, local_a),
            konst(SCRATCH_ADDR as i64),
            idx(op::LOCAL_GET, local_a),
            mem(op::I32_STORE, 0),
            konst(SCRATCH_ADDR as i64),
            konst(0),
            mem(op::I32_STORE, 4),
            konst(SCRATCH_ADDR as i64),
            mem(op::I32_LOAD, q),
        ];
        if r > 0 {
            instrs.extend(expand_div_u(1 << r, local_a));
        }
        return instrs;
    }

    expand_div_u(1 << c, local_a)
}

/// Signed `x >> C`: the byte-aligned cases reload the word with a sign-
/// extending load, which is the whole reason the scratch word exists.
fn expand_shr_s(c: u32, local_a: u32) -> Vec<Instr> {
    if c == 0 {
        return Vec::new();
    }
    if c >= 32 {
        return vec![
            idx(op::LOCAL_SET, local_a),
            konst(-1),
            konst(0),
            idx(op::LOCAL_GET, local_a),
            konst(0),
            nul(op::I32_LT_S),
            nul(op::SELECT),
        ];
    }

    let (q, r) = (c / 8, c % 8);

    if r == 0 {
        if q == 3 {
            return vec![
                idx(op::LOCAL_SET, local_a),
                konst(SCRATCH_ADDR as i64),
                idx(op::LOCAL_GET, local_a),
                mem(op::I32_STORE, 0),
                konst(SCRATCH_ADDR as i64),
                mem(op::I32_LOAD8_S, 3),
            ];
        }
        if q == 2 {
            return vec![
                idx(op::LOCAL_SET, local_a),
                konst(SCRATCH_ADDR as i64),
                idx(op::LOCAL_GET, local_a),
                mem(op::I32_STORE, 0),
                konst(SCRATCH_ADDR as i64),
                mem(op::I32_LOAD16_S, 2),
            ];
        }
        if q == 1 {
            // No `load24_s`, so the sign is put back by hand.
            let local_tmp = local_a + 1;
            return vec![
                idx(op::LOCAL_SET, local_a),
                konst(SCRATCH_ADDR as i64),
                idx(op::LOCAL_GET, local_a),
                mem(op::I32_STORE, 0),
                konst(SCRATCH_ADDR as i64),
                konst(0),
                mem(op::I32_STORE, 4),
                konst(SCRATCH_ADDR as i64),
                mem(op::I32_LOAD, 1),
                idx(op::LOCAL_SET, local_tmp),
                blk(op::BLOCK),
                idx(op::LOCAL_GET, local_tmp),
                konst(0x800000),
                nul(op::I32_LT_U),
                idx(op::BR_IF, 0),
                idx(op::LOCAL_GET, local_tmp),
                konst(0x1000000),
                nul(op::I32_SUB),
                idx(op::LOCAL_SET, local_tmp),
                nul(op::END),
                idx(op::LOCAL_GET, local_tmp),
            ];
        }
    }

    if q > 0 {
        let mut instrs = vec![
            idx(op::LOCAL_SET, local_a),
            konst(SCRATCH_ADDR as i64),
            idx(op::LOCAL_GET, local_a),
            mem(op::I32_STORE, 0),
            konst(SCRATCH_ADDR as i64),
            konst(0),
            mem(op::I32_STORE, 4),
            konst(SCRATCH_ADDR as i64),
            mem(op::I32_LOAD, q),
        ];
        if r > 0 {
            instrs.extend(expand_div_u(1 << r, local_a));
        }
        return instrs;
    }

    expand_div_u(1 << c, local_a)
}

/// `x rotl C` as `(x << c) + (x >> (32 - c))`.  `x` is on the stack.
fn expand_rotl_const(c: u32, local_a: u32) -> Vec<Instr> {
    let c = c & 31;
    if c == 0 {
        return Vec::new();
    }
    let local_saved_x = local_a + 2;
    let local_left = local_a + 3;

    let mut instrs = vec![
        idx(op::LOCAL_SET, local_a),
        idx(op::LOCAL_GET, local_a),
        idx(op::LOCAL_SET, local_saved_x),
    ];
    instrs.extend(expand_shl(c, local_a));
    instrs.push(idx(op::LOCAL_SET, local_left));
    instrs.push(idx(op::LOCAL_GET, local_saved_x));
    instrs.extend(expand_shr_u(32 - c, local_a));
    instrs.push(idx(op::LOCAL_GET, local_left));
    instrs.push(nul(op::I32_ADD));
    instrs
}

/// `x rotr C` as `(x >> c) + (x << (32 - c))`.  `x` is on the stack.
fn expand_rotr_const(c: u32, local_a: u32) -> Vec<Instr> {
    let c = c & 31;
    if c == 0 {
        return Vec::new();
    }
    let local_saved_x = local_a + 2;
    let local_right = local_a + 3;

    let mut instrs = vec![
        idx(op::LOCAL_SET, local_a),
        idx(op::LOCAL_GET, local_a),
        idx(op::LOCAL_SET, local_saved_x),
    ];
    instrs.push(idx(op::LOCAL_GET, local_a));
    instrs.extend(expand_shr_u(c, local_a));
    instrs.push(idx(op::LOCAL_SET, local_right));
    instrs.push(idx(op::LOCAL_GET, local_saved_x));
    instrs.push(idx(op::LOCAL_SET, local_a));
    instrs.extend(expand_shl(32 - c, local_a));
    instrs.push(idx(op::LOCAL_GET, local_right));
    instrs.push(nul(op::I32_ADD));
    instrs
}

// ── Bitwise ─────────────────────────────────────────────────────────

/// `x & 255`: store a byte and read it back.
fn expand_and_255(local_a: u32) -> Vec<Instr> {
    vec![
        idx(op::LOCAL_SET, local_a),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_a),
        mem(op::I32_STORE8, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD8_U, 0),
    ]
}

/// `i32.extend8_s`: the low byte, then `- 256` if it is at least `128`.
fn expand_extend8_s(local_a: u32) -> Vec<Instr> {
    vec![
        idx(op::LOCAL_SET, local_a),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_a),
        mem(op::I32_STORE8, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD8_U, 0),
        idx(op::LOCAL_SET, local_a),
        blk(op::BLOCK),
        idx(op::LOCAL_GET, local_a),
        konst(128),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 0),
        idx(op::LOCAL_GET, local_a),
        konst(256),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_a),
        nul(op::END),
        idx(op::LOCAL_GET, local_a),
    ]
}

/// `i32.extend16_s`: `store16` then `load16_s` does it in the memory.
fn expand_extend16_s(local_a: u32) -> Vec<Instr> {
    vec![
        idx(op::LOCAL_SET, local_a),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_a),
        mem(op::I32_STORE16, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD16_S, 0),
    ]
}

/// One byte of a bitwise operation against a constant mask.
///
/// The bits of `local_byte` come out most significant first: after bit `i` is
/// taken the value is below `2^i`, so `byte >= 2^i` is exactly that bit.  What
/// the result accumulates depends on the operation and on the mask bit —
/// `and` keeps the bit only where the mask is set, `or` sets it where the mask
/// is set and keeps it otherwise, `xor` flips it where the mask is set.
fn emit_byte_bitop(bitop: &str, mask_byte: u32, local_byte: u32, local_result: u32) -> Vec<Instr> {
    let mut out = vec![konst(0), idx(op::LOCAL_SET, local_result)];

    for i in (0..8).rev() {
        let mb = (mask_byte >> i) & 1;
        let add_if_set = (bitop == "and" && mb == 1)
            || (bitop == "or" && mb == 0)
            || (bitop == "xor" && mb == 0);
        let always_add = bitop == "or" && mb == 1;
        let add_if_clear = bitop == "xor" && mb == 1;

        let val = 1i64 << i;

        if add_if_clear {
            out.extend([
                idx(op::LOCAL_GET, local_byte),
                konst(val),
                nul(op::I32_GE_U),
                blk(op::IF),
                idx(op::LOCAL_GET, local_byte),
                konst(val),
                nul(op::I32_SUB),
                idx(op::LOCAL_SET, local_byte),
                nul(op::ELSE),
                idx(op::LOCAL_GET, local_result),
                konst(val),
                nul(op::I32_ADD),
                idx(op::LOCAL_SET, local_result),
                nul(op::END),
            ]);
        } else {
            out.extend([
                blk(op::BLOCK),
                idx(op::LOCAL_GET, local_byte),
                konst(val),
                nul(op::I32_LT_U),
                idx(op::BR_IF, 0),
                idx(op::LOCAL_GET, local_byte),
                konst(val),
                nul(op::I32_SUB),
                idx(op::LOCAL_SET, local_byte),
            ]);
            if add_if_set {
                out.extend([
                    idx(op::LOCAL_GET, local_result),
                    konst(val),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_result),
                ]);
            }
            out.push(nul(op::END));

            if always_add {
                out.extend([
                    idx(op::LOCAL_GET, local_result),
                    konst(val),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_result),
                ]);
            }
        }
    }
    out
}

/// A 32-bit `and` / `or` / `xor` against a constant: split the word into its
/// four bytes through the scratch address, do each byte that the mask does not
/// settle outright, and reload the word.
fn expand_bitop_general(bitop: &str, c: Val, local_a: u32) -> Vec<Instr> {
    let c_bytes = [c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, (c >> 24) & 0xFF];
    let local_x = local_a;
    let local_byte = local_a + 1;
    let local_result = local_a + 2;

    let mut instrs = vec![
        idx(op::LOCAL_SET, local_x),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_x),
        mem(op::I32_STORE, 0),
    ];

    for b in 0..4u32 {
        let cb = c_bytes[b as usize];

        if bitop == "and" {
            if cb == 0xFF {
                continue;
            }
            if cb == 0x00 {
                instrs.extend([konst(SCRATCH_ADDR as i64), konst(0), mem(op::I32_STORE8, b)]);
                continue;
            }
        } else if bitop == "or" {
            if cb == 0x00 {
                continue;
            }
            if cb == 0xFF {
                instrs.extend([konst(SCRATCH_ADDR as i64), konst(0xFF), mem(op::I32_STORE8, b)]);
                continue;
            }
        } else {
            if cb == 0x00 {
                continue;
            }
            if cb == 0xFF {
                instrs.extend([
                    konst(SCRATCH_ADDR as i64),
                    mem(op::I32_LOAD8_U, b),
                    idx(op::LOCAL_SET, local_byte),
                    konst(SCRATCH_ADDR as i64),
                    konst(255),
                    idx(op::LOCAL_GET, local_byte),
                    nul(op::I32_SUB),
                    mem(op::I32_STORE8, b),
                ]);
                continue;
            }
        }

        instrs.extend([
            konst(SCRATCH_ADDR as i64),
            mem(op::I32_LOAD8_U, b),
            idx(op::LOCAL_SET, local_byte),
        ]);
        instrs.extend(emit_byte_bitop(bitop, cb, local_byte, local_result));
        instrs.extend([
            konst(SCRATCH_ADDR as i64),
            idx(op::LOCAL_GET, local_result),
            mem(op::I32_STORE8, b),
        ]);
    }

    instrs.extend([konst(SCRATCH_ADDR as i64), mem(op::I32_LOAD, 0)]);
    instrs
}

fn expand_and_general(c: Val, local_a: u32) -> Vec<Instr> {
    if c == 0xFFFF_FFFE {
        return expand_and_fffffffe(local_a);
    }
    if c == 1 {
        return expand_and_1(local_a);
    }
    expand_bitop_general("and", c, local_a)
}

/// `x & 1`, which is the low byte modulo two.
fn expand_and_1(local_a: u32) -> Vec<Instr> {
    let local_x = local_a;
    let local_byte = local_a + 1;
    vec![
        idx(op::LOCAL_SET, local_x),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_x),
        mem(op::I32_STORE8, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD8_U, 0),
        idx(op::LOCAL_SET, local_byte),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_byte),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_byte),
    ]
}

/// `x & 0xFFFFFFFE`, which is `x - (x & 1)`.
fn expand_and_fffffffe(local_a: u32) -> Vec<Instr> {
    let local_x = local_a;
    let local_byte = local_a + 1;
    vec![
        idx(op::LOCAL_SET, local_x),
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_x),
        mem(op::I32_STORE8, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD8_U, 0),
        idx(op::LOCAL_SET, local_byte),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_byte),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_x),
        idx(op::LOCAL_GET, local_byte),
        nul(op::I32_SUB),
    ]
}

/// `x & 0x7FFFFFFE`: clear bit 0 as above, then bit 31 by a conditional
/// subtraction.
fn expand_and_7ffffffe_v2(local_a: u32) -> Vec<Instr> {
    let local_x = local_a;
    let local_byte = local_a + 1;

    let mut instrs = vec![idx(op::LOCAL_SET, local_x)];

    instrs.extend([
        konst(SCRATCH_ADDR as i64),
        idx(op::LOCAL_GET, local_x),
        mem(op::I32_STORE8, 0),
        konst(SCRATCH_ADDR as i64),
        mem(op::I32_LOAD8_U, 0),
        idx(op::LOCAL_SET, local_byte),
        blk(op::BLOCK),
        blk(op::LOOP),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 1),
        idx(op::LOCAL_GET, local_byte),
        konst(2),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_byte),
        idx(op::BR, 0),
        nul(op::END),
        nul(op::END),
        idx(op::LOCAL_GET, local_x),
        idx(op::LOCAL_GET, local_byte),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_x),
    ]);

    instrs.extend([
        blk(op::BLOCK),
        idx(op::LOCAL_GET, local_x),
        konst(-2147483648),
        nul(op::I32_LT_U),
        idx(op::BR_IF, 0),
        idx(op::LOCAL_GET, local_x),
        konst(-2147483648),
        nul(op::I32_SUB),
        idx(op::LOCAL_SET, local_x),
        nul(op::END),
    ]);

    instrs.push(idx(op::LOCAL_GET, local_x));
    instrs
}

fn expand_xor(c: Val, local_a: u32) -> Vec<Instr> {
    if c == 0xFFFF_FFFF {
        return vec![
            idx(op::LOCAL_SET, local_a),
            konst(-1),
            idx(op::LOCAL_GET, local_a),
            nul(op::I32_SUB),
        ];
    }
    if c == 1 {
        // `x ^ 1` is `x + 1 - 2 * (x & 1)`.
        let local_x = local_a;
        let local_bit = local_a + 1;
        let mut instrs = vec![idx(op::LOCAL_TEE, local_x)];
        instrs.extend(expand_and_1(local_a));
        instrs.extend([
            idx(op::LOCAL_SET, local_bit),
            idx(op::LOCAL_GET, local_x),
            konst(1),
            nul(op::I32_ADD),
            idx(op::LOCAL_GET, local_bit),
            idx(op::LOCAL_GET, local_bit),
            nul(op::I32_ADD),
            nul(op::I32_SUB),
        ]);
        return instrs;
    }
    expand_bitop_general("xor", c, local_a)
}

fn expand_or(c: Val, local_a: u32) -> Vec<Instr> {
    expand_bitop_general("or", c, local_a)
}

// ── The pass ────────────────────────────────────────────────────────

/// The opcodes the machine executes directly.
fn is_basic(opcode: u8) -> bool {
    use op::*;
    matches!(
        opcode,
        UNREACHABLE
            | NOP
            | BLOCK
            | LOOP
            | IF
            | ELSE
            | END
            | BR
            | BR_IF
            | BR_TABLE
            | RETURN
            | CALL
            | MEMORY_SIZE
            | DROP
            | SELECT
            | LOCAL_GET
            | LOCAL_SET
            | LOCAL_TEE
            | GLOBAL_GET
            | GLOBAL_SET
            | I32_LOAD
            | I32_LOAD8_S
            | I32_LOAD8_U
            | I32_LOAD16_S
            | I32_LOAD16_U
            | I32_STORE
            | I32_STORE8
            | I32_STORE16
            | I32_CONST
            | I32_EQZ
            | I32_EQ
            | I32_NE
            | I32_LT_S
            | I32_LT_U
            | I32_GT_S
            | I32_GT_U
            | I32_LE_S
            | I32_LE_U
            | I32_GE_S
            | I32_GE_U
            | I32_ADD
            | I32_SUB
    )
}

/// The binary operations a preceding constant lets us expand.
fn is_lowerable_binop(opcode: u8) -> bool {
    use op::*;
    matches!(
        opcode,
        I32_MUL
            | I32_DIV_S
            | I32_DIV_U
            | I32_REM_U
            | I32_REM_S
            | I32_AND
            | I32_OR
            | I32_SHL
            | I32_SHR_U
            | I32_SHR_S
            | I32_XOR
            | I32_ROTL
            | I32_ROTR
    )
}

fn is_lowerable_unary(opcode: u8) -> bool {
    use op::*;
    matches!(opcode, I32_EXTEND8_S | I32_EXTEND16_S | I32_CLZ | I32_CTZ | I32_POPCNT)
}

/// Locals that are only ever assigned one and the same literal.
///
/// clang hoists a constant into a register and then uses the register, so
/// without this the constant forms above would almost never fire.
fn find_const_locals(instrs: &[Instr]) -> HashMap<u32, Val> {
    // `None` marks a local seen with two different values, or with a value
    // that is not a literal at all.
    let mut const_map: HashMap<u32, Option<Val>> = HashMap::new();
    for (i, ins) in instrs.iter().enumerate() {
        if ins.opcode == op::LOCAL_SET || ins.opcode == op::LOCAL_TEE {
            let local_idx = ins.index();
            if i > 0 && instrs[i - 1].opcode == op::I32_CONST {
                let val = instrs[i - 1].i32();
                match const_map.get(&local_idx) {
                    None => {
                        const_map.insert(local_idx, Some(val));
                    }
                    Some(Some(seen)) if *seen != val => {
                        const_map.insert(local_idx, None);
                    }
                    _ => {}
                }
            } else {
                const_map.insert(local_idx, None);
            }
        }
    }
    const_map.into_iter().filter_map(|(k, v)| v.map(|v| (k, v))).collect()
}

/// Rewrite a function body so that only basic instructions remain.
///
/// `num_params` is where the function's own locals start, so the four
/// temporaries this appends do not collide with them.
pub fn lower_hard_ops(func: &FuncBody, num_params: u32) -> FuncBody {
    let instrs = &func.instructions;
    let needs_lowering = instrs
        .iter()
        .any(|ins| is_lowerable_binop(ins.opcode) || is_lowerable_unary(ins.opcode));
    if !needs_lowering {
        return func.clone();
    }

    let const_locals = find_const_locals(instrs);

    let temp_base = num_params + func.num_locals;
    let mut new_locals = func.locals.clone();
    new_locals.push((NUM_TEMPS, 0x7F));
    let new_num_locals = func.num_locals + NUM_TEMPS;

    let mut new_instrs: Vec<Instr> = Vec::new();
    let mut i = 0usize;
    while i < instrs.len() {
        let ins = &instrs[i];

        // Either a literal or a local standing for one, followed by the
        // operation it is the right-hand side of.
        let mut const_val: Option<Val> = None;
        let mut binop_idx = 0usize;

        if ins.opcode == op::I32_CONST
            && i + 1 < instrs.len()
            && is_lowerable_binop(instrs[i + 1].opcode)
        {
            const_val = Some(ins.i32());
            binop_idx = i + 1;
        } else if ins.opcode == op::LOCAL_GET
            && const_locals.contains_key(&ins.index())
            && i + 1 < instrs.len()
            && is_lowerable_binop(instrs[i + 1].opcode)
        {
            const_val = Some(const_locals[&ins.index()]);
            binop_idx = i + 1;
        }

        if let Some(c) = const_val {
            let opcode = instrs[binop_idx].opcode;
            let local_a = temp_base;
            let expansion: Option<Vec<Instr>> = match opcode {
                op::I32_AND => Some(match c {
                    255 => expand_and_255(local_a),
                    0xFFFF_FFFE => expand_and_fffffffe(local_a),
                    0x7FFF_FFFE => expand_and_7ffffffe_v2(local_a),
                    _ => expand_and_general(c, local_a),
                }),
                op::I32_MUL => {
                    let mut e = vec![idx(op::LOCAL_SET, local_a)];
                    e.extend(expand_mul(c, local_a));
                    Some(e)
                }
                op::I32_DIV_U => Some(expand_div_u(c, local_a)),
                op::I32_REM_U | op::I32_REM_S => Some(expand_rem_u(c, local_a)),
                op::I32_SHL => Some(expand_shl_from_stack(c, local_a)),
                op::I32_SHR_U => Some(expand_shr_u(c, local_a)),
                op::I32_SHR_S => Some(expand_shr_s(c, local_a)),
                op::I32_XOR => Some(expand_xor(c, local_a)),
                op::I32_OR => Some(expand_or(c, local_a)),
                op::I32_DIV_S => Some(expand_div_s(c, local_a)),
                op::I32_ROTL => Some(expand_rotl_const(c, local_a)),
                op::I32_ROTR => Some(expand_rotr_const(c, local_a)),
                _ => None,
            };

            if let Some(e) = expansion {
                new_instrs.extend(e);
                i = binop_idx + 1;
                continue;
            }
        }

        // What is left is an operation whose right-hand side is only known at
        // run time.  Each of these is a loop over the second operand.
        let local_a = temp_base;
        let local_b = temp_base + 1;
        let local_q = temp_base + 2;

        match ins.opcode {
            // `a << b`: double `a`, `b` times.
            op::I32_SHL => {
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_EQZ),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_a),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    konst(1),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_a),
                ]);
            }

            // `a >> b` unsigned: halve `a`, `b` times, each halving its own
            // subtraction loop.
            op::I32_SHR_U => {
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_EQZ),
                    idx(op::BR_IF, 1),
                    konst(0),
                    idx(op::LOCAL_SET, local_q),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_a),
                    konst(2),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    konst(2),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_q),
                    konst(1),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_q),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_q),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    konst(1),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_a),
                ]);
            }

            // `a >> b` signed, expanded as if it were unsigned: right for a
            // non-negative `a`, which is what the dynamic shifts here are.
            op::I32_SHR_S => {
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_EQZ),
                    idx(op::BR_IF, 1),
                    konst(0),
                    idx(op::LOCAL_SET, local_q),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_a),
                    konst(2),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    konst(2),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_q),
                    konst(1),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_q),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_q),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    konst(1),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_a),
                ]);
            }

            // `a * b`: add `a` to itself `b` times.
            op::I32_MUL => {
                let local_r = temp_base + 2;
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    konst(0),
                    idx(op::LOCAL_SET, local_r),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_EQZ),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_r),
                    idx(op::LOCAL_GET, local_a),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_r),
                    idx(op::LOCAL_GET, local_b),
                    konst(1),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_r),
                ]);
            }

            op::I32_DIV_U => {
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    konst(0),
                    idx(op::LOCAL_SET, local_q),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_q),
                    konst(1),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_q),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_q),
                ]);
            }

            op::I32_REM_U | op::I32_REM_S => {
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_a),
                ]);
            }

            // A dynamic `xor` becomes `ne`: right on the booleans clang
            // produces it for, and wrong on anything wider.
            op::I32_XOR => {
                new_instrs.push(nul(op::I32_NE));
            }

            // `a & b` as `select(a, 0, b)`, and `a | b` as `select(1, a, b)`:
            // the same restriction to zero and one.
            op::I32_AND => {
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_a),
                    konst(0),
                    idx(op::LOCAL_GET, local_a),
                    nul(op::SELECT),
                ]);
            }

            op::I32_OR => {
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_a),
                    konst(1),
                    idx(op::LOCAL_GET, local_a),
                    nul(op::SELECT),
                ]);
            }

            op::I32_DIV_S => {
                let local_neg = temp_base + 3;
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_a),
                    konst(0),
                    nul(op::I32_LT_S),
                    idx(op::LOCAL_GET, local_b),
                    konst(0),
                    nul(op::I32_LT_S),
                    nul(op::I32_NE),
                    idx(op::LOCAL_SET, local_neg),
                    blk(op::BLOCK),
                    idx(op::LOCAL_GET, local_a),
                    konst(0),
                    nul(op::I32_GE_S),
                    idx(op::BR_IF, 0),
                    konst(0),
                    idx(op::LOCAL_GET, local_a),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_a),
                    nul(op::END),
                    blk(op::BLOCK),
                    idx(op::LOCAL_GET, local_b),
                    konst(0),
                    nul(op::I32_GE_S),
                    idx(op::BR_IF, 0),
                    konst(0),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    nul(op::END),
                    konst(0),
                    idx(op::LOCAL_SET, local_q),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_q),
                    konst(1),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_q),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    blk(op::BLOCK),
                    idx(op::LOCAL_GET, local_neg),
                    nul(op::I32_EQZ),
                    idx(op::BR_IF, 0),
                    konst(0),
                    idx(op::LOCAL_GET, local_q),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_q),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_q),
                ]);
            }

            // `a rotl b`: reduce `b` below 32, then rotate by one that many
            // times, carrying bit 31 round by hand.
            op::I32_ROTL => {
                let local_bit = temp_base + 2;
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    konst(32),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_b),
                    konst(32),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_EQZ),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    konst(0),
                    nul(op::I32_LT_S),
                    idx(op::LOCAL_SET, local_bit),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_a),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_GET, local_bit),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    konst(1),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_a),
                ]);
            }

            // `a rotr b` is `a rotl (32 - b mod 32) mod 32`.
            op::I32_ROTR => {
                let local_bit = temp_base + 2;
                new_instrs.extend([
                    idx(op::LOCAL_SET, local_b),
                    idx(op::LOCAL_SET, local_a),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    konst(32),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_b),
                    konst(32),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    konst(32),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    konst(32),
                    nul(op::I32_LT_U),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_b),
                    konst(32),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    blk(op::BLOCK),
                    blk(op::LOOP),
                    idx(op::LOCAL_GET, local_b),
                    nul(op::I32_EQZ),
                    idx(op::BR_IF, 1),
                    idx(op::LOCAL_GET, local_a),
                    konst(0),
                    nul(op::I32_LT_S),
                    idx(op::LOCAL_SET, local_bit),
                    idx(op::LOCAL_GET, local_a),
                    idx(op::LOCAL_GET, local_a),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_GET, local_bit),
                    nul(op::I32_ADD),
                    idx(op::LOCAL_SET, local_a),
                    idx(op::LOCAL_GET, local_b),
                    konst(1),
                    nul(op::I32_SUB),
                    idx(op::LOCAL_SET, local_b),
                    idx(op::BR, 0),
                    nul(op::END),
                    nul(op::END),
                    idx(op::LOCAL_GET, local_a),
                ]);
            }

            op::I32_CLZ => new_instrs.extend(expand_clz(temp_base)),
            op::I32_CTZ => new_instrs.extend(expand_ctz(temp_base)),
            op::I32_POPCNT => new_instrs.extend(expand_popcnt(temp_base)),
            op::I32_EXTEND8_S => new_instrs.extend(expand_extend8_s(temp_base)),
            op::I32_EXTEND16_S => new_instrs.extend(expand_extend16_s(temp_base)),

            _ => new_instrs.push(ins.clone()),
        }
        i += 1;
    }

    FuncBody { locals: new_locals, num_locals: new_num_locals, instructions: new_instrs }
}

/// The instructions of a body that the machine cannot execute, with how many
/// times each occurs, in the order they are first met.  An empty result is the
/// property `lower_hard_ops` is supposed to establish.
pub fn check_basic_only(func: &FuncBody) -> Vec<(&'static str, usize)> {
    let mut bad: Vec<(&'static str, usize)> = Vec::new();
    for ins in &func.instructions {
        if !is_basic(ins.opcode) {
            let name = crate::decoder::op_name(ins.opcode);
            match bad.iter_mut().find(|(n, _)| *n == name) {
                Some(entry) => entry.1 += 1,
                None => bad.push((name, 1)),
            }
        }
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{decode, op_name};

    fn body(instrs: Vec<Instr>) -> FuncBody {
        FuncBody { locals: vec![(1, 0x7F)], num_locals: 1, instructions: instrs }
    }

    fn names(f: &FuncBody) -> Vec<&'static str> {
        f.instructions.iter().map(|i| op_name(i.opcode)).collect()
    }

    #[test]
    fn a_constant_shift_becomes_a_byte_offset_and_nothing_else() {
        // `x << 8` is a store at the scratch word and a load at byte 1 of it,
        // which is what little-endian memory gives for free.
        let f = lower_hard_ops(
            &body(vec![idx(op::LOCAL_GET, 0), konst(8), nul(op::I32_SHL)]),
            1,
        );
        assert_eq!(
            names(&f),
            [
                "local.get",
                "local.set",
                "i32.const",
                "i32.const",
                "i32.store",
                "i32.const",
                "local.get",
                "i32.store",
                "i32.const",
                "i32.load"
            ]
        );
        assert_eq!(f.instructions[7].mem_offset(), 1, "the shift is the store offset");
        // The four temporaries land after the function's own local.
        assert_eq!(f.num_locals, 5);
        assert_eq!(f.instructions[6].index(), 2, "and they start past the parameter");
    }

    #[test]
    fn a_constant_hoisted_into_a_local_is_still_a_constant() {
        // clang puts the divisor in a register first; without
        // `find_const_locals` this would fall to the dynamic form, which is a
        // loop around a loop.
        let hoisted = body(vec![
            konst(10),
            idx(op::LOCAL_SET, 1),
            idx(op::LOCAL_GET, 0),
            idx(op::LOCAL_GET, 1),
            nul(op::I32_DIV_U),
        ]);
        let divisors = |f: &FuncBody| -> usize {
            f.instructions.iter().filter(|i| i.opcode == op::I32_CONST && i.i32() == 10).count()
        };
        let lowered = lower_hard_ops(&hoisted, 1);
        assert_eq!(divisors(&lowered), 3, "the literal 10 is now the loop's own bound");

        // Assign the same local twice, differently, and it stops being one:
        // the divisor comes off the stack and the loop compares two locals.
        let mut twice = hoisted.clone();
        twice.instructions.splice(2..2, [konst(11), idx(op::LOCAL_SET, 1)]);
        let lowered = lower_hard_ops(&twice, 1);
        assert_eq!(divisors(&lowered), 1, "only its own assignment is left");
        assert!(
            lowered.instructions.iter().all(|i| i.opcode != op::I32_DIV_U),
            "the dynamic form lowers it too, just not to a constant divisor"
        );
    }

    #[test]
    fn every_hard_operation_lowers_to_the_dispatch_table() {
        // One body carrying every lowerable opcode, in both the constant and
        // the dynamic form.  What the pass promises is that nothing outside
        // the machine's table survives.
        let binops = [
            op::I32_MUL,
            op::I32_DIV_S,
            op::I32_DIV_U,
            op::I32_REM_U,
            op::I32_REM_S,
            op::I32_AND,
            op::I32_OR,
            op::I32_SHL,
            op::I32_SHR_U,
            op::I32_SHR_S,
            op::I32_XOR,
            op::I32_ROTL,
            op::I32_ROTR,
        ];
        let unary = [
            op::I32_EXTEND8_S,
            op::I32_EXTEND16_S,
            op::I32_CLZ,
            op::I32_CTZ,
            op::I32_POPCNT,
        ];

        let mut instrs = Vec::new();
        for c in [1u32, 255, 7, 0x7FFF_FFFE, 0xFFFF_FFFE, 0xFFFF_FFFF, 1 << 20] {
            for opcode in binops {
                instrs.extend([idx(op::LOCAL_GET, 0), konst(c as i64), nul(opcode), nul(op::DROP)]);
            }
        }
        for opcode in binops {
            instrs.extend([
                idx(op::LOCAL_GET, 0),
                idx(op::LOCAL_GET, 0),
                nul(opcode),
                nul(op::DROP),
            ]);
        }
        for opcode in unary {
            instrs.extend([idx(op::LOCAL_GET, 0), nul(opcode), nul(op::DROP)]);
        }

        let f = lower_hard_ops(&body(instrs), 1);
        assert_eq!(check_basic_only(&f), Vec::new(), "something is left for the machine to guess");
    }

    #[test]
    fn a_body_with_nothing_hard_in_it_is_returned_unchanged() {
        let f = body(vec![idx(op::LOCAL_GET, 0), konst(1), nul(op::I32_ADD)]);
        let lowered = lower_hard_ops(&f, 1);
        assert_eq!(lowered.instructions, f.instructions);
        assert_eq!(lowered.num_locals, f.num_locals, "and gains no temporaries");
    }

    /// The released examples, lowered.  The sources are in `programs/`; what
    /// is skipped is a machine with no clang that targets wasm32, since the
    /// modules are clang's output and are not in this repository.  The
    /// byte-for-byte comparison against `compilation/lower.py` is
    /// `tests/lowerdump.py`; what this pins is the property.
    #[test]
    fn the_released_programs_lower_to_the_dispatch_table() {
        let root = crate::programs("");
        let Ok(entries) = std::fs::read_dir(&root) else { return };
        // The `.wasm` are intermediates that the compiler removes after use,
        // so they are built here from the C, into a directory of our own.
        let scratch = std::env::temp_dir().join(format!("alm-lower-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).expect("a scratch directory");
        let mut sources: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("c"))
            .collect();
        sources.sort();

        let mut seen = 0;
        for src in &sources {
            let copied = scratch.join(src.file_name().unwrap());
            std::fs::copy(src, &copied).expect("the source copies");
            let wasm = match crate::emit::compile_c_to_wasm(&copied, &root.join("runtime.h")) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("skipped: {e}");
                    break;
                }
            };
            let m = decode(&std::fs::read(&wasm).unwrap()).expect("decode");
            for (fi, f) in m.functions.iter().enumerate() {
                let num_params = m.types[m.func_type_indices[fi] as usize].params.len() as u32;
                let lowered = lower_hard_ops(f, num_params);
                assert_eq!(check_basic_only(&lowered), Vec::new(), "{} function {fi}", src.display());
            }
            seen += 1;
        }
        let _ = std::fs::remove_dir_all(&scratch);
        assert!(seen == 0 || seen >= 7, "only {seen} of {} modules lowered", sources.len());
    }
}
