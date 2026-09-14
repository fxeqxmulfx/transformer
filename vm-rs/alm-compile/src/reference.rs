//! The reference run, ported from `wasm/reference.py`.
//!
//! The machine's trace is not a log of what it did: it *is* the computation.
//! Every instruction emits the tokens the transformer would have to produce —
//! the four bytes of a result, the carry marks of an addition, the commit that
//! says how the stack moved — and a program's `_ref.txt` is that token stream
//! with the dispatch table in front of it.  Running the model is then a
//! prediction task whose answer is known in advance, which is what makes the
//! whole construction checkable.
//!
//! So this is an ordinary interpreter of the dispatch table, written to emit
//! exactly the tokens the released `_ref.txt` files hold.

use std::fmt::Write;

const MASK32: u64 = 0xFFFF_FFFF;

/// The dispatch table's instruction set, resolved once so the run does not
/// compare strings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    InputBase,
    Halt,
    Const,
    LocalGet,
    LocalSet,
    LocalTee,
    GlobalGet,
    GlobalSet,
    Drop,
    Select,
    Add,
    Sub,
    Eqz,
    Eq,
    Ne,
    LtS,
    LtU,
    GtS,
    GtU,
    LeS,
    LeU,
    GeS,
    GeU,
    Load,
    Load8U,
    Load8S,
    Load16U,
    Load16S,
    Store,
    Store8,
    Store16,
    Br,
    BrIf,
    Call,
    Return,
    Output,
}

impl Op {
    pub fn from_name(name: &str) -> Option<Op> {
        use Op::*;
        Some(match name {
            "input_base" => InputBase,
            "halt" => Halt,
            "i32.const" => Const,
            "local.get" => LocalGet,
            "local.set" => LocalSet,
            "local.tee" => LocalTee,
            "global.get" => GlobalGet,
            "global.set" => GlobalSet,
            "drop" => Drop,
            "select" => Select,
            "i32.add" => Add,
            "i32.sub" => Sub,
            "i32.eqz" => Eqz,
            "i32.eq" => Eq,
            "i32.ne" => Ne,
            "i32.lt_s" => LtS,
            "i32.lt_u" => LtU,
            "i32.gt_s" => GtS,
            "i32.gt_u" => GtU,
            "i32.le_s" => LeS,
            "i32.le_u" => LeU,
            "i32.ge_s" => GeS,
            "i32.ge_u" => GeU,
            "i32.load" => Load,
            "i32.load8_u" => Load8U,
            "i32.load8_s" => Load8S,
            "i32.load16_u" => Load16U,
            "i32.load16_s" => Load16S,
            "i32.store" => Store,
            "i32.store8" => Store8,
            "i32.store16" => Store16,
            "br" => Br,
            "br_if" => BrIf,
            "call" => Call,
            "return" => Return,
            "output" => Output,
            _ => return None,
        })
    }
}

fn to_signed(v: u32) -> i64 {
    v as i32 as i64
}

// ── Reading a program file ──────────────────────────────────────────

/// The instructions between the braces, and the input tokens after them.
pub fn load_program(text: &str) -> Result<(Vec<(Op, u32)>, String), String> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.first() != Some(&"{") {
        return Err("the program does not open with '{'".into());
    }
    let end = tokens.iter().rposition(|t| *t == "}").ok_or("no closing '}' in the program")?;

    let body = &tokens[1..end];
    let mut program = Vec::with_capacity(body.len() / 5);
    let mut i = 0;
    while i < body.len() {
        let op = Op::from_name(body[i]).ok_or_else(|| format!("unknown op {:?}", body[i]))?;
        if i + 4 >= body.len() {
            return Err(format!("{} has no immediate", body[i]));
        }
        let mut imm: u32 = 0;
        for j in 0..4 {
            let b = u32::from_str_radix(body[i + 1 + j], 16)
                .map_err(|e| format!("bad immediate byte {:?}: {e}", body[i + 1 + j]))?;
            imm |= b << (8 * j);
        }
        program.push((op, imm));
        i += 5;
    }

    Ok((program, extract_input(&tokens[end + 1..])))
}

/// The input the program will be run on, decoded back from its tokens.  A
/// single-character token is that character, a two-digit one is a byte, and a
/// NUL ends the string.
fn extract_input(input_tokens: &[&str]) -> String {
    let mut chars = String::new();
    let without_commit = match input_tokens.last() {
        Some(t) if t.starts_with("commit(") => &input_tokens[..input_tokens.len() - 1],
        _ => input_tokens,
    };
    for tok in without_commit {
        match tok.len() {
            1 => chars.push_str(tok),
            2 => match u8::from_str_radix(tok, 16) {
                Ok(0) | Err(_) => break,
                Ok(b) => chars.push(b as char),
            },
            _ => break,
        }
    }
    chars
}

// ── The trace ───────────────────────────────────────────────────────

/// Tokens joined with spaces, one line per group, a group ending at whichever
/// token settles what the instruction did.
#[derive(Default)]
struct Sink {
    text: String,
    at_line_start: bool,
}

impl Sink {
    fn new() -> Sink {
        Sink { text: String::new(), at_line_start: true }
    }

    fn push(&mut self, token: &str) {
        if !self.at_line_start {
            self.text.push(' ');
        }
        self.text.push_str(token);
        self.at_line_start = false;
    }

    /// A terminal token: push it and end the line.
    fn end(&mut self, token: &str) {
        self.push(token);
        self.text.push('\n');
        self.at_line_start = true;
    }

    fn commit(&mut self, sd: i32, sts: u32, bt: u32) {
        let mut s = String::new();
        let _ = write!(s, "commit({sd:+},sts={sts},bt={bt})");
        self.end(&s);
    }

    /// The low `n` bytes of a value, each marked when its carry or borrow is
    /// set — the mark is how the trace records what a byte-wise adder knew.
    fn bytes(&mut self, value: u64, n: usize, marks: [u8; 4]) {
        for (i, mark) in marks.iter().enumerate().take(n) {
            let bv = (value >> (8 * i)) & 0xFF;
            let mut s = String::new();
            let _ = write!(s, "{bv:02x}");
            if *mark != 0 {
                s.push('\'');
            }
            self.push(&s);
        }
    }
}

const NO_MARKS: [u8; 4] = [0; 4];

fn add_carries(a: u64, b: u64) -> [u8; 4] {
    let mut out = [0u8; 4];
    let mut carry = 0u32;
    for (i, slot) in out.iter_mut().enumerate() {
        let s = ((a >> (8 * i)) & 0xFF) as u32 + ((b >> (8 * i)) & 0xFF) as u32 + carry;
        carry = u32::from(s >= 256);
        *slot = carry as u8;
    }
    out
}

fn sub_borrows(a: u64, b: u64) -> [u8; 4] {
    let mut out = [0u8; 4];
    let mut borrow = 0i32;
    for (i, slot) in out.iter_mut().enumerate() {
        let s = ((a >> (8 * i)) & 0xFF) as i32 - ((b >> (8 * i)) & 0xFF) as i32 - borrow;
        borrow = i32::from(s < 0);
        *slot = borrow as u8;
    }
    out
}

fn out_token(bv: u8) -> String {
    if bv > 0x20 && bv < 0x7F {
        format!("out({})", bv as char)
    } else {
        format!("out({bv:02x})")
    }
}

/// What a run produced.  `trace` is `None` unless it was asked for.
pub struct Run {
    pub instr_count: u64,
    pub token_count: u64,
    pub output: String,
    pub trace: Option<String>,
}

/// The machine's linear memory: 10 MiB, as the linker was told.
const MEM_SIZE: usize = 10 * 1024 * 1024;
const NUM_LOCALS: usize = 256;

/// Execute a dispatch table, optionally emitting the token trace.
pub fn run(
    program: &[(Op, u32)],
    input_str: &str,
    max_tokens: u64,
    trace: bool,
) -> Result<Run, String> {
    let mut mem = vec![0u8; MEM_SIZE];

    let input_base = match program.first() {
        Some((Op::InputBase, imm)) => Some(*imm),
        _ => None,
    };
    if let Some(base) = input_base {
        if !input_str.is_empty() {
            let mut bytes = input_str.as_bytes().to_vec();
            bytes.push(0);
            mem[base as usize..base as usize + bytes.len()].copy_from_slice(&bytes);
        }
    }

    let mut stack: Vec<u64> = Vec::new();
    let mut locals = vec![0u64; NUM_LOCALS];
    let mut call_stack: Vec<(i64, Vec<u64>, u32)> = Vec::new();
    let mut pc: i64 = 0;
    let mut instr_count: u64 = 0;
    let mut token_count: u64 = 0;
    let mut output: Vec<u8> = Vec::new();
    let mut sink = Sink::new();

    macro_rules! pop {
        () => {
            stack.pop().ok_or("the stack underflowed")?
        };
    }

    while pc >= 0 && (pc as usize) < program.len() && token_count < max_tokens {
        let (op, imm) = program[pc as usize];
        instr_count += 1;

        match op {
            // The input is not an instruction: it is the tokens the machine is
            // handed before it starts, counted here so the totals agree.
            Op::InputBase => {
                let mut bytes = input_str.as_bytes().to_vec();
                bytes.push(0);
                token_count += bytes.len() as u64 + 1;
                if trace {
                    for b in bytes {
                        if b > 0x20 && b < 0x7F {
                            sink.push(&(b as char).to_string());
                        } else {
                            sink.push(&format!("{b:02x}"));
                        }
                    }
                    sink.commit(0, 0, 0);
                }
                pc += 1;
            }

            Op::Halt => {
                token_count += 1;
                if trace {
                    sink.end("halt");
                }
                break;
            }

            Op::Const | Op::LocalGet | Op::GlobalGet => {
                let result = match op {
                    Op::Const => imm as u64 & MASK32,
                    _ => locals[imm as usize] & MASK32,
                };
                stack.push(result);
                token_count += 5;
                if trace {
                    sink.bytes(result, 4, NO_MARKS);
                    sink.commit(1, 1, 0);
                }
                pc += 1;
            }

            Op::LocalSet => {
                let val = pop!() & MASK32;
                locals[imm as usize] = val;
                token_count += 5;
                if trace {
                    sink.bytes(val, 4, NO_MARKS);
                    sink.commit(-1, 0, 0);
                }
                pc += 1;
            }

            Op::LocalTee => {
                let val = *stack.last().ok_or("the stack underflowed")? & MASK32;
                locals[imm as usize] = val;
                token_count += 5;
                if trace {
                    sink.bytes(val, 4, NO_MARKS);
                    sink.commit(0, 0, 0);
                }
                pc += 1;
            }

            // The compiler puts globals in memory, so this is unreachable in a
            // compiled program; it is kept because the trace format defines it.
            Op::GlobalSet => {
                locals[imm as usize] = pop!() & MASK32;
                token_count += 1;
                if trace {
                    sink.commit(-1, 0, 0);
                }
                pc += 1;
            }

            Op::Drop => {
                pop!();
                token_count += 1;
                if trace {
                    sink.commit(-1, 0, 0);
                }
                pc += 1;
            }

            Op::Select => {
                let c = pop!();
                let b = pop!();
                let a = pop!();
                let result = if c != 0 { a } else { b };
                stack.push(result);
                token_count += 5;
                if trace {
                    sink.bytes(result, 4, NO_MARKS);
                    sink.commit(-2, 1, 0);
                }
                pc += 1;
            }

            Op::Add | Op::Sub => {
                let b = pop!();
                let a = pop!();
                let (result, marks) = if op == Op::Add {
                    ((a + b) & MASK32, add_carries(a, b))
                } else {
                    ((a.wrapping_sub(b)) & MASK32, sub_borrows(a, b))
                };
                stack.push(result);
                token_count += 5;
                if trace {
                    sink.bytes(result, 4, marks);
                    sink.commit(-1, 1, 0);
                }
                pc += 1;
            }

            Op::Eqz => {
                let v = pop!();
                let result = u64::from(v == 0);
                stack.push(result);
                token_count += 5;
                if trace {
                    sink.bytes(result, 4, NO_MARKS);
                    sink.commit(0, 1, 0);
                }
                pc += 1;
            }

            Op::Eq | Op::Ne | Op::LtS | Op::LtU | Op::GtS | Op::GtU | Op::LeS | Op::LeU
            | Op::GeS | Op::GeU => {
                let b = pop!();
                let a = pop!();
                let (sa, sb) = (to_signed(a as u32), to_signed(b as u32));
                let result = u64::from(match op {
                    Op::Eq => a == b,
                    Op::Ne => a != b,
                    Op::LtS => sa < sb,
                    Op::LtU => a < b,
                    Op::GtS => sa > sb,
                    Op::GtU => a > b,
                    Op::LeS => sa <= sb,
                    Op::LeU => a <= b,
                    Op::GeS => sa >= sb,
                    _ => a >= b,
                });
                stack.push(result);
                token_count += 5;
                if trace {
                    sink.bytes(result, 4, NO_MARKS);
                    sink.commit(-1, 1, 0);
                }
                pc += 1;
            }

            Op::Load | Op::Load8U | Op::Load8S | Op::Load16U | Op::Load16S => {
                let addr = ((pop!() + imm as u64) & MASK32) as usize;
                let byte = |k: usize| -> u64 { mem[addr + k] as u64 };
                let (result, marks) = match op {
                    Op::Load => {
                        (byte(0) | byte(1) << 8 | byte(2) << 16 | byte(3) << 24, NO_MARKS)
                    }
                    Op::Load8U => (byte(0), NO_MARKS),
                    Op::Load8S => {
                        let v = byte(0);
                        let sign = u8::from(v >= 128);
                        let r = if v >= 128 { (v.wrapping_sub(256)) & MASK32 } else { v };
                        (r, [sign; 4])
                    }
                    Op::Load16U => (byte(0) | byte(1) << 8, NO_MARKS),
                    _ => {
                        let v = byte(0) | byte(1) << 8;
                        let sign = u8::from(v >= 32768);
                        let r = if v >= 32768 { (v.wrapping_sub(65536)) & MASK32 } else { v };
                        (r, [0, sign, sign, sign])
                    }
                };
                stack.push(result);
                token_count += 5;
                if trace {
                    sink.bytes(result, 4, marks);
                    sink.commit(0, 1, 0);
                }
                pc += 1;
            }

            Op::Store | Op::Store8 | Op::Store16 => {
                let val = pop!();
                let addr = ((pop!() + imm as u64) & MASK32) as usize;
                let width = match op {
                    Op::Store => 4,
                    Op::Store16 => 2,
                    _ => 1,
                };
                for k in 0..width {
                    mem[addr + k] = (val >> (8 * k)) as u8;
                }
                // A store costs one token per byte written, plus the commit.
                token_count += width as u64 + 1;
                if trace {
                    sink.bytes(val, width, NO_MARKS);
                    sink.commit(-2, 0, 0);
                }
                pc += 1;
            }

            Op::Br => {
                token_count += 6;
                if trace {
                    sink.end("branch_taken");
                    sink.bytes(imm as u64, 4, NO_MARKS);
                    sink.commit(0, 0, 1);
                }
                pc += 1 + to_signed(imm);
            }

            Op::BrIf => {
                let cond = pop!();
                if cond != 0 {
                    token_count += 6;
                    if trace {
                        sink.end("branch_taken");
                        sink.bytes(imm as u64, 4, NO_MARKS);
                        sink.commit(-1, 0, 1);
                    }
                    pc += 1 + to_signed(imm);
                } else {
                    token_count += 1;
                    if trace {
                        sink.commit(-1, 0, 0);
                    }
                    pc += 1;
                }
            }

            // A call saves the locals wholesale, which is what lets the callee
            // have its own frame without the machine having a stack pointer.
            Op::Call => {
                call_stack.push((pc + 1, std::mem::replace(&mut locals, vec![0; NUM_LOCALS]), imm));
                token_count += 6;
                if trace {
                    sink.end("branch_taken");
                    sink.bytes(imm as u64, 4, NO_MARKS);
                    sink.end("call_commit");
                }
                pc += 1 + to_signed(imm);
            }

            // The return's immediate is the complement of its distance from
            // the function's start; subtracting the call's own offset from it
            // is how the machine recovers where to go back to, and the borrows
            // of that subtraction are part of the trace.
            Op::Return => {
                let (ret_pc, ret_locals, call_imm) =
                    call_stack.pop().ok_or("return with no call to return from")?;
                token_count += 6;
                if trace {
                    let ret_offset = (ret_pc - pc - 1) as u64 & MASK32;
                    let borrows = sub_borrows(imm as u64, call_imm as u64);
                    sink.end("branch_taken");
                    sink.bytes(ret_offset, 4, borrows);
                    sink.end("return_commit");
                }
                pc = ret_pc;
                locals = ret_locals;
            }

            Op::Output => {
                let val = (pop!() & 0xFF) as u8;
                output.push(val);
                token_count += 1;
                if trace {
                    sink.end(&out_token(val));
                }
                pc += 1;
            }
        }
    }

    Ok(Run {
        instr_count,
        token_count,
        output: output.iter().map(|b| *b as char).collect(),
        trace: trace.then(|| {
            if !sink.at_line_start {
                sink.text.push('\n');
            }
            sink.text
        }),
    })
}

// ── The reference file ──────────────────────────────────────────────

/// The dispatch table, one instruction per line, exactly as it was read.
fn format_program_block(text: &str) -> Result<String, String> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.first() != Some(&"{") {
        return Err("the program does not open with '{'".into());
    }
    let end = tokens.iter().rposition(|t| *t == "}").ok_or("no closing '}' in the program")?;
    let mut out = String::from("{\n");
    // Five tokens to an instruction: the op and the four bytes of its
    // immediate.  A trailing partial group is not one, and is dropped.
    for chunk in tokens[1..end].as_chunks::<5>().0 {
        out.push_str(&chunk.join(" "));
        out.push('\n');
    }
    out.push_str("}\n");
    Ok(out)
}

/// Run a program file and return what its `_ref.txt` should hold.
pub fn generate_ref(prog_text: &str, max_tokens: u64) -> Result<(String, Run), String> {
    let (program, input_str) = load_program(prog_text)?;
    let run = run(&program, &input_str, max_tokens, true)?;
    let mut out = format_program_block(prog_text)?;
    out.push_str(run.trace.as_deref().unwrap_or(""));
    Ok((out, run))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A whole program in miniature: print `A`, then stop.
    fn hello_a() -> String {
        alm_compile_prefix(&[
            ("i32.const", 0x41),
            ("output", 0),
            ("halt", 0),
        ])
    }

    /// The `{ … }` block, written the way `emit::format_prefix` writes it.
    fn alm_compile_prefix(entries: &[(&'static str, u32)]) -> String {
        crate::emit::format_prefix(entries)
    }

    #[test]
    fn a_program_runs_and_its_trace_is_one_line_per_instruction() {
        let text = hello_a();
        let (formatted, run) = generate_ref(&text, 1000).expect("it runs");
        assert_eq!(run.output, "A");
        // 5 tokens for the constant, 1 for the output, 1 for the halt.
        assert_eq!(run.token_count, 7);
        assert_eq!(
            formatted,
            "{\ni32.const 41 00 00 00\noutput 00 00 00 00\nhalt 00 00 00 00\n}\n\
             41 00 00 00 commit(+1,sts=1,bt=0)\nout(A)\nhalt\n"
        );
    }

    /// The carry marks are what the trace adds over the result: they are the
    /// byte-wise adder's own state, and without them the addition would not be
    /// a function of the tokens alone.
    #[test]
    fn an_addition_marks_the_bytes_that_carried() {
        let text = alm_compile_prefix(&[
            ("i32.const", 0x00FF_00FF),
            ("i32.const", 1),
            ("i32.add", 0),
            ("halt", 0),
        ]);
        let (formatted, _) = generate_ref(&text, 1000).expect("it runs");
        let line = formatted.lines().nth(8).expect("the addition's line");
        // 0x00FF00FF + 1 = 0x00FF0100: byte 0 carried into byte 1, and no
        // further, so only the first mark is set.
        assert_eq!(line, "00' 01 ff 00 commit(-1,sts=1,bt=0)");
    }

    /// A subtraction marks borrows the same way, and the result wraps.
    #[test]
    fn a_subtraction_marks_the_bytes_that_borrowed() {
        let text = alm_compile_prefix(&[
            ("i32.const", 0),
            ("i32.const", 1),
            ("i32.sub", 0),
            ("halt", 0),
        ]);
        let (formatted, _) = generate_ref(&text, 1000).expect("it runs");
        let line = formatted.lines().nth(8).expect("the subtraction's line");
        assert_eq!(line, "ff' ff' ff' ff' commit(-1,sts=1,bt=0)");
    }

    /// `branch_taken` ends its own line, and the target's bytes begin the next
    /// one — the machine sees the decision before it sees where it goes.
    #[test]
    fn a_taken_branch_is_announced_on_a_line_of_its_own() {
        let text = alm_compile_prefix(&[("br", 1), ("halt", 0), ("halt", 0)]);
        let (formatted, run) = generate_ref(&text, 1000).expect("it runs");
        assert_eq!(run.instr_count, 2, "the branch skips the first halt");
        let body: Vec<&str> = formatted.lines().skip(5).collect();
        assert_eq!(body, ["branch_taken", "01 00 00 00 commit(+0,sts=0,bt=1)", "halt"]);
    }

    #[test]
    fn the_input_is_read_back_out_of_its_own_tokens() {
        let mut text = alm_compile_prefix(&[("input_base", 0x1400), ("halt", 0)]);
        text.push_str(&crate::emit::format_input_section("a b"));
        let (_program, input) = load_program(&text).expect("it parses");
        assert_eq!(input, "a b");
    }

    /// The machine is handed the input before it starts, so those tokens are
    /// part of the count and of the trace.
    #[test]
    fn the_input_tokens_come_first_and_are_counted() {
        let mut text = alm_compile_prefix(&[("input_base", 0x1400), ("halt", 0)]);
        text.push_str(&crate::emit::format_input_section("hi"));
        let (formatted, run) = generate_ref(&text, 1000).expect("it runs");
        // "h", "i", the NUL, the commit, and then the halt.
        assert_eq!(run.token_count, 5);
        assert!(formatted.ends_with("h i 00 commit(+0,sts=0,bt=0)\nhalt\n"), "{formatted}");
    }
}
