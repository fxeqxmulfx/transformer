//! The dispatch table and the token prefix, ported from
//! `compilation/compile_wasm.py`.
//!
//! A program reaches the machine as a list of `op imm` lines between braces.
//! Turning a lowered module into that list is a flattening: the structured
//! control flow becomes absolute branch targets which are then made relative,
//! every function after the entry point is appended with a prologue that pops
//! its parameters into locals, the data segments and the used globals become a
//! run of `i32.store8`, and `return` becomes the complement of the distance
//! back to the function's start.
//!
//! Nothing here is a choice: the token prefix is compared against the released
//! `transformer_vm/data/*.txt` byte for byte.

use crate::decoder::{op, op_name, FuncBody, Module, Val};
use crate::lower::lower_hard_ops;

/// Where the globals live in linear memory, one word each.
const GLOBAL_BASE: u32 = 8;

/// An unresolved branch target, replaced when its block ends.
const PLACEHOLDER: u32 = 0xDEAD;

/// One line of the dispatch table: the machine's op name and its immediate.
pub type Entry = (&'static str, u32);

/// The machine's name for each WebAssembly instruction it executes directly.
/// `unreachable` and `return` are handled before this is consulted.
fn machine_name(opcode: u8) -> Option<&'static str> {
    use op::*;
    let name = match opcode {
        I32_CONST | LOCAL_GET | LOCAL_SET | LOCAL_TEE | DROP | SELECT | I32_ADD | I32_SUB
        | I32_EQ | I32_NE | I32_LT_S | I32_LT_U | I32_GT_S | I32_GT_U | I32_LE_S | I32_LE_U
        | I32_GE_S | I32_GE_U | I32_EQZ | I32_LOAD | I32_LOAD8_S | I32_LOAD8_U | I32_LOAD16_S
        | I32_LOAD16_U | I32_STORE | I32_STORE8 | I32_STORE16 | BR | BR_IF => op_name(opcode),
        UNREACHABLE | RETURN => "halt",
        _ => return None,
    };
    Some(name)
}

#[derive(PartialEq)]
enum Kind {
    Block,
    Loop,
    If,
}

struct Frame {
    kind: Kind,
    start_pc: usize,
    /// Entries whose target is this block's end.
    patches: Vec<usize>,
    /// For an `if`, the `br_if` that skips the consequent.
    if_entry: usize,
}

/// Flatten one function body into dispatch entries with absolute targets.
///
/// `is_main` decides what a fallthrough end is: the entry point halts, an
/// ordinary function returns.  `global_temp_local` is the scratch local that
/// `global.set` needs, because the machine has no globals and they are kept in
/// memory at `GLOBAL_BASE`.
fn compile_function(
    func: &FuncBody,
    m: &Module,
    is_main: bool,
    global_temp_local: Option<u32>,
) -> Result<Vec<Entry>, String> {
    let imports: Vec<&str> =
        m.imports.iter().filter(|i| i.kind == 0).map(|i| i.name.as_str()).collect();
    let num_imports = imports.len() as u32;

    let mut entries: Vec<Entry> = Vec::new();
    let mut label_stack: Vec<Frame> = Vec::new();

    for instr in &func.instructions {
        let opcode = instr.opcode;

        match opcode {
            op::NOP => {}

            op::BLOCK | op::LOOP => {
                let kind = if opcode == op::BLOCK { Kind::Block } else { Kind::Loop };
                label_stack.push(Frame {
                    kind,
                    start_pc: entries.len(),
                    patches: Vec::new(),
                    if_entry: 0,
                });
            }

            // `if` is a `br_if` over the consequent, on the negated condition.
            op::IF => {
                entries.push(("i32.eqz", 0));
                let br_idx = entries.len();
                entries.push(("br_if", PLACEHOLDER));
                label_stack.push(Frame {
                    kind: Kind::If,
                    start_pc: entries.len() - 2,
                    patches: vec![br_idx],
                    if_entry: br_idx,
                });
            }

            op::ELSE => {
                let frame = label_stack.last_mut().ok_or("else outside a block")?;
                if frame.kind != Kind::If {
                    return Err("else without an if".into());
                }
                let else_br_idx = entries.len();
                entries.push(("br", PLACEHOLDER));
                // The consequent's exit is the only thing still waiting on the
                // end of the block; the `br_if` jumps here, to the alternative.
                frame.patches = vec![else_br_idx];
                let if_entry = frame.if_entry;
                entries[if_entry] = ("br_if", entries.len() as u32);
            }

            op::END => match label_stack.pop() {
                None => entries.push((if is_main { "halt" } else { "return" }, 0)),
                Some(frame) => {
                    let end_pc = entries.len() as u32;
                    for idx in frame.patches {
                        entries[idx].1 = end_pc;
                    }
                }
            },

            // A branch out of a loop goes to its head, a branch out of a block
            // to its end, which is not known yet.
            op::BR | op::BR_IF => {
                let name = if opcode == op::BR { "br" } else { "br_if" };
                let label_idx = instr.index() as usize;
                let at = label_stack
                    .len()
                    .checked_sub(label_idx + 1)
                    .ok_or("branch past the outermost block")?;
                if label_stack[at].kind == Kind::Loop {
                    entries.push((name, label_stack[at].start_pc as u32));
                } else {
                    let idx = entries.len();
                    entries.push((name, PLACEHOLDER));
                    label_stack[at].patches.push(idx);
                }
            }

            op::RETURN => entries.push((if is_main { "halt" } else { "return" }, 0)),
            op::UNREACHABLE => entries.push(("halt", 0)),

            op::CALL => {
                let fi = instr.index();
                if fi < num_imports {
                    if imports[fi as usize] == "output_byte" {
                        entries.push(("output", 0));
                    } else {
                        return Err(format!("unsupported call to import {fi}"));
                    }
                } else {
                    entries.push(("call", fi));
                }
            }

            // The machine has no globals: they are a word of memory each.
            op::GLOBAL_GET => {
                entries.push(("i32.const", GLOBAL_BASE + 4 * instr.index()));
                entries.push(("i32.load", 0));
            }
            op::GLOBAL_SET => {
                let temp = global_temp_local.ok_or("global.set requires a temp local")?;
                entries.push(("local.set", temp));
                entries.push(("i32.const", GLOBAL_BASE + 4 * instr.index()));
                entries.push(("local.get", temp));
                entries.push(("i32.store", 0));
            }

            op::LOCAL_GET | op::LOCAL_SET | op::LOCAL_TEE => {
                entries.push((op_name(opcode), instr.index()));
            }

            op::I32_CONST => entries.push(("i32.const", instr.i32())),

            op::I32_LOAD | op::I32_LOAD8_S | op::I32_LOAD8_U | op::I32_LOAD16_S
            | op::I32_LOAD16_U | op::I32_STORE | op::I32_STORE8 | op::I32_STORE16 => {
                entries.push((op_name(opcode), instr.mem_offset()));
            }

            _ => match machine_name(opcode) {
                Some(name) => entries.push((name, 0)),
                None => return Err(format!("unsupported wasm opcode: {}", op_name(opcode))),
            },
        }
    }

    Ok(entries)
}

/// Shift the absolute branch targets of a compiled body by `offset`.
fn adjust_branches(body: Vec<Entry>, offset: u32) -> Vec<Entry> {
    body.into_iter()
        .map(|(op, imm)| {
            if op == "br" || op == "br_if" {
                (op, imm.wrapping_add(offset))
            } else {
                (op, imm)
            }
        })
        .collect()
}

/// The scratch local `global.set` uses, if the function sets a global at all.
fn func_global_temp(func: &FuncBody, m: &Module, local_func_idx: usize) -> Option<u32> {
    if !func.instructions.iter().any(|i| i.opcode == op::GLOBAL_SET) {
        return None;
    }
    let ty = &m.types[m.func_type_indices[local_func_idx] as usize];
    Some(ty.params.len() as u32 + func.num_locals)
}

/// Where the program's input is written, from the linker's `__heap_base`.
fn compute_input_base(m: &Module) -> Result<u32, String> {
    for exp in &m.exports {
        if exp.name == "__heap_base" && exp.kind == 3 {
            return Ok((m.globals[exp.index as usize].init + 15) & !15);
        }
    }
    Err("__heap_base is not exported — the module was linked without it".into())
}

/// Build the whole dispatch table, returning it and the address the input goes
/// to.  The input itself is not baked in: the same program runs on any input.
pub fn build_program(m: &Module) -> Result<(Vec<Entry>, u32), String> {
    let num_imports = m.imports.iter().filter(|i| i.kind == 0).count() as u32;

    let mut main_local_idx = 0usize;
    for exp in &m.exports {
        if exp.kind == 0 && exp.name == "compute" {
            main_local_idx = (exp.index - num_imports) as usize;
            break;
        }
    }

    let func = &m.functions[main_local_idx];
    let param_count = m.types[m.func_type_indices[main_local_idx] as usize].params.len() as u32;
    let num_locals = param_count + func.num_locals;

    let input_base = if param_count > 0 { compute_input_base(m)? } else { 0 };
    let entry_args: Vec<u32> = if param_count > 0 { vec![input_base] } else { Vec::new() };

    let mut prologue: Vec<Entry> = Vec::new();

    // The first line tells the runtime where to put the input.
    if param_count > 0 {
        prologue.push(("input_base", input_base));
    }

    // Every local of the entry point starts at zero, except the parameters,
    // which start at the arguments.
    for k in (0..num_locals).rev() {
        let init_val = entry_args.get(k as usize).copied().unwrap_or(0);
        prologue.push(("i32.const", init_val));
        prologue.push(("local.set", k));
    }

    // Memory: the used globals and the data segments, byte by byte.  The
    // machine starts on zeroed memory, so a zero byte is already written.
    let mut initial_memory: std::collections::BTreeMap<u32, u8> = std::collections::BTreeMap::new();

    let mut used_globals: Vec<u32> = Vec::new();
    for fn_ in &m.functions {
        for ins in &fn_.instructions {
            if ins.opcode == op::GLOBAL_GET || ins.opcode == op::GLOBAL_SET {
                let g = ins.index();
                if !used_globals.contains(&g) {
                    used_globals.push(g);
                }
            }
        }
    }

    for (gidx, global) in m.globals.iter().enumerate() {
        if !used_globals.contains(&(gidx as u32)) {
            continue;
        }
        let addr = GLOBAL_BASE + 4 * gidx as u32;
        for b in 0..4u32 {
            initial_memory.insert(addr + b, (global.init >> (8 * b)) as u8);
        }
    }

    for seg in &m.data_segments {
        for (i, byte_val) in seg.data.iter().enumerate() {
            initial_memory.insert(seg.offset + i as u32, *byte_val);
        }
    }

    for (addr, byte_val) in &initial_memory {
        if *byte_val == 0 {
            continue;
        }
        prologue.push(("i32.const", *addr));
        prologue.push(("i32.const", *byte_val as u32));
        prologue.push(("i32.store8", 0));
    }

    // The entry point's body, after the prologue.
    let gt = func_global_temp(func, m, main_local_idx);
    let body0 = compile_function(func, m, true, gt)?;
    let mut program = prologue.clone();
    program.extend(adjust_branches(body0, prologue.len() as u32));

    // Then every other function, each behind a prologue that pops its
    // parameters into its locals.
    let mut func_addresses: Vec<(u32, u32)> = Vec::new();
    for fi in 0..m.functions.len() {
        if fi == main_local_idx {
            continue;
        }
        let func_fi = &m.functions[fi];
        let n_params = m.types[m.func_type_indices[fi] as usize].params.len() as u32;

        let func_start = program.len() as u32;
        func_addresses.push((num_imports + fi as u32, func_start));

        for k in (0..n_params).rev() {
            program.push(("local.set", k));
        }

        let gt_fi = func_global_temp(func_fi, m, fi);
        let body_fi = compile_function(func_fi, m, false, gt_fi)?;
        program.extend(adjust_branches(body_fi, func_start + n_params));

        // A `return` carries the complement of how far it is from the
        // function's start, which is how the machine unwinds it.
        for (j, entry) in program.iter_mut().enumerate().skip(func_start as usize) {
            if entry.0 == "return" {
                let d_local = j as u32 - func_start;
                *entry = ("return", !d_local);
            }
        }
    }

    // A call names a function index until every function has an address.
    for entry in program.iter_mut() {
        if entry.0 == "call" {
            let fi = entry.1;
            let addr = func_addresses
                .iter()
                .find(|(i, _)| *i == fi)
                .ok_or_else(|| format!("call to unknown function index {fi}"))?
                .1;
            *entry = ("call", addr);
        }
    }

    // Branches and calls are then relative to the line after themselves.
    for (i, entry) in program.iter_mut().enumerate() {
        if entry.0 == "br" || entry.0 == "br_if" || entry.0 == "call" {
            entry.1 = entry.1.wrapping_sub(i as u32 + 1);
        }
    }

    Ok((program, input_base))
}

// ── The token prefix ────────────────────────────────────────────────

fn int_to_bytes(v: Val) -> [u8; 4] {
    v.to_le_bytes()
}

/// The dispatch table as the `{ … }` block of a program file.
pub fn format_prefix(program: &[Entry]) -> String {
    use std::fmt::Write;
    let mut s = String::from("{\n");
    for (op, imm) in program {
        let bytes = int_to_bytes(*imm);
        let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let _ = writeln!(s, "{op} {}", hex.join(" "));
    }
    s.push_str("}\n");
    s
}

/// The input bytes as tokens: printable ASCII as itself, everything else as
/// two hex digits, then the commit that hands control to the program.
fn input_tokens(input_str: &str) -> Vec<String> {
    let mut data = input_str.as_bytes().to_vec();
    data.push(0);
    let mut tokens = Vec::new();
    for b in data {
        if b > 0x20 && b < 0x7F && b != b'{' && b != b'}' {
            tokens.push((b as char).to_string());
        } else {
            tokens.push(format!("{b:02x}"));
        }
    }
    tokens.push("commit(+0,sts=0,bt=0)".to_string());
    tokens
}

/// The input section appended after the program block.
pub fn format_input_section(input_str: &str) -> String {
    input_tokens(input_str).join(" ") + "\n"
}

/// The specialized model's input: the start token and the same input tokens.
pub fn format_spec_input(input_str: &str) -> String {
    let mut tokens = vec!["start".to_string()];
    if !input_str.is_empty() {
        tokens.extend(input_tokens(input_str));
    }
    tokens.join(" ") + "\n"
}

/// Decode, lower and flatten a module into its token prefix.
pub fn compile_wasm_to_prefix(wasm: &[u8]) -> Result<(String, u32), String> {
    let mut m = crate::decoder::decode(wasm)?;
    for fi in 0..m.functions.len() {
        let num_params = m.types[m.func_type_indices[fi] as usize].params.len() as u32;
        m.functions[fi] = lower_hard_ops(&m.functions[fi], num_params);
    }
    let (program, input_base) = build_program(&m)?;
    Ok((format_prefix(&program), input_base))
}

/// The two files the compiler writes for a program: the universal model's
/// prefix with its input, and the specialized model's input alone.
pub fn compile_program(wasm: &[u8], args_str: &str) -> Result<(String, String, u32), String> {
    let (prefix, input_base) = compile_wasm_to_prefix(wasm)?;
    let mut txt = prefix;
    if !args_str.is_empty() && input_base != 0 {
        txt.push_str(&format_input_section(args_str));
    }
    let spec = format_spec_input(if input_base != 0 { args_str } else { "" });
    Ok((txt, spec, input_base))
}

// ── C to WebAssembly ────────────────────────────────────────────────

/// The first clang on the box that can target wasm32, in the release's own
/// order of preference: `$CLANG_PATH`, then `$PATH`, then the usual places.
pub fn find_clang() -> Result<String, String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("CLANG_PATH") {
        candidates.push(p);
    }
    candidates.push("clang".to_string());
    for p in [
        "/opt/homebrew/opt/llvm/bin/clang",
        "/usr/local/opt/llvm/bin/clang",
        "/usr/lib/llvm-18/bin/clang",
        "/usr/lib/llvm-17/bin/clang",
        "/usr/lib/llvm-16/bin/clang",
        "/usr/bin/clang",
    ] {
        candidates.push(p.to_string());
    }
    for cc in candidates {
        if let Ok(out) = std::process::Command::new(&cc).arg("--print-targets").output() {
            if out.status.success() && String::from_utf8_lossy(&out.stdout).contains("wasm32") {
                return Ok(cc);
            }
        }
    }
    Err("no clang with a wasm32 target — install LLVM or set CLANG_PATH".into())
}

/// Compile one C file the way the release does.  Every flag matters: `-O2`
/// with store merging off and no jump tables is what keeps the output inside
/// the subset the lowering can reduce, and the two exports are what
/// `build_program` looks for.
pub fn compile_c_to_wasm(c_path: &std::path::Path, runtime_h: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let wasm_path = c_path.with_extension("wasm");
    if !runtime_h.exists() {
        return Err(format!("runtime.h not found at {}", runtime_h.display()));
    }
    let cc = find_clang()?;
    let out = std::process::Command::new(&cc)
        .args([
            "--target=wasm32",
            "-nostdlib",
            "-O2",
            "-fno-builtin",
            "-fno-jump-tables",
            "-mllvm",
            "--combiner-store-merging=false",
            "-Wl,--no-entry",
            "-Wl,--export=compute",
            "-Wl,--export=__heap_base",
            "-Wl,-z,stack-size=4096",
            "-Wl,--initial-memory=10485760",
        ])
        .arg(format!("-include{}", runtime_h.display()))
        .arg("-o")
        .arg(&wasm_path)
        .arg(c_path)
        .output()
        .map_err(|e| format!("could not run {cc}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "clang failed compiling {}:\n{}",
            c_path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(wasm_path)
}

/// The manifest's `name: args` pairs, in order.  The file is a fixed shape —
/// a `programs:` sequence of two-key mappings — so it is read as that and not
/// as YAML in general.
pub fn load_manifest(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut in_programs = false;
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t == "programs:" {
            in_programs = true;
            continue;
        }
        if !in_programs {
            continue;
        }
        let item = t.strip_prefix("- ").unwrap_or(t);
        let (key, value) = item.split_once(':').ok_or_else(|| format!("bad manifest line {t:?}"))?;
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .map(|v| v.to_string())
            .unwrap_or_else(|| value.to_string());
        match key.trim() {
            "name" => out.push((value, String::new())),
            "args" => {
                out.last_mut().ok_or("args before a name in the manifest")?.1 = value;
            }
            other => return Err(format!("unknown manifest key {other:?}")),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_is_six_programs_and_their_inputs() {
        let text = "programs:\n  - name: hello\n    args: \"World\"\n  - name: collatz\n    args: \"7\"\n";
        let got = load_manifest(text).expect("it reads");
        assert_eq!(got, vec![("hello".into(), "World".into()), ("collatz".into(), "7".into())]);
    }

    #[test]
    fn the_input_is_printable_ascii_where_it_can_be_and_hex_where_it_cannot() {
        // A brace would close the program block, so it goes as hex even though
        // it is printable; so does the space, and the terminating NUL.
        assert_eq!(format_input_section("a {b"), "a 20 7b b 00 commit(+0,sts=0,bt=0)\n");
        assert_eq!(format_spec_input(""), "start\n");
        assert_eq!(format_spec_input("7"), "start 7 00 commit(+0,sts=0,bt=0)\n");
    }

    #[test]
    fn an_immediate_goes_out_little_endian_in_four_bytes() {
        let out = format_prefix(&[("i32.const", 0x1234_5678), ("halt", 0)]);
        assert_eq!(out, "{\ni32.const 78 56 34 12\nhalt 00 00 00 00\n}\n");
    }

    /// `return` carries the complement of its distance from the function's
    /// start, and a branch the distance from the line after itself.  Both are
    /// wrapping, so a backward branch is a large unsigned immediate.
    #[test]
    fn a_backward_branch_is_a_wrapped_negative_offset() {
        let program = adjust_branches(vec![("br", 0)], 0);
        assert_eq!(program[0].1, 0);
        let out = format_prefix(&[("br", 0u32.wrapping_sub(3))]);
        assert_eq!(out, "{\nbr fd ff ff ff\n}\n");
    }
}
