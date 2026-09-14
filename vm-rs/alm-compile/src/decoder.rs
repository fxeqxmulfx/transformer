//! A decoder for the WebAssembly MVP subset, ported from
//! `compilation/decoder.py`.
//!
//! Only what clang emits for these programs is supported, and anything else is
//! an error rather than a silent skip: the token prefix the lowering produces
//! is a transcription of the instruction stream, so a misread immediate is a
//! wrong program and not a wrong diagnostic.

/// Signed LEB128 is read and then truncated to the width, so `i32.const -1`
/// arrives as `0xffff_ffff`.  The machine's values are unsigned 32-bit words
/// and the lowering never re-signs them.
pub type Val = u32;

#[derive(Clone, Debug, PartialEq)]
pub enum Imm {
    None,
    /// `block` / `loop` / `if`: the block type byte, `0x40` for void.
    Block(u8),
    /// `br`, `br_if`, `call`, `local.*`, `global.*`: one index.
    Index(u32),
    BrTable { targets: Vec<u32>, default: u32 },
    CallIndirect { type_idx: u32, table: u8 },
    /// A load or store: alignment and static offset.
    Mem { align: u32, offset: u32 },
    /// `memory.size` / `memory.grow`: the reserved byte.
    Reserved(u8),
    I32(Val),
    I64(u64),
    F32(f32),
    F64(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instr {
    pub opcode: u8,
    pub imm: Imm,
}

impl Instr {
    /// The single index of an instruction that carries one, panicking on any
    /// other shape — the lowering knows which opcodes it is asking about.
    pub fn index(&self) -> u32 {
        match self.imm {
            Imm::Index(i) => i,
            _ => panic!("{:#04x} carries no index", self.opcode),
        }
    }

    pub fn i32(&self) -> Val {
        match self.imm {
            Imm::I32(v) => v,
            _ => panic!("{:#04x} carries no i32", self.opcode),
        }
    }

    pub fn mem_offset(&self) -> u32 {
        match self.imm {
            Imm::Mem { offset, .. } => offset,
            _ => panic!("{:#04x} is not a memory access", self.opcode),
        }
    }
}

pub mod op {
    pub const UNREACHABLE: u8 = 0x00;
    pub const NOP: u8 = 0x01;
    pub const BLOCK: u8 = 0x02;
    pub const LOOP: u8 = 0x03;
    pub const IF: u8 = 0x04;
    pub const ELSE: u8 = 0x05;
    pub const END: u8 = 0x0b;
    pub const BR: u8 = 0x0c;
    pub const BR_IF: u8 = 0x0d;
    pub const BR_TABLE: u8 = 0x0e;
    pub const RETURN: u8 = 0x0f;
    pub const CALL: u8 = 0x10;
    pub const CALL_INDIRECT: u8 = 0x11;
    pub const DROP: u8 = 0x1a;
    pub const SELECT: u8 = 0x1b;
    pub const LOCAL_GET: u8 = 0x20;
    pub const LOCAL_SET: u8 = 0x21;
    pub const LOCAL_TEE: u8 = 0x22;
    pub const GLOBAL_GET: u8 = 0x23;
    pub const GLOBAL_SET: u8 = 0x24;

    pub const I32_LOAD: u8 = 0x28;
    pub const I32_LOAD8_S: u8 = 0x2c;
    pub const I32_LOAD8_U: u8 = 0x2d;
    pub const I32_LOAD16_S: u8 = 0x2e;
    pub const I32_LOAD16_U: u8 = 0x2f;
    pub const I32_STORE: u8 = 0x36;
    pub const I32_STORE8: u8 = 0x3a;
    pub const I32_STORE16: u8 = 0x3b;
    pub const MEMORY_SIZE: u8 = 0x3f;
    pub const MEMORY_GROW: u8 = 0x40;

    pub const I32_CONST: u8 = 0x41;
    pub const I64_CONST: u8 = 0x42;
    pub const F32_CONST: u8 = 0x43;
    pub const F64_CONST: u8 = 0x44;

    pub const I32_EQZ: u8 = 0x45;
    pub const I32_EQ: u8 = 0x46;
    pub const I32_NE: u8 = 0x47;
    pub const I32_LT_S: u8 = 0x48;
    pub const I32_LT_U: u8 = 0x49;
    pub const I32_GT_S: u8 = 0x4a;
    pub const I32_GT_U: u8 = 0x4b;
    pub const I32_LE_S: u8 = 0x4c;
    pub const I32_LE_U: u8 = 0x4d;
    pub const I32_GE_S: u8 = 0x4e;
    pub const I32_GE_U: u8 = 0x4f;

    pub const I32_CLZ: u8 = 0x67;
    pub const I32_CTZ: u8 = 0x68;
    pub const I32_POPCNT: u8 = 0x69;
    pub const I32_ADD: u8 = 0x6a;
    pub const I32_SUB: u8 = 0x6b;
    pub const I32_MUL: u8 = 0x6c;
    pub const I32_DIV_S: u8 = 0x6d;
    pub const I32_DIV_U: u8 = 0x6e;
    pub const I32_REM_S: u8 = 0x6f;
    pub const I32_REM_U: u8 = 0x70;
    pub const I32_AND: u8 = 0x71;
    pub const I32_OR: u8 = 0x72;
    pub const I32_XOR: u8 = 0x73;
    pub const I32_SHL: u8 = 0x74;
    pub const I32_SHR_S: u8 = 0x75;
    pub const I32_SHR_U: u8 = 0x76;
    pub const I32_ROTL: u8 = 0x77;
    pub const I32_ROTR: u8 = 0x78;

    pub const I32_EXTEND8_S: u8 = 0xc0;
    pub const I32_EXTEND16_S: u8 = 0xc1;
}

/// The name the dispatch table and the traces use for each opcode.
pub fn op_name(opcode: u8) -> &'static str {
    use op::*;
    match opcode {
        UNREACHABLE => "unreachable",
        NOP => "nop",
        BLOCK => "block",
        LOOP => "loop",
        IF => "if",
        ELSE => "else",
        END => "end",
        BR => "br",
        BR_IF => "br_if",
        BR_TABLE => "br_table",
        RETURN => "return",
        CALL => "call",
        CALL_INDIRECT => "call_indirect",
        DROP => "drop",
        SELECT => "select",
        LOCAL_GET => "local.get",
        LOCAL_SET => "local.set",
        LOCAL_TEE => "local.tee",
        GLOBAL_GET => "global.get",
        GLOBAL_SET => "global.set",
        I32_LOAD => "i32.load",
        I32_LOAD8_S => "i32.load8_s",
        I32_LOAD8_U => "i32.load8_u",
        I32_LOAD16_S => "i32.load16_s",
        I32_LOAD16_U => "i32.load16_u",
        I32_STORE => "i32.store",
        I32_STORE8 => "i32.store8",
        I32_STORE16 => "i32.store16",
        MEMORY_SIZE => "memory.size",
        MEMORY_GROW => "memory.grow",
        I32_CONST => "i32.const",
        I32_EQZ => "i32.eqz",
        I32_EQ => "i32.eq",
        I32_NE => "i32.ne",
        I32_LT_S => "i32.lt_s",
        I32_LT_U => "i32.lt_u",
        I32_GT_S => "i32.gt_s",
        I32_GT_U => "i32.gt_u",
        I32_LE_S => "i32.le_s",
        I32_LE_U => "i32.le_u",
        I32_GE_S => "i32.ge_s",
        I32_GE_U => "i32.ge_u",
        I32_CLZ => "i32.clz",
        I32_CTZ => "i32.ctz",
        I32_POPCNT => "i32.popcnt",
        I32_ADD => "i32.add",
        I32_SUB => "i32.sub",
        I32_MUL => "i32.mul",
        I32_DIV_S => "i32.div_s",
        I32_DIV_U => "i32.div_u",
        I32_REM_S => "i32.rem_s",
        I32_REM_U => "i32.rem_u",
        I32_AND => "i32.and",
        I32_OR => "i32.or",
        I32_XOR => "i32.xor",
        I32_SHL => "i32.shl",
        I32_SHR_S => "i32.shr_s",
        I32_SHR_U => "i32.shr_u",
        I32_ROTL => "i32.rotl",
        I32_ROTR => "i32.rotr",
        I32_EXTEND8_S => "i32.extend8_s",
        I32_EXTEND16_S => "i32.extend16_s",
        _ => "?",
    }
}

// ── The module ──────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct FuncType {
    pub params: Vec<u8>,
    pub results: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Import {
    pub module: String,
    pub name: String,
    /// 0 func, 1 table, 2 memory, 3 global.
    pub kind: u8,
    pub index: u32,
}

#[derive(Clone, Debug)]
pub struct Export {
    pub name: String,
    pub kind: u8,
    pub index: u32,
}

#[derive(Clone, Debug)]
pub struct FuncBody {
    pub locals: Vec<(u32, u8)>,
    pub num_locals: u32,
    pub instructions: Vec<Instr>,
}

#[derive(Clone, Debug)]
pub struct DataSegment {
    pub offset: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Global {
    pub valtype: u8,
    pub mutable: u8,
    pub init: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Module {
    pub types: Vec<FuncType>,
    pub imports: Vec<Import>,
    pub func_type_indices: Vec<u32>,
    pub exports: Vec<Export>,
    pub functions: Vec<FuncBody>,
    pub data_segments: Vec<DataSegment>,
    pub globals: Vec<Global>,
}

impl Module {
    pub fn num_imported_funcs(&self) -> usize {
        self.imports.iter().filter(|i| i.kind == 0).count()
    }
}

// ── The reader ──────────────────────────────────────────────────────

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> u8 {
        let b = self.data[self.pos];
        self.pos += 1;
        b
    }

    fn uleb(&mut self) -> u64 {
        let (mut result, mut shift) = (0u64, 0u32);
        loop {
            let b = self.byte();
            result |= u64::from(b & 0x7f) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                return result;
            }
        }
    }

    fn u32(&mut self) -> u32 {
        self.uleb() as u32
    }

    /// Signed LEB128, sign-extended and then masked to `bits` — the Python
    /// returns the two's-complement bit pattern, not a negative number.
    fn sleb(&mut self, bits: u32) -> u64 {
        let (mut result, mut shift) = (0u64, 0u32);
        let mut last;
        loop {
            last = self.byte();
            result |= u64::from(last & 0x7f) << shift;
            shift += 7;
            if last & 0x80 == 0 {
                break;
            }
        }
        if shift < bits && last & 0x40 != 0 {
            result |= (!0u64) << shift;
        }
        if bits >= 64 { result } else { result & ((1u64 << bits) - 1) }
    }

    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        s
    }

    fn name(&mut self) -> String {
        let n = self.u32() as usize;
        String::from_utf8(self.bytes(n).to_vec()).expect("a name is UTF-8")
    }
}

pub fn decode(data: &[u8]) -> Result<Module, String> {
    if data.len() < 8 {
        return Err("file too short to be a wasm module".into());
    }
    if &data[0..4] != b"\0asm" {
        return Err(format!("bad magic: {:02x?}", &data[0..4]));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 {
        return Err(format!("unsupported wasm version: {version}"));
    }

    let mut m = Module::default();
    let mut r = Reader { data, pos: 8 };
    while r.pos < data.len() {
        let id = r.byte();
        let size = r.u32() as usize;
        let end = r.pos + size;
        match id {
            1 => types(&mut r, &mut m),
            2 => imports(&mut r, &mut m),
            3 => {
                let n = r.u32();
                for _ in 0..n {
                    m.func_type_indices.push(r.u32());
                }
            }
            6 => globals(&mut r, &mut m),
            7 => exports(&mut r, &mut m),
            10 => code(&mut r, &mut m)?,
            11 => segments(&mut r, &mut m)?,
            // Custom, table, memory, start, element and datacount sections say
            // nothing the lowering reads.
            _ => {}
        }
        r.pos = end;
    }
    Ok(m)
}

fn types(r: &mut Reader, m: &mut Module) {
    let n = r.u32();
    for _ in 0..n {
        let form = r.byte();
        assert_eq!(form, 0x60, "expected a functype");
        let np = r.u32() as usize;
        let params = r.bytes(np).to_vec();
        let nr = r.u32() as usize;
        let results = r.bytes(nr).to_vec();
        m.types.push(FuncType { params, results });
    }
}

fn imports(r: &mut Reader, m: &mut Module) {
    let n = r.u32();
    for _ in 0..n {
        let module = r.name();
        let name = r.name();
        let kind = r.byte();
        let index = match kind {
            0 => r.u32(),
            1 | 2 => {
                if kind == 1 {
                    r.byte(); // element type
                }
                let flags = r.byte();
                r.uleb(); // minimum
                if flags & 1 != 0 {
                    r.uleb(); // maximum
                }
                0
            }
            _ => {
                r.byte(); // value type
                r.byte(); // mutability
                0
            }
        };
        m.imports.push(Import { module, name, kind, index });
    }
}

fn exports(r: &mut Reader, m: &mut Module) {
    let n = r.u32();
    for _ in 0..n {
        let name = r.name();
        let kind = r.byte();
        let index = r.u32();
        m.exports.push(Export { name, kind, index });
    }
}

fn globals(r: &mut Reader, m: &mut Module) {
    let n = r.u32();
    for _ in 0..n {
        let valtype = r.byte();
        let mutable = r.byte();
        let mut init = 0u32;
        if r.byte() == op::I32_CONST {
            init = r.sleb(32) as u32;
        }
        while r.data[r.pos] != op::END {
            r.pos += 1;
        }
        r.pos += 1;
        m.globals.push(Global { valtype, mutable, init });
    }
}

fn code(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let n = r.u32();
    for _ in 0..n {
        let size = r.u32() as usize;
        let end = r.pos + size;
        let decls = r.u32();
        let mut locals = Vec::new();
        let mut num_locals = 0;
        for _ in 0..decls {
            let count = r.u32();
            let ty = r.byte();
            locals.push((count, ty));
            num_locals += count;
        }
        let mut instructions = Vec::new();
        while r.pos < end {
            instructions.push(instruction(r)?);
        }
        m.functions.push(FuncBody { locals, num_locals, instructions });
    }
    Ok(())
}

fn segments(r: &mut Reader, m: &mut Module) -> Result<(), String> {
    let n = r.u32();
    for _ in 0..n {
        let flags = r.uleb();
        if flags != 0 {
            return Err(format!("unsupported data segment flags: {flags}"));
        }
        let opcode = r.byte();
        if opcode != op::I32_CONST {
            return Err(format!("expected i32.const in a data offset, got {opcode:#04x}"));
        }
        let offset = r.sleb(32) as u32;
        assert_eq!(r.byte(), op::END);
        let count = r.u32() as usize;
        m.data_segments.push(DataSegment { offset, data: r.bytes(count).to_vec() });
    }
    Ok(())
}

fn instruction(r: &mut Reader) -> Result<Instr, String> {
    use op::*;
    let opcode = r.byte();
    let imm = match opcode {
        UNREACHABLE | NOP | END | ELSE | RETURN | DROP | SELECT => Imm::None,
        I32_EQZ..=I32_GE_U => Imm::None,
        I32_CLZ..=I32_ROTR => Imm::None,
        I32_EXTEND8_S | I32_EXTEND16_S => Imm::None,
        BLOCK | LOOP | IF => Imm::Block(r.byte()),
        BR | BR_IF | CALL | LOCAL_GET | LOCAL_SET | LOCAL_TEE | GLOBAL_GET | GLOBAL_SET => {
            Imm::Index(r.u32())
        }
        BR_TABLE => {
            let n = r.u32();
            let targets = (0..n).map(|_| r.u32()).collect();
            Imm::BrTable { targets, default: r.u32() }
        }
        CALL_INDIRECT => Imm::CallIndirect { type_idx: r.u32(), table: r.byte() },
        0x28..=0x3e => Imm::Mem { align: r.u32(), offset: r.u32() },
        MEMORY_SIZE | MEMORY_GROW => Imm::Reserved(r.byte()),
        F32_CONST => Imm::F32(f32::from_le_bytes(r.bytes(4).try_into().unwrap())),
        F64_CONST => Imm::F64(f64::from_le_bytes(r.bytes(8).try_into().unwrap())),
        I32_CONST => Imm::I32(r.sleb(32) as u32),
        I64_CONST => Imm::I64(r.sleb(64)),
        _ => return Err(format!("unsupported wasm opcode {opcode:#04x} at {}", r.pos - 1)),
    };
    Ok(Instr { opcode, imm })
}

/// A canonical text form of a decoded module, so this decoder can be diffed
/// against `compilation/decoder.py` on the same binary.
pub fn dump(m: &Module) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for t in &m.types {
        let _ = writeln!(s, "type {:02x?} -> {:02x?}", t.params, t.results);
    }
    for i in &m.imports {
        let _ = writeln!(s, "import {} {} kind={} index={}", i.module, i.name, i.kind, i.index);
    }
    let _ = writeln!(s, "functypes {:?}", m.func_type_indices);
    for e in &m.exports {
        let _ = writeln!(s, "export {} kind={} index={}", e.name, e.kind, e.index);
    }
    for g in &m.globals {
        let _ = writeln!(s, "global type={:02x} mut={} init={}", g.valtype, g.mutable, g.init);
    }
    for d in &m.data_segments {
        let _ = writeln!(s, "data offset={} bytes={}", d.offset, hex(&d.data));
    }
    for (fi, f) in m.functions.iter().enumerate() {
        let _ = writeln!(s, "func {fi} locals={:?} n={}", f.locals, f.num_locals);
        for (i, ins) in f.instructions.iter().enumerate() {
            let _ = writeln!(s, "  {i} {} {}", op_name(ins.opcode), imm_text(&ins.imm));
        }
    }
    s
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn imm_text(i: &Imm) -> String {
    match i {
        Imm::None => String::new(),
        Imm::Block(b) => format!("block={b:#04x}"),
        Imm::Index(x) => format!("{x}"),
        Imm::BrTable { targets, default } => format!("targets={targets:?} default={default}"),
        Imm::CallIndirect { type_idx, table } => format!("type={type_idx} table={table}"),
        Imm::Mem { align, offset } => format!("align={align} offset={offset}"),
        Imm::Reserved(r) => format!("reserved={r}"),
        Imm::I32(v) => format!("{v}"),
        Imm::I64(v) => format!("{v}"),
        Imm::F32(v) => format!("{:08x}", v.to_bits()),
        Imm::F64(v) => format!("{:016x}", v.to_bits()),
    }
}
