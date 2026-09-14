//! The instruction set the machine dispatches on, and the circle it dispatches
//! with.  A port of the tables at the top of `transformer_vm/wasm/interpreter.py`.

/// Squared radius of the circle every dispatch point lies on.
pub const POINTS_R2: f64 = 32045.0;

/// Dispatch points, in the order the opcodes claim them.  Each opcode maps to
/// a unique lattice point on `x^2 + y^2 = 32045`, which turns "is this opcode?"
/// into a single dot product against the fetched `(opcode_x, opcode_y)`.
pub const POINTS: [(i32, i32); 64] = [
    (179, 2), (179, -2), (-179, 2), (-179, -2),
    (2, 179), (2, -179), (-2, 179), (-2, -179),
    (178, 19), (178, -19), (-178, 19), (-178, -19),
    (19, 178), (19, -178), (-19, 178), (-19, -178),
    (173, 46), (173, -46), (-173, 46), (-173, -46),
    (46, 173), (46, -173), (-46, 173), (-46, -173),
    (166, 67), (166, -67), (-166, 67), (-166, -67),
    (67, 166), (67, -166), (-67, 166), (-67, -166),
    (163, 74), (163, -74), (-163, 74), (-163, -74),
    (74, 163), (74, -163), (-74, 163), (-74, -163),
    (157, 86), (157, -86), (-157, 86), (-157, -86),
    (86, 157), (86, -157), (-86, 157), (-86, -157),
    (142, 109), (142, -109), (-142, 109), (-142, -109),
    (109, 142), (109, -142), (-109, 142), (-109, -142),
    (131, 122), (131, -122), (-131, 122), (-131, -122),
    (122, 131), (122, -131), (-122, 131), (-122, -131),
];

/// `(name, opcode byte, stack delta, stores to stack)`, in the order of the
/// `OPCODES` dict — the order is what assigns dispatch points, so it is part
/// of the weights.
pub const OPCODES: [(&str, u8, i32, bool); 36] = [
    ("halt",          0x00,  0, false),
    ("return",        0x0F,  0, false),
    ("call",          0x10,  0, false),
    ("br",            0x0C,  0, false),
    ("br_if",         0x0D, -1, false),
    ("drop",          0x1A, -1, false),
    ("select",        0x1B, -2, true),
    ("local.get",     0x20,  1, true),
    ("local.set",     0x21, -1, false),
    ("local.tee",     0x22,  0, false),
    ("global.get",    0x23,  1, true),
    ("global.set",    0x24, -1, false),
    ("i32.load",      0x28,  0, true),
    ("i32.load8_s",   0x2C,  0, true),
    ("i32.load8_u",   0x2D,  0, true),
    ("i32.load16_s",  0x2E,  0, true),
    ("i32.load16_u",  0x2F,  0, true),
    ("i32.store",     0x36, -2, false),
    ("i32.store8",    0x3A, -2, false),
    ("i32.store16",   0x3B, -2, false),
    ("i32.const",     0x41,  1, true),
    ("i32.eqz",       0x45,  0, true),
    ("i32.eq",        0x46, -1, true),
    ("i32.ne",        0x47, -1, true),
    ("i32.lt_s",      0x48, -1, true),
    ("i32.lt_u",      0x49, -1, true),
    ("i32.gt_s",      0x4A, -1, true),
    ("i32.gt_u",      0x4B, -1, true),
    ("i32.le_s",      0x4C, -1, true),
    ("i32.le_u",      0x4D, -1, true),
    ("i32.ge_s",      0x4E, -1, true),
    ("i32.ge_u",      0x4F, -1, true),
    ("i32.add",       0x6A, -1, true),
    ("i32.sub",       0x6B, -1, true),
    ("output",        0xFF, -1, false),
    ("input_base",    0xFE,  0, false),
];

/// Opcodes that write their operand somewhere other than the stack.
pub const WRITE_OPS: [&str; 5] =
    ["local.set", "local.tee", "i32.store8", "i32.store16", "i32.store"];

/// Address stride between local-variable slots per call depth: four bytes a
/// local, 256 bytes allowing 64 locals.
pub const LOCAL_STRIDE: f64 = 256.0;

fn index_of(op: &str) -> usize {
    OPCODES.iter().position(|&(n, ..)| n == op).unwrap_or_else(|| panic!("unknown opcode {op}"))
}

/// The dispatch point of an opcode.
pub fn point(op: &str) -> (f64, f64) {
    let (x, y) = POINTS[index_of(op)];
    (x as f64, y as f64)
}

pub fn stack_delta(op: &str) -> i32 {
    OPCODES[index_of(op)].2
}

pub fn stores_to_stack(op: &str) -> bool {
    OPCODES[index_of(op)].3
}

pub fn is_write_op(op: &str) -> bool {
    WRITE_OPS.contains(&op)
}

/// The distinct `(stack delta, stores to stack)` pairs, which is what the
/// `commit(...)` tokens are indexed by.  Python builds them as a set; the
/// order never reaches the weights, since the vocabulary is sorted, so they
/// come out sorted here.
pub fn commit_pairs() -> Vec<(i32, i32)> {
    let mut v: Vec<(i32, i32)> =
        OPCODES.iter().map(|&(_, _, sd, sts)| (sd, sts as i32)).collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dispatch_point_is_on_the_circle_and_unique() {
        let used = &POINTS[..OPCODES.len()];
        for &(x, y) in POINTS.iter() {
            assert_eq!((x * x + y * y) as f64, POINTS_R2);
        }
        for i in 0..used.len() {
            for j in i + 1..used.len() {
                assert_ne!(used[i], used[j]);
            }
        }
    }

    #[test]
    fn the_commit_tokens_cover_eight_shapes() {
        assert_eq!(
            commit_pairs(),
            vec![(-2, 0), (-2, 1), (-1, 0), (-1, 1), (0, 0), (0, 1), (1, 1)]
        );
    }
}
