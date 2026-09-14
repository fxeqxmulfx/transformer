//! The WASM interpreter as a computation graph.  A port of
//! `transformer_vm/wasm/interpreter.py::build()` for the universal machine
//! (the `program is None` branch; the Futamura-specialized branch is not here).
//!
//! Read alongside the Python: every binding below is the same binding, in the
//! same order, and the `bindings` list at the end is the `locals()` CPython
//! hands to `auto_name` — plain locals in order of first assignment, then the
//! variables a closure or a generator expression captures.

use crate::expr::{DimId, Expr};
use crate::graph::{Graph, TieBreak};
use crate::isa;
use crate::naming::{auto_name, Binding};

/// The built graph, with the embedding and unembedding tables it scores with.
pub struct MachineGraph {
    pub graph: Graph,
    /// Token name -> its row of the embedding.
    pub input_tokens: Vec<(String, Expr)>,
    /// Token name -> the expression the output head scores it with.
    pub output_tokens: Vec<(String, Expr)>,
}

/// Name of the byte token for a value, with the `'` suffix marking a carry.
fn byte_token(bv: u32, carry: bool) -> String {
    if carry {
        format!("{bv:02x}'")
    } else {
        format!("{bv:02x}")
    }
}

/// Name of the output token for a byte: printable ASCII as itself.
fn out_token(bv: u32) -> String {
    if (0x20..0x7f).contains(&bv) && bv != 0x20 {
        format!("out({})", char::from_u32(bv).unwrap())
    } else {
        format!("out({bv:02x})")
    }
}

fn commit_token(sd: i32, sts: i32, bt: i32) -> String {
    format!("commit({sd:+},sts={sts},bt={bt})")
}

/// An ordered token table with a name index, standing in for a Python dict.
struct Tokens {
    order: Vec<(String, Expr)>,
    index: std::collections::HashMap<String, usize>,
}

impl Tokens {
    fn new() -> Self {
        Tokens { order: Vec::new(), index: std::collections::HashMap::new() }
    }

    fn set(&mut self, name: impl Into<String>, e: Expr) {
        let name = name.into();
        match self.index.get(&name) {
            Some(&i) => self.order[i].1 = e,
            None => {
                self.index.insert(name.clone(), self.order.len());
                self.order.push((name, e));
            }
        }
    }

    fn get(&self, name: &str) -> Option<&Expr> {
        self.index.get(name).map(|&i| &self.order[i].1)
    }

    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

pub fn build() -> MachineGraph {
    let mut g = Graph::new();
    let one = g.one;
    let position = g.position;
    assert_eq!(one, 0, "`one` is dimension zero; the constant helper assumes it");

    // A literal, as the Python coercion of an `int` to an `Expression` would
    // produce it: zero collapses to the empty expression.
    let k = |c: f64| Expr::scaled(one, c);
    let d = Expr::dim;

    // ── Input dimensions ────────────────────────────────────────────
    let byte_number = g.input("byte_number");
    let carry = g.input("carry");
    let delta_cursor = g.input("delta_cursor");
    let delta_stack = g.input("delta_stack");
    let is_jump = g.input("is_jump");
    let store_to_stack = g.input("store_to_stack");
    let is_branch_taken = g.input("is_branch_taken");
    let delta_call_depth = g.input("delta_call_depth");
    let is_return_commit = g.input("is_return_commit");

    let delta_stack_prefix = g.input("delta_stack_prefix");
    let store_to_stack_prefix = g.input("store_to_stack_prefix");
    let opcode_x = g.input("opcode_x");
    let opcode_y = g.input("opcode_y");
    let is_write = g.input("is_write");

    // ── Input tokens ────────────────────────────────────────────────
    let mut input_tokens = Tokens::new();

    for bv in 0u32..256 {
        for c in 0..2 {
            let e = Expr::scaled(byte_number, (bv + 1) as f64)
                .add(&Expr::scaled(carry, c as f64));
            input_tokens.set(byte_token(bv, c == 1), e);
        }
    }
    for (sd, sts) in isa::commit_pairs() {
        for bt in 0..2 {
            let e = d(delta_cursor)
                .add(&Expr::scaled(delta_stack, sd as f64))
                .add(&Expr::scaled(store_to_stack, sts as f64))
                .add(&Expr::scaled(is_jump, bt as f64));
            input_tokens.set(commit_token(sd, sts, bt), e);
        }
    }
    for bv in 0u32..256 {
        input_tokens.set(out_token(bv), d(delta_cursor));
    }
    input_tokens.set("branch_taken", d(is_branch_taken));
    input_tokens.set("call_commit", d(delta_cursor).add(&d(delta_call_depth)).add(&d(is_jump)));
    input_tokens.set(
        "return_commit",
        d(delta_cursor).sub(&d(delta_call_depth)).add(&d(is_return_commit)).add(&d(is_jump)),
    );

    input_tokens.set("{", Expr::new());
    input_tokens.set("}", Expr::scaled(delta_stack, 3.0));
    for &(op, _, sd, sts) in isa::OPCODES.iter() {
        let (px, py) = isa::point(op);
        let embedding = Expr::scaled(opcode_x, px)
            .add(&Expr::scaled(opcode_y, py))
            .add(&Expr::scaled(delta_stack_prefix, sd as f64))
            .add(&Expr::scaled(store_to_stack_prefix, sts as i32 as f64))
            .add(&Expr::scaled(is_write, isa::is_write_op(op) as i32 as f64));
        input_tokens.set(op, embedding);
    }

    // Printable ASCII aliases: 'a' gets the embedding of '61'.  Space is not a
    // valid token name, so 0x20 is skipped.
    for bv in 0x21u32..0x7f {
        let ch = char::from_u32(bv).unwrap().to_string();
        if input_tokens.contains(&ch) {
            continue;
        }
        let e = input_tokens.get(&byte_token(bv, false)).unwrap().clone();
        input_tokens.set(ch, e);
    }

    // Every token but the start token carries `one`.
    let start_token = "{";
    for i in 0..input_tokens.order.len() {
        if input_tokens.order[i].0 != start_token {
            input_tokens.order[i].1.set(one, 1.0);
        }
    }

    // ── store_value and the branch offset ───────────────────────────
    let store_bytes: Vec<DimId> = (1..5)
        .map(|i| {
            let value = d(byte_number).sub(&k(1.0));
            let query = d(position).sub(&k(i as f64));
            g.fetch1(&value, Some(&query), Some(&d(position)), None)
        })
        .collect();
    let mut store_value = Expr::new();
    for i in 1..5u32 {
        store_value = store_value.add(&Expr::scaled(store_bytes[(i - 1) as usize], (1u64 << (8 * (4 - i))) as f64));
    }
    let store_value = g.persist(&store_value, None);
    let msb = store_bytes[0];
    let unsigned_branch = g.reglu(&store_value, &d(is_jump));
    let jump_sign = {
        let b = d(msb).add(&Expr::scaled(is_jump, 128.0)).sub(&k(256.0));
        g.stepglu(&d(one), &b)
    };
    let delta_cursor_expr =
        d(delta_cursor).add(&unsigned_branch).sub(&jump_sign.mul((1u64 << 32) as f64));

    let byte_index = {
        let lu = g.fetch1(&d(position), Some(&d(one)), Some(&d(one)), Some(&d(byte_number)));
        d(position).sub(&d(lu))
    };
    let is_boundary = g.stepglu(&d(one), &d(byte_number).neg());

    // ── Cumulative state ────────────────────────────────────────────
    let sums = g.fetch_sum(&[d(delta_stack), delta_cursor_expr.clone(), d(delta_call_depth)]);
    let (stack_depth, cursor, call_depth) = (sums[0].clone(), sums[1].clone(), sums[2].clone());

    // ── Instruction fetch ───────────────────────────────────────────
    let instruction_position = cursor.mul(5.0).add(&k(1.0));
    let fetched = g.fetch(
        &[d(opcode_x), d(opcode_y), d(delta_stack_prefix), d(store_to_stack_prefix), d(is_write)],
        Some(&instruction_position),
        Some(&d(position)),
        None,
        TieBreak::Latest,
    );
    let (fetched_opcode_x, fetched_opcode_y) = (fetched[0], fetched[1]);
    let (fetched_stack_delta, fetched_store_to_stack, fetched_is_write) =
        (fetched[2], fetched[3], fetched[4]);

    let mut immediate = Expr::new();
    for i in 1..5u32 {
        let value = d(byte_number).sub(&k(1.0));
        let query = instruction_position.add(&k(i as f64));
        let lu = g.fetch1(&value, Some(&query), Some(&d(position)), None);
        immediate = immediate.add(&Expr::scaled(lu, (1u64 << (8 * (i - 1))) as f64));
    }
    let immediate = g.persist(&immediate, None);

    // The opcode gate: 1 when the fetched opcode is this one, <= -1 otherwise.
    let op_dot = |op: &str| -> Expr {
        let (px, py) = isa::point(op);
        Expr::scaled(fetched_opcode_x, px)
            .add(&Expr::scaled(fetched_opcode_y, py))
            .sub(&Expr::scaled(one, isa::POINTS_R2))
            .add(&Expr::scaled(one, 1.0))
    };
    macro_rules! is_op {
        ($g:expr, $op:expr) => {{
            let dot = op_dot($op);
            $g.reglu(&d(one), &dot)
        }};
    }

    // ── Derived instruction properties ──────────────────────────────
    let is_output = is_op!(g, "output");
    let memory_write_gate = {
        let e = is_op!(g, "i32.store")
            .add(&is_op!(g, "i32.store8"))
            .add(&is_op!(g, "i32.store16"))
            .add(&is_op!(g, "input_base"));
        g.persist(&e, None)
    };
    let uses_top_byte = d(fetched_is_write).add(&is_output);
    let is_producing_bytes = d(fetched_store_to_stack).add(&d(fetched_is_write));

    // ── Local variables ─────────────────────────────────────────────
    let local_write_key_dim = call_depth
        .mul(isa::LOCAL_STRIDE)
        .add(&immediate.mul(4.0))
        .add(&byte_index);
    // `1 - x` is Python's `Expression.__rsub__`: negate, then append the
    // constant.  The term order it leaves is the order the weight builder
    // sums the coefficients in, so it is reproduced rather than tidied.
    let not_local_write = is_op!(g, "local.set")
        .neg()
        .add(&k(1.0))
        .sub(&is_op!(g, "local.tee"))
        .add(&is_boundary);
    let local_byte = {
        let value = d(byte_number).sub(&k(1.0));
        let query = local_write_key_dim.add(&k(1.0));
        g.fetch1(&value, Some(&query), Some(&local_write_key_dim), Some(&not_local_write))
    };

    // ── Stack access ────────────────────────────────────────────────
    let not_store_to_stack = d(store_to_stack).neg().add(&k(1.0));
    let top = g.fetch(
        &[store_value.clone(), d(position).sub(&k(4.0))],
        Some(&stack_depth),
        Some(&stack_depth),
        Some(&not_store_to_stack),
        TieBreak::Latest,
    );
    let (stack_top_value, stack_top_position) = (top[0], top[1]);

    let second = g.fetch(
        &[store_value.clone(), d(position).sub(&k(4.0))],
        Some(&stack_depth.sub(&k(1.0))),
        Some(&stack_depth),
        Some(&not_store_to_stack),
        TieBreak::Latest,
    );
    let (stack_second_value, stack_second_position) = (second[0], second[1]);

    let stack_third_position = g.fetch1(
        &d(position).sub(&k(4.0)),
        Some(&stack_depth.sub(&k(2.0))),
        Some(&stack_depth),
        Some(&not_store_to_stack),
    );

    let byte_at = |g: &mut Graph, at: DimId| {
        let value = d(byte_number).sub(&k(1.0));
        let query = d(at).add(&byte_index);
        g.fetch1(&value, Some(&query), Some(&d(position)), None)
    };
    let top_byte = byte_at(&mut g, stack_top_position);
    let second_byte = byte_at(&mut g, stack_second_position);
    let third_byte = byte_at(&mut g, stack_third_position);

    // ── Memory ──────────────────────────────────────────────────────
    let memory_read_address = d(stack_top_value).add(&immediate).add(&byte_index);
    let memory_write_address =
        d(stack_second_value).add(&immediate).add(&byte_index).sub(&k(1.0));
    let not_memory_write_byte = k(1.0).add(&is_boundary).sub(&memory_write_gate);
    let mem = g.fetch(
        &[d(byte_number).sub(&k(1.0)), memory_write_address.clone()],
        Some(&memory_read_address),
        Some(&memory_write_address),
        Some(&not_memory_write_byte),
        TieBreak::Latest,
    );
    let (memory_byte_dirty, memory_byte_dirty_position) = (mem[0], mem[1]);
    let diff = d(memory_byte_dirty_position).sub(&memory_read_address);
    let memory_byte = {
        let a = d(memory_byte_dirty);
        let up = diff.add(&k(1.0));
        let dn = diff.sub(&k(1.0));
        let r0 = g.reglu(&a, &up);
        let r1 = g.reglu(&a, &diff);
        let r2 = g.reglu(&a, &dn);
        r0.sub(&r1.mul(2.0)).add(&r2)
    };
    let memory_sign = {
        let b = memory_byte.sub(&k(128.0));
        g.stepglu(&d(one), &b)
    };

    // ── Arithmetic ──────────────────────────────────────────────────
    let carry_late = g.persist(&d(carry), None);

    let add_value = d(second_byte).add(&d(top_byte)).add(&carry_late);
    let add_carry = {
        let b = add_value.sub(&k(256.0));
        g.stepglu(&d(one), &b)
    };
    let add_byte = add_value.sub(&add_carry.mul(256.0));

    let sub_value = d(second_byte).sub(&d(top_byte)).sub(&carry_late);
    let sub_borrow = g.stepglu(&d(one), &sub_value).neg().add(&k(1.0));
    let sub_byte = sub_value.add(&sub_borrow.mul(256.0));

    // ── Comparisons ─────────────────────────────────────────────────
    let a_gt_b_u = {
        let b = d(stack_second_value).sub(&d(stack_top_value)).sub(&k(1.0));
        g.stepglu(&d(one), &b)
    };
    let a_lt_b_u = {
        let b = d(stack_top_value).sub(&d(stack_second_value)).sub(&k(1.0));
        g.stepglu(&d(one), &b)
    };
    let a_eq_b = d(one).sub(&a_gt_b_u).sub(&a_lt_b_u);

    let sign_diff = {
        let sign_bit = (1u64 << 31) as f64;
        let t = d(stack_top_value).sub(&k(sign_bit));
        let s = d(stack_second_value).sub(&k(sign_bit));
        let r0 = g.reglu(&d(one), &t.add(&k(1.0)));
        let r1 = g.reglu(&d(one), &t);
        let r2 = g.reglu(&d(one), &s.add(&k(1.0)));
        let r3 = g.reglu(&d(one), &s);
        let e = r0.sub(&r1).sub(&r2).add(&r3);
        g.persist(&e, None)
    };
    let a_gt_b_s = {
        let b = sign_diff.add(&a_gt_b_u).sub(&k(1.0));
        g.stepglu(&d(one), &b)
    };
    let a_lt_b_s = {
        let b = sign_diff.neg().add(&a_lt_b_u).sub(&k(1.0));
        g.stepglu(&d(one), &b)
    };

    let cond_nonzero = {
        let b = d(stack_top_value).sub(&k(1.0));
        g.stepglu(&d(one), &b)
    };

    // ── Call stack, byte by byte ────────────────────────────────────
    let call_stack_write_key = call_depth.mul(4.0).add(&byte_index).sub(&k(1.0));
    let call_stack_read_key = call_depth.sub(&k(1.0)).mul(4.0).add(&byte_index);
    let not_call_byte = {
        let a = d(one).sub(&is_boundary);
        let dot = op_dot("call");
        let r = g.reglu(&a, &dot);
        r.neg().add(&k(1.0))
    };
    let call_stack_byte = g.fetch1(
        &d(byte_number).sub(&k(1.0)),
        Some(&call_stack_read_key),
        Some(&call_stack_write_key),
        Some(&not_call_byte),
    );

    // ── Result byte ─────────────────────────────────────────────────
    let const_byte = {
        let value = d(byte_number).sub(&k(1.0));
        let query = instruction_position.add(&byte_index).add(&k(1.0));
        g.fetch1(&value, Some(&query), Some(&d(position)), None)
    };

    // ── Branch/return byte ──────────────────────────────────────────
    let csb = {
        let dot = op_dot("return");
        g.reglu(&d(call_stack_byte), &dot)
    };
    let cc = {
        let dot = op_dot("return");
        g.reglu(&carry_late, &dot)
    };
    let branch_sub_val = d(const_byte).sub(&csb).sub(&cc);
    let branch_sub_val = g.persist(&branch_sub_val, None);
    let branch_sub_borrow = g.stepglu(&d(one), &branch_sub_val).neg().add(&k(1.0));
    let branch_byte = branch_sub_val.add(&branch_sub_borrow.mul(256.0));
    let branch_carry = branch_sub_borrow.clone();

    let byte_at_2 = {
        let b = byte_index.sub(&k(2.0));
        g.stepglu(&d(one), &b)
    };

    let result_byte_early = {
        let mut e = {
            let dot = op_dot("local.get");
            g.reglu(&d(local_byte), &dot)
        };
        e = e.add(&g.reglu(&d(top_byte), &uses_top_byte));
        let sel = op_dot("select");
        let b = sel.add(&cond_nonzero).sub(&k(1.0));
        e = e.add(&g.reglu(&d(third_byte), &b));
        let b = op_dot("select").sub(&cond_nonzero);
        e = e.add(&g.reglu(&d(second_byte), &b));
        g.persist(&e, None)
    };

    let result_byte = {
        let mut e = {
            let dot = op_dot("i32.const");
            g.reglu(&branch_sub_val, &dot)
        };
        let dot = op_dot("i32.add");
        e = e.add(&g.reglu(&add_byte, &dot));
        let dot = op_dot("i32.sub");
        e = e.add(&g.reglu(&sub_byte, &dot));
        e = e.add(&result_byte_early);
        let dot = op_dot("i32.load");
        e = e.add(&g.reglu(&memory_byte, &dot));
        let b = op_dot("i32.load8_u").add(&is_boundary).sub(&k(1.0));
        e = e.add(&g.reglu(&memory_byte, &b));
        let b = op_dot("i32.load8_s").add(&is_boundary).sub(&k(1.0));
        e = e.add(&g.reglu(&memory_byte, &b));
        let b = op_dot("i32.load8_s").sub(&is_boundary);
        e = e.add(&g.reglu(&carry_late, &b).mul(255.0));
        let b = op_dot("i32.load16_u").sub(&byte_at_2);
        e = e.add(&g.reglu(&memory_byte, &b));
        let b = op_dot("i32.load16_s").sub(&byte_at_2);
        e = e.add(&g.reglu(&memory_byte, &b));
        let b = op_dot("i32.load16_s").add(&byte_at_2).sub(&k(1.0));
        e = e.add(&g.reglu(&carry_late, &b).mul(255.0));
        let dot = op_dot("br");
        e = e.add(&g.reglu(&branch_sub_val, &dot));
        let dot = op_dot("br_if");
        e = e.add(&g.reglu(&branch_sub_val, &dot));
        let dot = op_dot("call");
        e = e.add(&g.reglu(&branch_sub_val, &dot));
        let dot = op_dot("return");
        e = e.add(&g.reglu(&branch_byte, &dot));
        let first = g.persist(&e, None);

        let cmp = |g: &mut Graph, a: &Expr, op: &str| {
            let b = op_dot(op).add(&is_boundary).sub(&k(1.0));
            g.reglu(a, &b)
        };
        let not = |e: &Expr| e.neg().add(&k(1.0));
        let mut e2 = cmp(&mut g, &a_eq_b, "i32.eq");
        e2 = e2.add(&cmp(&mut g, &not(&a_eq_b), "i32.ne"));
        e2 = e2.add(&cmp(&mut g, &a_gt_b_u, "i32.gt_u"));
        e2 = e2.add(&cmp(&mut g, &not(&a_gt_b_u), "i32.le_u"));
        e2 = e2.add(&cmp(&mut g, &a_lt_b_u, "i32.lt_u"));
        e2 = e2.add(&cmp(&mut g, &not(&a_lt_b_u), "i32.ge_u"));
        e2 = e2.add(&cmp(&mut g, &a_gt_b_s, "i32.gt_s"));
        e2 = e2.add(&cmp(&mut g, &not(&a_gt_b_s), "i32.le_s"));
        e2 = e2.add(&cmp(&mut g, &a_lt_b_s, "i32.lt_s"));
        e2 = e2.add(&cmp(&mut g, &not(&a_lt_b_s), "i32.ge_s"));
        e2 = e2.add(&cmp(&mut g, &not(&cond_nonzero), "i32.eqz"));
        let second = g.persist(&e2, None);
        first.add(&second)
    };

    let result_carry = {
        let dot = op_dot("i32.add");
        let mut e = g.reglu(&add_carry, &dot);
        let dot = op_dot("i32.sub");
        e = e.add(&g.reglu(&sub_borrow, &dot));
        let b = op_dot("i32.load8_s").add(&is_boundary).sub(&k(1.0));
        e = e.add(&g.reglu(&memory_sign, &b));
        let b = op_dot("i32.load8_s").sub(&is_boundary);
        e = e.add(&g.reglu(&carry_late, &b));
        let b = op_dot("i32.load16_s").sub(&is_boundary);
        e = e.add(&g.reglu(&memory_sign, &b));
        let b = op_dot("i32.load16_s").add(&byte_at_2).sub(&k(1.0));
        e = e.add(&g.reglu(&carry_late.sub(&memory_sign), &b));
        let dot = op_dot("return");
        e = e.add(&g.reglu(&branch_carry, &dot));
        g.persist(&e, None)
    };

    // ── Next-token prediction ───────────────────────────────────────
    let byte_index_4 = {
        let b = byte_index.sub(&k(4.0));
        g.stepglu(&d(one), &b)
    };
    let early_done = {
        let a = is_boundary.neg().add(&k(1.0));
        let dot = op_dot("i32.store8");
        let r0 = g.reglu(&a, &dot);
        let dot = op_dot("i32.store16");
        let r1 = g.reglu(&byte_at_2, &dot);
        r0.add(&r1)
    };
    let early_done = g.persist(&early_done, None);
    let byte_done = byte_index_4.add(&early_done);
    let is_byte_seq = is_boundary.neg().add(&k(1.0)).sub(&byte_done);

    let emit_halt = {
        let dot = op_dot("halt");
        g.reglu(&is_boundary, &dot)
    };
    let emit_branch_taken = {
        let b = op_dot("br").sub(&d(is_branch_taken));
        let mut e = g.reglu(&is_boundary, &b);
        let b = cond_nonzero.add(&op_dot("br_if")).sub(&d(is_branch_taken)).sub(&k(1.0));
        e = e.add(&g.reglu(&is_boundary, &b));
        let b = op_dot("return").sub(&d(is_branch_taken));
        e = e.add(&g.reglu(&is_boundary, &b));
        let b = op_dot("call").sub(&d(is_branch_taken));
        e = e.add(&g.reglu(&is_boundary, &b));
        e
    };
    let emit_return_commit = {
        let dot = op_dot("return");
        g.reglu(&byte_index_4, &dot)
    };
    let emit_out = {
        let dot = op_dot("output");
        g.reglu(&is_boundary, &dot)
    };
    let emit_byte_start = g.reglu(&is_producing_bytes, &is_boundary);
    let emit_byte = emit_byte_start.add(&is_byte_seq).add(&d(is_branch_taken));
    let emit_call_commit = {
        let dot = op_dot("call");
        g.reglu(&byte_index_4, &dot)
    };
    let emit_bt = {
        let dot = op_dot("br");
        let r0 = g.reglu(&byte_done, &dot);
        let dot = op_dot("br_if");
        let r1 = g.reglu(&byte_done, &dot);
        r0.add(&r1)
    };
    let emit_commit = byte_done
        .add(&is_boundary)
        .sub(&emit_halt)
        .sub(&emit_branch_taken)
        .sub(&emit_return_commit)
        .sub(&emit_out)
        .sub(&emit_byte_start)
        .sub(&d(is_branch_taken))
        .sub(&emit_call_commit);

    // ── Output tokens ───────────────────────────────────────────────
    const H: f64 = 1e5;
    let mut output_tokens = Tokens::new();
    output_tokens.set("halt", emit_halt.mul(H));
    output_tokens.set("branch_taken", emit_branch_taken.mul(H));
    output_tokens.set("call_commit", emit_call_commit.mul(H));
    output_tokens.set("return_commit", emit_return_commit.mul(H));

    for bv in 0u32..256 {
        let e = emit_out
            .mul(H)
            .add(&Expr::scaled(top_byte, (2 * bv) as f64))
            .sub(&k((bv * bv) as f64));
        output_tokens.set(out_token(bv), e);
    }

    for (sd, sts) in isa::commit_pairs() {
        for bt in 0..2 {
            let e = emit_commit
                .mul(H)
                .add(&Expr::scaled(fetched_stack_delta, (2 * sd) as f64))
                .sub(&k((sd * sd) as f64))
                .add(&Expr::scaled(fetched_store_to_stack, (2 * sts) as f64))
                .sub(&k((sts * sts) as f64))
                .add(&emit_bt.mul((2 * bt) as f64))
                .sub(&k((bt * bt) as f64));
            output_tokens.set(commit_token(sd, sts, bt), e);
        }
    }

    let mut bv_base = Expr::new();
    let mut score = Expr::new();
    for bv in 0u32..256 {
        bv_base = emit_byte
            .mul(H)
            .add(&result_byte.mul((2 * bv) as f64))
            .sub(&k((bv * bv) as f64));
        for c in 0..2u32 {
            score = bv_base
                .add(&result_carry.mul((2 * c) as f64))
                .sub(&k((c * c) as f64));
            output_tokens.set(byte_token(bv, c == 1), score.clone());
        }
    }

    // ── Naming ──────────────────────────────────────────────────────
    // The `locals()` CPython hands `auto_name`: plain locals in order of first
    // assignment, then the four names a closure or a generator expression
    // captures (`fetched_opcode_x`, `fetched_opcode_y`, `instruction_position`,
    // `store_bytes`).  `embedding`, `bv_base` and `score` leak from their loops
    // in Python and are carried here for the same reason.
    let embedding = input_tokens.get("input_base").unwrap().clone();
    let bindings: Vec<(&str, Binding)> = vec![
        ("embedding", embedding.into()),
        ("store_value", (&store_value).into()),
        ("msb", msb.into()),
        ("unsigned_branch", (&unsigned_branch).into()),
        ("jump_sign", (&jump_sign).into()),
        ("delta_cursor_expr", (&delta_cursor_expr).into()),
        ("byte_index", (&byte_index).into()),
        ("is_boundary", (&is_boundary).into()),
        ("stack_depth", (&stack_depth).into()),
        ("cursor", (&cursor).into()),
        ("call_depth", (&call_depth).into()),
        ("fetched_stack_delta", fetched_stack_delta.into()),
        ("fetched_store_to_stack", fetched_store_to_stack.into()),
        ("fetched_is_write", fetched_is_write.into()),
        ("immediate", (&immediate).into()),
        ("is_output", (&is_output).into()),
        ("memory_write_gate", (&memory_write_gate).into()),
        ("uses_top_byte", (&uses_top_byte).into()),
        ("is_producing_bytes", (&is_producing_bytes).into()),
        ("local_write_key_dim", (&local_write_key_dim).into()),
        ("not_local_write", (&not_local_write).into()),
        ("local_byte", local_byte.into()),
        ("not_store_to_stack", (&not_store_to_stack).into()),
        ("stack_top_value", stack_top_value.into()),
        ("stack_top_position", stack_top_position.into()),
        ("stack_second_value", stack_second_value.into()),
        ("stack_second_position", stack_second_position.into()),
        ("stack_third_position", stack_third_position.into()),
        ("top_byte", top_byte.into()),
        ("second_byte", second_byte.into()),
        ("third_byte", third_byte.into()),
        ("memory_read_address", (&memory_read_address).into()),
        ("memory_write_address", (&memory_write_address).into()),
        ("not_memory_write_byte", (&not_memory_write_byte).into()),
        ("memory_byte_dirty", memory_byte_dirty.into()),
        ("memory_byte_dirty_position", memory_byte_dirty_position.into()),
        ("diff", (&diff).into()),
        ("memory_byte", (&memory_byte).into()),
        ("memory_sign", (&memory_sign).into()),
        ("carry_late", (&carry_late).into()),
        ("add_value", (&add_value).into()),
        ("add_carry", (&add_carry).into()),
        ("add_byte", (&add_byte).into()),
        ("sub_value", (&sub_value).into()),
        ("sub_borrow", (&sub_borrow).into()),
        ("sub_byte", (&sub_byte).into()),
        ("a_gt_b_u", (&a_gt_b_u).into()),
        ("a_lt_b_u", (&a_lt_b_u).into()),
        ("a_eq_b", (&a_eq_b).into()),
        ("sign_diff", (&sign_diff).into()),
        ("a_gt_b_s", (&a_gt_b_s).into()),
        ("a_lt_b_s", (&a_lt_b_s).into()),
        ("cond_nonzero", (&cond_nonzero).into()),
        ("call_stack_write_key", (&call_stack_write_key).into()),
        ("call_stack_read_key", (&call_stack_read_key).into()),
        ("not_call_byte", (&not_call_byte).into()),
        ("call_stack_byte", call_stack_byte.into()),
        ("const_byte", const_byte.into()),
        ("csb", (&csb).into()),
        ("cc", (&cc).into()),
        ("branch_sub_val", (&branch_sub_val).into()),
        ("branch_sub_borrow", (&branch_sub_borrow).into()),
        ("branch_byte", (&branch_byte).into()),
        ("branch_carry", (&branch_carry).into()),
        ("byte_at_2", (&byte_at_2).into()),
        ("result_byte_early", (&result_byte_early).into()),
        ("result_byte", (&result_byte).into()),
        ("result_carry", (&result_carry).into()),
        ("byte_index_4", (&byte_index_4).into()),
        ("early_done", (&early_done).into()),
        ("byte_done", (&byte_done).into()),
        ("is_byte_seq", (&is_byte_seq).into()),
        ("emit_halt", (&emit_halt).into()),
        ("emit_branch_taken", (&emit_branch_taken).into()),
        ("emit_return_commit", (&emit_return_commit).into()),
        ("emit_out", (&emit_out).into()),
        ("emit_byte_start", (&emit_byte_start).into()),
        ("emit_byte", (&emit_byte).into()),
        ("emit_call_commit", (&emit_call_commit).into()),
        ("emit_bt", (&emit_bt).into()),
        ("emit_commit", (&emit_commit).into()),
        ("bv_base", (&bv_base).into()),
        ("score", (&score).into()),
        ("fetched_opcode_x", fetched_opcode_x.into()),
        ("fetched_opcode_y", fetched_opcode_y.into()),
        ("instruction_position", (&instruction_position).into()),
        ("store_bytes", Binding::Seq(store_bytes.iter().map(|&x| Binding::Dim(x)).collect())),
    ];
    auto_name(&mut g, &bindings);

    MachineGraph { graph: g, input_tokens: input_tokens.order, output_tokens: output_tokens.order }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The graph is checked against Python by dumping both and diffing
    /// (`alm-graph-dump`); what is worth pinning here is the handful of
    /// generated names `plan.yaml` resolves by, since a rename would break the
    /// schedule silently rather than loudly.
    #[test]
    fn the_names_the_plan_resolves_by_come_out_as_python_spells_them() {
        let mg = build();
        let names: std::collections::HashSet<&str> =
            mg.graph.dims.iter().map(|d| d.name.as_str()).collect();
        for expected in [
            "store_bytes[0]",
            "store_bytes[3]",
            "store_value",
            "unsigned_branch+",
            "byte_index_lu",
            "is_boundary",
            "jump_sign",
            "persist_33",
            "reglu_24",
            "stack_depth+",
            "cursor+",
            "call_depth+",
            "immediate",
            "result_byte",
            "result_byte$1",
            "result_carry",
            "emit_branch_taken+",
            "emit_branch_taken+3",
            "emit_bt+1",
            "lookup_5_v0",
            "lookup_10_v0",
        ] {
            assert!(names.contains(expected), "plan.yaml names {expected}, the graph does not");
        }
    }

    #[test]
    fn the_vocabulary_is_the_shipped_one() {
        let mg = build();
        let mut all: Vec<&str> = mg
            .input_tokens
            .iter()
            .chain(mg.output_tokens.iter())
            .map(|(n, _)| n.as_str())
            .collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 915, "model.bin ships a 915-token vocabulary");
    }
}
