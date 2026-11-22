//! Learned diagonal recurrence with a gated read. The bounded variant
//! implements Transformer.Phasor.gatedMemoryStep and its adjoint.
//! One lane owns one state coordinate of one sequence in both directions.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let ptr: u64;
        mad.wide.u32(ptr, i, 4, p);
        ld.global.f32(v, [ptr]);
    }
    fn store(p: u64, i: u32, v: f32) {
        let ptr: u64;
        mad.wide.u32(ptr, i, 4, p);
        st.global.f32([ptr], v);
    }
    fn sigmoid(v: f32) {
        mul.rn.f32(v, v, -1.4426950408889634);
        ex2.approx.f32(v, v);
        add.rn.f32(v, v, 1.0);
        div.rn.f32(v, 1.0, v);
    }
    fn tanh(v: f32) {
        add.rn.f32(v, v, v);
        sigmoid(v);
        fma.rn.f32(v, v, 2.0, -1.0);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn forward<const BOUNDED: bool>(
        read: *mut f32,
        states: *mut f32,
        gates: *mut f32,
        length: u32,
    ) {
        let (row, end, lane, at): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        let (a, b, c, s, output): (f32, f32, f32, f32, f32);
        mov.f32(s, 0.0);
        label(scan);
        mad.lo.u32(at, row, 96, lane);
        load(a, gates, at);
        if BOUNDED {
            sigmoid(a);
        } else {
            tanh(a);
        }
        store(gates, at, a);
        add.u32(at, at, 32);
        load(b, gates, at);
        tanh(b);
        store(gates, at, b);
        add.u32(at, at, 32);
        load(c, gates, at);
        sigmoid(c);
        store(gates, at, c);
        if BOUNDED {
            sub.rn.f32(output, s, b);
            fma.rn.f32(s, a, output, b);
        } else {
            fma.rn.f32(s, a, s, b);
        }
        mad.lo.u32(at, row, 32, lane);
        store(states, at, s);
        mul.rn.f32(output, s, c);
        store(read, at, output);
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn backward<const BOUNDED: bool>(
        dgates: *mut f32,
        gates: *const f32,
        states: *const f32,
        dread: *const f32,
        length: u32,
    ) {
        let (row, start, lane, at): (u32, u32, u32, u32);
        mov.u32(start, ctaid.x);
        mov.u32(lane, tid.x);
        mul.lo.u32(start, start, length);
        add.u32(row, start, length);
        let (carry, dy, ds, s, prev, a, b, c, d): (f32, f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(carry, 0.0);
        label(scan);
        sub.u32(row, row, 1);
        mad.lo.u32(at, row, 32, lane);
        load(dy, dread, at);
        load(s, states, at);
        mov.f32(prev, 0.0);
        let more: pred;
        setp.gt.u32(more, row, start);
        if more {
            sub.u32(at, at, 32);
            load(prev, states, at);
        }
        mad.lo.u32(at, row, 96, lane);
        add.u32(at, at, 64);
        load(c, gates, at);
        fma.rn.f32(ds, dy, c, carry);
        sub.rn.f32(d, 1.0, c);
        mul.rn.f32(d, d, c);
        mul.rn.f32(d, d, s);
        mul.rn.f32(d, d, dy);
        store(dgates, at, d);
        sub.u32(at, at, 32);
        load(b, gates, at);
        let forget_at: u32;
        sub.u32(forget_at, at, 32);
        load(a, gates, forget_at);
        mul.rn.f32(d, b, b);
        sub.rn.f32(d, 1.0, d);
        let part: f32;
        if BOUNDED {
            sub.rn.f32(part, 1.0, a);
            mul.rn.f32(d, d, part);
        }
        mul.rn.f32(d, d, ds);
        store(dgates, at, d);
        mul.rn.f32(carry, ds, a);
        if BOUNDED {
            sub.rn.f32(d, 1.0, a);
            mul.rn.f32(d, d, a);
            sub.rn.f32(part, prev, b);
            mul.rn.f32(d, d, part);
        } else {
            mul.rn.f32(d, a, a);
            sub.rn.f32(d, 1.0, d);
            mul.rn.f32(d, d, prev);
        }
        mul.rn.f32(d, d, ds);
        store(dgates, forget_at, d);
        if more {
            bra(scan);
        }
    }
}
