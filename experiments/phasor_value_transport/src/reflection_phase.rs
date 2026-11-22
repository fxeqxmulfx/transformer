//! Shared learned increments drive a rotation and a reflection. The gate
//! is shared across all positions and both read coordinates of its channel.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(value: f32, base: u64, index: u32) {
        let address: u64;
        mad.wide.u32(address, index, 4, base);
        ld.global.f32(value, [address]);
    }
    fn store(base: u64, index: u32, value: f32) {
        let address: u64;
        mad.wide.u32(address, index, 4, base);
        st.global.f32([address], value);
    }
    pub unsafe extern "ptx-kernel" fn forward(
        read: *mut f32,
        rotation: *mut f32,
        reflection: *mut f32,
        gates: *const f32,
        length: u32,
    ) {
        let (row, end, lane, at, out): (u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        mov.u32(lane, tid.x);
        let (plus, minus, step, gate, s, c, sr, cr, mixed): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        load(gate, gates, lane);
        mov.f32(plus, 0.0);
        mov.f32(minus, 0.0);
        label(scan);
        mad.lo.u32(at, row, 32, lane);
        mad.lo.u32(out, row, 64, lane);
        load(step, rotation, at);
        add.rn.f32(plus, plus, step);
        sub.rn.f32(minus, step, minus);
        store(rotation, at, plus);
        store(reflection, at, minus);
        sin.approx.f32(s, plus);
        sin.approx.f32(sr, minus);
        sub.rn.f32(mixed, sr, s);
        fma.rn.f32(mixed, gate, mixed, s);
        store(read, out, mixed);
        cos.approx.f32(c, plus);
        cos.approx.f32(cr, minus);
        sub.rn.f32(mixed, cr, c);
        sub.rn.f32(c, c, 1.0);
        fma.rn.f32(mixed, gate, mixed, c);
        add.u32(out, out, 32);
        store(read, out, mixed);
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }

    pub unsafe extern "ptx-kernel" fn backward(
        increments: *mut f32,
        partial: *mut f32,
        rotation: *const f32,
        reflection: *const f32,
        dread: *const f32,
        gates: *const f32,
        length: u32,
    ) {
        let (seq, row, start, lane, at, out): (u32, u32, u32, u32, u32, u32);
        mov.u32(seq, ctaid.x);
        mul.lo.u32(start, seq, length);
        add.u32(row, start, length);
        mov.u32(lane, tid.x);
        let (plus, minus, gate, rotate_weight, gs, gc, s, c, sr, cr): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (dplus, dminus, dgate, temp, local): (f32, f32, f32, f32, f32);
        load(gate, gates, lane);
        sub.rn.f32(rotate_weight, 1.0, gate);
        mov.f32(dplus, 0.0);
        mov.f32(dminus, 0.0);
        mov.f32(dgate, 0.0);
        label(scan);
        sub.u32(row, row, 1);
        mad.lo.u32(at, row, 32, lane);
        mad.lo.u32(out, row, 64, lane);
        load(plus, rotation, at);
        load(minus, reflection, at);
        load(gs, dread, out);
        add.u32(out, out, 32);
        load(gc, dread, out);
        sin.approx.f32(s, plus);
        cos.approx.f32(c, plus);
        sin.approx.f32(sr, minus);
        cos.approx.f32(cr, minus);
        sub.rn.f32(temp, sr, s);
        fma.rn.f32(dgate, gs, temp, dgate);
        sub.rn.f32(temp, cr, c);
        fma.rn.f32(dgate, gc, temp, dgate);
        mul.rn.f32(local, gc, s);
        neg.f32(local, local);
        fma.rn.f32(local, gs, c, local);
        fma.rn.f32(dplus, rotate_weight, local, dplus);
        mul.rn.f32(local, gc, sr);
        neg.f32(local, local);
        fma.rn.f32(local, gs, cr, local);
        neg.f32(dminus, dminus);
        fma.rn.f32(dminus, gate, local, dminus);
        add.rn.f32(temp, dplus, dminus);
        store(increments, at, temp);
        let more: pred;
        setp.gt.u32(more, row, start);
        if more {
            bra(scan);
        }
        mad.lo.u32(at, seq, 32, lane);
        store(partial, at, dgate);
    }

    // A single lane sums the partials for one gate in stable batch order.
    pub unsafe extern "ptx-kernel" fn gate_sum(out: *mut f32, partial: *const f32, batch: u32) {
        let (lane, seq, at): (u32, u32, u32);
        mov.u32(lane, tid.x);
        mov.u32(seq, 0);
        let (sum, value): (f32, f32);
        mov.f32(sum, 0.0);
        label(scan);
        mad.lo.u32(at, seq, 32, lane);
        load(value, partial, at);
        add.rn.f32(sum, sum, value);
        add.u32(seq, seq, 1);
        let more: pred;
        setp.lt.u32(more, seq, batch);
        if more {
            bra(scan);
        }
        store(out, lane, sum);
    }
}
