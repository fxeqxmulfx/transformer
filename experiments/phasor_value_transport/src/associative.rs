//! Eight key coordinates and four value rows fit in one warp. Each lane
//! owns one matrix entry for an entire sequence, including its reverse pass.
pub const KEYS: usize = 8;
pub const VALUES: usize = 4;
pub const PROJECTED: usize = 2 * KEYS + VALUES;

#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.f32(x, [p]);
    }
    fn store(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }
    fn dot8(sum: f32, a: f32, b: f32) {
        let part: f32;
        mul.rn.f32(sum, a, b);
        for s in 0..3 {
            shfl.sync.bfly.b32(part, sum, 1 << s, 31, 0xffffffff);
            add.rn.f32(sum, sum, part);
        }
    }
    fn sum4(total: f32, val: f32) {
        let part: f32;
        shfl.sync.bfly.b32(part, val, 8, 31, 0xffffffff);
        add.rn.f32(total, val, part);
        shfl.sync.bfly.b32(part, total, 16, 31, 0xffffffff);
        add.rn.f32(total, total, part);
    }

    // Normalize the key and query in place. Cache their inverse norms for
    // the transpose of x -> x / sqrt(1 + dot(x,x)). Values pass through.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn bound<const BACK: bool>(
        projected: *mut f32,
        gradient: *mut f32,
        inverse: *mut f32,
    ) {
        let (row, lane, i, addr, group, slot): (u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        and.b32(i, lane, 15);
        shr.u32(group, i, 3);
        mad.lo.u32(slot, row, 2, group);
        mad.lo.u32(addr, row, 20, i);
        let (v, g, scale, dot, out): (f32, f32, f32, f32, f32);
        load(v, projected, addr);
        if BACK {
            load(g, gradient, addr);
            load(scale, inverse, slot);
            dot8(dot, g, v);
            neg.f32(dot, dot);
            fma.rn.f32(out, dot, v, g);
            mul.rn.f32(out, out, scale);
        } else {
            dot8(dot, v, v);
            add.rn.f32(dot, dot, 1.0);
            rsqrt.approx.f32(scale, dot);
            mul.rn.f32(out, v, scale);
        }
        let discard: pred;
        setp.ge.u32(discard, lane, 16);
        if discard {
            ret();
        }
        if BACK {
            store(gradient, addr, out);
        } else {
            store(projected, addr, out);
            and.b32(i, lane, 7);
            setp.ne.u32(discard, i, 0);
            if discard {
                ret();
            }
            store(inverse, slot, scale);
        }
    }

    pub unsafe extern "ptx-kernel" fn forward(
        read: *mut f32,
        states: *mut f32,
        projected: *const f32,
        length: u32,
    ) {
        let (row, end, lane, key, value, base, addr): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        mov.u32(lane, tid.x);
        and.b32(key, lane, 7);
        shr.u32(value, lane, 3);
        let (state, previous_key, v, q, dot, error): (f32, f32, f32, f32, f32, f32);
        mov.f32(state, 0.0);
        mov.f32(previous_key, 0.0);
        label(scan);
        mul.lo.u32(base, row, 20);
        add.u32(addr, base, value);
        add.u32(addr, addr, 16);
        load(v, projected, addr);
        dot8(dot, state, previous_key);
        sub.rn.f32(error, v, dot);
        fma.rn.f32(state, error, previous_key, state);
        mad.lo.u32(addr, row, 32, lane);
        store(states, addr, state);
        add.u32(addr, base, key);
        load(previous_key, projected, addr);
        add.u32(addr, addr, 8);
        load(q, projected, addr);
        dot8(dot, state, q);
        let writer: pred;
        setp.eq.u32(writer, key, 0);
        if writer {
            mad.lo.u32(addr, row, 4, value);
            store(read, addr, dot);
        }
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }

    // Reverse every write and its following query. The key of the last
    // position has no successor. Each projected gradient has one writer.
    pub unsafe extern "ptx-kernel" fn backward(
        gradient: *mut f32,
        upstream: *const f32,
        states: *const f32,
        projected: *const f32,
        length: u32,
    ) {
        let (row, start, lane, key, value, base, addr): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(start, ctaid.x);
        mul.lo.u32(start, start, length);
        add.u32(row, start, length);
        sub.u32(row, row, 1);
        mov.u32(lane, tid.x);
        and.b32(key, lane, 7);
        shr.u32(value, lane, 3);
        let (first_value, first_key, earlier): (pred, pred, pred);
        setp.eq.u32(first_value, value, 0);
        setp.eq.u32(first_key, key, 0);
        let (adj, state, old, k, q, v, g, tmp, dq, dv, dk, error): (
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
            f32,
            f32,
        );
        mov.f32(adj, 0.0);
        if first_value {
            mad.lo.u32(addr, row, 20, key);
            store(gradient, addr, adj);
        }
        label(scan);
        mul.lo.u32(base, row, 20);
        mad.lo.u32(addr, row, 4, value);
        load(g, upstream, addr);
        mad.lo.u32(addr, row, 32, lane);
        load(state, states, addr);
        mov.f32(old, 0.0);
        mov.f32(k, 0.0);
        setp.gt.u32(earlier, row, start);
        if earlier {
            sub.u32(addr, addr, 32);
            load(old, states, addr);
            add.u32(addr, base, key);
            sub.u32(addr, addr, 20);
            load(k, projected, addr);
        }
        add.u32(addr, base, key);
        add.u32(addr, addr, 8);
        load(q, projected, addr);
        fma.rn.f32(adj, g, q, adj);
        mul.rn.f32(tmp, g, state);
        sum4(dq, tmp);
        if first_value {
            store(gradient, addr, dq);
        }
        dot8(dv, adj, k);
        add.u32(addr, base, value);
        add.u32(addr, addr, 16);
        load(v, projected, addr);
        if first_key {
            store(gradient, addr, dv);
        }
        dot8(tmp, old, k);
        sub.rn.f32(error, v, tmp);
        neg.f32(dv, dv);
        mul.rn.f32(tmp, dv, old);
        fma.rn.f32(tmp, error, adj, tmp);
        sum4(dk, tmp);
        fma.rn.f32(adj, dv, k, adj);
        if !earlier {
            ret();
        }
        if first_value {
            add.u32(addr, base, key);
            sub.u32(addr, addr, 20);
            store(gradient, addr, dk);
        }
        sub.u32(row, row, 1);
        bra(scan);
    }
}
