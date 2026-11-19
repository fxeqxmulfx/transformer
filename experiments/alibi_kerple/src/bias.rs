//! Distance biases added after the SSMax gain. KERPLE's tables, gradients and
//! clamps run here; the fixed tables (ALiBi, -ln(1+r), NoPE) are written by
//! the host, `layout::fixed_bias`.
//!
//! A table holds b_h(r) at `h * length + r` for the current length. KERPLE's
//! parameters are one pair (r1, r2) per head, shared by all layers (KERPLE
//! Figure 1; released code `megatron/mpu/layers.py` at 25be3d71):
//! log: b(r) = -r1 ln(1 + r2 r); power: b(r) = -r1 r^r2.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(v, [a]);
    }
    fn store(p: u64, i: u32, v: f32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], v);
    }
    /// The derivative of b(r) in r1 and r2 at distance `x`, with b(r) itself.
    /// At r = 0 both kinds give b = -0, so the forward adds nothing.
    fn kernel<const POWER: bool>(b: f32, d1: f32, d2: f32, x: f32, scale: f32, shape: f32) {
        let (y, l): (f32, f32);
        if POWER {
            // r^r2 = 2^(r2 log2 r); ln r = log2 r ln 2.
            lg2.approx.f32(l, x);
            mul.rn.f32(y, shape, l);
            ex2.approx.f32(y, y);
            mul.rn.f32(l, l, 0.6931471805599453);
            neg.f32(d1, y);
            mul.rn.f32(d2, scale, y);
            mul.rn.f32(d2, d2, l);
            neg.f32(d2, d2);
        } else {
            // 1 + r2 r and its natural logarithm.
            fma.rn.f32(y, shape, x, 1.0);
            lg2.approx.f32(l, y);
            mul.rn.f32(l, l, 0.6931471805599453);
            neg.f32(d1, l);
            mul.rn.f32(d2, scale, x);
            div.rn.f32(d2, d2, y);
            neg.f32(d2, d2);
        }
        mul.rn.f32(b, scale, d1);
    }
    /// table[h * length + r] = b_h(r) from (r1, r2) = parameters[2h], [2h + 1].
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn table<const POWER: bool>(
        table: *mut f32,
        parameters: *const f32,
        length: u32,
        count: u32,
    ) {
        let (i, block, thread, n, h, r, at): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(thread, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, block, n, thread);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        div.u32(h, i, length);
        rem.u32(r, i, length);
        let (scale, shape, x, b, d1, d2): (f32, f32, f32, f32, f32, f32);
        mul.lo.u32(at, h, 2);
        load(scale, parameters, at);
        add.u32(at, at, 1);
        load(shape, parameters, at);
        cvt.rn.f32.u32(x, r);
        kernel::<POWER>(b, d1, d2, x, scale, shape);
        store(table, i, b);
    }
    /// parts[h * length + r] = the sum over rows b and queries t >= r of the
    /// logit gradient at key t - r, read as ds / gain(t) with the forward's
    /// own SSMax gain ln(t + 1) / D; r = 0 holds 0. One warp per (h, r):
    /// lanes stride the (b, t) pairs in a fixed order, then a butterfly.
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn distance_sums<const D: usize>(
        parts: *mut f32,
        ds: *const f32,
        heads: u32,
        length: u32,
        batch: u32,
    ) {
        let (id, lane, h, r, b, t, span, pairs, k, at): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        div.u32(h, id, length);
        rem.u32(r, id, length);
        sub.u32(span, length, r);
        mul.lo.u32(pairs, batch, span);
        let origin: pred;
        setp.eq.u32(origin, r, 0);
        if origin {
            mov.u32(pairs, 0);
        }
        let (s, v, gain, x): (f32, f32, f32, f32);
        mov.f32(s, 0.0);
        mov.u32(k, lane);
        label(pairs_loop);
        let done: pred;
        setp.ge.u32(done, k, pairs);
        if done {
            bra(pairs_done);
        }
        div.u32(b, k, span);
        rem.u32(t, k, span);
        add.u32(t, t, r);
        mad.lo.u32(at, b, heads, h);
        mad.lo.u32(at, at, length, t);
        mul.lo.u32(at, at, length);
        add.u32(at, at, t);
        sub.u32(at, at, r);
        load(v, ds, at);
        add.u32(at, t, 1);
        cvt.rn.f32.u32(gain, at);
        lg2.approx.f32(gain, gain);
        mul.rn.f32(gain, gain, 0.6931471805599453 / (D as f32));
        div.rn.f32(v, v, gain);
        add.rn.f32(s, s, v);
        add.u32(k, k, 32);
        bra(pairs_loop);
        label(pairs_done);
        bar.warp.sync(0xffffffff);
        for j in 0..5 {
            shfl.sync.bfly.b32(x, s, 1 << j, 31, 0xffffffff);
            add.rn.f32(s, s, x);
        }
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            store(parts, id, s);
        }
    }
    /// gradient[2h], [2h + 1] = the sum over 1 <= r < length of parts[h *
    /// length + r] times d b_h(r) / d(r1, r2), plus their old values unless
    /// `first`. One thread per head; layers add in the backward's order.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn parameter_gradient<const POWER: bool>(
        gradient: *mut f32,
        parts: *const f32,
        parameters: *const f32,
        heads: u32,
        length: u32,
        first: u32,
    ) {
        let (h, r, at): (u32, u32, u32);
        mov.u32(h, tid.x);
        let past: pred;
        setp.ge.u32(past, h, heads);
        if past {
            ret();
        }
        let (scale, shape, x, b, d1, d2, g, g1, g2): (f32, f32, f32, f32, f32, f32, f32, f32, f32);
        mul.lo.u32(at, h, 2);
        load(scale, parameters, at);
        add.u32(at, at, 1);
        load(shape, parameters, at);
        mov.f32(g1, 0.0);
        mov.f32(g2, 0.0);
        mov.u32(r, 1);
        label(distances);
        let done: pred;
        setp.ge.u32(done, r, length);
        if done {
            bra(distances_done);
        }
        mad.lo.u32(at, h, length, r);
        load(g, parts, at);
        cvt.rn.f32.u32(x, r);
        kernel::<POWER>(b, d1, d2, x, scale, shape);
        fma.rn.f32(g1, g, d1, g1);
        fma.rn.f32(g2, g, d2, g2);
        add.u32(r, r, 1);
        bra(distances);
        label(distances_done);
        mul.lo.u32(at, h, 2);
        let accumulate: pred;
        setp.eq.u32(accumulate, first, 0);
        if accumulate {
            load(g, gradient, at);
            add.rn.f32(g1, g1, g);
            add.u32(at, at, 1);
            load(g, gradient, at);
            add.rn.f32(g2, g2, g);
            mul.lo.u32(at, h, 2);
        }
        store(gradient, at, g1);
        add.u32(at, at, 1);
        store(gradient, at, g2);
    }
    /// KERPLE's clamps, applied after each update as its forward applies them
    /// before each use: every parameter >= 0.01, the power exponent r2 <= 2.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn project<const POWER: bool>(parameters: *mut f32, count: u32) {
        let (i, parity): (u32, u32);
        mov.u32(i, tid.x);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (v, capped): (f32, f32);
        load(v, parameters, i);
        max.f32(v, v, 0.01);
        if POWER {
            and.b32(parity, i, 1);
            let exponent: pred;
            setp.eq.u32(exponent, parity, 1);
            min.f32(capped, v, 2.0);
            selp.f32(v, capped, v, exponent);
        }
        store(parameters, i, v);
    }
}
