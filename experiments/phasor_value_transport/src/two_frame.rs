//! Half-split rotations before and after one shared dense map.
//! The scalar controls come from the previous context in the same sequence.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, p: u64, i: u32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        ld.global.f32(x, [addr]);
    }
    fn store(p: u64, i: u32, x: f32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        st.global.f32([addr], x);
    }
    pub unsafe extern "ptx-kernel" fn positions(
        input: *mut f32,
        output: *mut f32,
        gain: *mut f32,
        controls: *const f32,
        length: u32,
    ) {
        let (row, t, at): (u32, u32, u32);
        mov.u32(row, ctaid.x);
        rem.u32(t, row, length);
        let (a, b, s): (f32, f32, f32);
        mov.f32(a, 0.0);
        mov.f32(b, 0.0);
        mov.f32(s, 1.0);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if first {
            bra(write);
        }
        sub.u32(at, row, 1);
        mul.lo.u32(at, at, 3);
        load(a, controls, at);
        add.u32(at, at, 1);
        load(b, controls, at);
        add.u32(at, at, 1);
        load(s, controls, at);
        add.rn.f32(s, s, 1.0);
        label(write);
        store(input, row, a);
        store(output, row, b);
        store(gain, row, s);
    }

    // rot stores the unscaled result for the exact reverse rotation. The
    // outer stage also applies the row gain and the strict prefix's bias.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn forward<const OUTER: bool>(
        rot: *mut f32,
        feature: *mut f32,
        v: *const f32,
        angles: *const f32,
        gains: *const f32,
        bias: *const f32,
        length: u32,
    ) {
        let (r, j, m, d, lo, hi, i, tmp): (u32, u32, u32, u32, u32, u32, u32, u32);
        mov.u32(r, ctaid.x);
        mov.u32(j, tid.x);
        mov.u32(m, ntid.x);
        shl.b32(d, m, 1);
        mad.lo.u32(lo, r, d, j);
        add.u32(hi, lo, m);
        mad.lo.u32(i, r, m, j);
        let (a, s, c, x, y, re, im, gr, br, bi): (
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
        load(a, angles, i);
        sin.approx.f32(s, a);
        cos.approx.f32(c, a);
        load(x, v, lo);
        load(y, v, hi);
        mul.rn.f32(re, s, y);
        neg.f32(re, re);
        fma.rn.f32(re, c, x, re);
        mul.rn.f32(im, s, x);
        fma.rn.f32(im, c, y, im);
        store(rot, lo, re);
        store(rot, hi, im);
        if OUTER {
            load(gr, gains, r);
            mov.f32(br, 0.0);
            mov.f32(bi, 0.0);
            rem.u32(tmp, r, length);
            let prior: pred;
            setp.ne.u32(prior, tmp, 0);
            if prior {
                sub.u32(i, lo, d);
                sub.u32(tmp, hi, d);
                load(br, bias, i);
                load(bi, bias, tmp);
            }
            fma.rn.f32(re, gr, re, br);
            fma.rn.f32(im, gr, im, bi);
            store(feature, lo, re);
            store(feature, hi, im);
        }
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn backward<const OUTER: bool>(
        dx: *mut f32,
        dangle: *mut f32,
        grad: *const f32,
        rotated: *const f32,
        angles: *const f32,
        gains: *const f32,
    ) {
        let (r, j, m, d, lo, hi, i): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(r, ctaid.x);
        mov.u32(j, tid.x);
        mov.u32(m, ntid.x);
        shl.b32(d, m, 1);
        mad.lo.u32(lo, r, d, j);
        add.u32(hi, lo, m);
        mad.lo.u32(i, r, m, j);
        let (a, s, c, x, y, g, h, re, im, torque): (
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
        load(g, grad, lo);
        load(h, grad, hi);
        if OUTER {
            load(a, gains, r);
            mul.rn.f32(g, g, a);
            mul.rn.f32(h, h, a);
        }
        load(a, angles, i);
        sin.approx.f32(s, a);
        cos.approx.f32(c, a);
        load(x, rotated, lo);
        load(y, rotated, hi);
        mul.rn.f32(torque, g, y);
        neg.f32(torque, torque);
        fma.rn.f32(torque, h, x, torque);
        store(dangle, i, torque);
        mul.rn.f32(re, s, h);
        fma.rn.f32(re, c, g, re);
        mul.rn.f32(im, s, g);
        neg.f32(im, im);
        fma.rn.f32(im, c, h, im);
        store(dx, lo, re);
        store(dx, hi, im);
    }

    // Control row r contributes to r+1. Its gain derivative is the dot
    // product with the unscaled output. Every terminal gradient is zero.
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "ptx-kernel" fn controls_back(
        dcontrols: *mut f32,
        dbias: *mut f32,
        input: *const f32,
        output: *const f32,
        upstream: *const f32,
        rot: *const f32,
        d: u32,
        length: u32,
    ) {
        let (r, t, next, at, lo, hi, j): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(r, ctaid.x);
        add.u32(next, r, 1);
        rem.u32(t, next, length);
        mul.lo.u32(lo, r, d);
        mul.lo.u32(hi, next, d);
        let (a, b, s, x, y): (f32, f32, f32, f32, f32);
        mov.f32(a, 0.0);
        mov.f32(b, 0.0);
        mov.f32(s, 0.0);
        let prior: pred;
        setp.ne.u32(prior, t, 0);
        if prior {
            load(a, input, next);
            load(b, output, next);
        }
        mov.u32(j, 0);
        label(dot);
        mov.f32(x, 0.0);
        if prior {
            load(x, upstream, hi);
            load(y, rot, hi);
            fma.rn.f32(s, x, y, s);
        }
        store(dbias, lo, x);
        add.u32(lo, lo, 1);
        add.u32(hi, hi, 1);
        add.u32(j, j, 1);
        let more: pred;
        setp.lt.u32(more, j, d);
        if more {
            bra(dot);
        }
        mul.lo.u32(at, r, 3);
        store(dcontrols, at, a);
        add.u32(at, at, 1);
        store(dcontrols, at, b);
        add.u32(at, at, 1);
        store(dcontrols, at, s);
    }
}
